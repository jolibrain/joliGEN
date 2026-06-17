import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import math


class B2BGenerator(nn.Module):
    """
    Handles both target generation and model forward prediction internally.
    """

    def __init__(
        self,
        b2b_model,
        sampling_method,
        image_size,
        G_ngf,
        opt=None,
    ):
        super().__init__()
        self.b2b_model = b2b_model
        self.sampling_method = sampling_method
        self.image_size = image_size
        self.opt = opt
        self.P_mean = float(getattr(opt, "alg_b2b_P_mean", -0.8)) if opt else -0.8
        self.P_std = float(getattr(opt, "alg_b2b_P_std", 0.8)) if opt else 0.8
        self.timestep_uniform_mix_prob = (
            float(getattr(opt, "alg_b2b_timestep_uniform_mix_prob", 0.1))
            if opt
            else 0.1
        )
        requested_noise_scale = (
            getattr(opt, "alg_b2b_noise_scale", -1.0) if opt else -1.0
        )
        if requested_noise_scale > 0:
            self.noise_scale = float(requested_noise_scale)
        else:
            self.noise_scale = 1.0 if int(self.image_size) <= 256 else 2.0
        self.t_eps = float(getattr(opt, "alg_b2b_t_eps", 5e-2)) if opt else 5e-2
        self.current_t = 1
        self.cfg_scale = float(getattr(opt, "alg_b2b_cfg_scale", 1.0)) if opt else 1.0
        self.cfg_interval = (0.1, 1.0)  # value used in paper training examples
        self.clip_denoised_default = (
            bool(getattr(opt, "alg_b2b_clip_denoised", False)) if opt else False
        )
        self.disable_inference_clipping = (
            bool(getattr(opt, "alg_b2b_disable_inference_clipping", False))
            if opt
            else False
        )
        self.mask_size_conditioning = (
            bool(getattr(opt, "alg_b2b_mask_size_conditioning", False))
            if opt
            else False
        )
        self.global_context_conditioning = (
            bool(getattr(opt, "alg_b2b_global_context_conditioning", False))
            if opt
            else False
        )

        self.denoise_timesteps = (
            getattr(opt, "alg_b2b_denoise_timesteps", 50) if opt else 50
        )
        self.num_classes = getattr(opt, "G_vit_num_classes", 1) if opt else 1

        self.label_drop_prob = (
            float(getattr(opt, "alg_diffusion_dropout_prob", 0.0)) if opt else 0.0
        )
        self.label_drop_prob = min(max(self.label_drop_prob, 0.0), 1.0)

    def _mask_size_condition(self, mask, reference):
        if not self.mask_size_conditioning:
            return None
        if mask is None:
            return None

        if reference.ndim == 4:
            B, _, H, W = reference.shape
            if mask.ndim != 4:
                raise RuntimeError(
                    f"Expected image mask rank 4, got {tuple(mask.shape)}"
                )
            mask = mask[:, :1]
            target_shape = (B, 1, H, W)
            leading_shape = (B,)
        elif reference.ndim == 5:
            B, F, _, H, W = reference.shape
            if mask.ndim == 4:
                mask = mask[:, None]
            if mask.ndim != 5:
                raise RuntimeError(
                    f"Expected video mask rank 5, got {tuple(mask.shape)}"
                )
            mask = mask[:, :, :1]
            target_shape = (B, F, 1, H, W)
            leading_shape = (B, F)
        else:
            raise RuntimeError(
                f"Expected reference rank 4 or 5, got {tuple(reference.shape)}"
            )

        leading_target_shape = target_shape[:-2]
        if any(
            src not in (1, dst)
            for src, dst in zip(mask.shape[:-2], leading_target_shape)
        ):
            raise RuntimeError(
                f"Mask shape {tuple(mask.shape)} cannot broadcast to {target_shape}"
            )

        if mask.shape[-2:] != (H, W):
            flat_mask = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1])
            flat_mask = Fnn.interpolate(flat_mask.float(), size=(H, W), mode="nearest")
            mask = flat_mask.reshape(*mask.shape[:-2], H, W)

        mask = mask.expand(target_shape)
        flat = (mask > 0.5).reshape(-1, H, W)
        dtype = reference.dtype if reference.is_floating_point() else torch.float32
        device = reference.device

        positive = flat.any(dim=(1, 2))
        area = flat.to(dtype=dtype).mean(dim=(1, 2))

        y_any = flat.any(dim=2)
        x_any = flat.any(dim=1)
        y_idx = torch.arange(H, device=device, dtype=dtype)
        x_idx = torch.arange(W, device=device, dtype=dtype)
        y_min = (
            torch.where(y_any, y_idx[None], torch.full_like(y_idx[None], H))
            .min(1)
            .values
        )
        y_max = (
            torch.where(y_any, y_idx[None], torch.full_like(y_idx[None], -1))
            .max(1)
            .values
        )
        x_min = (
            torch.where(x_any, x_idx[None], torch.full_like(x_idx[None], W))
            .min(1)
            .values
        )
        x_max = (
            torch.where(x_any, x_idx[None], torch.full_like(x_idx[None], -1))
            .max(1)
            .values
        )

        width = (x_max - x_min + 1.0) / max(1, W)
        height = (y_max - y_min + 1.0) / max(1, H)
        cx = (x_min + x_max + 1.0) / (2.0 * max(1, W))
        cy = (y_min + y_max + 1.0) / (2.0 * max(1, H))
        log_aspect = torch.log((width + 1e-6) / (height + 1e-6)).clamp(-3.0, 3.0) / 3.0

        features = torch.stack([cx, cy, width, height, area, log_aspect], dim=1)
        features = torch.where(
            positive[:, None],
            features,
            torch.zeros_like(features),
        )
        return features.reshape(*leading_shape, 6)

    def _match_prediction_channels(self, pred, reference):
        if pred.ndim != reference.ndim:
            raise RuntimeError(
                f"Prediction dims {pred.ndim} do not match reference dims {reference.ndim}"
            )

        channel_dim = 2 if pred.ndim == 5 else 1
        pred_channels = pred.shape[channel_dim]
        ref_channels = reference.shape[channel_dim]
        if pred_channels == ref_channels:
            return pred
        if pred_channels > ref_channels:
            if channel_dim == 1:
                return pred[:, -ref_channels:, :, :]
            return pred[:, :, -ref_channels:, :, :]
        raise RuntimeError(
            f"Prediction channels ({pred_channels}) are fewer than reference channels ({ref_channels})"
        )

    def sample_t(self, B: int, device, F=None):
        # returns t_cont in [0,1]
        if F is None:
            t_z = torch.randn(B, device=device) * self.P_std + self.P_mean
            t = torch.sigmoid(t_z)  # (B,)
        else:
            t_z = torch.randn(B, F, device=device) * self.P_std + self.P_mean
            t = torch.sigmoid(t_z)  # (B,F)

        if self.timestep_uniform_mix_prob <= 0.0:
            return t
        if self.timestep_uniform_mix_prob >= 1.0:
            return torch.rand_like(t)

        t_uniform = torch.rand_like(t)
        use_uniform = torch.rand_like(t) < self.timestep_uniform_mix_prob
        return torch.where(use_uniform, t_uniform, t)

    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.label_drop_prob <= 0.0:
            return labels
        drop = torch.rand(labels.shape, device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def _model_kwargs(self, mask_size_cond, global_context):
        kwargs = {}
        if self.mask_size_conditioning:
            kwargs["mask_size_cond"] = mask_size_cond
        if self.global_context_conditioning:
            kwargs["global_context"] = global_context
        return kwargs

    def b2b_forward(
        self,
        x,
        mask,
        x_cond=None,
        label=None,
        use_gt=None,
        ref_idx=None,
        global_context=None,
    ):
        labels_dropped = (
            self.drop_labels(label) if self.training and label is not None else label
        )

        # 1) sample timestep
        B = x.size(0)
        is_video = x.ndim == 5

        if not is_video:
            t_cont = self.sample_t(B, device=x.device)  # (B,)
            t = t_cont.view(B, *([1] * (x.ndim - 1)))  # (B,1,1,1)
            t_flat = t_cont
        else:
            F = x.size(1)
            t_base = self.sample_t(B, device=x.device)  # (B,)
            t_cont = t_base[:, None].repeat(1, F)  # (B,F)

            if use_gt is not None and ref_idx is not None and use_gt.any():
                b_idx = torch.arange(B, device=x.device)
                t_cont[b_idx[use_gt], ref_idx[use_gt]] = 1.0

            t = t_cont.view(B, F, 1, 1, 1)
            t_flat = t_cont.reshape(B * F)

        # 2) clean mask
        if mask is not None:
            mask = torch.clamp(mask, min=0.0, max=1.0)

        # 3) noise image only
        e = torch.randn_like(x) * self.noise_scale
        z_t = t * x + (1.0 - t) * e

        if mask is not None:
            z = z_t * mask + (1.0 - mask) * x
        else:
            z = z_t

        # 4) concatenate clean condition after noising
        if x_cond is None:
            z_model = z
        elif len(x.shape) != 5:
            z_model = torch.cat([x_cond, z], dim=1)
        else:
            z_model = torch.cat([x_cond, z], dim=2)

        # 5) target velocity in image space only
        v = (x - z) / (1.0 - t).clamp_min(self.t_eps)

        # 6) predict image
        mask_size_cond = self._mask_size_condition(mask, z_model)
        x_pred = self.b2b_model(
            z_model,
            t_flat,
            labels_dropped,
            **self._model_kwargs(mask_size_cond, global_context),
        )
        x_pred = self._match_prediction_channels(x_pred, x)

        return x_pred, z, v, t, x

    def forward(
        self,
        x,
        mask=None,
        x_cond=None,
        label=None,
        use_gt=None,
        ref_idx=None,
        global_context=None,
        return_x_pred=False,
        return_raw_x_pred=False,
    ):
        x_pred, z, v, t, x_target = self.b2b_forward(
            x, mask, x_cond, label, use_gt, ref_idx, global_context
        )
        raw_x_pred = x_pred
        if mask is not None:
            x_pred = x_pred * mask + (1 - mask) * x_target
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        if return_raw_x_pred:
            return v_pred, v, x_pred, raw_x_pred
        if return_x_pred:
            return v_pred, v, x_pred
        return v_pred, v

    def _project_known_pixels(self, x, y_known, mask):
        if mask is None or y_known is None:
            return x
        return x * mask + y_known * (1.0 - mask)

    def _restoration_model_timesteps(self, t, x, use_gt=None, ref_idx=None):
        if x.ndim != 5:
            return t.flatten()

        B, F = x.shape[:2]
        t_model = t
        while t_model.ndim > 2 and t_model.shape[-1] == 1:
            t_model = t_model.squeeze(-1)

        if t_model.ndim == 1:
            if t_model.shape[0] != B:
                raise RuntimeError(
                    f"Expected timestep batch dim {B}, got {tuple(t_model.shape)}"
                )
            t_model = t_model[:, None].expand(B, F).clone()
        elif t_model.ndim == 2:
            if t_model.shape == (B, 1):
                t_model = t_model.expand(B, F).clone()
            elif t_model.shape == (B, F):
                t_model = t_model.clone()
            else:
                raise RuntimeError(
                    f"Expected video timesteps with shape {(B, F)} or {(B, 1)}, "
                    f"got {tuple(t_model.shape)}"
                )
        else:
            raise RuntimeError(
                "Expected restoration timesteps with 1 or 2 non-singleton dims, "
                f"got {tuple(t.shape)}"
            )

        if use_gt is not None and ref_idx is not None:
            use_gt = use_gt.to(device=x.device, dtype=torch.bool).reshape(-1)
            ref_idx = ref_idx.to(device=x.device, dtype=torch.long).reshape(-1)
            if use_gt.shape[0] != B or ref_idx.shape[0] != B:
                raise RuntimeError(
                    f"Expected use_gt/ref_idx batch dim {B}, got "
                    f"{tuple(use_gt.shape)} and {tuple(ref_idx.shape)}"
                )
            if use_gt.any():
                selected_ref_idx = ref_idx[use_gt]
                if torch.any((selected_ref_idx < 0) | (selected_ref_idx >= F)):
                    raise RuntimeError(
                        f"ref_idx must be in [0, {F - 1}], got {selected_ref_idx}"
                    )
                b_idx = torch.arange(B, device=x.device)
                t_model[b_idx[use_gt], selected_ref_idx] = 1.0

        return t_model.reshape(B * F)

    @torch.no_grad()
    def restoration(
        self,
        y,
        y_cond=None,
        denoise_timesteps=None,
        mask=None,
        labels=None,
        clip_denoised=None,
        use_gt=None,
        ref_idx=None,
        init_noise=None,
        global_context=None,
    ):
        B = y.shape[0]
        device = y.device
        y_known = y if mask is not None else None

        if mask is not None:
            mask = torch.clamp(mask, 0.0, 1.0)
            y_background = y * (1.0 - mask)
        else:
            y_background = y

        if init_noise is None:
            noise = torch.randn_like(y)
        else:
            if init_noise.shape != y.shape:
                raise RuntimeError(
                    f"init_noise shape {init_noise.shape} does not match input shape {y.shape}"
                )
            noise = init_noise.to(device=y.device, dtype=y.dtype)

        x = y_background + noise * self.noise_scale

        if mask is not None:
            x = x * mask + y * (1.0 - mask)

        steps = int(denoise_timesteps)
        if clip_denoised is None:
            clip_denoised = self.clip_denoised_default

        timesteps = (
            torch.linspace(0.0, 1.0, steps + 1, device=device)
            .view(steps + 1, 1, *([1] * (x.ndim - 1)))
            .expand(steps + 1, x.shape[0], *([1] * (x.ndim - 1)))
        )

        # ODE integration
        for i in range(steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            x = self._heun_step_restoration(
                x,
                t,
                t_next,
                y_cond,
                mask,
                labels,
                y_known,
                use_gt,
                ref_idx,
                global_context,
            )

            if clip_denoised:
                x = x.clamp(-1.0, 1.0)
            if mask is not None:
                x = x * mask + y * (1.0 - mask)

        # Last step with euler
        x = self._euler_step_restoration(
            x,
            timesteps[-2],
            timesteps[-1],
            y_cond,
            mask,
            labels,
            y_known,
            use_gt,
            ref_idx,
            global_context,
        )
        if clip_denoised:
            x = x.clamp(-1.0, 1.0)
        if mask is not None:
            x = x * mask + y * (1.0 - mask)

        # Always clamp the final restored sample to a valid image range.
        x = x.clamp(-1.0, 1.0)

        return x

    @torch.no_grad()
    def _forward_sample_restoration(
        self,
        x,
        t,
        y_cond,
        mask,
        labels,
        y_known,
        use_gt=None,
        ref_idx=None,
        global_context=None,
    ):
        """
        JIT-equivalent CFG:
          v = v_uncond + scale(t) * (v_cond - v_uncond)
        """
        x_in = self._project_known_pixels(x, y_known, mask)
        if labels is None:
            labels = torch.zeros(x_in.shape[0], dtype=torch.long, device=x_in.device)
        if y_cond is None:
            model_input = x_in
        elif len(x_in.shape) != 5:
            model_input = torch.cat([y_cond, x_in], dim=1)
        else:
            model_input = torch.cat([y_cond, x_in], dim=2)

        model_t = self._restoration_model_timesteps(t, model_input, use_gt, ref_idx)

        mask_size_cond = self._mask_size_condition(mask, model_input)
        model_kwargs = self._model_kwargs(mask_size_cond, global_context)

        # --- conditional ---
        x_cond = self.b2b_model(model_input, model_t, labels, **model_kwargs)
        x_cond = self._match_prediction_channels(x_cond, x_in)
        x_cond = self._project_known_pixels(x_cond, y_known, mask)

        den = 1.0 - t
        if not self.disable_inference_clipping:
            den = den.clamp_min(self.t_eps)
        v_cond = (x_cond - x_in) / den

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_is_neutral = math.isclose(self.cfg_scale, 1.0, rel_tol=0.0, abs_tol=1e-12)

        # When guidance is neutral (or inactive for this t), CFG exactly equals v_cond.
        if cfg_is_neutral or not torch.any(interval_mask):
            return v_cond

        # --- unconditional ---
        num_classes = int(self.b2b_model.num_classes)
        uncond_labels = torch.full_like(labels, num_classes)
        x_uncond = self.b2b_model(model_input, model_t, uncond_labels, **model_kwargs)
        x_uncond = self._match_prediction_channels(x_uncond, x_in)
        x_uncond = self._project_known_pixels(x_uncond, y_known, mask)
        v_uncond = (x_uncond - x_in) / den

        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    #    @torch.no_grad() # no use CFG
    #    def _forward_sample_restoration(self, x, t, y_cond, mask, labels):
    #        x_pred = self.b2b_model(x, t.flatten(), labels)
    #        return (x_pred - x) / (1.0 - t).clamp_min(self.t_eps)

    @torch.no_grad()
    def _euler_step_restoration(
        self,
        x,
        t,
        t_next,
        y_cond,
        mask,
        labels,
        y_known,
        use_gt=None,
        ref_idx=None,
        global_context=None,
    ):
        v = self._forward_sample_restoration(
            x, t, y_cond, mask, labels, y_known, use_gt, ref_idx, global_context
        )
        return x + (t_next - t) * v

    @torch.no_grad()
    def _heun_step_restoration(
        self,
        x,
        t,
        t_next,
        y_cond,
        mask,
        labels,
        y_known,
        use_gt=None,
        ref_idx=None,
        global_context=None,
    ):
        v_t = self._forward_sample_restoration(
            x, t, y_cond, mask, labels, y_known, use_gt, ref_idx, global_context
        )
        x_euler = x + (t_next - t) * v_t
        v_t_next = self._forward_sample_restoration(
            x_euler,
            t_next,
            y_cond,
            mask,
            labels,
            y_known,
            use_gt,
            ref_idx,
            global_context,
        )
        v = 0.5 * (v_t + v_t_next)
        return x + (t_next - t) * v
