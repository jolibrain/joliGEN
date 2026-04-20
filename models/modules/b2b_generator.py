import torch
import torch.nn as nn
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

        self.denoise_timesteps = (
            getattr(opt, "alg_b2b_denoise_timesteps", 50) if opt else 50
        )
        self.num_classes = getattr(opt, "G_vit_num_classes", 1) if opt else 1

        self.label_drop_prob = (
            float(getattr(opt, "alg_diffusion_dropout_prob", 0.0)) if opt else 0.0
        )
        self.label_drop_prob = min(max(self.label_drop_prob, 0.0), 1.0)

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
            return torch.sigmoid(t_z)  # (B,)
        else:
            t_z = torch.randn(B, F, device=device) * self.P_std + self.P_mean
            return torch.sigmoid(t_z)  # (B,F)

    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.label_drop_prob <= 0.0:
            return labels
        drop = torch.rand(labels.shape, device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def b2b_forward(self, x, mask, x_cond=None, label=None, use_gt=None, ref_idx=None):
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
        x_pred = self.b2b_model(z_model, t_flat, labels_dropped)
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
        return_x_pred=False,
    ):
        x_pred, z, v, t, x_target = self.b2b_forward(
            x, mask, x_cond, label, use_gt, ref_idx
        )
        if mask is not None:
            x_pred = x_pred * mask + (1 - mask) * x_target
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        if return_x_pred:
            return v_pred, v, x_pred
        return v_pred, v

    def _project_known_pixels(self, x, y_known, mask):
        if mask is None or y_known is None:
            return x
        return x * mask + y_known * (1.0 - mask)

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
            x = self._heun_step_restoration(x, t, t_next, y_cond, mask, labels, y_known)

            if clip_denoised:
                x = x.clamp(-1.0, 1.0)
            if mask is not None:
                x = x * mask + y * (1.0 - mask)

        # Last step with euler
        x = self._euler_step_restoration(
            x, timesteps[-2], timesteps[-1], y_cond, mask, labels, y_known
        )
        if clip_denoised:
            x = x.clamp(-1.0, 1.0)
        if mask is not None:
            x = x * mask + y * (1.0 - mask)

        # Always clamp the final restored sample to a valid image range.
        x = x.clamp(-1.0, 1.0)

        return x

    @torch.no_grad()
    def _forward_sample_restoration(self, x, t, y_cond, mask, labels, y_known):
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

        # --- conditional ---
        x_cond = self.b2b_model(model_input, t.flatten(), labels)
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
        x_uncond = self.b2b_model(
            model_input, t.flatten(), torch.full_like(labels, num_classes)
        )
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
    def _euler_step_restoration(self, x, t, t_next, y_cond, mask, labels, y_known):
        v = self._forward_sample_restoration(x, t, y_cond, mask, labels, y_known)
        return x + (t_next - t) * v

    @torch.no_grad()
    def _heun_step_restoration(self, x, t, t_next, y_cond, mask, labels, y_known):
        v_t = self._forward_sample_restoration(x, t, y_cond, mask, labels, y_known)
        x_euler = x + (t_next - t) * v_t
        v_t_next = self._forward_sample_restoration(
            x_euler, t_next, y_cond, mask, labels, y_known
        )
        v = 0.5 * (v_t + v_t_next)
        return x + (t_next - t) * v
