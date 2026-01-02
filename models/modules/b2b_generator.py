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
        self.P_mean = -0.8
        self.P_std = 0.8
        self.noise_scale = 2.0
        self.t_eps = 5e-2
        self.current_t = 1
        self.cfg_scale = 1.0  # guidance strength
        self.cfg_interval = (0.0, 1.0)  # apply for all t in (0,1)

        self.denoise_timesteps = (
            getattr(opt, "alg_b2b_denoise_timesteps", 2) if opt else 2
        )
        self.num_classes = getattr(opt, "G_vit_num_classes", 1) if opt else 1
        self.label_drop_prob = 0.0  # float(getattr(opt, "label_drop_prob", 0.0) or 0.1)

    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def b2b_forward(self, x, x_cond=None, label=None):
        if not x_cond is None:
            if len(x.shape) != 5:
                x_with_cond = torch.cat([x_cond, x], dim=1)
            else:
                x_with_cond = torch.cat([x_cond, x], dim=2)
        else:
            x_with_cond = x
        B = x_with_cond.size(0)
        # 2) sample continuous timestep t in (0,1): shape [B]
        z_t = torch.randn(B, device=x_with_cond.device) * self.P_std + self.P_mean
        t_cont = torch.sigmoid(z_t)
        # broadcast t to match x dims for mixing
        t = t_cont.view(-1, *([1] * (x_with_cond.ndim - 1)))

        # 3) noise + mixing
        e = torch.randn_like(x_with_cond) * self.noise_scale
        z = t * x_with_cond + (1.0 - t) * e
        v = (x_with_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # 4) labels safety (JiT label embedder needs long indices)
        num_classes = int(self.num_classes)
        labels_dropped = self.drop_labels(label) if self.training else label

        # 5) model forward
        x_pred = self.b2b_model(z, t_cont, labels_dropped)

        return x_pred, z, v, t

    def forward(self, x, mask=None, x_cond=None, label=None):
        device = x.device
        x_pred, z, v, t = self.b2b_forward(x, x_cond, label)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        return v_pred, v

    @torch.no_grad()
    def restoration(
        self,
        y,
        y_cond=None,
        denoise_timesteps=None,
        mask=None,
        labels=None,
        clip_denoised=True,
    ):
        B, C, H, W = y.shape
        device = y.device

        if mask is not None:
            mask = torch.clamp(mask, 0.0, 1.0)
            y_background = y * (1.0 - mask)
        else:
            y_background = y

        x = y_background + torch.randn_like(y) * self.noise_scale

        if mask is not None:
            x = x * mask + y * (1.0 - mask)

        steps = int(denoise_timesteps)

        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)
        timesteps = timesteps.view(-1, *([1] * (x.ndim))).expand(-1, B, -1, -1, -1)

        if labels is None:
            num_classes = int(self.b2b_model.num_classes)
            labels = torch.full((B,), num_classes, device=device, dtype=torch.long)

        # ODE integration
        for i in range(steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            x = self._heun_step_restoration(x, t, t_next, y_cond, labels)

            if clip_denoised:
                x = x.clamp(-1.0, 1.0)
            if mask is not None:
                x = x * mask + y * (1.0 - mask)

        # Last step with euler
        x = self._euler_step_restoration(
            x, timesteps[-2], timesteps[-1], y_cond, labels
        )
        if clip_denoised:
            x = x.clamp(-1.0, 1.0)
        if mask is not None:
            x = x * mask + y * (1.0 - mask)

        return x

    @torch.no_grad()
    def _forward_sample_restoration(self, x, t, y_cond, labels):
        """
        JIT-equivalent CFG:
          v = v_uncond + scale(t) * (v_cond - v_uncond)
        """
        num_classes = int(self.b2b_model.num_classes)
        B = x.size(0)

        # labels must exist for conditional pass; if None, fall back to unconditional-only
        if labels is None:
            labels = torch.full((B,), num_classes, device=x.device, dtype=torch.long)
        else:
            labels = labels.view(-1).long().to(x.device)
            if labels.numel() != B:
                raise ValueError(
                    f"labels must have B elements; got {labels.numel()} vs B={B}"
                )

        uncond_labels = torch.full_like(labels, num_classes)

        # t comes in as shape [B,1,1,1,1] sometimes; JIT uses t.flatten()
        t_flat = t.flatten().float()

        # --- conditional ---
        x_cond = self.b2b_model(x, t_flat, labels)
        v_cond = (x_cond - x) / (1.0 - t).clamp_min(self.t_eps)

        # --- unconditional ---
        x_uncond = self.b2b_model(x, t_flat, uncond_labels)
        v_uncond = (x_uncond - x) / (1.0 - t).clamp_min(self.t_eps)

        # --- cfg interval scaling ---
        low, high = self.cfg_interval

        # interval_mask uses scalar per-sample t; use t_flat (shape [B])
        interval_mask = (t_flat < high) & ((low == 0.0) | (t_flat > low))

        # scale per sample (shape [B]); broadcast to v dims
        cfg_scale_interval = torch.where(
            interval_mask,
            torch.full_like(t_flat, float(self.cfg_scale)),
            torch.ones_like(t_flat),
        )

        # broadcast cfg_scale_interval to match v tensors (B, C, H, W) or (B,T,C,H,W)
        while cfg_scale_interval.ndim < v_uncond.ndim:
            cfg_scale_interval = cfg_scale_interval.view(
                -1, *([1] * (v_uncond.ndim - 1))
            )

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step_restoration(self, x, t, t_next, y_cond, labels):
        v = self._forward_sample_restoration(x, t, y_cond, labels)
        return x + (t_next - t) * v

    @torch.no_grad()
    def _heun_step_restoration(self, x, t, t_next, y_cond, labels):
        v_t = self._forward_sample_restoration(x, t, y_cond, labels)
        x_euler = x + (t_next - t) * v_t
        v_t_next = self._forward_sample_restoration(x_euler, t_next, y_cond, labels)
        v = 0.5 * (v_t + v_t_next)
        return x + (t_next - t) * v


#
# @torch.no_grad()
# def _forward_sample_restoration(self, x, t, y_cond, labels):
#     num_classes = int(self.b2b_model.num_classes)
#     if labels is None:  # Should not happen if called from restoration
#         B = x.size(0)
#         labels = torch.full((B,), num_classes, device=x.device, dtype=torch.long)

#     uncond_labels = torch.full_like(labels, num_classes)

#     t_vec = t.view(-1).float()

#     x_pred = self.b2b_model(x, t_vec, uncond_labels)

#     v_pred = (x_pred - x) / (1.0 - t).clamp_min(self.t_eps)
#     return v_pred

# @torch.no_grad()
# def _euler_step_restoration(self, x, t, t_next, y_cond, labels):
#     v = self._forward_sample_restoration(x, t, y_cond, labels)
#     x_next = x + (t_next - t) * v
#     return x_next

# @torch.no_grad()
# def _heun_step_restoration(self, x, t, t_next, y_cond, labels):
#     v_t = self._forward_sample_restoration(x, t, y_cond, labels)
#     x_euler = x + (t_next - t) * v_t
#     v_t_next = self._forward_sample_restoration(x_euler, t_next, y_cond, labels)
#     v = 0.5 * (v_t + v_t_next)
#     x_next = x + (t_next - t) * v
#     return x_next
