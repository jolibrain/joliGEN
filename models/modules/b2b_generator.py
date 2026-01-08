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
        self.noise_scale = 1.0  # 2.0 when image size 512
        self.t_eps = 5e-2
        self.current_t = 1
        self.cfg_scale = 2.9  # guidance strength as indicated in paper
        self.cfg_interval = (0.1, 1.0)  # value used in paper training examples

        self.denoise_timesteps = (
            getattr(opt, "alg_b2b_denoise_timesteps", 50) if opt else 50
        )
        self.num_classes = getattr(opt, "G_vit_num_classes", 1) if opt else 1
        if self.num_classes != 1:
            raise RuntimeError(
                f"Expected G_vit_num_classes == 1, but got {self.num_classes}. "
                "Stopping because this run only supports num_classes=1."
            )
        self.label_drop_prob = 0.1  # default  value used in paper

    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def b2b_forward(self, x, mask, x_cond=None, label=None):
        labels_dropped = self.drop_labels(label) if self.training else label

        # 2) sample_t timestep
        z_t = torch.randn(x.size(0), device=x.device) * self.P_std + self.P_mean
        t_cont = torch.sigmoid(z_t)
        t = t_cont.view(-1, *([1] * (x.ndim - 1)))

        if not x_cond is None:
            if len(x.shape) != 5:
                x_with_cond = torch.cat([x_cond, x], dim=1)
            else:
                x_with_cond = torch.cat([x_cond, x], dim=2)
        else:
            x_with_cond = x

        # 3) noise + mixing
        if mask is not None:
            mask = torch.clamp(mask, min=0.0, max=1.0)

        e = torch.randn_like(x_with_cond) * self.noise_scale
        z_t = t * x_with_cond + (1 - t) * e

        if mask is not None:
            z = z_t * mask + (1 - mask) * x_with_cond

        v = (x_with_cond - z) / (1 - t).clamp_min(self.t_eps)

        # 4) model predict img
        x_pred = self.b2b_model(z, t.flatten(), labels_dropped)

        return x_pred, z, v, t

    def forward(self, x, mask=None, x_cond=None, label=None):
        device = x.device
        x_pred, z, v, t = self.b2b_forward(x, mask, x_cond, label)
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

        timesteps = (
            torch.linspace(0.0, 1.0, steps + 1, device=device)
            .view(-1, *([1] * (x.ndim)))
            .expand(-1, B, -1, -1, -1)
        )

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
        # --- conditional ---
        x_cond = self.b2b_model(x, t.flatten(), labels)
        v_cond = (x_cond - x) / (1.0 - t).clamp_min(self.t_eps)

        # --- unconditional ---
        x_uncond = self.b2b_model(x, t.flatten(), torch.full_like(labels, num_classes))
        v_uncond = (x_uncond - x) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

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
