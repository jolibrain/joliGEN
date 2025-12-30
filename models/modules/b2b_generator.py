import torch
import torch.nn as nn
import math


class B2BGenerator(nn.Module):
    """
    Shortcut Consistency Generator (CM-style)
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
        self.noise_scale = 1.0
        self.t_eps = 5e-2
        self.current_t = 1
        self.denoise_timesteps = (
            getattr(opt, "alg_b2b_denoise_timesteps", 2) if opt else 2
        )

    def b2b_forward(self, x, x_cond=None, y=None):
        # 1) build conditioned input
        if not x_cond is None:
            if len(x.shape) != 5:
                x_with_cond = torch.cat([x_cond, x], dim=1)
            else:
                x_with_cond = torch.cat([x_cond, x], dim=2)
        else:
            x_with_cond = torch.cat([x, x], dim=-3)
        B = x_with_cond.size(0)
        # 2) sample continuous timestep t in (0,1): shape [B]
        z_t = torch.randn(B, device=x_with_cond.device) * self.P_std + self.P_mean
        t_cont = torch.sigmoid(z_t)  # [B] float

        # broadcast t to match x dims for mixing
        t = t_cont.view(-1, *([1] * (x_with_cond.ndim - 1)))  # [B,1,1,1] or [B,1,1,1,1]

        # 3) noise + mixing
        e = torch.randn_like(x_with_cond) * self.noise_scale
        z = t * x_with_cond + (1.0 - t) * e
        v = (x_with_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # 4) labels safety (JiT label embedder needs long indices)
        num_classes = int(self.b2b_model.num_classes)

        if y is None:
            # unconditional / dropped label
            y = torch.full((B,), num_classes, device=x.device, dtype=torch.long)
        else:
            y = y.view(-1).long()
            if y.numel() != B:
                raise ValueError(f"y must have B elements; got {y.numel()} vs B={B}")
            y = y.clamp(0, num_classes)

        # 5) model forward
        # JiT expects: x: [B,C,H,W], t: [B] (float), y: [B] (long)
        x_pred = self.b2b_model(z, t_cont, y=y)

        return x_pred, z, v, t

    def forward(self, x, mask=None, x_cond=None, y=None):
        device = x.device
        x_pred, z, v, t = self.b2b_forward(x, x_cond, y=y)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        return v_pred, v


#
#    @torch.no_grad()
#    def restoration(self, y, denoise_timesteps=None, y_cond=None, mask=None, clip_denoised=True, labels=None):
#        B, C, H, W = y.shape
#        device = y.device
#
#        # Start from pure noise, but keep known areas from `y`.
#        x = torch.randn_like(y)
#        if mask is not None:
#            mask = torch.clamp(mask, min=0.0, max=1.0)
#            x = x * mask + y * (1.0 - mask)
#
#        if denoise_timesteps is None:
#            denoise_timesteps = self.denoise_timesteps
#        if isinstance(denoise_timesteps, list):
#            denoise_timesteps = denoise_timesteps[0]
#        steps = int(denoise_timesteps)
#
#        # Timesteps from 0 to 1, as in user's snippet for generation.
#        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)
#        timesteps = timesteps.view(-1, *([1] * (x.ndim - 1))).expand(-1, B, -1, -1, -1)
#
#        if labels is None:
#            num_classes = int(self.b2b_model.num_classes)
#            labels = torch.full((B,), num_classes, device=device, dtype=torch.long)
#
#        stepper = self._heun_step_restoration
#
#        # ODE integration
#        for i in range(steps - 1):
#            t = timesteps[i]
#            t_next = timesteps[i + 1]
#            x = stepper(x, t, t_next, y_cond, labels)
#
#            if clip_denoised:
#                x = x.clamp(-1.0, 1.0)
#            if mask is not None:
#                x = x * mask + y * (1.0 - mask)
#
#        # Last step with euler
#        x = self._euler_step_restoration(x, timesteps[-2], timesteps[-1], y_cond, labels)
#        if clip_denoised:
#            x = x.clamp(-1.0, 1.0)
#        if mask is not None:
#            x = x * mask + y * (1.0 - mask)
#
#        return x
#
#    @torch.no_grad()
#    def _forward_sample_restoration(self, x, t, y_cond, labels):
#        # The user wants unconditional, so I will ignore `labels` and use a dummy one.
#        num_classes = int(self.b2b_model.num_classes)
#        if labels is None: # Should not happen if called from restoration
#            B = x.size(0)
#            labels = torch.full((B,), num_classes, device=x.device, dtype=torch.long)
#
#        uncond_labels = torch.full_like(labels, num_classes)
#
#        # The model input needs `y_cond`.
#        if y_cond is not None:
#            if len(x.shape) != 5:
#                model_input = torch.cat([y_cond, x], dim=1)
#            else: # video
#                model_input = torch.cat([y_cond, x], dim=2)
#        else:
#            # This is based on `b2b_forward` and `in_channel` logic.
#            # It assumes the model always wants concatenated input.
#            model_input = torch.cat([x, x], dim=1)
#
#        # `t` is a broadcasted tensor, model wants a vector.
#        t_vec = t.view(-1).float()
#
#        x_pred = self.b2b_model(model_input, t_vec, y=uncond_labels)
#
#        # Velocity calculation from user's snippet (adapted)
#        # v = (x_pred - z) / (1-t) where z is the input to network.
#        # Here, the "denoised" part of the input is `x`.
#        v_pred = (x_pred - x) / (1.0 - t).clamp_min(self.t_eps)
#        return v_pred
#
#    @torch.no_grad()
#    def _euler_step_restoration(self, x, t, t_next, y_cond, labels):
#        v = self._forward_sample_restoration(x, t, y_cond, labels)
#        x_next = x + (t_next - t) * v
#        return x_next
#
#    @torch.no_grad()
#    def _heun_step_restoration(self, x, t, t_next, y_cond, labels):
#        v_t = self._forward_sample_restoration(x, t, y_cond, labels)
#        x_euler = x + (t_next - t) * v_t
#        v_t_next = self._forward_sample_restoration(x_euler, t_next, y_cond, labels)
#        v = 0.5 * (v_t + v_t_next)
#        x_next = x + (t_next - t) * v
#        return x_next
#
#
#
