import torch

from models.modules.cafm_jit_discriminator import (
    CAFMDiscriminatorJVP,
    CAFMJiTDiscriminator,
)


def _tiny_cafm_discriminator():
    return CAFMDiscriminatorJVP(
        CAFMJiTDiscriminator(
            input_size=8,
            patch_size=4,
            in_channels=3,
            hidden_size=16,
            depth=1,
            num_heads=2,
            num_classes=1,
            bottleneck_dim=8,
            in_context_len=0,
            in_context_start=0,
        )
    )


def test_cafm_jit_discriminator_jvp_shapes():
    discriminator = _tiny_cafm_discriminator()
    x = torch.randn(2, 3, 8, 8)
    y = torch.zeros(2, dtype=torch.long)
    t = torch.rand(2)
    dx = torch.randn_like(x)
    dt = torch.ones_like(t)

    out, out_jvp = discriminator(x=x, y=y, t=t, dx=dx, dt=dt)

    assert out.shape == (2,)
    assert out_jvp.shape == (2,)


def test_cafm_jit_discriminator_batched_jvp_shapes():
    discriminator = _tiny_cafm_discriminator()
    x = torch.randn(2, 3, 8, 8)
    y = torch.zeros(2, dtype=torch.long)
    t = torch.rand(2)
    dx = torch.randn(2, 2, 3, 8, 8)
    dt = torch.ones(2, 2)

    out, out_jvp = discriminator(x=x, y=y, t=t, dx=dx, dt=dt)

    assert out.shape == (2, 2)
    assert out_jvp.shape == (2, 2)


def test_cafm_jit_discriminator_jvp_keeps_tangent_grad():
    discriminator = _tiny_cafm_discriminator()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)

    x = torch.randn(2, 3, 8, 8)
    y = torch.zeros(2, dtype=torch.long)
    t = torch.rand(2)
    dx = torch.randn_like(x, requires_grad=True)
    dt = torch.ones_like(t)

    _, out_jvp = discriminator(x=x.detach(), y=y, t=t.detach(), dx=dx, dt=dt)
    out_jvp.mean().backward()

    assert dx.grad is not None
    assert dx.grad.abs().sum() > 0
