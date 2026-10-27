from types import SimpleNamespace
import copy
import os
import random

import numpy as np
import pytest
import torch
from torch import nn

from models.modules.convrefiner import ConvRefiner, load_refiner_generator_state
from models.modules.b2b_generator import B2BGenerator
from models.b2b_model import B2BModel
from models.b2b_refiner_training import validate_refiner_options
from scripts.b2b_export_onnx import B2BDenoiserWrapper
from scripts.b2b_pth_denoiser_infer_autoregressive_progress_bbox import (
    PthDenoiserSession,
)


class Backbone(nn.Module):
    num_classes = 2

    def __init__(self, masks=True):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.masks = masks

    def forward(self, image, timesteps, labels, **kwargs):
        video = image.ndim == 5
        x = image[..., -3:, :, :]
        flat = x.flatten(0, 1) if video else x
        output = self.conv(flat).reshape_as(x)
        if labels is not None:
            output = output + labels.reshape(-1, *([1] * (output.ndim - 1))) * 0.01
        return (output, output[..., :1, :, :]) if self.masks else output


def options(tmp_path=None, **overrides):
    values = dict(
        alg_b2b_convrefiner="temporal",
        alg_b2b_convrefiner_width=32,
        alg_b2b_convrefiner_finetune=True,
        alg_b2b_convrefiner_source_ema=False,
        alg_b2b_convrefiner_gan_weight=0.02,
        alg_b2b_lora=False,
        model_output_nc=3,
        G_vit_num_classes=2,
        G_netG="vit_vid",
        alg_b2b_mask_prediction=True,
        alg_b2b_perceptual_loss=[""],
        alg_b2b_loss="MSE",
        alg_b2b_loss_masked_region_only=True,
        alg_diffusion_lambda_G=1.0,
        alg_b2b_lambda_perceptual=0.1,
        isTrain=True,
        train_D_lr=1e-4,
        train_beta1=0.9,
        train_beta2=0.95,
        train_optim_weight_decay=0.0,
        train_optim_eps=1e-8,
        train_iter_size=2,
        train_G_ema=True,
        train_G_ema_beta=0.999,
        train_continue_from="",
        train_continue=True,
        train_epoch_count=1,
        model_load_no_strictness=False,
        total_iters=0,
    )
    values.update(overrides)
    values["optim"] = lambda opt, parameters, **kwargs: torch.optim.AdamW(
        parameters, **kwargs
    )
    return SimpleNamespace(**values)


def generator(opt=None):
    opt = opt or options()
    return B2BGenerator(Backbone(), "", 32, 32, opt)


def training_model(tmp_path, **overrides):
    model = B2BModel.__new__(B2BModel)
    model.opt = options(tmp_path, **overrides)
    model.isTrain = True
    model.device = torch.device("cpu")
    model.gpu_ids = []
    model.use_cuda = model.with_amp = False
    model.scaler = torch.amp.GradScaler("cpu", enabled=False)
    model.netG_A = generator(model.opt)
    model.model_names = ["G_A"]
    model.save_dir = str(tmp_path)
    model.optimizer_G = torch.optim.AdamW(
        model._iter_trainable_generator_parameters(), lr=1e-4
    )
    model.optimizers = [model.optimizer_G]
    model.niter = 0
    model._init_refiner_training()
    model.schedulers = [
        torch.optim.lr_scheduler.LambdaLR(o, lambda _: 1.0) for o in model.optimizers
    ]
    model.loss_fn = nn.MSELoss()
    model.num_classes = 2
    model.label_cls = torch.zeros(1, dtype=torch.long)
    model.gt_image = torch.randn(1, 2, 3, 32, 32)
    model.mask = torch.ones(1, 2, 1, 32, 32)
    model.mask[:, 0] = 0
    model.cond_image = None
    model.use_gt = torch.tensor([True])
    model.ref_idx = torch.tensor([0])
    return model


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize(
    "shape", [(2, 3, 16, 16), (2, 1, 3, 16, 16), (2, 3, 3, 16, 16)]
)
def test_identity_initialization_and_time_shapes(temporal, shape):
    refiner = ConvRefiner(width=32, temporal=temporal)
    x = torch.randn(shape)
    assert torch.equal(refiner(x, torch.tensor([0.1, 0.9])), x)
    if x.ndim == 5:
        assert torch.equal(refiner(x, torch.rand(shape[0] * shape[1])), x)


def test_temporal_branch_processes_neighboring_frames_jointly():
    torch.manual_seed(7)
    refiner = ConvRefiner(width=32, temporal=True)
    nn.init.normal_(refiner.output_conv.weight, std=0.05)
    for block in refiner.blocks:
        nn.init.normal_(block.temporal_conv.weight, std=0.02)
    x = torch.randn(1, 2, 3, 16, 16)
    altered = x.clone()
    altered[:, 0] = torch.randn_like(altered[:, 0])
    t = torch.tensor([0.5])
    assert not torch.allclose(refiner(x, t)[:, 1], refiner(altered, t)[:, 1])


def test_shared_prediction_preserves_mask_logits_and_projects_reference():
    net = generator()
    nn.init.normal_(net.convrefiner.output_conv.weight, std=0.05)
    x = torch.randn(1, 2, 3, 32, 32)
    t, labels = torch.tensor([0.9, 1.0]), torch.tensor([0])
    base, base_mask = net.b2b_model(x, t, labels)
    refined, refined_mask = net.predict_clean(x, t, labels)
    assert torch.equal(refined_mask, base_mask)
    assert not torch.equal(refined, base)
    mask = torch.ones(1, 2, 1, 32, 32)
    mask[:, 0] = 0
    result = net(
        x,
        mask,
        return_training_details=True,
        use_gt=torch.tensor([True]),
        ref_idx=torch.tensor([0]),
    )
    assert torch.equal(result["x_pred"][:, 0], x[:, 0])
    assert result["t"][0, 0].item() == 1
    net.train()
    assert not net.b2b_model.training
    assert net.convrefiner.training


def test_training_accumulation_freezes_backbone_and_updates_ema_only_on_step(tmp_path):
    model = training_model(tmp_path)
    backbone = copy.deepcopy(model.netG_A.b2b_model.state_dict())
    refiner = copy.deepcopy(model.netG_A.convrefiner.state_dict())
    disc = copy.deepcopy(model.netD_refiner.state_dict())
    model.optimize_parameters()
    assert model.refiner_optimizer_steps == 0
    assert not hasattr(model, "netG_A_ema")
    model.optimize_parameters()
    assert model.refiner_optimizer_steps == 1
    assert all(
        torch.equal(v, model.netG_A.b2b_model.state_dict()[k])
        for k, v in backbone.items()
    )
    assert all(
        p.grad is None and not p.requires_grad
        for p in model.netG_A.b2b_model.parameters()
    )
    assert any(
        not torch.equal(v, model.netG_A.convrefiner.state_dict()[k])
        for k, v in refiner.items()
    )
    assert any(
        not torch.equal(v, model.netD_refiner.state_dict()[k]) for k, v in disc.items()
    )
    assert hasattr(model, "netG_A_ema")


@pytest.mark.parametrize("schedule", ["near_clean", "sampler_aligned"])
def test_empty_masks_backward_and_partial_epoch_flush(tmp_path, schedule):
    model = training_model(
        tmp_path,
        alg_b2b_convrefiner_gan_schedule=schedule,
        alg_b2b_denoise_timesteps=[2, 5],
    )
    model.mask.zero_()
    model.optimize_parameters()
    assert model.loss_G_tot == 0
    assert model.loss_D_refiner == 0
    model.finish_refiner_epoch()
    assert model.refiner_optimizer_steps == 1
    assert model._refiner_accumulated == 0


def test_checkpoint_fresh_strict_resume_and_source_ema(tmp_path):
    model = training_model(tmp_path)
    raw = {
        k: v
        for k, v in model.netG_A.state_dict().items()
        if not k.startswith("convrefiner.")
    }
    load_refiner_generator_state(model.netG_A, raw, fresh=True)
    with pytest.raises(RuntimeError, match="Incompatible"):
        load_refiner_generator_state(model.netG_A, raw)
    missing_base = dict(raw)
    del missing_base[next(iter(missing_base))]
    with pytest.raises(RuntimeError, match="Incompatible"):
        load_refiner_generator_state(model.netG_A, missing_base, fresh=True)
    model.optimize_parameters()
    model.finish_refiner_epoch()
    model.prepare_refiner_checkpoint(3, 0, end_epoch=True)
    model.save_networks("latest")
    restored = training_model(tmp_path)
    restored.load_networks("latest")
    assert restored.opt.train_epoch_count == 4
    assert restored.refiner_optimizer_steps == model.refiner_optimizer_steps
    assert all(
        torch.equal(v, restored.netG_A.state_dict()[k])
        for k, v in model.netG_A.state_dict().items()
    )
    assert restored.optimizer_G.state_dict()["state"]
    restored.restore_refiner_rng()
    restored.gt_image = model.gt_image.clone()
    torch.manual_seed(123)
    model.optimize_parameters()
    model.optimize_parameters()
    torch.manual_seed(123)
    restored.optimize_parameters()
    restored.optimize_parameters()
    for key, value in model.netG_A.state_dict().items():
        torch.testing.assert_close(
            value, restored.netG_A.state_dict()[key], rtol=0, atol=0
        )
    source = tmp_path / "source"
    source.mkdir()
    torch.save(raw, source / "320_net_G_A_ema.pth")
    fresh = training_model(
        tmp_path,
        train_continue_from=str(source),
        train_continue=False,
        alg_b2b_convrefiner_source_ema=True,
    )
    fresh.load_networks("320", load_dir=str(source))
    assert fresh.refiner_source_checkpoint.endswith("320_net_G_A_ema.pth")


def test_sampler_aligned_timestep_distribution_and_default(tmp_path):
    model = training_model(tmp_path)
    before = random.getstate()
    assert model._refiner_adversarial_timestep_range() == (0.9, 0.98)
    assert random.getstate() == before  # Legacy schedule does not consume extra RNG.
    model.opt.alg_b2b_convrefiner_gan_schedule = "sampler_aligned"
    model.opt.alg_b2b_denoise_timesteps = [2, 5, 5]
    counts = {0.5: 0, 0.8: 0, 0.94: 0}
    random.seed(42)
    for _ in range(10000):
        lo, hi = model._refiner_adversarial_timestep_range()
        key = round((lo + hi) / 2, 2)
        counts[key] += 1
        assert hi - lo == pytest.approx(0.08 if key == 0.94 else 0.1)
    assert counts[0.5] / 10000 == pytest.approx(0.4, abs=0.02)
    assert counts[0.8] / 10000 == pytest.approx(0.4, abs=0.02)
    assert counts[0.94] / 10000 == pytest.approx(0.2, abs=0.02)


@pytest.mark.parametrize("steps", [2, 5])
def test_sampler_aligned_forward_and_frozen_backbone(tmp_path, steps):
    model = training_model(
        tmp_path,
        alg_b2b_convrefiner_gan_schedule="sampler_aligned",
        alg_b2b_convrefiner_gan_near_clean_prob=0.0,
        alg_b2b_convrefiner_gan_timestep_radius=0.0,
        alg_b2b_denoise_timesteps=[steps],
    )
    output = model._refiner_forward(near=True)
    assert output["t"][0, 0].item() == 1.0  # Clean reference remains clean.
    assert output["t"][0, 1].item() == pytest.approx(1 - 1 / steps)
    backbone = {k: v.clone() for k, v in model.netG_A.b2b_model.state_dict().items()}
    model.optimize_parameters()
    model.optimize_parameters()
    assert model.refiner_optimizer_steps == 1
    assert "G_refiner_adv" in model.loss_names
    assert torch.isfinite(model.loss_G_refiner_adv_avg)
    assert model.netG_A.convrefiner.output_conv.weight.abs().sum() > 0
    for k, v in model.netG_A.b2b_model.state_dict().items():
        assert torch.equal(v, backbone[k])


@pytest.mark.parametrize(
    "overrides",
    [
        {"alg_b2b_convrefiner_gan_near_clean_prob": float("nan")},
        {"alg_b2b_convrefiner_gan_near_clean_prob": 1.1},
        {"alg_b2b_convrefiner_gan_timestep_radius": -0.1},
        {"alg_b2b_convrefiner_gan_timestep_radius": float("inf")},
        {"alg_b2b_denoise_timesteps": [1, 5]},
        {"alg_b2b_denoise_timesteps": []},
        {"alg_b2b_convrefiner_resume_from": "source", "train_continue_from": "fresh"},
    ],
)
def test_sampler_aligned_option_validation(overrides):
    values = dict(
        alg_b2b_convrefiner_gan_schedule="sampler_aligned",
        alg_b2b_denoise_timesteps=[2, 5],
    )
    values.update(overrides)
    with pytest.raises(ValueError):
        validate_refiner_options(options(**values))


def test_sampler_aligned_branch_resume_rng_and_optimizers(tmp_path):
    source = tmp_path / "source"
    branch = tmp_path / "branch"
    source.mkdir()
    branch.mkdir()
    original = training_model(source)
    original.optimize_parameters()
    original.finish_refiner_epoch()
    original.prepare_refiner_checkpoint(5, 0, end_epoch=True)
    original.save_networks("5")
    model = training_model(
        branch,
        alg_b2b_convrefiner_resume_from=str(source),
        alg_b2b_convrefiner_gan_schedule="sampler_aligned",
        alg_b2b_denoise_timesteps=[2, 5],
    )
    model.load_networks("5")
    assert model.save_dir == str(branch)
    assert model.opt.train_epoch_count == 6
    assert model.refiner_optimizer_steps == original.refiner_optimizer_steps
    for current, previous in zip(model.optimizers, original.optimizers):
        assert (
            current.state_dict()["param_groups"]
            == previous.state_dict()["param_groups"]
        )
        for k, state in current.state_dict()["state"].items():
            for field, value in state.items():
                torch.testing.assert_close(
                    value, previous.state_dict()["state"][k][field]
                )
    model.restore_refiner_rng()
    model.prepare_refiner_checkpoint(6, 0)
    model.save_networks("latest")
    with pytest.raises(ValueError, match="Branch already contains"):
        model.load_networks("5")
    expected = [model._refiner_adversarial_timestep_range() for _ in range(20)]
    # Subsequent resumes must clear the original branch-source option.
    resumed = training_model(
        branch,
        alg_b2b_convrefiner_gan_schedule="sampler_aligned",
        alg_b2b_denoise_timesteps=[2, 5],
    )
    resumed.load_networks("latest")
    resumed.restore_refiner_rng()
    assert [
        resumed._refiner_adversarial_timestep_range() for _ in range(20)
    ] == expected
    state = torch.load(branch / "latest_refiner_training.pth", weights_only=False)
    assert state["adversarial_schedule"]["schedule"] == "sampler_aligned"
    resumed.gt_image = model.gt_image.clone()
    rng = model._refiner_rng_state()
    model.optimize_parameters()
    model.optimize_parameters()
    resumed._refiner_resume_rng = rng
    resumed.restore_refiner_rng()
    resumed.optimize_parameters()
    resumed.optimize_parameters()
    for key, value in model.netG_A.state_dict().items():
        torch.testing.assert_close(
            value, resumed.netG_A.state_dict()[key], rtol=0, atol=0
        )
    for key, value in model.netD_refiner.state_dict().items():
        torch.testing.assert_close(
            value, resumed.netD_refiner.state_dict()[key], rtol=0, atol=0
        )


def test_pth_and_export_wrapper_use_nonzero_refinement():
    net = generator().eval()
    nn.init.normal_(net.convrefiner.output_conv.weight, std=0.05)
    x, t, labels = torch.randn(1, 2, 3, 32, 32), torch.tensor([0.5]), torch.tensor([0])
    expected = net.predict_clean(x, t, labels)
    wrapper = B2BDenoiserWrapper(net)
    actual = wrapper(x, t, labels)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    session = PthDenoiserSession(net, torch.device("cpu"), mask_prediction=False)
    # SmartBrush session requires its conditioning inputs even with a tiny backbone.
    session.mask_prediction = True
    inputs = dict(
        model_input=x.numpy(),
        timesteps=t.numpy(),
        labels=labels.numpy(),
        mask_precision_mode=np.zeros((1, 2), dtype=np.int64),
        mask_precision_severity=np.zeros((1, 2), dtype=np.float32),
    )
    actual = session.run(["output", "mask_logits"], inputs)
    for a, b in zip(expected, actual):
        np.testing.assert_allclose(a.detach().numpy(), b)


def test_onnx_dynamic_batch_frames_and_nonzero_temporal_refiner(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    net = generator().eval()
    nn.init.normal_(net.convrefiner.output_conv.weight, std=0.01)
    for block in net.convrefiner.blocks:
        nn.init.normal_(block.temporal_conv.weight, std=0.01)
    wrapper = B2BDenoiserWrapper(net)
    path = str(tmp_path / "refiner.onnx")
    inputs = (torch.randn(1, 2, 3, 32, 32), torch.tensor([0.5]), torch.tensor([0]))
    torch.onnx.export(
        wrapper,
        inputs,
        path,
        opset_version=18,
        dynamo=False,
        input_names=["model_input", "timesteps", "labels"],
        output_names=["output", "mask_logits"],
        dynamic_axes={
            "model_input": {0: "batch", 1: "frames"},
            "timesteps": {0: "times"},
            "labels": {0: "batch"},
            "output": {0: "batch", 1: "frames"},
            "mask_logits": {0: "batch", 1: "frames"},
        },
    )
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for batch, frames in [(1, 1), (2, 3)]:
        x, t, labels = (
            torch.randn(batch, frames, 3, 32, 32),
            torch.rand(batch * frames),
            torch.zeros(batch, dtype=torch.long),
        )
        expected = wrapper(x, t, labels)
        values = {
            "model_input": x.numpy(),
            "timesteps": t.numpy(),
            "labels": labels.numpy(),
        }
        actual = session.run(
            None, {i.name: values[i.name] for i in session.get_inputs()}
        )
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(a.detach().numpy(), b, atol=2e-5, rtol=2e-4)

    # Three overlapping autoregressive windows, through both actual samplers.
    from scripts import b2b_onnx_denoiser_infer_autoregressive_progress_bbox as runner

    params = runner.get_b2b_params(
        {"alg": {"b2b_mask_prediction": True}, "G": {"vit_num_classes": 2}}, 32
    )
    params.update(
        noise_scale=net.noise_scale,
        t_eps=net.t_eps,
        disable_inference_clipping=net.disable_inference_clipping,
        cfg_scale=1.0,
        clip_denoised=False,
    )
    previous = torch.rand(1, 3, 32, 32) * 2 - 1
    previous_ort = previous.numpy().copy()

    class ToySession:
        def run(self, names, values):
            # This toy backbone ignores precision conditioning, so ONNX prunes it.
            return session.run(
                names, {item.name: values[item.name] for item in session.get_inputs()}
            )

    mask = torch.ones(1, 2, 1, 32, 32)
    mask[:, 0] = 0
    labels = torch.zeros(1, dtype=torch.long)
    for _ in range(3):
        current = torch.rand_like(previous) * 2 - 1
        y = torch.stack([previous, current], dim=1)
        y_ort = np.stack([previous_ort, current.numpy()], axis=1)
        noise = torch.randn_like(y)
        expected = net.restoration(
            y, None, 2, mask, labels, init_noise=noise, source_image=y
        )
        actual = runner.restoration_with_denoiser(
            ToySession(),
            y_ort,
            None,
            2,
            mask.numpy(),
            labels.numpy(),
            params,
            noise.numpy(),
        )
        np.testing.assert_allclose(expected.numpy(), actual, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(expected[:, 0], previous, rtol=0, atol=0)
        previous, previous_ort = expected[:, 1], actual[:, 1]


@pytest.mark.parametrize("cfg", [1.0, 2.0])
def test_cfg_combines_refined_predictions_before_velocity_projection(cfg):
    net = generator(options(alg_b2b_cfg_scale=cfg)).eval()
    nn.init.normal_(net.convrefiner.output_conv.weight, std=0.01)
    x, known = torch.randn(1, 2, 3, 32, 32), torch.randn(1, 2, 3, 32, 32)
    mask = torch.ones(1, 2, 1, 32, 32)
    mask[:, 0] = 0
    t = torch.full((1, 2, 1, 1, 1), 0.5)
    labels = torch.zeros(1, dtype=torch.long)
    x_in = net._project_known_pixels(x, known, mask)
    cond, _ = net.predict_clean(x_in, t.flatten(), labels)
    uncond, _ = net.predict_clean(x_in, t.flatten(), torch.full_like(labels, 2))
    expected = (
        net._project_known_pixels(uncond + cfg * (cond - uncond), known, mask) - x_in
    ) / 0.5
    actual, _ = net._forward_sample_restoration(x, t, None, mask, labels, known)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def _ddp_worker(rank, rendezvous, result_dir, schedule):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2
    )
    try:
        torch.manual_seed(19)
        model = training_model(
            result_dir,
            alg_b2b_convrefiner_gan_schedule=schedule,
            alg_b2b_denoise_timesteps=[2, 5],
        )
        model.netG_A = nn.parallel.DistributedDataParallel(
            model.netG_A, broadcast_buffers=False
        )
        model.netD_refiner = nn.parallel.DistributedDataParallel(
            model.netD_refiner, broadcast_buffers=False
        )
        torch.manual_seed(100 + rank)
        if rank == 0:
            model.mask.zero_()
        for _ in range(3):
            model.optimize_parameters()
        model.finish_refiner_epoch()
        assert model.refiner_optimizer_steps == 2
        backbone = model.netG_A.module.b2b_model
        assert all(parameter.grad is None for parameter in backbone.parameters())
        flat = torch.cat(
            [p.detach().flatten() for p in model.netG_A.module.convrefiner.parameters()]
        )
        gathered = [torch.empty_like(flat), torch.empty_like(flat)]
        torch.distributed.all_gather(gathered, flat)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("schedule", ["near_clean", "sampler_aligned"])
def test_two_rank_ddp_accumulation_and_empty_rank(tmp_path, schedule):
    torch.multiprocessing.spawn(
        _ddp_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path), schedule),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(
    os.environ.get("JOLIGEN_TEST_TRT") != "1", reason="opt-in GPU/TensorRT execution"
)
def test_tensorrt_nonzero_temporal_refiner(tmp_path):
    trt = pytest.importorskip("tensorrt")
    from scripts.b2b_trt_denoiser_infer_autoregressive_progress_bbox import (
        TensorRTDenoiserSession,
    )

    net = generator().eval()
    nn.init.normal_(net.convrefiner.output_conv.weight, std=0.01)
    for block in net.convrefiner.blocks:
        nn.init.normal_(block.temporal_conv.weight, std=0.01)
    wrapper = B2BDenoiserWrapper(net)
    inputs = (torch.randn(1, 2, 3, 32, 32), torch.tensor([0.5]), torch.tensor([1]))
    onnx_path = tmp_path / "refiner.onnx"
    torch.onnx.export(
        wrapper,
        inputs,
        str(onnx_path),
        opset_version=18,
        dynamo=False,
        input_names=["model_input", "timesteps", "labels"],
        output_names=["output", "mask_logits"],
    )
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    assert parser.parse(onnx_path.read_bytes()), "\n".join(
        str(parser.get_error(i)) for i in range(parser.num_errors)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
    config.clear_flag(trt.BuilderFlag.TF32)
    engine = builder.build_serialized_network(network, config)
    assert engine is not None
    engine_path = tmp_path / "refiner.engine"
    engine_path.write_bytes(bytes(engine))
    session = TensorRTDenoiserSession(str(engine_path))
    try:
        expected = wrapper(*inputs)
        actual = session.run(
            ["output", "mask_logits"],
            dict(
                zip(["model_input", "timesteps", "labels"], [x.numpy() for x in inputs])
            ),
        )
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(a.detach().numpy(), b, rtol=3e-4, atol=3e-5)
    finally:
        session.close()


@pytest.mark.skipif(
    not os.environ.get("JOLIGEN_REFINER_SOURCE_CONFIG"),
    reason="opt-in real checkpoint smoke test",
)
def test_real_checkpoint_training_and_evaluation(tmp_path):
    import json
    from data import create_dataset, create_dataloader, list_test_sets
    from options.train_options import TrainOptions
    from util.convrefiner_metrics import evaluate_refiner

    config = json.loads(open(os.environ["JOLIGEN_REFINER_SOURCE_CONFIG"]).read())
    config["gpu_ids"] = "0"
    config["checkpoints_dir"] = str(tmp_path)
    config["train"]["batch_size"] = 1
    config["data"]["num_threads"] = 0
    config["output"]["display_id"] = 0
    opt = TrainOptions().parse_json(config, save_config=False, set_device=False)
    opt.optim = lambda opt, parameters, **kwargs: torch.optim.AdamW(
        parameters, **kwargs
    )
    opt.total_iters = 0
    opt.jg_dir = os.getcwd()
    opt.use_cuda = True
    opt.train_iter_size = 1
    torch.cuda.set_device(0)
    model = B2BModel(opt, 0)
    model.setup(opt)
    model.single_gpu()
    # Real source config, real source weights, real transformed training examples.
    dataset = create_dataset(opt, "train")
    loader = create_dataloader(opt, 0, dataset, 1)
    model.set_input(next(iter(loader)))
    before = {
        key: value.detach().clone()
        for key, value in model.netG_A.b2b_model.state_dict().items()
    }
    steps_before = model.refiner_optimizer_steps
    model.optimize_parameters()
    assert torch.isfinite(model.loss_G_tot)
    assert model.refiner_optimizer_steps == steps_before + 1
    assert all(
        torch.equal(value, model.netG_A.b2b_model.state_dict()[key])
        for key, value in before.items()
    )
    assert model.netG_A.convrefiner.output_conv.weight.abs().sum() > 0
    test_name = list_test_sets(opt)[0]
    test_dataset = create_dataset(opt, "test", test_name)
    test_loader = create_dataloader(opt, 0, test_dataset, 1)
    report = evaluate_refiner(model, test_loader, max_batches=1)
    assert {"2", "5"}.issubset(report)
    assert report["2"]["refined"]["masked_PSNR"] is not None
    assert report["5"]["refined"]["HF_absolute_deficit_dB"] >= 0
