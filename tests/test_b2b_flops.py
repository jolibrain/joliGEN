import copy
import json
import pytest
import torch
from torch import nn
from scripts import profile_b2b_flops as p


@pytest.fixture(scope="module")
def thop():
    directory = p.ROOT / "results/thop_deps"
    if not directory.exists():
        pytest.skip("Install the optional profiler as described in docs/b2b_flops.md")
    return p.load_thop(directory)


@pytest.mark.parametrize(
    "model,x,expected",
    [
        (nn.Linear(5, 7, bias=False), torch.zeros(2, 5), 70),
        (nn.Conv2d(3, 4, 3, bias=False), torch.zeros(1, 3, 8, 8), 3888),
    ],
)
def test_analytical_layers(thop, model, x, expected):
    result = p.measure(model, (x,), thop)
    assert result["thop_ops"] == expected
    assert result["supplemental_baddbmm_macs"] == 0
    assert result["estimated_gflops"] == 2 * expected / 1e9


def test_attention_products_counted_once(thop):
    class Attention(nn.Module):
        def forward(self, x):
            weights = torch.baddbmm(torch.zeros(2, 3, 3), x, x.transpose(1, 2))
            return torch.bmm(weights, x)

    result = p.measure(Attention(), (torch.zeros(2, 3, 4),), thop)
    assert result["thop_ops"] == 72
    assert result["supplemental_baddbmm_macs"] == 72
    assert result["unsupported_leaf_modules"] == ["Attention"]
    assert result["estimated_gflops"] is None


def test_matmul_without_supplement(thop):
    class Matmul(nn.Module):
        def forward(self, x):
            return x @ x.transpose(-1, -2)

    result = p.measure(Matmul(), (torch.zeros(2, 3, 4),), thop)
    assert result["thop_ops"] == 72
    assert result["supplemental_baddbmm_macs"] == 0


def test_hybrid_sum_and_shared_parameters():
    stages = {
        k: dict(estimated_gflops=v, params=100, complete=True)
        for k, v in [("baseline", 1), ("coarse", 2), ("refinement", 3)]
    }
    rows = p.summary("example", stages)
    assert rows[2]["representative_gflops"] == 5
    assert rows[2]["params"] == 100
    stages["refinement"]["complete"] = False
    assert p.summary("example", stages)[2]["representative_gflops"] is None


def test_inputs_conditioning():
    config = dict(
        model=dict(input_nc=3),
        G=dict(vit_num_classes=20),
        data=dict(crop_size=32, temporal_number_frames=2),
        alg=dict(
            b2b_mask_as_channel=True,
            b2b_mask_prediction=True,
            b2b_mask_size_conditioning=True,
            b2b_global_context_mode="tokens",
            b2b_global_context_size=16,
            b2b_temporal_frame_step_conditioning=True,
        ),
    )
    x, t, label, cond = p.inputs(config, 17, "coarse", "cpu")
    assert x.shape == (1, 2, 4, 32, 32)
    assert cond["global_context"].shape == (1, 2, 3, 16, 16)
    assert cond["mask_precision_mode"].tolist() == [[0, 2]]
    assert p.inputs(config, 17, "refinement", "cpu")[3][
        "mask_precision_mode"
    ].tolist() == [[0, 0]]
    assert label.tolist() == [17]
    with pytest.raises(ValueError):
        p.inputs(config, 20, "coarse", "cpu")


def test_adapter_and_no_parameter_changes(thop):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(2, 2)

        def predict_clean(self, x, times, labels, **kwargs):
            return self.projection(x) + kwargs["offset"]

    model = Model().eval()
    state = copy.deepcopy(model.state_dict())
    result = p.measure(
        p.Predict(model),
        (torch.zeros(1, 2), torch.zeros(1), torch.zeros(1), {"offset": 1}),
        thop,
    )
    assert result["params"] == 6
    assert result["complete"]
    assert state.keys() == model.state_dict().keys()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())


def test_unsupported_functional_product_is_reported(thop):
    class Product(nn.Module):
        def __init__(self):
            super().__init__()
            self.identity = nn.Identity()

        def forward(self, x):
            return torch.einsum("ij,jk->ik", x, x)

    result = p.measure(Product(), (torch.zeros(2, 2),), thop)
    assert result["unsupported_functional_products"]
    assert not result["complete"]
    assert result["estimated_gflops"] is None


def test_cli_outputs_without_checkpoint_loading(tmp_path, monkeypatch, thop):
    from types import SimpleNamespace
    from scripts import b2b_pth_denoiser_infer_autoregressive_progress_bbox as loader

    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"fake checkpoint")
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            dict(model_type="b2b", G=dict(vit_num_classes=1), data=dict(crop_size=8))
        )
    )
    spec = dict(weights=str(weights), config=str(config))
    local = tmp_path / "local.json"
    local.write_text(
        json.dumps(
            dict(model_sets=dict(example=dict(label=0, baseline=spec, smartbrush=spec)))
        )
    )

    def load(*args):
        model = nn.Sequential(nn.Identity())
        return SimpleNamespace(model=model, mask_prediction=True), None, None

    monkeypatch.setattr(loader, "load_pth_session", load)
    monkeypatch.setattr(
        p.Predict, "forward", lambda self, x, t, labels, conditions: self.model(x)
    )
    output = tmp_path / "out"
    p.main(["--config", str(local), "--output-dir", str(output)])
    import csv

    with (output / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    details = json.loads((output / "details.json").read_text())
    assert details["model_sets"]["example"]["coarse"]["input_shape"] == [1, 2, 3, 8, 8]
    with pytest.raises(FileExistsError):
        p.main(["--config", str(local), "--output-dir", str(output)])


def test_jit_spatial_attention_analytical_count(thop):
    from models.modules.vit.vit_vid import Attention

    model = Attention(dim=8, num_heads=2).eval()
    result = p.measure(model, (torch.zeros(1, 3, 8), lambda x: x), thop)
    # QKV + output projections, then QK and AV products; RMSNorm excluded.
    assert result["thop_ops"] == 3 * 8 * 24 + 3 * 8 * 8 + 2 * 2 * 3 * 3 * 4
    assert result["supplemental_baddbmm_macs"] == 0
    assert result["complete"]


@pytest.mark.parametrize("source_crop", [288, 320, 512])
def test_source_crop_does_not_override_network_resolution(source_crop):
    config = dict(
        data=dict(
            crop_size=256,
            load_size=384,
            online_creation=dict(crop_size_A=source_crop),
            temporal_number_frames=2,
        )
    )
    image, _, _, _ = p.inputs(config, 0, "coarse", "meta")
    assert image.shape == (1, 2, 3, 256, 256)


@pytest.mark.parametrize("whole_block", [False, True])
def test_actual_temporal_attention_and_transformer_cost(thop, whole_block):
    from models.modules.vit.vit_vid_per_layer_motion import (
        TemporalTransformerBlock,
        VersatileAttention,
    )

    frames, positions, dim = 2, 3, 8
    tokens = positions * frames
    qk = positions * frames * frames * dim
    projections = 4 * tokens * dim**2
    native = projections + qk  # projections and attention-value product
    if whole_block:
        model = TemporalTransformerBlock(dim, 2, 4, ("Temporal_Self",))
        # GEGLU input/output projections and two affine LayerNorms.
        native += tokens * (dim * 8 * dim + 4 * dim * dim) + 2 * 4 * tokens * dim
    else:
        model = VersatileAttention(
            attention_mode="Temporal", query_dim=dim, heads=2, dim_head=4
        )
    result = p.measure(
        model.eval(), (torch.zeros(frames, positions, dim), frames), thop
    )
    assert result["complete"]
    assert result["thop_ops"] == native
    assert result["supplemental_baddbmm_macs"] == qk
    assert result["estimated_gflops"] == 2 * (native + qk) / 1e9
