from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import b2b_pth_denoiser_infer_autoregressive_progress_bbox as pth


def test_runner_uses_full_precision_bbox_background(monkeypatch, tmp_path):
    runner = pth.onnx_runner
    monkeypatch.setattr(runner, "get_train_shape", lambda _: (2, 2, 2, 2, 2))
    monkeypatch.setattr(
        runner,
        "get_b2b_params",
        lambda *_: {"mask_as_channel": False, "mask_prediction": True},
    )
    monkeypatch.setattr(runner, "resolve_mask_precision", lambda *_: (2, 0.0))
    monkeypatch.setattr(runner, "resolve_temporal_frame_step", lambda *_: 1.0)
    monkeypatch.setattr(runner, "resolve_b2b_label", lambda *_: 0)
    seen_flags = []

    def preprocess(**kwargs):
        seen_flags.append(kwargs["no_mandatory_bbox"])
        return dict(
            y_t=torch.zeros(1, 3, 2, 2),
            y0_tensor=torch.zeros(1, 3, 2, 2),
            mask=torch.ones(1, 1, 2, 2),
            mandatory_mask=torch.zeros(1, 1, 2, 2),
            cond_image=torch.zeros(1, 3, 2, 2),
            label_cls=0,
        )

    monkeypatch.setattr(runner, "preprocess_with_repo_crop", preprocess)
    bbox = np.full((1, 2, 3, 2, 2), 0.25, dtype=np.float32)
    captured = []

    def restore(**kwargs):
        np.testing.assert_array_equal(kwargs["source_image"], bbox)
        result = runner.RestorationResult(
            generated_pixels=bbox,
            generated_mask=np.ones((1, 2, 1, 2, 2), dtype=np.float32),
            generated_applied=bbox,
            coarse_applied=bbox,
        )
        captured.append(result)
        return result

    monkeypatch.setattr(runner, "restoration_with_denoiser", restore)
    monkeypatch.setattr(runner, "write_frame", lambda *_, **__: None)
    callbacks = []
    written = runner.run_sequence(
        session=object(),
        pairs=[("0.jpg", "0.txt"), ("1.jpg", "1.txt")],
        dataset_root=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        bbox_index=0,
        label=0,
        seed=7,
        train_json={},
        denoise_steps=2,
        debug_dump_dir=None,
        autoregressive_reinject_patch=False,
        no_mandatory_bbox=True,
        known_image_override=bbox,
        result_callback=callbacks.append,
    )
    assert written == [0, 1]
    assert seen_flags == [True, True]
    assert callbacks == captured


def test_two_stage_passes_bbox_result_to_refinement(monkeypatch, tmp_path):
    bbox = np.full((1, 2, 3, 2, 2), 0.25, dtype=np.float32)
    predicted_mask = np.zeros((1, 2, 1, 2, 2), dtype=np.float32)
    predicted_mask[..., 0, 0] = 1.0
    calls = []

    def run_sequence(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            kwargs["result_callback"](SimpleNamespace(coarse_applied=bbox))
        else:
            np.testing.assert_array_equal(kwargs["known_image_override"], bbox)
            refined = bbox.copy()
            refined[..., 0, 0] = 0.75
            kwargs["result_callback"](
                SimpleNamespace(
                    generated_applied=refined, generated_mask=predicted_mask
                )
            )
        return [0, 1]

    monkeypatch.setattr(pth.onnx_runner, "run_sequence", run_sequence)
    assert pth.run_two_stage_hybrid({"seed": 7}, str(tmp_path / "output"), None) == [
        0,
        1,
    ]
    assert calls[0]["apply_predicted_mask"] is False
    assert calls[1]["apply_predicted_mask"] is True
    assert calls[1]["use_predicted_mask_during_denoising"] is True
    assert calls[0]["output_dir"].endswith("/bbox")
    assert calls[1]["output_dir"].endswith("/hybrid")


def test_two_stage_rejects_changed_background(monkeypatch, tmp_path):
    bbox = np.zeros((1, 2, 3, 2, 2), dtype=np.float32)
    calls = 0

    def run_sequence(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            kwargs["result_callback"](SimpleNamespace(coarse_applied=bbox))
        else:
            kwargs["result_callback"](
                SimpleNamespace(
                    generated_applied=np.ones_like(bbox),
                    generated_mask=np.zeros((1, 2, 1, 2, 2), dtype=np.float32),
                )
            )
        return [0, 1]

    monkeypatch.setattr(pth.onnx_runner, "run_sequence", run_sequence)
    with pytest.raises(RuntimeError, match="outside the predicted mask"):
        pth.run_two_stage_hybrid({"seed": 7}, str(tmp_path / "output"), None)
