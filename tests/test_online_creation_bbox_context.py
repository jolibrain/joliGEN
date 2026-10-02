import random
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from data.online_creation import (
    compute_bbox_context_crop,
    crop_image,
    sample_bbox_context_crop_state,
    validate_bbox_context_crop_options,
)


EXACT_STATE = {
    "context_fraction": 0.1,
    "center_x_fraction": 0.0,
    "center_y_fraction": 0.0,
    "scale": 1.0,
    "aspect": 1.0,
}


def _compute(bbox, *, mask_aspect_ratio=0.0, orientation="fixed"):
    return compute_bbox_context_crop(
        bbox,
        frame_width=1920,
        frame_height=1080,
        model_width=256,
        model_height=256,
        state=EXACT_STATE,
        mask_aspect_ratio=mask_aspect_ratio,
        mask_aspect_ratio_orientation=orientation,
    )


def test_bbox_context_matches_measured_runtime_crop_sizes():
    assert _compute((100, 100, 192, 260))["source_rect"][2:] == [200, 200]
    assert _compute((100, 100, 256, 342))["source_rect"][2:] == [303, 303]


def test_zero_mask_aspect_ratio_leaves_detector_box_unchanged():
    result = _compute((100, 100, 192, 260), mask_aspect_ratio=0.0)

    assert result["effective_mask_aspect_ratio"] == 0.0
    assert result["mask_box"] == result["detector_box"]


def test_bbox_orientation_flips_portrait_ratio_for_landscape_box():
    result = _compute(
        (100, 100, 260, 192),
        mask_aspect_ratio=0.8,
        orientation="bbox",
    )
    mask_width = result["mask_box"][2] - result["mask_box"][0]
    mask_height = result["mask_box"][3] - result["mask_box"][1]

    assert result["effective_mask_aspect_ratio"] == pytest.approx(1.25)
    assert mask_width / mask_height == pytest.approx(1.25)


def test_bbox_context_shifts_source_rect_inside_frame_at_edges():
    result = _compute((0, 0, 92, 160))

    assert result["source_rect"] == [0, 0, 200, 200]


def test_bbox_context_sampling_is_seed_reproducible_and_bounded():
    first = sample_bbox_context_crop_state(
        [0.05, 0.20],
        0.05,
        [0.90, 1.10],
        [0.90, 1.10],
        rng=random.Random(7),
    )
    second = sample_bbox_context_crop_state(
        [0.05, 0.20],
        0.05,
        [0.90, 1.10],
        [0.90, 1.10],
        rng=random.Random(7),
    )

    assert first == second
    assert 0.05 <= first["context_fraction"] <= 0.20
    assert -0.05 <= first["center_x_fraction"] <= 0.05
    assert -0.05 <= first["center_y_fraction"] <= 0.05
    assert 0.90 <= first["scale"] <= 1.10
    assert 0.90 <= first["aspect"] <= 1.10


def test_bbox_context_options_require_zero_legacy_context_pixels():
    opt = SimpleNamespace(
        data_online_creation_crop_mode_A="bbox_context",
        data_online_creation_crop_context_fraction_range_A=[0.05, 0.20],
        data_online_creation_crop_bbox_scale_range_A=[0.90, 1.10],
        data_online_creation_crop_bbox_aspect_range_A=[0.90, 1.10],
        data_online_creation_crop_bbox_center_jitter_A=0.05,
        data_online_creation_crop_mask_aspect_ratio_A=0.0,
        data_online_creation_crop_mask_aspect_ratio_orientation_A="fixed",
        data_online_context_pixels=1,
    )

    with pytest.raises(ValueError, match="data_online_context_pixels=0"):
        validate_bbox_context_crop_options(opt)


def test_crop_image_bbox_context_uses_runtime_rect_and_metadata(tmp_path):
    image_path = tmp_path / "image.png"
    bbox_path = tmp_path / "bbox.txt"
    gradient = np.zeros((400, 400, 3), dtype=np.uint8)
    gradient[:, :, 0] = np.arange(400, dtype=np.uint16)[None, :] % 256
    Image.fromarray(gradient).save(image_path)
    bbox_path.write_text("1 100 100 192 260\n", encoding="utf-8")

    image, mask, _, _, metadata = crop_image(
        str(image_path),
        str(bbox_path),
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=False,
        crop_dim=256,
        output_dim=256,
        context_pixels=0,
        load_size=[],
        crop_center=True,
        fixed_mask_min_unmasked_border_model=0,
        return_meta=True,
        crop_mode="bbox_context",
        bbox_context_state=EXACT_STATE,
    )

    assert image.size == (256, 256)
    assert mask.size == (256, 256)
    assert metadata["crop_mode"] == "bbox_context"
    assert metadata["crop_size"] == 200
    assert metadata["bbox_context"]["source_rect"] == [46, 80, 200, 200]
