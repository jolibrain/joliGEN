import numpy as np
from PIL import Image

import data.online_creation as online_creation
from data.online_creation import crop_image


def _write_sample(tmp_path, bbox_line):
    img_path = tmp_path / "image.png"
    bbox_path = tmp_path / "bbox.txt"
    Image.new("RGB", (256, 256), color=(127, 127, 127)).save(img_path)
    bbox_path.write_text(bbox_line)
    return str(img_path), str(bbox_path)


def _write_sized_sample(tmp_path, size, bbox_line):
    img_path = tmp_path / "image.png"
    bbox_path = tmp_path / "bbox.txt"
    Image.new("RGB", size, color=(127, 127, 127)).save(img_path)
    bbox_path.write_text(bbox_line)
    return str(img_path), str(bbox_path)


def _mask_bbox_size(mask):
    bbox = Image.fromarray(np.array(mask)).getbbox()
    assert bbox is not None
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _mask_bbox(mask):
    bbox = Image.fromarray(np.array(mask)).getbbox()
    assert bbox is not None
    return bbox


def test_crop_image_fixed_model_mask_size_exact_square_with_crop_coordinates(tmp_path):
    img_path, bbox_path = _write_sample(tmp_path, "1 96 96 116 116\n")

    crop_coordinates = crop_image(
        img_path,
        bbox_path,
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=False,
        crop_dim=128,
        output_dim=128,
        context_pixels=0,
        load_size=[],
        get_crop_coordinates=True,
        crop_center=True,
        fixed_mask_size_model=64,
    )
    _, mask, _, _ = crop_image(
        img_path,
        bbox_path,
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=False,
        crop_dim=128,
        output_dim=128,
        context_pixels=0,
        load_size=[],
        crop_coordinates=crop_coordinates,
        crop_center=True,
        fixed_mask_size_model=64,
    )

    assert _mask_bbox_size(mask) == (64, 64)


def test_crop_image_keep_ratio_load_size_scales_crop_without_distortion(tmp_path):
    img_path, bbox_path = _write_sized_sample(tmp_path, (200, 100), "1 50 25 90 65\n")
    load_size = [100, 100]

    crop_coordinates = crop_image(
        img_path,
        bbox_path,
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=False,
        crop_dim=100,
        output_dim=100,
        context_pixels=0,
        load_size=load_size,
        load_size_keep_ratio=True,
        get_crop_coordinates=True,
        crop_center=True,
    )
    _, _, _, _, meta = crop_image(
        img_path,
        bbox_path,
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=False,
        crop_dim=100,
        output_dim=100,
        context_pixels=0,
        load_size=load_size,
        load_size_keep_ratio=True,
        crop_coordinates=crop_coordinates,
        crop_center=True,
        return_meta=True,
    )

    assert load_size == [100, 100]
    assert meta["loaded_width"] == 100
    assert meta["loaded_height"] == 50
    assert meta["crop_size"] == 50


def test_crop_image_keep_ratio_load_size_uses_largest_side_for_portrait(tmp_path):
    img_path, bbox_path = _write_sized_sample(
        tmp_path, (1080, 1920), "1 400 700 700 1200\n"
    )

    _, _, _, _, meta = crop_image(
        img_path,
        bbox_path,
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=False,
        crop_dim=512,
        output_dim=512,
        context_pixels=0,
        load_size=[1920, 1080],
        load_size_keep_ratio=True,
        crop_center=True,
        return_meta=True,
    )

    assert meta["loaded_width"] == 1080
    assert meta["loaded_height"] == 1920


def test_crop_image_fixed_model_mask_size_keeps_larger_containing_square(tmp_path):
    img_path, bbox_path = _write_sample(tmp_path, "1 64 80 164 120\n")

    _, mask, _, _ = crop_image(
        img_path,
        bbox_path,
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=True,
        crop_dim=128,
        output_dim=128,
        context_pixels=0,
        load_size=[],
        crop_center=True,
        fixed_mask_size_model=64,
    )

    assert _mask_bbox_size(mask) == (100, 100)


def test_crop_image_fixed_model_mask_size_leaves_default_unmasked_border(tmp_path):
    img_path, bbox_path = _write_sample(tmp_path, "1 48 48 80 80\n")

    _, mask, _, _ = crop_image(
        img_path,
        bbox_path,
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=False,
        crop_dim=128,
        output_dim=128,
        context_pixels=0,
        load_size=[],
        crop_center=True,
        fixed_mask_size_model=128,
    )

    assert _mask_bbox(mask) == (4, 4, 124, 124)


def test_crop_image_fixed_model_mask_size_shifts_edge_mask_inside_border(tmp_path):
    img_path, bbox_path = _write_sample(tmp_path, "1 0 0 20 20\n")

    _, mask, _, _ = crop_image(
        img_path,
        bbox_path,
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=False,
        crop_dim=128,
        output_dim=128,
        context_pixels=0,
        load_size=[],
        crop_center=True,
        fixed_mask_size_model=128,
        fixed_mask_min_unmasked_border_model=4,
    )

    assert _mask_bbox(mask) == (4, 4, 124, 124)


def test_crop_image_fixed_model_mask_size_clamps_large_bbox_to_border(tmp_path):
    img_path, bbox_path = _write_sample(tmp_path, "1 0 0 256 256\n")

    _, mask, _, _ = crop_image(
        img_path,
        bbox_path,
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=True,
        crop_dim=128,
        output_dim=128,
        context_pixels=0,
        load_size=[],
        crop_center=True,
        fixed_mask_size_model=64,
        fixed_mask_min_unmasked_border_model=4,
    )

    assert _mask_bbox(mask) == (4, 4, 124, 124)


def test_crop_image_broaden_rect_aug_disabled_is_noop(tmp_path):
    img_path, bbox_path = _write_sample(tmp_path, "1 96 96 116 116\n")

    _, mask, _, _ = crop_image(
        img_path,
        bbox_path,
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=False,
        crop_dim=128,
        output_dim=128,
        context_pixels=0,
        load_size=[],
        crop_center=True,
        broaden_rect_aug=False,
    )

    assert _mask_bbox_size(mask) == (20, 20)


def test_crop_image_broaden_rect_aug_side_expand_contains_original(
    tmp_path, monkeypatch
):
    img_path, bbox_path = _write_sample(tmp_path, "1 96 96 116 116\n")
    monkeypatch.setattr(online_creation.random, "random", lambda: 0.30)
    monkeypatch.setattr(online_creation.random, "uniform", lambda _low, _high: 0.50)

    _, mask, _, _ = crop_image(
        img_path,
        bbox_path,
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=False,
        crop_dim=128,
        output_dim=128,
        context_pixels=0,
        load_size=[],
        crop_center=True,
        broaden_rect_aug=True,
    )

    assert _mask_bbox_size(mask) == (40, 40)


def test_crop_image_broaden_rect_aug_target_area(tmp_path, monkeypatch):
    img_path, bbox_path = _write_sample(tmp_path, "1 96 96 116 116\n")
    monkeypatch.setattr(online_creation.random, "random", lambda: 0.60)
    monkeypatch.setattr(online_creation.random, "uniform", lambda _low, _high: 4.0)

    _, mask, _, _ = crop_image(
        img_path,
        bbox_path,
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=False,
        crop_dim=128,
        output_dim=128,
        context_pixels=0,
        load_size=[],
        crop_center=True,
        broaden_rect_aug=True,
    )

    assert _mask_bbox_size(mask) == (40, 40)


def test_crop_image_broaden_rect_aug_target_aspect(tmp_path, monkeypatch):
    img_path, bbox_path = _write_sample(tmp_path, "1 96 96 116 116\n")
    monkeypatch.setattr(online_creation.random, "random", lambda: 0.80)
    monkeypatch.setattr(online_creation.random, "uniform", lambda _low, _high: 2.0)

    _, mask, _, _ = crop_image(
        img_path,
        bbox_path,
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=False,
        crop_dim=128,
        output_dim=128,
        context_pixels=0,
        load_size=[],
        crop_center=True,
        broaden_rect_aug=True,
    )

    assert _mask_bbox_size(mask) == (40, 20)


def test_crop_image_broaden_rect_aug_high_roll_uses_bounded_aspect(
    tmp_path, monkeypatch
):
    img_path, bbox_path = _write_sample(tmp_path, "1 32 96 52 116\n")
    monkeypatch.setattr(online_creation.random, "random", lambda: 0.95)
    monkeypatch.setattr(online_creation.random, "uniform", lambda _low, _high: 2.85)

    _, mask, _, _ = crop_image(
        img_path,
        bbox_path,
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=False,
        crop_dim=128,
        output_dim=128,
        context_pixels=0,
        load_size=[],
        crop_center=True,
        broaden_rect_aug=True,
    )

    width, height = _mask_bbox_size(mask)
    assert width > 20
    assert height == 20
    assert width < 128


def test_crop_image_broaden_rect_aug_reuses_crop_coordinate_state(
    tmp_path, monkeypatch
):
    img_path, bbox_path = _write_sample(tmp_path, "1 96 96 116 116\n")
    monkeypatch.setattr(online_creation.random, "random", lambda: 0.30)
    monkeypatch.setattr(online_creation.random, "uniform", lambda _low, _high: 0.50)

    crop_coordinates = crop_image(
        img_path,
        bbox_path,
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=False,
        crop_dim=128,
        output_dim=128,
        context_pixels=0,
        load_size=[],
        get_crop_coordinates=True,
        crop_center=True,
        broaden_rect_aug=True,
    )

    monkeypatch.setattr(online_creation.random, "random", lambda: 0.0)
    _, mask, _, _ = crop_image(
        img_path,
        bbox_path,
        mask_random_offset=[0.0],
        mask_delta=[[]],
        crop_delta=0,
        mask_square=False,
        crop_dim=128,
        output_dim=128,
        context_pixels=0,
        load_size=[],
        crop_coordinates=crop_coordinates,
        crop_center=True,
        broaden_rect_aug=True,
    )

    assert _mask_bbox_size(mask) == (40, 40)
