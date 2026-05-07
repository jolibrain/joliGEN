import numpy as np
from PIL import Image

from data.online_creation import crop_image


def _write_sample(tmp_path, bbox_line):
    img_path = tmp_path / "image.png"
    bbox_path = tmp_path / "bbox.txt"
    Image.new("RGB", (256, 256), color=(127, 127, 127)).save(img_path)
    bbox_path.write_text(bbox_line)
    return str(img_path), str(bbox_path)


def _mask_bbox_size(mask):
    bbox = Image.fromarray(np.array(mask)).getbbox()
    assert bbox is not None
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


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
