from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from data.base_dataset import build_masked_global_context_image
from data.online_creation import crop_image, prepare_online_image
from util.diff_aug import DiffAugment


def _identity_camera_plan(**overrides):
    plan = {
        "rgb_gains": [1.0, 1.0, 1.0],
        "exposure": 1.0,
        "channel_offsets": [0.0, 0.0, 0.0],
        "black_point": 0.0,
        "white_point": 1.0,
        "gamma": 1.0,
        "saturation": 1.0,
        "tone": 0.0,
        "color_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "gradient_x": 0.0,
        "gradient_y": 0.0,
        "vignette": 0.0,
        "vignette_center": (0.0, 0.0),
    }
    plan.update(overrides)
    return plan


def _detail_plan(mode, **overrides):
    plan = {
        "mode": mode,
        "strength": 1.0,
        "soften_sigma": 0.8,
        "unsharp_sigma": 0.8,
        "unsharp_amount": 1.5,
        "clarity_sigma": 2.0,
        "clarity_amount": 0.5,
        "phone_denoise_sigma": 0.3,
        "phone_unsharp_sigma": 0.7,
        "phone_unsharp_amount": 1.0,
        "phone_clarity_sigma": 2.0,
        "phone_clarity_amount": 0.25,
    }
    plan.update(overrides)
    return plan


@pytest.mark.parametrize("name", ["camera_color", "detail"])
def test_new_policy_strength_must_be_in_unit_interval(name):
    kwargs = {f"{name}_strength": 1.01}
    with pytest.raises(ValueError, match="must be in"):
        DiffAugment(name, p=1.0, **kwargs)


def test_zero_strength_is_exact_identity():
    augment = DiffAugment(
        "camera_color,detail",
        p=1.0,
        camera_color_strength=0.0,
        detail_strength=0.0,
    )
    image = torch.rand(2, 3, 17, 19) * 2.0 - 1.0

    images, _ = augment.apply_synchronized([image], [])

    assert torch.equal(images[0], image)


def test_camera_color_plan_is_shared_across_frames_and_resolutions():
    augment = DiffAugment("camera_color", p=1.0)
    plan = _identity_camera_plan(
        rgb_gains=[1.10, 0.90, 0.80],
        exposure=0.8,
        gamma=1.2,
        gradient_x=0.1,
        vignette=0.2,
    )
    local = torch.zeros(1, 2, 3, 16, 16)
    context = torch.zeros(1, 2, 3, 8, 8)

    with patch.object(augment, "_sample_camera_color_params", return_value=plan):
        images, _ = augment.apply_synchronized([local, context], [])

    assert torch.allclose(images[0][0, 0], images[0][0, 1])
    assert torch.allclose(images[1][0, 0], images[1][0, 1])
    assert not torch.equal(images[0], local)
    assert images[0].min() >= -1.0 and images[0].max() <= 1.0


def test_camera_matrix_preserves_neutral_gray():
    augment = DiffAugment("camera_color", p=1.0)
    plan = _identity_camera_plan(
        color_matrix=[
            [0.97, 0.02, 0.01],
            [-0.01, 1.03, -0.02],
            [0.04, -0.01, 0.97],
        ]
    )
    gray = torch.full((3, 12, 12), 0.25)

    output = augment.apply_camera_color_plan(gray, plan)

    assert torch.allclose(output[0], output[1], atol=1e-6)
    assert torch.allclose(output[1], output[2], atol=1e-6)


def test_sampled_camera_parameters_stay_in_documented_ranges():
    augment = DiffAugment("camera_color", p=1.0)

    for _ in range(250):
        plan = augment._sample_camera_color_params()
        assert 0.68 <= plan["rgb_gains"][0] <= 1.416
        assert 0.765 <= plan["rgb_gains"][1] <= 1.298
        assert 0.68 <= plan["rgb_gains"][2] <= 1.416
        assert 0.5 <= plan["exposure"] <= 2.0
        assert all(-20 / 255 <= value <= 20 / 255 for value in plan["channel_offsets"])
        assert -0.04 <= plan["black_point"] <= 0.04
        assert 0.90 <= plan["white_point"] <= 1.10
        assert 0.70 <= plan["gamma"] <= 1.40
        assert 0.80 <= plan["saturation"] <= 1.20
        assert -0.50 <= plan["tone"] <= 0.50
        assert -0.15 <= plan["gradient_x"] <= 0.15
        assert -0.15 <= plan["gradient_y"] <= 0.15
        assert 0.0 <= plan["vignette"] <= 0.35
        assert all(sum(row) == pytest.approx(1.0) for row in plan["color_matrix"])


def test_camera_and_detail_restore_excluded_black_pixels():
    augment = DiffAugment("camera_color,detail", p=1.0)
    image = torch.full((1, 3, 16, 16), 0.25)
    exclusion = torch.zeros(1, 1, 16, 16)
    exclusion[:, :, 4:12, 4:12] = 1.0
    image = image.masked_fill(exclusion.bool(), -1.0)

    images, _ = augment.apply_synchronized(
        [image], [], image_exclusion_masks=[exclusion]
    )

    assert torch.all(images[0][:, :, 4:12, 4:12] == -1.0)


def test_detail_unsharp_increases_and_soften_decreases_edge_energy():
    augment = DiffAugment("detail", p=1.0)
    checker = torch.tensor(
        [[(x + y) % 2 for x in range(32)] for y in range(32)], dtype=torch.float32
    )
    image = checker.unsqueeze(0).repeat(3, 1, 1) * 1.2 - 0.6

    sharpened = augment._apply_detail(image, _detail_plan("unsharp"), image.shape[-2:])
    softened = augment._apply_detail(image, _detail_plan("soften"), image.shape[-2:])

    def edge_energy(value):
        return (value[..., 1:, :] - value[..., :-1, :]).abs().mean() + (
            value[..., :, 1:] - value[..., :, :-1]
        ).abs().mean()

    assert edge_energy(sharpened) > edge_energy(image)
    assert edge_energy(softened) < edge_energy(image)


def test_excluded_policy_is_not_applied():
    augment = DiffAugment("camera_color,detail", p=1.0)
    image = torch.rand(1, 3, 16, 16) * 2.0 - 1.0
    augment._sample_camera_color_params = pytest.fail
    augment._sample_detail_params = lambda: _detail_plan("soften")

    images, _ = augment.apply_synchronized(
        [image], [], excluded_policies={"camera_color"}
    )

    assert not torch.equal(images[0], image)


def test_b2b_pre_crop_route_excludes_model_side_camera_policy():
    from models.b2b_model import B2BModel

    class CapturingAugment:
        def __init__(self):
            self.excluded_policies = None

        def apply_synchronized(
            self,
            *,
            image_tensors,
            mask_tensors,
            image_exclusion_masks,
            excluded_policies=None,
        ):
            self.excluded_policies = excluded_policies
            return image_tensors, mask_tensors

    model = B2BModel.__new__(B2BModel)
    model.opt = SimpleNamespace(
        isTrain=True,
        alg_b2b_mask_prediction=False,
        dataaug_diff_aug_camera_color_pre_crop=True,
    )
    model.diff_augment = CapturingAugment()
    model.gt_image = torch.zeros(1, 2, 3, 8, 8)
    model.y_t = torch.zeros(1, 2, 3, 8, 8)
    model.mask = torch.ones(1, 2, 1, 8, 8)
    model.global_context = None

    model._apply_b2b_diff_augment()

    assert model.diff_augment.excluded_policies == {"camera_color"}


def test_prepared_image_is_reused_by_crop_and_global_context(tmp_path):
    pixels = np.full((16, 16, 3), 96, dtype=np.uint8)
    pixels[:, 8:, 0] = 160
    image_path = tmp_path / "frame.png"
    bbox_path = tmp_path / "frame.txt"
    Image.fromarray(pixels).save(image_path)
    bbox_path.write_text("1 6 6 10 10\n")

    augment = DiffAugment("camera_color", p=1.0)
    prepared = prepare_online_image(
        str(image_path),
        [],
        camera_augment=augment,
        camera_color_plan=_identity_camera_plan(exposure=0.75),
    )

    crop_kwargs = {
        "img_path": str(image_path),
        "bbox_path": str(bbox_path),
        "mask_random_offset": [0.0],
        "mask_delta": [[]],
        "crop_delta": 0,
        "mask_square": False,
        "crop_dim": 8,
        "output_dim": 8,
        "context_pixels": 0,
        "load_size": [],
        "crop_center": True,
        "fixed_mask_min_unmasked_border_model": 1,
        "prepared_image": prepared,
        "return_meta": True,
    }
    with patch("data.online_creation.load_image", side_effect=AssertionError):
        crop, _, _, _, crop_meta = crop_image(**crop_kwargs)
    with patch("data.base_dataset.load_image", side_effect=AssertionError):
        context = build_masked_global_context_image(
            str(image_path), crop_meta, [], prepared_image=prepared
        )

    assert crop.size == (8, 8)
    assert context.size == prepared.image.size
    assert np.array(context)[0, 0].tolist() == np.array(prepared.image)[0, 0].tolist()


def _pre_crop_option_config(**overrides):
    config = {
        "gpu_ids": "-1",
        "model_type": "b2b",
        "data_dataset_mode": "self_supervised_vid_mask_online",
        "dataaug_diff_aug_policy": "camera_color,detail,wild",
        "dataaug_diff_aug_camera_color_pre_crop": True,
    }
    config.update(overrides)
    return config


def test_pre_crop_camera_options_parse_for_supported_b2b_loader():
    from options.train_options import TrainOptions

    opt = TrainOptions().parse_json(
        _pre_crop_option_config(), save_config=False, set_device=False
    )

    assert opt.dataaug_diff_aug_camera_color_pre_crop is True
    assert opt.dataaug_diff_aug_camera_color_strength == 1.0
    assert opt.dataaug_diff_aug_detail_strength == 1.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"dataaug_diff_aug_camera_color_strength": 1.1},
            "camera_color_strength",
        ),
        ({"dataaug_diff_aug_policy": "detail"}, "requires camera_color"),
        ({"data_dataset_mode": "unaligned"}, "requires --data_dataset_mode"),
    ],
)
def test_pre_crop_camera_options_reject_invalid_combinations(overrides, message):
    from options.train_options import TrainOptions

    with pytest.raises(ValueError, match=message):
        TrainOptions().parse_json(
            _pre_crop_option_config(**overrides),
            save_config=False,
            set_device=False,
        )
