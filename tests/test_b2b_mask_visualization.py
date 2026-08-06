import torch

from models.b2b_model import B2BModel
from util.util import tensor2im


def test_b2b_semantic_mask_visual_uses_class_palette():
    model = B2BModel.__new__(B2BModel)
    mask = torch.tensor([[[[0, 1], [2, 0]]]], dtype=torch.int64)

    for name in ("mask_", "augmented_mask_"):
        visual = model._b2b_visual_mask_tensor(name, mask)

        assert visual.shape == (1, 2, 2)
        assert torch.equal(visual, mask.squeeze(0))
        assert tensor2im(visual).tolist() == [
            [[0, 0, 0], [0, 255, 0]],
            [[255, 0, 0], [0, 0, 0]],
        ]


def test_b2b_predicted_mask_visual_stays_binary():
    model = B2BModel.__new__(B2BModel)
    mask = torch.tensor([[[[-1.0, 0.0], [0.25, 1.0]]]])

    visual = model._b2b_visual_mask_tensor("predicted_mask_2_steps_", mask)
    expected = (mask > 0).float().repeat(1, 3, 1, 1) * 2.0 - 1.0

    assert visual.shape == (1, 3, 2, 2)
    assert torch.equal(visual, expected)
