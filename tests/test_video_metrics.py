from types import SimpleNamespace

import torch

import models.base_model as base_model_module
from models.base_model import BaseModel
from util.util import MAX_INT


class VideoMetricModel:
    def __init__(self):
        self.opt = SimpleNamespace(
            train_nb_img_max_fid=MAX_INT,
            model_type="palette",
            isTrain=True,
            test_batch_size=2,
            G_netG="vit_vid",
            data_direction="AtoB",
            train_metrics_list=["PSNR"],
        )
        self.use_temporal = False
        self.use_inception = False
        self.visual_names = []
        self.inference_offsets = []

    def _is_b2b_validation_loss_enabled(self):
        return False

    def set_input(self, data):
        self.gt_image = data

    def inference(self, _batch_size, offset=0):
        self.inference_offsets.append(offset)
        self.fake_B = self.gt_image.clone()


def test_video_metrics_process_every_test_batch(monkeypatch):
    monkeypatch.setattr(
        base_model_module, "ssim", lambda _real, _fake: torch.tensor(1.0)
    )

    def fake_psnr(real, _fake, reduction="mean"):
        if reduction == "none":
            return torch.ones(real.shape[0])
        return torch.tensor(1.0)

    monkeypatch.setattr(base_model_module, "psnr", fake_psnr)

    model = VideoMetricModel()
    test_batches = [
        (torch.zeros(2, 2, 3, 4, 4),),
        (torch.zeros(2, 2, 3, 4, 4),),
        (torch.zeros(1, 2, 3, 4, 4),),
    ]

    BaseModel.compute_metrics_test(model, test_batches, n_epoch=1, n_iter=8000)

    assert model.inference_offsets == [0, 2, 4]
