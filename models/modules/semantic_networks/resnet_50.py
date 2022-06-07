import os
from torch import nn


class Resnet50Segmentor(nn.Module):
    def __init__(
        self,
        jg_dir,
        f_s_config,
        img_size,
        num_classes=10,
        final_conv=False,
        padding_type="zeros",
    ):
        super().__init__()
        import mmcv

        cfg = mmcv.Config.fromfile(os.path.join(jg_dir, f_s_config))

        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        # cfg.model.decode_head.num_classes = num_classes
        from mmseg.models import build_segmentor

        self.resnet50 = build_segmentor(
            cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg")
        )

    def forward(self, x):
        return self.resnet50.forward_dummy(x)
