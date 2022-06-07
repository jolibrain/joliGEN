import os
from torch import nn
import torch


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


def convert_batchnorm(module):
    r"""Helper function to convert all :attr:`BatchNorm*D` layers in the model to
    :class:`torch.nn.SyncBatchNorm` layers.

    Args:
        module (nn.Module): module containing one or more attr:`BatchNorm*D` layers
        process_group (optional): process group to scope synchronization,
            default is the whole world

    Returns:
        The original :attr:`module` with the converted :class:`torch.nn.SyncBatchNorm`
        layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
        a new :class:`torch.nn.SyncBatchNorm` layer object will be returned
        instead.

    Example::

        >>> # Network with nn.BatchNorm layer
        >>> module = torch.nn.Sequential(
        >>>            torch.nn.Linear(20, 100),
        >>>            torch.nn.BatchNorm1d(100),
        >>>          ).cuda()
        >>> # creating process group (optional)
        >>> # process_ids is a list of int identifying rank ids.
        >>> process_group = torch.distributed.new_group(process_ids)
        >>> sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)

    """
    module_output = module
    print(type(module))
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, convert_batchnorm(child))
    del module
    return module_output
