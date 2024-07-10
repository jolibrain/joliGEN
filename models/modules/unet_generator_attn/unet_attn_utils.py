"""
Various utilities for neural networks.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from .switchable_norm import SwitchNorm2d


class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[1]
        input_channels = x.shape[2]
        x = rearrange(x, "b f c h w -> (b f) c h w")
        expected_channels = self.in_channels
        if input_channels != expected_channels:
            raise ValueError(
                f"Expected input channels: {expected_channels}, but got: {input_channels}"
            )

        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b f c h w", f=video_length)

        return x


class InflatedGroupNorm(nn.GroupNorm):
    def forward(self, x):
        video_length = x.shape[1]

        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b f c h w", f=video_length)

        return x


class GroupNorm(nn.Module):
    def __init__(self, group_size, channels):
        super().__init__()
        self.norm = nn.GroupNorm(group_size, channels)

    def forward(self, x):
        return self.norm(x.float()).type(x.dtype)


class BatchNorm2dC(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels, track_running_stats=False)

    def forward(self, x):
        return self.norm(x.float()).type(x.dtype)


class BatchInstanceNorm1dC(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.InstanceNorm1d(channels)

    def forward(self, x):
        return self.norm(x.float()).type(x.dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels, norm="groupnorm32"):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    if "groupnorm" in norm:
        group_norm_size = int(norm.split("groupnorm")[1])
        return GroupNorm(group_norm_size, channels)
    elif norm == "instancenorm":
        return GroupNorm(channels, channels)
    elif norm == "layernorm":
        return GroupNorm(1, channels)
    elif norm == "batchnorm":
        return BatchNorm2dC(channels)
    elif norm == "switchablenorm":
        return SwitchNorm2d(channels)
    else:
        raise ValueError("%s is not implemented for unet attn generator" % norm)


def normalization1d(channels):
    return BatchInstanceNorm1dC(channels)


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class BatchNormXd(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
        # is this method that is overwritten by the sub-class
        # This original goal of this method was for tensor sanity checks
        # If you're ok bypassing those sanity checks (eg. if you trust your inference
        # to provide the right dimensional inputs), then you can just use this method
        # for easy conversion from SyncBatchNorm
        # (unfortunately, SyncBatchNorm does not store the original class - if it did
        #  we could return the one that was originally created)
        return


def revert_sync_batchnorm(module):
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, nn.modules.batchnorm.SyncBatchNorm):
        new_cls = BatchNormXd
        module_output = BatchNormXd(
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
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output
