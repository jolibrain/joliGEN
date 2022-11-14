from packaging import version
import torch
from torch import nn

from .base_NCE import BaseNCELoss


class PatchNCELoss(BaseNCELoss):
    def __init__(self, opt):
        super().__init__(opt)
