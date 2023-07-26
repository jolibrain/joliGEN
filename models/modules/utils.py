import functools
import os

import numpy as np
import torch
import wget
from torch import nn
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.models import vgg

from util.util import tensor2im

from .op import upfirdn2d

##########################################################
# Fonctions used for networks initialisation
##########################################################


def init_net(net, init_type="normal", init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """
    init_weights(net, init_type, init_gain=init_gain)
    return net


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def get_weights(weight_path):
    try:
        weights = torch.jit.load(weight_path).state_dict()
        print("Torch script weights are detected and loaded in %s" % weight_path)
    except:
        weights = torch.load(weight_path)
    return weights


##########################################################
# Fonctions used for networks construction
##########################################################


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "none":

        def norm_layer(x):
            return Identity()

    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.train_lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(
                0, epoch + opt.train_epoch_count - opt.train_n_epochs
            ) / float(opt.train_n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.train_lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.train_lr_decay_iters, gamma=0.1
        )
    elif opt.train_lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif opt.train_lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.train_n_epochs, eta_min=0
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", opt.train_lr_policy
        )
    return scheduler


def make_layers(cfg, batch_norm=False):
    """This is almost verbatim from torchvision.models.vgg, except that the
    MaxPool2d modules are configured with ceil_mode=True.
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            modules = [conv2d, nn.ReLU(inplace=True)]
            if batch_norm:
                modules.insert(1, nn.BatchNorm2d(v))
            layers.extend(modules)
            in_channels = v
    return nn.Sequential(*layers)


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


##########################################################
# Fonctions used for
##########################################################


def _crop(input, shape, offset=0):
    _, _, h, w = shape.size()
    return input[:, :, offset : offset + h, offset : offset + w].contiguous()


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


segformer_weights = {
    "segformer_mit-b0.pth": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth",
    "segformer_mit-b1.pth": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b1_512x512_160k_ade20k/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth",
    "segformer_mit-b2.pth": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_512x512_160k_ade20k/segformer_mit-b2_512x512_160k_ade20k_20210726_112103-cbd414ac.pth",
    "segformer_mit-b3.pth": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b3_512x512_160k_ade20k/segformer_mit-b3_512x512_160k_ade20k_20210726_081410-962b98d2.pth",
    "segformer_mit-b4.pth": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b4_512x512_160k_ade20k/segformer_mit-b4_512x512_160k_ade20k_20210728_183055-7f509d7d.pth",
    "segformer_mit-b5.pth": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth",
    "segformer_mit-b5_640.pth": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_640x640_160k_ade20k/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth",
}


def download_segformer_weight(path):
    for i in range(2, len(path.split("/"))):
        temp = path.split("/")[:i]
        cur_path = "/".join(temp)
        if not os.path.isdir(cur_path):
            os.mkdir(cur_path)
    model_name = path.split("/")[-1]
    if model_name in segformer_weights:
        wget.download(segformer_weights[model_name], path)
    else:
        raise NameError(
            "There is no pretrained weight to download for %s, you need to provide a path to segformer weights."
            % model_name
        )


def download_midas_weight(model_type="DPT_Large"):
    midas = torch.hub.load("intel-isl/MiDaS:v3_1", model_type, skip_validation=True)
    midas.requires_grad_(False)
    midas.eval()
    return midas


def download_sam_weight(path):
    if not os.path.exists(path):
        if not "mobile_sam" in path:
            sam_weights = {
                "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            }
            for i in range(2, len(path.split("/"))):
                temp = path.split("/")[:i]
                cur_path = "/".join(temp)
                if not os.path.isdir(cur_path):
                    os.mkdir(cur_path)
            model_name = path.split("/")[-1]
            if model_name in sam_weights:
                wget.download(sam_weights[model_name], path)
            else:
                raise NameError(
                    "There is no pretrained weight to download for %s, you need to provide a path to sam weights."
                    % model_name
                )
        else:
            download_mobile_sam_weight(path)


def download_mobile_sam_weight(path):
    if not os.path.exists(path):
        sam_weights = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        for i in range(2, len(path.split("/"))):
            temp = path.split("/")[:i]
            cur_path = "/".join(temp)
            if not os.path.isdir(cur_path):
                os.mkdir(cur_path)
        model_name = path.split("/")[-1]
        if model_name in sam_weights:
            wget.download(sam_weights, path)
        else:
            raise NameError(
                "There is no pretrained weight to download for %s, you need to provide a path to mobileSam weights."
                % model_name
            )


def predict_depth(img, midas, model_type):
    # img must be RGB
    input_size = 384
    if model_type == "MiDas_small" or model_type == "DPT_SwinV2_T_256":
        input_size = 256
    elif model_type == "DPT_BEiT_L_512":
        input_size = 512
    elif model_type == "DPT_LeViT_224":
        input_size = 224
    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
        ]
    )
    prediction = midas(transform(img))  # .unsqueeze(dim=1)
    return prediction


### functions for frequency space


def get_haar_wavelet(in_channels):
    haar_wav_l = 1 / (2**0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2**0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h

    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh


class HaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer("ll", ll)
        self.register_buffer("lh", lh)
        self.register_buffer("hl", hl)
        self.register_buffer("hh", hh)

    def forward(self, input):
        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)

        return torch.cat((ll, lh, hl, hh), 1)


class InverseHaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer("ll", ll)
        self.register_buffer("lh", -lh)
        self.register_buffer("hl", -hl)
        self.register_buffer("hh", hh)

    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)
        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))

        return ll + lh + hl + hh
