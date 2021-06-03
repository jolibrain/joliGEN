import torch.nn as nn
import functools
from torch.optim import lr_scheduler
import math
import torchvision.models as models

from .modules.utils import spectral_norm,init_net,init_weights,get_norm_layer

from .modules.resnet_architecture.resnet_generator import ResnetGenerator
from .modules.resnet_architecture.mobile_resnet_generator import MobileResnetGenerator,MobileResnetGenerator_attn
from .modules.unet_architecture.unet_generator import UnetGenerator
from .modules.resnet_architecture.resnet_generator import ResnetGenerator_attn
from .modules.discriminators import NLayerDiscriminator
from .modules.discriminators import PixelDiscriminator
from .modules.classifiers import Classifier, VGG16_FCN8s, torch_model,model_classes
from .modules.UNet_classification import UNet
from .modules.classifiers import Classifier_w
from .modules.fid.pytorch_fid.inception import InceptionV3
from .modules.stylegan_networks import StyleGAN2Discriminator, StyleGAN2Generator, TileStyleGAN2Discriminator
from .modules.cut_networks import PatchSampleF

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

class Identity(nn.Module):
    def forward(self, x):
        return x

def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, use_spectral=False, init_type='normal', init_gain=0.02, gpu_ids=[], decoder=True, wplus=True, wskip=False, init_weight=True, img_size=128, img_size_dec=128,nb_attn = 10,nb_mask_input=1,padding_type='reflect',opt=None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        use_spectral (bool) -- if use spectral norm.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_spectral=use_spectral, n_blocks=9, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, padding_type=padding_type)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_spectral=use_spectral, n_blocks=6, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, padding_type=padding_type)
    elif netG == 'resnet_12blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_spectral=use_spectral, n_blocks=12, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, padding_type=padding_type)
    elif netG == 'mobile_resnet_9blocks':
        net = MobileResnetGenerator(input_nc, output_nc, ngf=ngf, norm_layer=norm_layer,
                                        dropout_rate=0.0, n_blocks=9, wplus=wplus,
                                        init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids,
                                        img_size=img_size, img_size_dec=img_size_dec)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'resnet_attn':
        net = ResnetGenerator_attn(input_nc, output_nc, ngf, n_blocks=9, use_spectral=use_spectral,nb_attn = nb_attn,nb_mask_input=nb_mask_input,padding_type=padding_type)
    elif netG == 'mobile_resnet_attn':
        net = MobileResnetGenerator_attn(input_nc, output_nc, ngf, n_blocks=9, use_spectral=use_spectral,nb_attn = nb_attn,nb_mask_input=nb_mask_input,padding_type=padding_type)
    elif netG == 'stylegan2':
        net = StyleGAN2Generator(input_nc, output_nc,ngf, use_dropout=use_dropout, opt=opt)
    elif netG == 'smallstylegan2':
        net = StyleGAN2Generator(input_nc, output_nc,ngf, use_dropout=use_dropout, n_blocks=2, opt=opt)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids,init_weight=init_weight and ('stylegan2' not in netG))

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_dropout=False, use_spectral=False, init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[],opt=None):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        use_dropout (bool) -- whether to use dropout layers
        use_spectral(bool) -- whether to use spectral norm
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_dropout=use_dropout, use_spectral=use_spectral)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_dropout=use_dropout, use_spectral=use_spectral)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif 'stylegan2' in netD: # global D from sty2 repo
        net = StyleGAN2Discriminator(input_nc, ndf, n_layers_D, no_antialias=no_antialias, opt=opt)
    elif netD in model_classes : # load torchvision model
        nclasses=1
        template=netD
        net = torch_model(input_nc, ndf, nclasses,opt.crop_size, template, pretrained=False)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids,init_weight= 'stylegan2' not in netD)

def define_C(input_nc, ndf,img_size, init_type='normal', init_gain=0.02, gpu_ids=[], nclasses=10, template='basic', pretrained=False):
    if template == 'basic':
        netC = Classifier(input_nc, ndf, nclasses,img_size)
    else:
        netC = torch_model(input_nc, ndf, nclasses, img_size, template, pretrained)
    return init_net(netC, init_type, init_gain, gpu_ids)

def define_f(input_nc, nclasses, init_type='normal', init_gain=0.02, gpu_ids=[], fs_light=False):
    if not fs_light:
        net = VGG16_FCN8s(nclasses,pretrained = False, weights_init =None,output_last_ft=False)
    else:
        net = UNet(classes=nclasses,input_nc=input_nc)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_classifier_w(pretrained=False, weights_init='', init_type='normal', init_gain=0.02, gpu_ids=[],init_weight=True,img_size_dec=256):
    net = Classifier_w(img_size_dec=img_size_dec)
    return init_net(net, init_type, init_gain, gpu_ids,init_weight=init_weight)

def define_inception(device,dims):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    return model

def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[], opt=None):
    if netF == 'global_pool':
        net = PoolingF()
    elif netF == 'reshape':
        net = ReshapeF()
    elif netF == 'sample':
        net = PatchSampleF(use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'mlp_sample':
        net = PatchSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'strided_conv':
        net = StridedConvF(init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return init_net(net, init_type, init_gain, gpu_ids)

