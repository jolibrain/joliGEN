import os
import torch.nn as nn
import functools
from torch.optim import lr_scheduler
import math
import torchvision.models as models

from .modules.utils import spectral_norm,init_net,init_weights,get_norm_layer,get_weights

from .modules.resnet_architecture.resnet_generator import ResnetGenerator
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
from .modules.projected_d.discriminator import ProjectedDiscriminator
from .modules.segformer.segformer_generator import Segformer,SegformerGenerator_attn

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

class Identity(nn.Module):
    def forward(self, x):
        return x

def define_G(model_input_nc, model_output_nc, G_ngf, G_netG, G_norm, G_dropout, G_spectral, model_init_type, model_init_gain,G_padding_type,data_crop_size,G_attn_nb_mask_attn,G_attn_nb_mask_input,jg_dir,G_config_segformer,G_stylegan2_num_downsampling,**unused_options):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        G_netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        G_dropout (bool) -- if use dropout layers.
        G_spectral (bool) -- if use spectral norm.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

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
    norm_layer = get_norm_layer(norm_type=G_norm)
    
    if G_netG == 'resnet_9blocks':
        net = ResnetGenerator(model_input_nc, model_output_nc, G_ngf, norm_layer=norm_layer, use_dropout=G_dropout, use_spectral=G_spectral, n_blocks=9, padding_type=G_padding_type)
    elif G_netG == 'resnet_6blocks':
        net = ResnetGenerator(model_input_nc, model_output_nc, G_ngf, norm_layer=norm_layer, use_dropout=G_dropout, use_spectral=G_spectral, n_blocks=6, padding_type=G_padding_type)
    elif G_netG == 'resnet_12blocks':
        net = ResnetGenerator(model_input_nc, model_output_nc, G_ngf, norm_layer=norm_layer, use_dropout=G_dropout, use_spectral=G_spectral, n_blocks=12, padding_type=G_padding_type)
    elif G_netG == 'resnet_3blocks':
        net = ResnetGenerator(model_input_nc, model_output_nc, G_ngf, norm_layer=norm_layer, use_dropout=G_dropout, use_spectral=G_spectral, n_blocks=3, padding_type=G_padding_type)
    elif G_netG == 'mobile_resnet_9blocks':
        net = ResnetGenerator(model_input_nc, model_output_nc, ngf=G_ngf, norm_layer=norm_layer,
                              n_blocks=9,mobile=True)
    elif G_netG == 'mobile_resnet_3blocks':
        net = ResnetGenerator(model_input_nc, model_output_nc, ngf=G_ngf, norm_layer=norm_layer,
                              n_blocks=3,mobile=True)
    elif G_netG == 'unet_128':
        net = UnetGenerator(model_input_nc, model_output_nc, 7, G_ngf, norm_layer=norm_layer, use_dropout=G_dropout)
    elif G_netG == 'unet_256':
        net = UnetGenerator(model_input_nc, model_output_nc, 8, G_ngf, norm_layer=norm_layer, use_dropout=G_dropout)
    elif G_netG == 'resnet_attn':
        net = ResnetGenerator_attn(model_input_nc, model_output_nc,G_attn_nb_mask_attn,G_attn_nb_mask_input, G_ngf, n_blocks=9, use_spectral=G_spectral,padding_type=G_padding_type)
    elif G_netG == 'mobile_resnet_attn':
        net = ResnetGenerator_attn(model_input_nc, model_output_nc,G_attn_nb_mask_attn,G_attn_nb_mask_input, G_ngf, n_blocks=9, use_spectral=G_spectral,padding_type=G_padding_type,mobile=True)
    elif G_netG == 'stylegan2':
        net = StyleGAN2Generator(model_input_nc, model_output_nc,G_ngf, use_dropout=G_dropout,stylegan2_num_downsampling=G_stylegan2_num_downsampling,img_size=data_crop_size)
        return net
    elif G_netG == 'smallstylegan2':
        net = StyleGAN2Generator(model_input_nc, model_output_nc,G_ngf, use_dropout=G_dropout, n_blocks=2,stylegan2_num_downsampling=G_stylegan2_num_downsampling,img_size=data_crop_size)
        return net
    elif G_netG == 'segformer_attn_conv':
        net = SegformerGenerator_attn(jg_dir,G_config_segformer,img_size=data_crop_size,nb_mask_attn=G_attn_nb_mask_attn,nb_mask_input=G_attn_nb_mask_input,final_conv=True)
        return net
    elif G_netG == 'segformer_conv':
        net = Segformer(jg_dir,G_config_segformer,img_size=data_crop_size,num_classes=256,final_conv=True)
        return net
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % G_netG)
    return init_net(net, model_init_type, model_init_gain)

def define_D(netD, model_input_nc, D_ndf, D_n_layers, D_norm, D_dropout, D_spectral, model_init_type, model_init_gain, D_no_antialias,data_crop_size,D_proj_network_type,D_proj_interp,D_proj_config_segformer,D_proj_weight_segformer,jg_dir,**unused_options):
    """Create a discriminator

    Parameters:
        model_input_nc (int)     -- the number of channels in input images
        D_ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        D_n_layers (int)   -- the number of conv layers in the discriminator; effective when D_netD=='n_layers'
        D_norm (str)         -- the type of normalization layers used in the network.
        D_dropout (bool) -- whether to use dropout layers
        D_spectral(bool) -- whether to use spectral norm
        model_init_type (str)    -- the name of the initialization method.
        model_init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <D_n_layers> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=D_norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(model_input_nc, D_ndf, n_layers=3, norm_layer=norm_layer, use_dropout=D_dropout, use_spectral=D_spectral)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(model_input_nc, D_ndf, D_n_layers, norm_layer=norm_layer, use_dropout=D_dropout, use_spectral=D_spectral)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(model_input_nc, D_ndf, norm_layer=norm_layer)
    elif 'stylegan2' in netD: # global D from sty2 repo
        net = StyleGAN2Discriminator(model_input_nc, D_ndf, D_n_layers, no_antialias=D_no_antialias, img_size=data_crop_size,netD=netD)
    elif netD in model_classes : # load torchvision model
        nclasses=1
        template=netD
        net = torch_model(model_input_nc, D_ndf, nclasses,opt.data_crop_size, template, pretrained=False)
        return net
    elif netD == 'projected_d': # D in projected feature space
        net = ProjectedDiscriminator(D_proj_network_type,interp=224 if data_crop_size < 224 else D_proj_interp,config_path=os.path.join(jg_dir,D_proj_config_segformer),weight_path=os.path.join(jg_dir,D_proj_weight_segformer))
        return net # no init since custom frozen backbone
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, model_init_type, model_init_gain)

def define_C(model_output_nc, f_s_nf,data_crop_size, f_s_semantic_nclasses, train_sem_cls_template, model_init_type, model_init_gain, train_sem_cls_pretrained,**unused_options):
    img_size = data_crop_size
    if train_sem_cls_template == 'basic':
        netC = Classifier(model_output_nc, f_s_nf, f_s_semantic_nclasses,img_size)
    else:
        netC = torch_model(model_output_nc, f_s_nf, f_s_semantic_nclasses, img_size, train_sem_cls_template, train_sem_cls_pretrained)
    return init_net(netC, model_init_type, model_init_gain)

def define_f(f_s_net,model_input_nc, f_s_semantic_nclasses, model_init_type, model_init_gain,f_s_config_segformer,f_s_weight_segformer,jg_dir,data_crop_size,**unused_options):
    if f_s_net == 'vgg':
        net = VGG16_FCN8s(f_s_semantic_nclasses,pretrained = False, weights_init =None,output_last_ft=False)
    elif f_s_net =='unet':
        net = UNet(classes=f_s_semantic_nclasses,input_nc=model_input_nc)
    elif f_s_net == 'segformer':
        net = Segformer(jg_dir,f_s_config_segformer,img_size=data_crop_size,num_classes=f_s_semantic_nclasses,final_conv=False)
        weights = get_weights(os.path.join(jg_dir,f_s_weight_segformer))
        net.net.load_state_dict(weights,strict=False)
        return net
        
    return init_net(net, model_init_type, model_init_gain)

def define_classifier_w(pretrained=False, weights_init='', init_type='normal', init_gain=0.02,init_weight=True,img_size_dec=256):
    net = Classifier_w(img_size_dec=img_size_dec)
    return init_net(net, init_type, init_gain)

def define_inception(device,dims):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    return model

def define_F(alg_cut_netF_nc,alg_cut_netF, alg_cut_netF_norm, alg_cut_netF_dropout, model_init_type, model_init_gain,**unused_options):
    if alg_cut_netF == 'global_pool':
        net = PoolingF()
    elif alg_cut_netF == 'sample':
        net = PatchSampleF(use_mlp=False, init_type=model_init_type, init_gain=model_init_gain, nc=alg_cut_netF_nc)
    elif alg_cut_netF == 'mlp_sample':
        net = PatchSampleF(use_mlp=True, init_type=model_init_type, init_gain=model_init_gain, nc=alg_cut_netF_nc)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return init_net(net, model_init_type, model_init_gain)

