import functools

import torch
from torch import nn

from models.modules.mobile_modules import SeparableConv2d

#from models.networks import WBlock, NBlock
from ...networks import init_net
from ..utils import spectral_norm,normal_init

import math
import sys

import torch.nn.functional as F

class MobileResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, dropout_rate, use_bias):
        super(MobileResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, dropout_rate, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, dropout_rate, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zeros':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            SeparableConv2d(in_channels=dim, out_channels=dim,
                            kernel_size=3, padding=p, stride=1),
            norm_layer(dim), nn.ReLU(True)
        ]
        conv_block += [nn.Dropout(dropout_rate)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zeros':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            SeparableConv2d(in_channels=dim, out_channels=dim,
                            kernel_size=3, padding=p, stride=1),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class MobileResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer=nn.InstanceNorm2d,
                 dropout_rate=0.0, n_blocks=9, padding_type='reflect',
                 wplus=True,
                 img_size=128, img_size_dec=128):
        assert (n_blocks >= 0)
        super(MobileResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        self.encoder=MobileResnetEncoder(input_nc, output_nc, ngf, norm_layer,
                 dropout_rate, n_blocks, padding_type,
                 wplus,
                 img_size, img_size_dec)

        self.decoder=MobileResnetDecoder(input_nc, output_nc, ngf, norm_layer,
                 dropout_rate, n_blocks, padding_type,
                 wplus,
                 img_size, img_size_dec)
                
    def forward(self, input, extract_layer_ids=[], encode_only=False):
        if -1 in extract_layer_ids: #if -1 is in extract_layer_ids, the output of the encoder will be returned (features just after the last layer) 
            extract_layer_ids.append(len(self.encoder))
        if len(extract_layer_ids) > 0:
            feat,feats=self.encoder(input,extract_layer_ids=extract_layer_ids, encode_only=encode_only)
            if encode_only:
                return feats
            else:
                output=self.decoder(feat)
                return output, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            output = self.encoder(input)
            output = self.decoder(output)
            return output    

class MobileResnetEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer=nn.InstanceNorm2d,
                 dropout_rate=0, n_blocks=9, padding_type='reflect',
                 wplus=True,
                 img_size=128, img_size_dec=128):
        assert (n_blocks >= 0)
        self.wplus = wplus
        super(MobileResnetEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling

        n_blocks1 = n_blocks // 3
        n_blocks2 = n_blocks1
        n_blocks3 = n_blocks - n_blocks1 - n_blocks2

        for i in range(n_blocks1):
            model += [MobileResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                        dropout_rate=dropout_rate,
                                        use_bias=use_bias)]

        for i in range(n_blocks2):
            model += [MobileResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                        dropout_rate=dropout_rate,
                                        use_bias=use_bias)]

        for i in range(n_blocks3):
            model += [MobileResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                        dropout_rate=dropout_rate,
                                        use_bias=use_bias)]
        self.model = nn.Sequential(*model)
                
    def forward(self, input, extract_layer_ids=[],encode_only=False):
        if -1 in extract_layer_ids:
            extract_layer_ids.append(len(self.encoder))
        if len(extract_layer_ids) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in extract_layer_ids:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == extract_layer_ids[-1] and encode_only:
                    # print('encoder only return features')
                    return None,feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            output = self.model(input)
            return output

class MobileResnetDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer=nn.InstanceNorm2d,
                 dropout_rate=0, n_blocks=9, padding_type='reflect', decoder=True,
                 wplus=True,
                 img_size=128, img_size_dec=128):
        assert (n_blocks >= 0)
        super(MobileResnetDecoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []

        n_downsampling = 2

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
                
    def forward(self, input):
        return self.model(input)


class MobileResnetBlock_attn(nn.Module):
    def __init__(self, channel, kernel, stride, padding_type):
        super(MobileResnetBlock_attn, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.stride = stride
        self.padding = 1
        self.padding_type = padding_type
        self.conv1 = SeparableConv2d(channel, channel, kernel, stride, padding=self.padding, padding_mode=self.padding_type)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = SeparableConv2d(channel, channel, kernel, stride, padding=self.padding, padding_mode=self.padding_type)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.relu(self.conv1_norm(self.conv1(input)))
        x = self.conv2_norm(self.conv2(x))

        return input + x

class MobileResnetGenerator_attn(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, use_spectral=False,size=128,nb_attn = 10,nb_mask_input=1,padding_type='reflect'): #nb_attn : nombre de masques d'attention, nb_mask_input : nb de masques d'attention qui vont etre appliqués a l'input
        super(MobileResnetGenerator_attn, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.nb = n_blocks
        self.nb_attn = nb_attn
        self.padding_type = padding_type
        self.nb_mask_input = nb_mask_input
        self.conv1 = spectral_norm(nn.Conv2d(input_nc, ngf, 7, 1, 0),use_spectral)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = spectral_norm(nn.Conv2d(ngf, ngf * 2, 3, 2, 1),use_spectral)
        self.conv2_norm = nn.InstanceNorm2d(ngf * 2)
        self.conv3 = spectral_norm(nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),use_spectral)
        self.conv3_norm = nn.InstanceNorm2d(ngf * 4)

        self.resnet_blocks = []
        for i in range(n_blocks):
            self.resnet_blocks.append(MobileResnetBlock_attn(ngf * 4, 3, 1, self.padding_type))
            self.resnet_blocks[i].weight_init(0, 0.02)

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.deconv1_content = spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1),use_spectral)
        self.deconv1_norm_content = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_content = spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1),use_spectral)
        self.deconv2_norm_content = nn.InstanceNorm2d(ngf)        
        self.deconv3_content = spectral_norm(nn.Conv2d(ngf, 3 * (self.nb_attn-nb_mask_input), 7, 1, 0),use_spectral)#self.nb_attn-nb_mask_input: nombre d'images générées ou les masques d'attention vont etre appliqués

        self.deconv1_attention = spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1),use_spectral)
        self.deconv1_norm_attention = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_attention = spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1),use_spectral)
        self.deconv2_norm_attention = nn.InstanceNorm2d(ngf)
        self.deconv3_attention = nn.Conv2d(ngf,self.nb_attn, 1, 1, 0)
        
        self.tanh = nn.Tanh()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, extract_layer_ids=[], encode_only=False):
        if self.padding_type == 'reflect':
            x = F.pad(input, (3, 3, 3, 3), 'reflect')
        else:
            x = F.pad(input, (3, 3, 3, 3), 'constant', 0)
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))

        if -1 in extract_layer_ids: #if -1 is in extract_layer_ids, the output of the encoder will be returned (features just after the last layer) 
            extract_layer_ids.append(len(self.resnet_blocks))
        if len(extract_layer_ids) > 0:
            feat=x
            feats=[]
            for layer_id, layer in enumerate(self.resnet_blocks):
                feat = layer(feat)
                if layer_id in extract_layer_ids:
                    feats.append(feat)
            if encode_only:
                return feats
            else:
                x=feat
        else:
            x = self.resnet_blocks(x)
        
        x_content = F.relu(self.deconv1_norm_content(self.deconv1_content(x)))
        x_content = F.relu(self.deconv2_norm_content(self.deconv2_content(x_content)))
        if self.padding_type == 'reflect':
            x_content = F.pad(x_content, (3, 3, 3, 3), 'reflect')
        else:
            x_content = F.pad(x_content, (3, 3, 3, 3), 'constant', 0)
        content = self.deconv3_content(x_content)
        image = self.tanh(content)

        images = []

        for i in range(self.nb_attn - self.nb_mask_input):
            images.append(image[:, 3*i:3*(i+1), :, :])

        x_attention = F.relu(self.deconv1_norm_attention(self.deconv1_attention(x)))
        x_attention = F.relu(self.deconv2_norm_attention(self.deconv2_attention(x_attention)))
        attention = self.deconv3_attention(x_attention)

        softmax_ = nn.Softmax(dim=1)
        attention = softmax_(attention)

        attentions =[]
        
        for i in range(self.nb_attn):
            attentions.append(attention[:, i:i+1, :, :].repeat(1, 3, 1, 1))

        outputs = []
        
        for i in range(self.nb_attn-self.nb_mask_input):
            outputs.append(images[i]*attentions[i])
        for i in range(self.nb_attn-self.nb_mask_input,self.nb_attn):
            outputs.append(input * attentions[i])
        
        o = outputs[0]
        for i in range(1,self.nb_attn):
            o += outputs[i]
        return o

