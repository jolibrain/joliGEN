import functools

from torch import nn
import torch
from ..utils import spectral_norm,normal_init,init_net,init_weights
import torch.nn.functional as F
import math


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, use_spectral=False):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, use_spectral)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, use_spectral):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),use_spectral), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),use_spectral), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', use_spectral=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        self.encoder=ResnetEncoder(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_spectral, init_type, init_gain, gpu_ids)
        self.decoder=ResnetDecoder(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_spectral, init_type, init_gain, gpu_ids)
        
    def forward(self, input):
        """Standard forward"""
        output=self.encoder(input)
        output=self.decoder(output)
        return output

class ResnetEncoderSty2(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', use_spectral=False, init_type='normal', init_gain=0.02, gpu_ids=[],img_size=128,img_size_dec=128):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetEncoderSty2, self).__init__()

        self.encoder=ResnetEncoder(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_spectral, init_type, init_gain, gpu_ids)

        n_feat = 2**(2*int(math.log(img_size,2)-2))
        self.n_wplus = (2*int(math.log(img_size_dec,2)-1))
        n_downsampling = 2
        mult = 2 ** n_downsampling
        self.wblocks = nn.ModuleList()
        for n in range(0,self.n_wplus):
            self.wblocks += [WBlock(ngf*mult,n_feat,init_type,init_gain,gpu_ids)]
        self.nblocks = nn.ModuleList()
        noise_map = [4,8,8,16,16,32,32,64,64,128,128,256,256,512,512,1024,1024]
        for n in range(0,self.n_wplus-1):
            self.nblocks += [NBlock(ngf*mult,n_feat,noise_map[n],init_type,init_gain,gpu_ids)]
        
    def forward(self, input):
        """Standard forward"""
        features=self.encoder(input)
        outputs = []
        noutputs = []
        for wc in self.wblocks:
            outputs.append(wc(features))
        outputs=torch.stack(outputs).unsqueeze(0)
        for nc in self.nblocks:
            noutputs.append(nc(features))            
        return outputs, noutputs


class ResnetEncoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', use_spectral=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
        """Construct a Resnet-based encoder

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()

        model = []
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        fl = [nn.ReflectionPad2d(3),
                 spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),use_spectral),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        model += fl
        
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            dsp = [spectral_norm(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),use_spectral),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            model += dsp

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            resblockl = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            model += resblockl

        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        """Standard forward"""
        output = self.model(input)
        return output

class ResnetDecoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', use_spectral=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetDecoder, self).__init__()

        model = []
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        n_downsampling = 2

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [spectral_norm(nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                       kernel_size=3, stride=2,
                                                       padding=1, output_padding=1,
                                                       bias=use_bias),use_spectral),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        """Standard forward"""
        output = self.model(input)
        return output

class resnet_block_attn(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block_attn, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = self.conv2_norm(self.conv2(x))

        return input + x

class ResnetGenerator_attn(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, use_spectral=False, init_type='normal', init_gain=0.02, gpu_ids=[],size=128,nb_attn = 10,nb_mask_input=1): #nb_attn : nombre de masques d'attention, nb_mask_input : nb de masques d'attention qui vont etre appliqués a l'input
        super(ResnetGenerator_attn, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.nb = n_blocks
        self.nb_attn = nb_attn
        self.nb_mask_input = nb_mask_input
        self.conv1 = spectral_norm(nn.Conv2d(input_nc, ngf, 7, 1, 0),use_spectral)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = spectral_norm(nn.Conv2d(ngf, ngf * 2, 3, 2, 1),use_spectral)
        self.conv2_norm = nn.InstanceNorm2d(ngf * 2)
        self.conv3 = spectral_norm(nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),use_spectral)
        self.conv3_norm = nn.InstanceNorm2d(ngf * 4)

        self.resnet_blocks = []
        for i in range(n_blocks):
            self.resnet_blocks.append(resnet_block_attn(ngf * 4, 3, 1, 1))
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
        x = F.pad(input, (3, 3, 3, 3), 'reflect')
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
        x_content = F.pad(x_content, (3, 3, 3, 3), 'reflect')
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

class WBlock(nn.Module):
    """Define a linear block for W"""
    def __init__(self, dim, n_feat, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super(WBlock, self).__init__()
        self.conv2d = nn.Conv2d(dim,1,kernel_size=1)
        self.lin1 = nn.Linear(n_feat,32,bias=True)
        self.lin2 = nn.Linear(32,512,bias=True)
        #self.lin = nn.Linear(n_feat,512)
        w_block = []
        w_block += [self.conv2d,nn.InstanceNorm2d(1),nn.Flatten(),self.lin1,nn.ReLU(True),self.lin2]
        self.w_block = init_net(nn.Sequential(*w_block), init_type, init_gain, gpu_ids)
        
    def forward(self, x):
        out = self.w_block(x)
        return out.squeeze(0)
    
class NBlock(nn.Module):
    """Define a linear block for N"""
    def __init__(self, dim, n_feat, out_feat, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super(NBlock, self).__init__()
        self.out_feat = out_feat
        if out_feat < 32: # size of input
            self.conv2d = nn.Conv2d(dim,1,kernel_size=1)
            self.lin = nn.Linear(n_feat,out_feat**2)
            n_block = []
            n_block += [self.conv2d,nn.InstanceNorm2d(1),nn.Flatten(),self.lin]
            self.n_block = init_net(nn.Sequential(*n_block), init_type, init_gain, gpu_ids)
        else:
            self.n_block = []
            self.n_block = [nn.Conv2d(256,64,kernel_size=3,stride=1,padding=1),
                            nn.InstanceNorm2d(1),
                            nn.ReLU(True)]
            self.n_block += [nn.Upsample((out_feat,out_feat))]
            self.n_block += [nn.Conv2d(64,1,kernel_size=1)]
            self.n_block += [nn.Flatten()]
            self.n_block = init_net(nn.Sequential(*self.n_block), init_type, init_gain, gpu_ids)
                    
    def forward(self, x):
        out = self.n_block(x)
        return torch.reshape(out.unsqueeze(1),(1,1,self.out_feat,self.out_feat))
