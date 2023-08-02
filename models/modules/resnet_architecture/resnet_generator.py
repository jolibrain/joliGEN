import functools
from torch import nn
import torch
from ..utils import spectral_norm, normal_init, init_net, init_weights
import torch.nn.functional as F
import math
from models.modules.attn_network import BaseGenerator_attn
from models.modules.mobile_modules import SeparableConv2d


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(
        self,
        dim,
        padding_type,
        norm_layer,
        use_dropout,
        use_bias,
        use_spectral=False,
        conv=nn.Conv2d,
    ):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv = conv
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias, use_spectral
        )

    def build_conv_block(
        self, dim, padding_type, norm_layer, use_dropout, use_bias, use_spectral
    ):
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
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zeros":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            spectral_norm(
                self.conv(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                use_spectral,
            ),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zeros":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            spectral_norm(
                self.conv(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                use_spectral,
            ),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
        use_spectral=False,
        mobile=False,
    ):
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
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()

        if mobile:
            self.conv = SeparableConv2d
        else:
            self.conv = nn.Conv2d

        self.encoder = ResnetEncoder(
            input_nc,
            output_nc,
            ngf,
            norm_layer,
            use_dropout,
            n_blocks,
            padding_type,
            use_spectral,
            self.conv,
        )
        self.decoder = ResnetDecoder(
            input_nc,
            output_nc,
            ngf,
            norm_layer,
            use_dropout,
            n_blocks,
            padding_type,
            use_spectral,
        )

    def get_feats(self, input, extract_layer_ids=[]):
        return self.encoder.get_feats(input, extract_layer_ids)

    def forward(self, input):
        """Standard forward"""
        output = self.encoder(input)
        output = self.decoder(output)
        return output


class ResnetEncoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
        use_spectral=False,
        conv=nn.Conv2d,
    ):
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
        assert n_blocks >= 0
        super(ResnetEncoder, self).__init__()

        model = []
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        fl = [
            nn.ReflectionPad2d(3),
            spectral_norm(
                nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                use_spectral,
            ),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        model += fl

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2**i
            dsp = [
                spectral_norm(
                    nn.Conv2d(
                        ngf * mult,
                        ngf * mult * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=use_bias,
                    ),
                    use_spectral,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]
            model += dsp

        mult = 2**n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            resblockl = [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                    conv=conv,
                )
            ]
            model += resblockl

        self.model = nn.Sequential(*model)

    def compute_feats(self, input, extract_layer_ids=[]):
        if -1 in extract_layer_ids:
            extract_layer_ids.append(len(self.encoder))
        feat = input
        feats = []
        for layer_id, layer in enumerate(self.model):

            feat = layer(feat)
            if layer_id in extract_layer_ids:
                feats.append(feat)
        return feat, feats  # return both output and intermediate features

    def forward(self, input):
        """Standard forward"""
        output, _ = self.compute_feats(input)
        return output

    def get_feats(self, input, extract_layer_ids=[]):
        _, feats = self.compute_feats(input, extract_layer_ids)
        return feats


class ResnetDecoder(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
        use_spectral=False,
    ):
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
        assert n_blocks >= 0
        super(ResnetDecoder, self).__init__()

        model = []
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_downsampling = 2

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                spectral_norm(
                    nn.ConvTranspose2d(
                        ngf * mult,
                        int(ngf * mult / 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=use_bias,
                    ),
                    use_spectral,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        p = 0
        if padding_type == "reflect":
            model += [nn.ReflectionPad2d(3)]
        elif padding_type == "replicate":
            model += [nn.ReplicationPad2d(3)]
        elif padding_type == "zeros":
            p = 3
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=p)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        output = self.model(input)
        return output


class resnet_block_attn(nn.Module):
    def __init__(self, channel, kernel, stride, padding_type, conv=nn.Conv2d):
        super(resnet_block_attn, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.stride = stride
        self.padding = 1
        self.padding_type = padding_type
        self.conv1 = conv(
            channel,
            channel,
            kernel,
            stride,
            padding=self.padding,
            padding_mode=self.padding_type,
        )
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = conv(
            channel,
            channel,
            kernel,
            stride,
            padding=self.padding,
            padding_mode=self.padding_type,
        )
        self.conv2_norm = nn.InstanceNorm2d(channel)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.relu(self.conv1_norm(self.conv1(input)))
        x = self.conv2_norm(self.conv2(x))
        return input + x


class ResnetGenerator_attn(BaseGenerator_attn):
    # initializers
    def __init__(
        self,
        input_nc,
        output_nc,
        nb_mask_attn,
        nb_mask_input,
        ngf=64,
        n_blocks=9,
        use_spectral=False,
        size=128,
        padding_type="reflect",
        mobile=False,
        twice_resnet_blocks=False,
        freq_space=False,
    ):
        super(ResnetGenerator_attn, self).__init__(
            nb_mask_attn=nb_mask_attn, nb_mask_input=nb_mask_input
        )
        if mobile:
            conv = SeparableConv2d
        else:
            conv = nn.Conv2d

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.nb = n_blocks
        self.padding_type = padding_type
        self.twice_resnet_blocks = twice_resnet_blocks
        self.freq_space = freq_space

        if freq_space:
            from ..freq_utils import InverseHaarTransform, HaarTransform

            self.iwt = InverseHaarTransform(self.input_nc)
            self.dwt = HaarTransform(self.input_nc)
            self.input_nc = input_nc * 4

        self.conv1 = spectral_norm(
            nn.Conv2d(self.input_nc, self.ngf, 7, 1, 0), use_spectral
        )
        self.input_nc = output_nc  # hack
        self.conv1_norm = nn.InstanceNorm2d(self.ngf)
        self.conv2 = spectral_norm(
            nn.Conv2d(self.ngf, self.ngf * 2, 3, 2, 1), use_spectral
        )
        self.conv2_norm = nn.InstanceNorm2d(self.ngf * 2)
        self.conv3 = spectral_norm(
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 3, 2, 1), use_spectral
        )
        self.conv3_norm = nn.InstanceNorm2d(self.ngf * 4)

        self.resnet_blocks = []
        for i in range(n_blocks):
            self.resnet_blocks.append(
                resnet_block_attn(self.ngf * 4, 3, 1, self.padding_type, conv=conv)
            )
            self.resnet_blocks[i].weight_init(0, 0.02)

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.deconv1_content = spectral_norm(
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 3, 2, 1, 1), use_spectral
        )
        self.deconv1_norm_content = nn.InstanceNorm2d(self.ngf * 2)
        self.deconv2_content = spectral_norm(
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 3, 2, 1, 1), use_spectral
        )
        self.deconv2_norm_content = nn.InstanceNorm2d(self.ngf)
        if self.freq_space:
            deconv3_ngf = int(self.ngf / 4)
        else:
            deconv3_ngf = self.ngf
        self.deconv3_content = spectral_norm(
            nn.Conv2d(
                deconv3_ngf,
                self.input_nc * (self.nb_mask_attn - self.nb_mask_input),
                7,
                1,
                0,
            ),
            use_spectral,
        )

        self.deconv1_attention = spectral_norm(
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 3, 2, 1, 1), use_spectral
        )
        self.deconv1_norm_attention = nn.InstanceNorm2d(self.ngf * 2)
        self.deconv2_attention = spectral_norm(
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 3, 2, 1, 1), use_spectral
        )
        self.deconv2_norm_attention = nn.InstanceNorm2d(self.ngf)
        self.deconv3_attention = nn.Conv2d(self.ngf, self.nb_mask_attn, 1, 1, 0)

        self.tanh = nn.Tanh()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def compute_feats(self, input, extract_layer_ids=[]):
        if self.padding_type == "reflect":
            x = F.pad(input, (3, 3, 3, 3), "reflect")
        else:
            x = F.pad(input, (3, 3, 3, 3), "constant", 0)

        if self.freq_space:
            x = self.dwt(x)

        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))

        if (
            -1 in extract_layer_ids
        ):  # if -1 is in extract_layer_ids, the output of the encoder will be returned (features just after the last layer)
            extract_layer_ids.append(len(self.resnet_blocks))
        feat = x
        feats = []
        for layer_id, layer in enumerate(self.resnet_blocks):
            feat = layer(feat)
            if layer_id in extract_layer_ids:
                feats.append(feat)

        return feat, feats

    def compute_attention_content(self, feat):
        if self.twice_resnet_blocks:
            x = self.resnet_blocks(feat)
        else:
            x = feat

        x_content = F.relu(self.deconv1_norm_content(self.deconv1_content(x)))
        x_content = F.relu(self.deconv2_norm_content(self.deconv2_content(x_content)))

        if self.freq_space:
            x_content = self.iwt(x_content)

        if self.padding_type == "reflect":
            x_content = F.pad(x_content, (3, 3, 3, 3), "reflect")
        else:
            x_content = F.pad(x_content, (3, 3, 3, 3), "constant", 0)
        content = self.deconv3_content(x_content)

        image = self.tanh(content)

        images = []

        for i in range(self.nb_mask_attn - self.nb_mask_input):
            images.append(image[:, self.input_nc * i : self.input_nc * (i + 1), :, :])

        x_attention = F.relu(self.deconv1_norm_attention(self.deconv1_attention(x)))
        x_attention = F.relu(
            self.deconv2_norm_attention(self.deconv2_attention(x_attention))
        )
        attention = self.deconv3_attention(x_attention)

        softmax_ = nn.Softmax(dim=1)
        attention = softmax_(attention)

        attentions = []

        for i in range(self.nb_mask_attn):
            attentions.append(
                attention[:, i : i + 1, :, :].repeat(1, self.input_nc, 1, 1)
            )

        return attentions, images
