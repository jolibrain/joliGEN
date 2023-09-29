import functools
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from .utils import spectral_norm, normal_init


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        use_spectral=False,
        freq_space=False,
    ):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
            use_dropout (bool) -- whether to use dropout layers
            use_spectral (bool) -- whether to use spectral norm
        """
        super(NLayerDiscriminator, self).__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.freq_space = freq_space
        if self.freq_space:
            from .freq_utils import InverseHaarTransform, HaarTransform

            self.iwt = InverseHaarTransform(input_nc)
            self.dwt = HaarTransform(input_nc)
            input_nc *= 4

        kw = 4
        padw = 1
        sequence = [
            spectral_norm(
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                use_spectral,
            ),
            nn.LeakyReLU(0.2, True),
        ]
        if use_dropout:
            sequence += [nn.Dropout(0.5)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                spectral_norm(
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                        bias=use_bias,
                    ),
                    use_spectral,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
            if use_dropout:
                sequence += [nn.Dropout(0.5)]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            spectral_norm(
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                    bias=use_bias,
                ),
                use_spectral,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        if use_dropout:
            sequence += [nn.Dropout(0.5)]

        sequence += [
            spectral_norm(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
                use_spectral,
            )
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        if self.freq_space:
            x = self.dwt(input)
        else:
            x = input
        x = self.model(x)
        return x


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class UnetDiscriminator(nn.Module):
    """Create a Unet-based discriminator"""

    def __init__(
        self,
        input_nc,
        output_nc,
        D_num_downs,
        D_ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        """Construct a Unet discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            D_num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            D_ngf (int)       -- the number of filters in the last conv layer, here  ngf=64, so inner_nc=64*8=512
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetDiscriminator, self).__init__()
        # construct unet structure
        # add the innermost layer
        unet_block = UnetSkipConnectionBlock(
            D_ngf * 8,
            D_ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )
        # add intermediate layers with ngf * 8 filters
        for i in range(D_num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(
                D_ngf * 8,
                D_ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            D_ngf * 4,
            D_ngf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
        )
        unet_block = UnetSkipConnectionBlock(
            D_ngf * 2,
            D_ngf * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
        )
        unet_block = UnetSkipConnectionBlock(
            D_ngf, D_ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )

        # add the outermost layer
        self.model = UnetSkipConnectionBlock(
            output_nc,
            D_ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )

    def compute_feats(self, input, extract_layer_ids=[]):
        output, feats = self.model(input, feats=[])
        return_feats = []
        for i, feat in enumerate(feats):
            if i in extract_layer_ids:
                return_feats.append(feat)

        return output, return_feats

    def forward(self, input):
        output, _ = self.compute_feats(input)
        return output

    def get_feats(self, input, extract_layer_ids=[]):
        _, feats = self.compute_feats(input, extract_layer_ids)

        return feats


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x, feats):
        output = self.model[0](x)
        return_feats = feats + [output]

        for layer in self.model[1:]:
            if isinstance(layer, UnetSkipConnectionBlock):
                output, return_feats = layer(output, return_feats)
            else:
                output = layer(output)

        if not self.outermost:  # add skip connections
            output = torch.cat([x, output], 1)

        return output, return_feats
