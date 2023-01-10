from abc import abstractmethod
import torch
from torch import nn
import torch.nn.functional as F

from .resnet_generator import BaseGenerator_attn
from ..utils import normal_init
from models.modules.diffusion_utils import gamma_embedding
from models.modules.mobile_modules import SeparableConv2d
from models.modules.unet_generator_attn.unet_attn_utils import normalization


class EmbedBlock(nn.Module):
    """
    Any module where forward() takes embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` embeddings.
        """


class EmbedSequential(nn.Sequential, EmbedBlock):
    """
    A sequential module that passes embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class resnet_block_attn(EmbedBlock):
    def __init__(
        self,
        channel,
        emb_channels,
        kernel,
        stride,
        padding_type,
        conv=nn.Conv2d,
        use_scale_shift_norm=True,
    ):
        super(resnet_block_attn, self).__init__()
        self.channel = channel
        self.emb_channels = emb_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = 1
        self.padding_type = padding_type
        self.use_scale_shift_norm = use_scale_shift_norm

        self.conv1 = conv(
            channel,
            channel,
            kernel,
            stride,
            padding=self.padding,
            padding_mode=self.padding_type,
        )
        self.conv1_norm = normalization(channel)
        self.conv2 = conv(
            channel,
            channel,
            kernel,
            stride,
            padding=self.padding,
            padding_mode=self.padding_type,
        )
        self.conv2_norm = normalization(channel)

        self.emb_layers = nn.Sequential(
            torch.nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * channel if use_scale_shift_norm else self.out_channel,
            ),
        )

        self.out_layers = nn.Sequential(
            normalization(channel),
            torch.nn.SiLU(),
            nn.Conv2d(channel, channel, 3, padding=1),
        )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, emb):
        x = F.relu(self.conv1_norm(self.conv1(input)))
        x = self.conv2_norm(self.conv2(x))

        emb_out = self.emb_layers(emb).type(input.dtype)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out.unsqueeze(-1)

        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            out = x * (1 + scale) + shift
        else:
            out = x + emb_out

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            out = out_norm(x)
            out = out_rest(out) * (1 + scale) + shift
        else:
            out = self.layers(x) + emb_out
            out = self.out_layers(out)

        return input + out


class embed_block(EmbedBlock):
    def __init__(self, layers, cond_embed, out_layers, use_scale_shift_norm):
        super(embed_block, self).__init__()
        self.layers = layers
        self.cond_embed = cond_embed
        self.out_layers = out_layers
        self.use_scale_shift_norm = use_scale_shift_norm

    def forward(self, x, emb):
        emb_out = self.cond_embed(emb)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out.unsqueeze(-1)

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            out = self.layers(x)
            out = out_norm(out)
            out = out_rest(out) * (1 + scale) + shift
        else:
            out = self.layers(x) + emb_out
            out = self.out_layers(out)

        return out


class ResnetGenerator_attn_diff(BaseGenerator_attn):
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
        padding_type="reflect",
        mobile=False,
        use_scale_shift_norm=True,
    ):
        super(ResnetGenerator_attn_diff, self).__init__(
            nb_mask_attn=nb_mask_attn, nb_mask_input=nb_mask_input
        )
        if mobile:
            conv = SeparableConv2d
        else:
            conv = nn.Conv2d

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.inner_channel = ngf
        self.ngf = ngf
        self.nb = n_blocks
        self.padding_type = padding_type
        self.embed_channel_ratio = 2 if use_scale_shift_norm else 1

        # encoder
        cond_embed_dim = self.inner_channel  # * 4

        self.cond_embed = nn.Sequential(
            nn.Linear(self.inner_channel, cond_embed_dim),
            torch.nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )

        cur_chan = input_nc
        next_chan = ngf

        self.encoder = [
            # Layer 1
            embed_block(
                nn.Sequential(nn.Conv2d(input_nc, ngf, 7, 1, 0), normalization(ngf)),
                nn.Sequential(
                    nn.Linear(cond_embed_dim, cond_embed_dim),
                    torch.nn.SiLU(),
                    nn.Linear(cond_embed_dim, ngf * self.embed_channel_ratio),
                ),
                nn.Sequential(
                    normalization(ngf),
                    torch.nn.SiLU(),
                    nn.Conv2d(ngf, ngf, 3, padding=1),
                ),
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # Layer 2
            embed_block(
                nn.Sequential(nn.Conv2d(ngf, ngf * 2, 3, 2, 1), normalization(ngf * 2)),
                nn.Sequential(
                    nn.Linear(cond_embed_dim, cond_embed_dim),
                    torch.nn.SiLU(),
                    nn.Linear(cond_embed_dim, ngf * 2 * self.embed_channel_ratio),
                ),
                nn.Sequential(
                    normalization(ngf * 2),
                    torch.nn.SiLU(),
                    nn.Conv2d(ngf * 2, ngf * 2, 3, padding=1),
                ),
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # Layer 3
            embed_block(
                nn.Sequential(
                    nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1), normalization(ngf * 4)
                ),
                nn.Sequential(
                    nn.Linear(cond_embed_dim, cond_embed_dim),
                    torch.nn.SiLU(),
                    nn.Linear(cond_embed_dim, ngf * 4 * self.embed_channel_ratio),
                ),
                nn.Sequential(
                    normalization(ngf * 4),
                    torch.nn.SiLU(),
                    nn.Conv2d(ngf * 4, ngf * 4, 3, padding=1),
                ),
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        ]

        self.encoder = EmbedSequential(*self.encoder)

        self.resnet_blocks = []
        for i in range(n_blocks):
            self.resnet_blocks.append(
                resnet_block_attn(
                    channel=ngf * 4,
                    emb_channels=cond_embed_dim,
                    kernel=3,
                    stride=1,
                    padding_type=self.padding_type,
                    conv=conv,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )
            self.resnet_blocks[i].weight_init(0, 0.02)

        # self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
        self.resnet_blocks = EmbedSequential(*self.resnet_blocks)

        self.decoder_content = [
            # Layer 1
            embed_block(
                nn.Sequential(
                    nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1),
                    normalization(ngf * 2),
                ),
                nn.Sequential(
                    nn.Linear(cond_embed_dim, cond_embed_dim),
                    torch.nn.SiLU(),
                    nn.Linear(cond_embed_dim, ngf * 2 * self.embed_channel_ratio),
                ),
                nn.Sequential(
                    normalization(ngf * 2),
                    torch.nn.SiLU(),
                    nn.Conv2d(ngf * 2, ngf * 2, 3, padding=1),
                ),
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # Layer 2
            embed_block(
                nn.Sequential(
                    nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1), normalization(ngf)
                ),
                nn.Sequential(
                    nn.Linear(cond_embed_dim, cond_embed_dim),
                    torch.nn.SiLU(),
                    nn.Linear(cond_embed_dim, ngf * self.embed_channel_ratio),
                ),
                nn.Sequential(
                    normalization(ngf),
                    torch.nn.SiLU(),
                    nn.Conv2d(ngf, ngf, 3, padding=1),
                ),
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # Layer 3
            embed_block(
                nn.Sequential(
                    nn.Conv2d(
                        ngf,
                        ngf,  # self.output_nc * (self.nb_mask_attn - self.nb_mask_input),
                        7,
                        1,
                        0,
                    ),
                    nn.Identity(),
                ),
                nn.Sequential(
                    nn.Linear(cond_embed_dim, cond_embed_dim),
                    torch.nn.SiLU(),
                    nn.Linear(
                        cond_embed_dim,
                        ngf
                        * self.embed_channel_ratio,  # self.output_nc * (self.nb_mask_attn - self.nb_mask_input) * self.embed_channel_ratio,
                    ),
                ),
                nn.Sequential(
                    nn.InstanceNorm2d(
                        self.output_nc * (self.nb_mask_attn - self.nb_mask_input)
                    ),
                    torch.nn.SiLU(),
                    nn.Conv2d(
                        ngf,  # self.output_nc * (self.nb_mask_attn - self.nb_mask_input),
                        ngf,  # self.output_nc * (self.nb_mask_attn - self.nb_mask_input),
                        3,
                        padding=1,
                    ),
                ),
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        ]

        self.decoder_content = EmbedSequential(*self.decoder_content)

        self.decoder_attention = [
            # Layer 1
            embed_block(
                nn.Sequential(
                    nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1),
                    nn.InstanceNorm2d(ngf * 2),
                ),
                nn.Sequential(
                    nn.Linear(cond_embed_dim, ngf),
                    torch.nn.SiLU(),
                    nn.Linear(ngf, ngf * 2 * self.embed_channel_ratio),
                ),
                nn.Sequential(
                    nn.InstanceNorm2d(ngf * 2),
                    torch.nn.SiLU(),
                    nn.Conv2d(ngf * 2, ngf * 2, 3, padding=1),
                ),
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # Layer 2
            embed_block(
                nn.Sequential(
                    nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1),
                    nn.InstanceNorm2d(ngf),
                ),
                nn.Sequential(
                    nn.Linear(cond_embed_dim, ngf * 2),
                    torch.nn.SiLU(),
                    nn.Linear(ngf * 2, ngf * self.embed_channel_ratio),
                ),
                nn.Sequential(
                    nn.InstanceNorm2d(ngf),
                    torch.nn.SiLU(),
                    nn.Conv2d(ngf, ngf, 3, padding=1),
                ),
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # Layer 3
            embed_block(
                nn.Sequential(
                    nn.Conv2d(ngf, self.nb_mask_attn, 1, 1, 0),
                    nn.Identity(),
                ),
                nn.Sequential(
                    nn.Linear(cond_embed_dim, ngf * 4),
                    torch.nn.SiLU(),
                    nn.Linear(ngf * 4, self.nb_mask_attn * self.embed_channel_ratio),
                ),
                nn.Sequential(
                    nn.InstanceNorm2d(self.nb_mask_attn),
                    torch.nn.SiLU(),
                    nn.Conv2d(self.nb_mask_attn, self.nb_mask_attn, 3, padding=1),
                ),
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        ]

        self.decoder_attention = EmbedSequential(*self.decoder_attention)

        self.output_layer = nn.Sequential(
            normalization(ngf),
            torch.nn.SiLU(),
            nn.Conv2d(
                ngf,
                self.output_nc * (self.nb_mask_attn - self.nb_mask_input),
                3,
                padding=1,
            ),
        )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def compute_feats(self, input, gammas, extract_layer_ids=[]):
        if gammas is None:
            b = input.shape[0]
            gammas = torch.ones((b,)).to(input.device)
        gammas = gammas.view(-1)

        emb = self.cond_embed(gamma_embedding(gammas, self.inner_channel))

        if self.padding_type == "reflect":
            x = F.pad(input, (3, 3, 3, 3), "reflect")
        else:
            x = F.pad(input, (3, 3, 3, 3), "constant", 0)

        # encoder
        for layer_id, layer in enumerate(self.encoder):
            x = F.relu(layer(x, emb))

        if (
            -1 in extract_layer_ids
        ):  # if -1 is in extract_layer_ids, the output of the encoder will be returned (features just after the last layer)
            extract_layer_ids.append(len(self.resnet_blocks))
        feat = x
        feats = []
        for layer_id, layer in enumerate(self.resnet_blocks):
            feat = layer(feat, emb)
            if layer_id in extract_layer_ids:
                feats.append(feat)

        return feat, feats, emb

    def compute_attention_content(self, feat, emb):

        x_content = feat

        for layer_id, layer in enumerate(self.decoder_content):
            if layer_id == len(self.decoder_content) - 1:  # last layer
                if self.padding_type == "reflect":
                    x_content = F.pad(x_content, (3, 3, 3, 3), "reflect")
                else:
                    x_content = F.pad(x_content, (3, 3, 3, 3), "constant", 0)

            x_content = F.relu(layer(x_content, emb))

        content = x_content

        image = self.output_layer(content)

        images = []

        for i in range(self.nb_mask_attn - self.nb_mask_input):
            images.append(image[:, self.output_nc * i : self.output_nc * (i + 1), :, :])

        x_attention = feat

        for layer_id, layer in enumerate(self.decoder_attention):
            x_attention = layer(x_attention, emb)
            if layer_id != len(self.decoder_attention):  # not last layer
                x_attention = F.relu(x_attention)

        attention = x_attention

        softmax_ = nn.Softmax(dim=1)
        attention = softmax_(attention)

        attentions = []

        for i in range(self.nb_mask_attn):
            attentions.append(
                attention[:, i : i + 1, :, :].repeat(1, self.output_nc, 1, 1)
            )

        return attentions, images

    def forward(self, input, gammas=None):
        feat, _, emb = self.compute_feats(input, gammas=gammas)
        attentions, images = self.compute_attention_content(feat, emb)

        _, _, outputs = self.compute_outputs(input, attentions, images)

        o = outputs[0]
        for i in range(1, self.nb_mask_attn):
            o += outputs[i]
        return o

    def extract(self, a, t, x_shape=(1, 1, 1, 1)):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def get_attention_masks(self, input, gammas):
        feat, _, emb = self.compute_feats(input, gammas)
        attentions, images = self.compute_attention_content(feat, emb)
        return self.compute_outputs(input, attentions, images)

    def get_feats(self, input, gammas, extract_layer_ids):
        _, feats = self.compute_feats(input, gammas, extract_layer_ids)
        return feats
