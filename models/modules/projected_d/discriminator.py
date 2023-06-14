from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DownBlock, DownBlockPatch, conv2d

# from pg_modules.diffaug import DiffAugment
from .projector import Proj


class SingleDisc(nn.Module):
    def __init__(
        self,
        nc=None,
        ndf=None,
        start_sz=256,
        end_sz=8,
        head=None,
        separable=False,
        patch=False,
    ):
        super().__init__()
        channel_dict = {
            4: 512,
            8: 512,
            16: 256,
            32: 128,
            64: 64,
            128: 64,
            256: 32,
            512: 16,
            1024: 8,
        }

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [
                conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Down Blocks
        DB = (
            partial(DownBlockPatch, separable=separable)
            if patch
            else partial(DownBlock, separable=separable)
        )
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz], nfc[start_sz // 2]))
            start_sz = start_sz // 2

        layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class SingleDiscCond(nn.Module):
    def __init__(
        self,
        nc=None,
        ndf=None,
        start_sz=256,
        end_sz=8,
        head=None,
        separable=False,
        patch=False,
        c_dim=1000,
        cmap_dim=64,
        embedding_dim=128,
    ):
        super().__init__()
        self.cmap_dim = cmap_dim

        # midas channels
        channel_dict = {
            4: 512,
            8: 512,
            16: 256,
            32: 128,
            64: 64,
            128: 64,
            256: 32,
            512: 16,
            1024: 8,
        }

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [
                conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Down Blocks
        DB = (
            partial(DownBlockPatch, separable=separable)
            if patch
            else partial(DownBlock, separable=separable)
        )
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz], nfc[start_sz // 2]))
            start_sz = start_sz // 2
        self.main = nn.Sequential(*layers)

        # additions for conditioning on class information
        self.cls = conv2d(nfc[end_sz], self.cmap_dim, 4, 1, 0, bias=False)
        self.embed = nn.Embedding(num_embeddings=c_dim, embedding_dim=embedding_dim)
        self.embed_proj = nn.Sequential(
            nn.Linear(self.embed.embedding_dim, self.cmap_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        h = self.main(x)
        out = self.cls(h)

        # conditioning via projection
        cmap = self.embed_proj(self.embed(c.argmax(1))).unsqueeze(-1).unsqueeze(-1)
        out = (out * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out


class MultiScaleD(nn.Module):
    def __init__(
        self,
        channels,
        resolutions,
        conv,
        feats,
        num_discs=4,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        cond=0,
        separable=False,  # was false
        patch=False,
        **kwargs,
    ):
        super().__init__()

        assert num_discs in [1, 2, 3, 4]

        # the first disc is on the lowest level of the backbone
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]
        Disc = SingleDiscCond if cond else SingleDisc

        mini_discs = []

        if conv:
            for i, (cin, res) in enumerate(
                zip(self.disc_in_channels, self.disc_in_res)
            ):
                start_sz = res if not patch else 16
                mini_discs += (
                    [
                        str(i),
                        Disc(
                            nc=cin,
                            start_sz=start_sz,
                            end_sz=8,
                            separable=separable,
                            patch=patch,
                        ),
                    ],
                )
        else:
            for i in range(num_discs):
                n_feats = channels[i] * resolutions[i]
                mlp = [
                    nn.Flatten(),
                    nn.Linear(n_feats, 100),
                    nn.ReLU(),
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100, 100),
                ]
                mlp = nn.Sequential(*mlp)
                mini_discs += ([str(i), mlp],)

        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features):
        all_logits = []
        for k, disc in self.mini_discs.items():
            all_logits.append(disc(features[k]).view(features[k].size(0), -1))

        all_logits = torch.cat(all_logits, dim=1)
        return all_logits


class ProjectedDiscriminator(torch.nn.Module):
    def __init__(
        self,
        projector_model,
        interp=-1,
        backbone_kwargs={"cout": 64, "expand": True},
        config_path="",
        weight_path="",
        img_size=256,
        diffusion_aug=False,
        **kwargs,
    ):
        super().__init__()
        self.interp = interp

        self.freeze_feature_network = Proj(
            projector_model,
            config_path=config_path,
            weight_path=weight_path,
            interp=self.interp,
            img_size=img_size,
            diffusion_aug=diffusion_aug,
            **backbone_kwargs,
        )
        self.freeze_feature_network.requires_grad_(False)

        self.discriminator = MultiScaleD(
            channels=self.freeze_feature_network.CHANNELS,
            resolutions=self.freeze_feature_network.RESOLUTIONS,
            feats=self.freeze_feature_network.FEATS,
            conv="vit" not in projector_model,
            **backbone_kwargs,
        )

    def train(self, mode=True):
        self.freeze_feature_network = self.freeze_feature_network.train(False)
        self.discriminator = self.discriminator.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x):
        if self.interp > 0:
            x = F.interpolate(x, self.interp, mode="bilinear", align_corners=False)

        features = self.freeze_feature_network(x)
        logits = self.discriminator(features)

        return logits


class MultiScaleLSTM(torch.nn.Module):
    def __init__(
        self,
        channels,
        resolutions,
        number_frames,
    ):
        super().__init__()
        self.channels = channels
        self.resolutions = resolutions
        self.number_frames = number_frames
        self.h0 = {}
        self.c0 = {}
        self.num_feats = len(self.channels)
        self.flatt = {}
        self.unflatt = {}
        lstms = []

        for i, (channel, resolution) in enumerate(zip(self.channels, self.resolutions)):
            feat_dim = channel * resolution * resolution

            self.flatt[str(i)] = nn.Flatten(start_dim=2)
            self.unflatt[str(i)] = nn.Unflatten(
                dim=2, unflattened_size=(channel, resolution, resolution)
            )
            cur_net = torch.nn.LSTM(
                input_size=feat_dim, hidden_size=16, batch_first=True
            )

            lstms += ([str(i), cur_net],)

            self.h0[str(i)] = torch.randn(1, self.number_frames, feat_dim)
            self.c0[str(i)] = torch.randn(1, self.number_frames, feat_dim)

        self.lstms = nn.ModuleDict(lstms)

    def forward(self, features):
        out_features = {str(k): [] for k in range(self.num_feats)}
        for k, lstm in self.lstms.items():
            (h0, c0) = (self.h0[k], self.c0[k])
            temp = features[k]
            temp = self.flatt[k](temp)
            temp, (_, _) = lstm(temp)  # , (h0, c0))
            temp = self.unflatt[k](temp)
            out_features[k] = temp

        return out_features


class TemporalProjectedDiscriminator(torch.nn.Module):
    def __init__(
        self,
        projector_model,
        interp,
        config_path,
        weight_path,
        data_temporal_number_frames,
        data_temporal_frame_step,
        backbone_kwargs={"cout": 64, "expand": True},
        img_size=256,
        **kwargs,
    ):
        super().__init__()
        self.data_temporal_number_frames = data_temporal_number_frames
        self.interp = interp

        self.freeze_feature_network = Proj(
            projector_model,
            config_path=config_path,
            weight_path=weight_path,
            interp=interp,
            img_size=img_size,
            **backbone_kwargs,
        )
        self.freeze_feature_network.requires_grad_(False)

        channels = [
            channel * data_temporal_number_frames
            for channel in self.freeze_feature_network.CHANNELS
        ]

        self.num_feats = len(channels)

        self.discriminator = MultiScaleD(
            channels=self.freeze_feature_network.CHANNELS,
            resolutions=self.freeze_feature_network.RESOLUTIONS,
            feats=self.freeze_feature_network.FEATS,
            conv="vit" not in projector_model,
            **backbone_kwargs,
        )

        if self.interp > 0:
            input_size = self.interp
        else:
            input_size = img_size
        dumb_input = torch.zeros([1, 3, input_size, input_size])

        temp = self.freeze_feature_network(dumb_input)
        temp = self.discriminator(temp)
        lstm_size = temp.shape[-1]

        self.lstm = torch.nn.LSTM(
            input_size=lstm_size, hidden_size=lstm_size, batch_first=True
        )

    def train(self, mode=True):
        self.freeze_feature_network = self.freeze_feature_network.train(False)
        self.discriminator = self.discriminator.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, images):
        features = {str(k): [] for k in range(self.num_feats)}
        logits_frames = []

        for i in range(self.data_temporal_number_frames):
            x = images[:, i]
            if self.interp > 0:
                x = F.interpolate(x, self.interp, mode="bilinear", align_corners=False)

            cur_feat = self.freeze_feature_network(x)

            logits_frames.append(self.discriminator(cur_feat).unsqueeze(1))

            for k in range(self.num_feats):
                features[str(k)].append(cur_feat[str(k)])

        logits_frames = torch.cat(logits_frames, dim=1)

        logits, _ = self.lstm(logits_frames)

        return logits
