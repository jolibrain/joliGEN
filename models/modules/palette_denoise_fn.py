import torch
from torch import nn
from torchvision import transforms
from einops import rearrange

from .image_bind import imagebind_model
from .image_bind.imagebind_model import ModalityType
import clip

from inspect import signature
from models.modules.diffusion_utils import rearrange_5dto4d_bf, rearrange_4dto5d_bf


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """

    def __init__(self, num_classes, hidden_size):
        super().__init__()

        self.embedding_table = nn.Embedding(
            num_classes,
            hidden_size,
            max_norm=1.0,
            scale_grad_by_freq=True,
        )
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class PaletteDenoiseFn(nn.Module):
    def __init__(self, model, cond_embed_dim, ref_embed_net, conditioning, nclasses):
        super().__init__()

        self.model = model
        model_sig = signature(model.forward)
        self.model_nargs = len(model_sig.parameters)

        self.conditioning = conditioning
        self.cond_embed_dim = cond_embed_dim
        self.ref_embed_net = ref_embed_net
        self.cond_embed_gammas = cond_embed_dim

        # Label embedding
        if "class" in conditioning:
            if type(nclasses) == list:
                # TODO this is arbitrary, half for class & half for detector
                cond_embed_class = cond_embed_dim // (len(nclasses) + 1)
                self.netl_embedders_class = nn.ModuleList(
                    [LabelEmbedder(nc, cond_embed_class) for nc in nclasses]
                )
                for embed in self.netl_embedders_class:
                    self.cond_embed_gammas -= cond_embed_class
                    nn.init.normal_(embed.embedding_table.weight, std=0.02)
            else:
                # TODO this can be included in the general case
                cond_embed_class = cond_embed_dim // 2
                self.netl_embedder_class = LabelEmbedder(
                    nclasses,
                    cond_embed_class,  # * image_size * image_size
                )
                self.cond_embed_gammas -= cond_embed_class
                nn.init.normal_(
                    self.netl_embedder_class.embedding_table.weight, std=0.02
                )

        if "mask" in conditioning:
            cond_embed_mask = cond_embed_dim
            self.netl_embedder_mask = LabelEmbedder(
                nclasses,
                cond_embed_mask,  # * image_size * image_size
            )
            self.cond_embed_gammas -= cond_embed_class
            nn.init.normal_(self.netl_embedder_mask.embedding_table.weight, std=0.02)

        # Instantiate model
        if "ref" in conditioning:
            cond_embed_class = cond_embed_dim // 2

            self.ref_transform = transforms.Compose(
                [
                    transforms.Resize(
                        224, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(224),
                ]
            )

            if ref_embed_net == "clip":
                model_name = "ViT-B/16"
                self.freezenetClip, _ = clip.load(model_name)
                self.freezenetClip = self.freezenetClip.visual.float()
                ref_embed_dim = 512

            elif ref_embed_net == "imagebind":
                self.freezenetImageBin = imagebind_model.imagebind_huge(pretrained=True)
                self.freezenetImageBin.eval()
                ref_embed_dim = 1024

            else:
                raise NotImplementedError(ref_embed_net)

            self.emb_layers = nn.Sequential(
                torch.nn.SiLU(), nn.Linear(ref_embed_dim, cond_embed_class)
            )
            self.cond_embed_gammas -= cond_embed_class

    def forward(self, input, embed_noise_level, cls, mask, ref):
        cls_embed, mask_embed, ref_embed = self.compute_cond(input, cls, mask, ref)
        if "class" in self.conditioning:
            embedding = torch.cat((embed_noise_level, cls_embed), dim=1)
        else:
            embedding = embed_noise_level

        if "ref" in self.conditioning:
            embedding = torch.cat((embedding, ref_embed), dim=1)

        if "mask" in self.conditioning:
            if len(input.shape) == 5:  # video input
                input = torch.cat([input, mask_embed], dim=2)
            else:
                input = torch.cat([input, mask_embed], dim=1)

        if self.model_nargs == 3:  # ref from dataloader with reference image
            out = self.model(input, embedding, ref)
        else:
            out = self.model(input, embedding)
        return out

    def compute_cond(self, input, cls, mask, ref):
        if "class" in self.conditioning and cls is not None:
            if hasattr(self, "netl_embedders_class"):
                cls_embed = []
                for i in range(len(self.netl_embedders_class)):
                    cls_embed.append(self.netl_embedders_class[i](cls[:, i]))
                cls_embed = torch.cat(cls_embed, dim=1)
            else:
                # TODO general case
                cls_embed = self.netl_embedder_class(cls)
        else:
            cls_embed = None

        if "mask" in self.conditioning and mask is not None:
            video_input = False
            if len(input.shape) == 5:  # video input (b,f,1,h,w)
                video_input = True
                b, f = mask.shape[:2]
                mask = rearrange_5dto4d_bf(mask)[0]

            data_crop_size = mask.shape[-1]
            mask_embed = mask.to(torch.int32).squeeze(1)
            mask_embed = rearrange(mask_embed, "b h w -> b (h w)")
            mask_embed = self.netl_embedder_mask(mask_embed)
            mask_embed = rearrange(mask_embed, "b (h w) c -> b c h w", h=data_crop_size)
            if video_input:
                mask_embed = rearrange_4dto5d_bf(f, mask_embed)[0]

        else:
            mask_embed = None

        if "ref" in self.conditioning:
            ref = self.ref_transform(ref)

            if self.ref_embed_net == "clip":
                ref_embed = self.freezenetClip(ref)

            elif self.ref_embed_net == "imagebind":
                input_ref = {ModalityType.VISION: ref}
                ref_embed = self.freezenetImageBin(input_ref)["vision"]

            else:
                raise NotImplementedError(ref_embed_net)

            ref_embed = self.emb_layers(ref_embed)

        else:
            ref_embed = None

        return cls_embed, mask_embed, ref_embed
