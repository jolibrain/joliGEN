import torch
from torch import nn

from einops import rearrange


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
    def __init__(self, model, cond_embed_dim, conditioning, nclasses):
        super().__init__()

        self.model = model
        self.conditioning = conditioning
        self.cond_embed_dim = cond_embed_dim

        # Label embedding
        if "class" in conditioning:
            cond_embed_class = cond_embed_dim // 2
            self.netl_embedder_class = LabelEmbedder(
                nclasses,
                cond_embed_class,  # * image_size * image_size
            )
            nn.init.normal_(self.netl_embedder_class.embedding_table.weight, std=0.02)

        if "mask" in conditioning:
            # TODO make a new option
            cond_embed_mask = cond_embed_dim
            self.netl_embedder_mask = LabelEmbedder(
                nclasses,
                cond_embed_mask,  # * image_size * image_size
            )
            nn.init.normal_(self.netl_embedder_mask.embedding_table.weight, std=0.02)

    def forward(self, input, embed_noise_level, cls, mask):
        cls_embed, mask_embed = self.compute_cond(input, cls, mask)

        if "class" in self.conditioning:
            embedding = torch.cat((embed_noise_level, cls_embed), dim=1)
        else:
            embedding = embed_noise_level

        if "mask" in self.conditioning:
            input = torch.cat([input, mask_embed], dim=1)

        return self.model(input, embedding)

    def compute_cond(self, input, cls, mask):
        if "class" in self.conditioning and cls is not None:
            cls_embed = self.netl_embedder_class(cls)
        else:
            cls_embed = None

        if "mask" in self.conditioning and mask is not None:
            data_crop_size = mask.shape[-1]
            mask_embed = mask.to(torch.int32).squeeze(1)
            mask_embed = rearrange(mask_embed, "b h w -> b (h w)")
            mask_embed = self.netl_embedder_mask(mask_embed)
            mask_embed = rearrange(mask_embed, "b (h w) c -> b c h w", h=data_crop_size)

        else:
            mask_embed = None

        return cls_embed, mask_embed
