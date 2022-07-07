from torch import nn
import torch.nn.functional as F
import warnings


class BaseGenerator_attn(nn.Module):
    # initializers
    def __init__(self, nb_mask_attn, nb_mask_input):
        super(BaseGenerator_attn, self).__init__()
        self.nb_mask_attn = nb_mask_attn
        self.nb_mask_input = nb_mask_input

    def compute_outputs(self, input, attentions, images):
        outputs = []

        if input.shape[1] > 3:
            input = input[:, :3, :, :]

        for i in range(self.nb_mask_attn - self.nb_mask_input):
            if images[i].shape != attentions[i].shape:
                attentions[i] = F.interpolate(
                    attentions[i], size=(images[i].shape[2], images[i].shape[3])
                )

            outputs.append(images[i] * attentions[i])

        for i in range(self.nb_mask_attn - self.nb_mask_input, self.nb_mask_attn):
            if input.shape != attentions[i].shape:
                attentions[i] = F.interpolate(
                    attentions[i], size=(input.shape[2], input.shape[3])
                )

            outputs.append(input * attentions[i])

        return images, attentions, outputs

    # forward method
    def forward(self, input):
        feat, _ = self.compute_feats(input)
        attentions, images = self.compute_attention_content(feat)
        _, _, outputs = self.compute_outputs(input, attentions, images)

        o = outputs[0]
        for i in range(1, self.nb_mask_attn):
            o += outputs[i]
        return o

    def get_attention_masks(self, input):
        feat, _ = self.compute_feats(input)
        attentions, images = self.compute_attention_content(feat)
        return self.compute_outputs(input, attentions, images)

    def get_feats(self, input, extract_layer_ids):
        _, feats = self.compute_feats(input, extract_layer_ids)
        return feats
