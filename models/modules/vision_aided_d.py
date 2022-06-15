import torch
from torch import nn
import vision_aided_loss


class VisionAidedDiscriminator(nn.Module):
    """Defines a vision-aided discriminator"""

    def __init__(
        self,
        cv_type="clip+dino",
    ):
        super(VisionAidedDiscriminator, self).__init__()
        loss_type = ""  # loss is computed elsewhere
        self.model = vision_aided_loss.Discriminator(
            cv_type,
            loss_type,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.model.cv_ensemble.requires_grad_(False)  # freeze feature extractor

    def forward(self, input):
        return self.model(input)[0]
