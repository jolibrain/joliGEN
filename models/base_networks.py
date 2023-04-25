import torch
from torch import nn


def define_inception(device, dims):
    model = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=True)

    model.fc = nn.Identity()

    model = model.to(device)

    model.eval()

    return model
