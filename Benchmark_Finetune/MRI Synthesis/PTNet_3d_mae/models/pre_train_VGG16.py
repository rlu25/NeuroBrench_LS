import torch
import torch.nn as nn
from torchvision.models import vgg19
from collections import namedtuple


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg19(pretrained=True).features)[:33]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 7, 8, 15, 22}:
                results.append(x)

        return results