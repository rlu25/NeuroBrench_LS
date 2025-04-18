import torch
import torch.nn as nn
from .mae_cnn import MAE_CNN
from .sub_module import CompetitiveDenseBlock, ClassifierBlock

'''
Class definition of MAE_UNet, leveraing a 2D encoder pretrained via MAE 



'''


class MAE_UNet(nn.Module):
    # use softmax = True for segmentation
    def __init__(self, num_classes, softmax=False):
        super().__init__()

        self.enc = MAE_CNN()

        # connector_1 to upsample the skip connection to 2x its resolution
        self.connector_1 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.ReLU(True)
        )
        # connector_2 to upsample the skip connection to 4x its resolution
        self.connector_2 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=4,
                               padding=0, bias=True),
            nn.ReLU(True)
        )

        # config_dec1: config for the first decoder after encoding
        config_dec1 = {
            'kernel_h': 3,
            'kernel_w': 3,
            'num_channels': 512,
            'num_filters': 512,
            'num_filters_last': 256,
            'stride_conv': 1,
        }
        # config_dec2, config for the second decoder (final one)
        config_dec2 = {
            'kernel_h': 3,
            'kernel_w': 3,
            'num_channels': 512,
            'num_filters': 512,
            'num_filters_last': 256,
            'stride_conv': 1,
        }

        config_classifier = {
            'num_channels': 256,
            'num_classes': num_classes,
            'kernel_c': 1,
            'stride_conv': 1
        }
        # upsample the bottleneck feat by 2
        self.upsample_1 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.ReLU(True)
        )
        # upsample the output of first decoder by 2
        self.upsample_2 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.ReLU(True)
        )

        self.dec1 = CompetitiveDenseBlock(config_dec1)
        self.dec2 = CompetitiveDenseBlock(config_dec2)
        self.classfier = ClassifierBlock(config_classifier)
        self.use_softmax = softmax
        if self.use_softmax:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, img):
        _skip, feat = self.enc(img)

        feat = self.upsample_1(feat)

        feat = self.dec1(torch.concat([self.connector_1(_skip), feat], dim=1))

        feat = self.upsample_2(feat)

        feat = self.dec2(torch.concat([self.connector_2(_skip), feat], dim=1))

        opt = self.classfier(feat)

        if self.use_softmax:
            return self.softmax(opt)

        return opt
