import torch
import torch.nn as nn
from models.blocks import create_encoders, ExtResNetBlock, _ntuple, res_decoders
import numpy as np


class MAE_CNN(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, cfg):
        super().__init__()
        # --------------------------------------------------------------------------
        # ResNet encoder specifics
        self.cfg = cfg
        embed_dim = cfg.model.embed_dim
        depth = cfg.model.depth
        decoder_embed_dim = embed_dim // 16
        to_tuple = _ntuple(depth)
        # encoder
        self.local_encoder = create_encoders(in_channels=1, f_maps=to_tuple(embed_dim), basic_module=ExtResNetBlock,
                                             conv_kernel_size=4, conv_stride_size=4, conv_padding=0, layer_order='gcr',
                                             num_groups=32)

        # upsample
        self.local_upsample = nn.ConvTranspose3d(in_channels=embed_dim, out_channels=decoder_embed_dim, kernel_size=4,
                                                 stride=4)

        # decoder

        self.local_decoder = res_decoders(in_channels=decoder_embed_dim, f_maps=[16],
                                          basic_module=ExtResNetBlock, conv_kernel_size=3, conv_stride_size=1,
                                          conv_padding=0, layer_order='gcr', num_groups=8)

        # norm layers
        self.final_projection_local_recon = nn.Conv3d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.final_norm_local_recon = nn.GroupNorm(
            num_groups=8, num_channels=16)

        self.avgpool = nn.AdaptiveAvgPool3d((3, 1, 1))




    # def forward(self, local_patch, global_img):
    def forward(self, x):
        x0 = self.local_encoder[0](x)

        # apply encoder blocks        # apply encoder blocks
        for idx, blk in enumerate(self.local_encoder[1:]):       
            if idx == 0:
                x1 = blk(x0)
            else:
                x1 = blk(x1)

        return x0, x1
