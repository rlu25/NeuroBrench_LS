import torch
import torch.nn as nn
from .blocks import create_encoders, ExtResNetBlock, _ntuple, res_decoders
import numpy as np


class MAE_CNN(nn.Module):
    """ MAE Encoder 
    """

    def __init__(self):
        super().__init__()
        # --------------------------------------------------------------------------
        # LOG: WE HARDCODE CFG HERE
        # embed_dim = 512
        embed_dim = 256
        depth = 8
        decoder_embed_dim = embed_dim // 16
        to_tuple = _ntuple(depth)
        # encoder
        self.local_encoder = create_encoders(in_channels=1, f_maps=to_tuple(embed_dim), basic_module=ExtResNetBlock,
                                             conv_kernel_size=4, conv_stride_size=4, conv_padding=0, layer_order='gcr',
                                             num_groups=32)

    def forward(self, x):

        # masking: length -> length * mask_ratio

        # apply encoder blocks
        for blk in self.local_encoder:
            x = blk(x)

        return x
