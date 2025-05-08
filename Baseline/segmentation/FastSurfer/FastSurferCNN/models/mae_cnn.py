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
        embed_dim = 512
        depth = 8
        decoder_embed_dim = embed_dim // 16
        to_tuple = _ntuple(depth)
        # encoder
        self.local_encoder = create_encoders(in_channels=1, f_maps=to_tuple(embed_dim), basic_module=ExtResNetBlock,
                                             conv_kernel_size=4, conv_stride_size=4, conv_padding=0, layer_order='gcr',
                                             num_groups=32)

    def forward(self, x):

        # x0 for skip connection
        # x1 as the encoded feature
        x0 = self.local_encoder[0](x)

        # apply encoder blocks
        for idx, blk in enumerate(self.local_encoder[1:]):
            if idx == 0:
                x1 = blk(x0)
            else:
                x1 = blk(x1)

        return x0, x1
