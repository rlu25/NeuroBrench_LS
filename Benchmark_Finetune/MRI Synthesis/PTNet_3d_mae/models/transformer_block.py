# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Borrow from timm(https://github.com/rwightman/pytorch-image-models)
"""
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath
from models.token_performer import Token_performer
from unfoldNd import UnfoldNd
from models.token_transformer import Token_transformer
from models.pixelshuffle3d import PixelShuffle3d
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.LeakyReLU(negative_slope=0.2),
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention2(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, N = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        x = v.squeeze(1) + x  # because the original x has different size with current x, use v to do skip connection

        return x


# class Token_transformer(nn.Module):
#
#     def __init__(self, dim, in_dim, num_heads=4, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim=dim,
#             num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(in_dim)
#         self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio), out_features=in_dim,
#                        act_layer=act_layer, drop=drop)
#
#     def forward(self, x):
#         print(x.shape)
#         x = self.attn(self.norm1(x))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU(), norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class SWT(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, img_size, in_channel=1, out_channel=64, patch=4, stride=2, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        num_patches = (img_size // stride) * (img_size // stride)
        self.transformer = Token_performer(dim=out_channel, in_dim=out_channel)
        self.proj = nn.Linear(in_channel * patch * patch, out_channel)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=out_channel),
                               requires_grad=True)
        self.LN = nn.LayerNorm(out_channel)

    def forward(self, x):
        x = self.sw(x).transpose(1, 2)
        x = self.proj(x)
        x = self.LN(x) + self.PE

        x = self.transformer(x)

        return x


class SWT_up(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, up_scale, img_size, in_channel=1, out_channel=64, patch=4, stride=1, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        num_patches = (img_size // stride) * (img_size // stride)

        self.transformer = Token_performer(dim=in_channel * patch * patch, in_dim=out_channel)

        # self.PE = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=in_channel*patch*patch))
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=up_scale)

    def forward(self, x):
        x = self.sw(x).transpose(1, 2)
        x = self.transformer(x)
        B, HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))

        return self.up_sample(x)


class SWT2(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, img_size, in_channel=1, out_channel=64, patch=4, stride=2, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        num_patches = (img_size // stride) * (img_size // stride)
        self.transformer = Token_performer(dim=in_channel * patch * patch, in_dim=out_channel)
        self.transformer2 = Token_performer(dim=out_channel, in_dim=out_channel)

    def forward(self, x, attention=True):
        x = self.sw(x).transpose(1, 2)
        if attention:
            x = self.transformer(x)
            x = self.transformer2(x)

        return x


class SWT_up2(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, up_scale, img_size, in_channel=1, out_channel=64, patch=4, stride=1, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        num_patches = (img_size // stride) * (img_size // stride)

        self.transformer = Token_performer(dim=in_channel, in_dim=out_channel)
        self.transformer2 = Token_performer(dim=out_channel, in_dim=out_channel)

        # self.PE = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=in_channel*patch*patch))
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=up_scale)

    def forward(self, x, SC=None, reshape=True):
        x = self.up_sample(x)
        if SC is not None:
            x = torch.cat((x, SC), dim=1)
        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.transformer(x)
        x = self.transformer2(x)
        if reshape:
            B, HW, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))
        return x


class SWT3(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, img_size, in_channel=1, out_channel=64, patch=4, stride=2, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        num_patches = (img_size // stride) * (img_size // stride)
        self.transformer = Token_performer(dim=out_channel, in_dim=out_channel)
        self.proj = nn.Linear(in_channel * patch * patch, out_channel)
        self.transformer2 = Token_performer(dim=out_channel, in_dim=out_channel)
        self.LN = nn.LayerNorm(out_channel)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=out_channel),
                               requires_grad=False)

    def forward(self, x, attention=True):
        x = self.sw(x).transpose(1, 2)
        if attention:
            x = self.proj(x)
            x = self.LN(x) + self.PE
            x = self.transformer(x)
            x = self.transformer2(x)

        return x


class SWT_up3(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, up_scale, img_size, in_channel=1, out_channel=64, patch=4, stride=1, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        # num_patches = (img_size // stride) * (img_size // stride)

        self.transformer = Token_performer(dim=in_channel, in_dim=out_channel)
        # self.transformer2 = Token_performer(dim=out_channel, in_dim=out_channel)
        # self.proj = nn.Linear(in_channel,out_channel)
        # self.LN = nn.LayerNorm(out_channel)
        # self.PE = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=out_channel),requires_grad=False)
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=up_scale)

    def forward(self, x, SC=None, reshape=True):
        x = self.up_sample(x)
        if SC is not None:
            x = torch.cat((x, SC), dim=1)
        x = x.flatten(start_dim=2).transpose(1, 2)
        # x = self.proj(x)
        # x = self.LN(x) + self.PE
        x = self.transformer(x)
        # x = self.transformer2(x)
        if reshape:
            B, HW, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(HW)), int(np.sqrt(HW)))
        return x


class SWT4(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, in_channel=1, out_channel=64, patch=4, stride=2, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        self.transformer = Token_performer(dim=in_channel * patch * patch , in_dim=out_channel)

    def forward(self, x, attention=True):
        x = self.sw(x).transpose(1, 2)
        if attention:
            x = self.transformer(x)

        return x


class SWT_up4(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, up_scale, in_channel=1, out_channel=64, patch=4, stride=1, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        # num_patches = (img_size // stride) * (img_size // stride)

        self.transformer = Token_performer(dim=in_channel, in_dim=out_channel)

        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=up_scale)
        self.factor = up_scale

    def forward(self, x, size=None, SC=None, reshape=True):
        x = self.up_sample(x)
        if SC is not None:
            x = torch.cat((x, SC), dim=1)
        x = x.flatten(start_dim=2).transpose(1, 2)

        x = self.transformer(x)
        if reshape:
            B, HW, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, int(size[0] * self.factor), int(size[1] * self.factor))
        return x


class SWT_up4_3D(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, up_scale, in_channel=1, out_channel=64, patch=4, stride=1, padding=1):
        super().__init__()
        self.sw = UnfoldNd(kernel_size=patch, stride=stride, padding=padding)
        # num_patches = (img_size // stride) * (img_size // stride)

        self.transformer = Token_performer(dim=in_channel, in_dim=out_channel)
        # self.transformer = Token_transformer(dim=in_channel, in_dim=out_channel,num_heads=1)

        self.up_sample = nn.Upsample(scale_factor=up_scale, mode='trilinear', align_corners=True)
        self.factor = up_scale

    def forward(self, x, size=None, SC=None, reshape=True):
        x = self.up_sample(x)
        if SC is not None:
            x = torch.cat((x, SC), dim=1)
        x = x.flatten(start_dim=2).transpose(1, 2)

        x = self.transformer(x)
        if reshape:
            B, HW, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, int(size[0] * self.factor), int(size[1] * self.factor), int(size[2] * self.factor))
        return x

class SWT4_3D(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, in_channel=1, out_channel=64, patch=4, stride=2, padding=1):
        super().__init__()
        self.sw = UnfoldNd(kernel_size=patch, stride=stride, padding=padding)
        self.transformer = Token_performer(dim=in_channel * patch * patch * patch, in_dim=out_channel)
        # self.transformer = Token_transformer(dim=in_channel * patch * patch * patch, in_dim=out_channel, num_heads=1)

    def forward(self, x, attention=True):
        x = self.sw(x).transpose(1, 2)
        if attention:
            x = self.transformer(x)

        return x

class SWT_up4_ps(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, up_scale, in_channel=1, out_channel=64, patch=4, stride=1, padding=1):
        super().__init__()
        self.sw = nn.Unfold(kernel_size=patch, stride=stride, padding=padding)
        # num_patches = (img_size // stride) * (img_size // stride)

        self.transformer = Token_performer(dim=in_channel, in_dim=out_channel)

        self.up_sample = nn.PixelShuffle(upscale_factor=up_scale)
        self.factor = up_scale


    def forward(self, x, size=None, SC=None, reshape=True):
        x = self.up_sample(x)
        if SC is not None:
            x = torch.cat((x, SC), dim=1)
        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.transformer(x)
        if reshape:
            B, HW, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, int(size[0] * self.factor), int(size[1] * self.factor))
        return x

class SWT_up4_3D_ps(nn.Module):
    """
    sliding window tokenization
    """

    def __init__(self, up_scale, in_channel=1, out_channel=64, patch=4, stride=1, padding=1):
        super().__init__()
        self.sw = UnfoldNd(kernel_size=patch, stride=stride, padding=padding)
        # num_patches = (img_size // stride) * (img_size // stride)

        self.transformer = Token_performer(dim=in_channel, in_dim=out_channel)

        self.up_sample = PixelShuffle3d(scale=up_scale)
        self.factor = up_scale

    def forward(self, x, size=None, SC=None, reshape=True):
        x = self.up_sample(x)
        if SC is not None:
            x = torch.cat((x, SC), dim=1)
        x = x.flatten(start_dim=2).transpose(1, 2)

        x = self.transformer(x)
        if reshape:
            B, HW, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, int(size[0] * self.factor), int(size[1] * self.factor), int(size[2] * self.factor))
        return x