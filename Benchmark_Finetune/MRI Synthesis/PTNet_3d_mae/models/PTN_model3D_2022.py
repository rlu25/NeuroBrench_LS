# Copyright (c) 2021, Xuzhe Zhang, Xinzi He, Yun Wang
# MIT License

from timm.models.registry import register_model
from .transformer_block import Block, get_sinusoid_encoding, SWT4_3D, SWT_up4_3D, SWT4_3DT,SWT_up4_3DT
from torch import nn
import torch
import torch.nn.functional as F
from unfoldNd import UnfoldNd


class PTNet(nn.Module):
    """
    PTNet class
    """

    def __init__(self, pyramid=False, pyramid_dim=32,img_size=[64, 64, 64], trans_type='performer', down_ratio=[1, 1, 2, 4],
                 channels=[1, 32, 64, 128],
                 patch=[7, 3, 3], embed_dim=128, depth=5, num_heads=2, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm,
                 skip_connection=True, individual_use=False):
        super().__init__()
        self.individual = individual_use
        self.down_blocks = nn.ModuleList(
            [SWT4_3D(in_channel=1, out_channel=8, patch=3,stride=1,padding=1), # 64
             SWT4_3D(in_channel=8, out_channel=16, patch=3, stride=2, padding=1), # 16
             SWT4_3D(in_channel=16, out_channel=32, patch=3, stride=2, padding=1),  # 8
             SWT4_3D(in_channel=32, out_channel=64, patch=3, stride=2, padding=1)  # 8*8*8*128  4*4*4
             ])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=nn.LeakyReLU(negative_slope=0.2))
            for i in range(depth)])

        self.up_blocks = nn.ModuleList([
            SWT_up4_3D(in_channel=128,out_channel=32,up_scale=2), #16
            SWT_up4_3D(in_channel=64, out_channel=16, up_scale=2),# 32
            SWT_up4_3D(in_channel=32, out_channel=16, up_scale=2),# 64
            SWT_up4_3D(in_channel=24, out_channel=16, up_scale=1),  # 64
            SWT_up4_3D(in_channel=17, out_channel=16, up_scale=1),  # 64
            # SWT_up4_3D(in_channel=17, out_channel=16, up_scale=1),  # 64*64*64*16
        ])

        # self.sw = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(
            n_position=(img_size[0] //8) * (img_size[1] // 8) * (img_size[2] // 8),
            d_hid=embed_dim), requires_grad=False)

        self.final_proj = nn.Linear(16, 1)
        # self.final_LN = nn.LayerNorm(channels[1])

        self.bot_proj = nn.Linear(64, embed_dim)
        # self.bot_LN = nn.LayerNorm(channels[-1])
        self.bot_proj2 = nn.Linear(embed_dim, 64)
        self.sw = UnfoldNd(kernel_size=3,stride=1,padding=1)
        self.tanh = nn.Tanh()
        self.sc = skip_connection
        self.size = img_size
        self.ratio = down_ratio

    def forward(self, x):
        sc = []
        sc.append(x)
        for i,down in enumerate(self.down_blocks):
            x = down(x)
            B,Size_cube, C = x.shape
            x = x.transpose(1,2).reshape(B,C, round(Size_cube**(1./3.)),round(Size_cube**(1./3.)),round(Size_cube**(1./3.)))
            sc.append(x)

        # x = self.sw(x).transpose(1,2)
        x = torch.flatten(x,start_dim=2).transpose(1,2)
        x = self.bot_proj(x)
        x += self.PE

        for blk in self.bottleneck:
            x = blk(x)

        x = self.bot_proj2(x).transpose(1, 2)
        B, C, HW = x.shape
        x = x.reshape(B, C, round(HW**(1/3)), round(HW**(1/3)), round(HW**(1/3)))

        for i, up in enumerate(self.up_blocks):

            x = up(torch.cat((x,sc[-(i+1)]),dim=1),reshape=True if i<len(self.up_blocks)-1 else False)
        x = self.final_proj(x)
        B, HW, C = x.shape
        x = x.reshape(B, C, round(HW ** (1 / 3)), round(HW ** (1 / 3)), round(HW ** (1 / 3)))
        return self.tanh(x)



class Pyramid_layer(nn.Module):
    """
    PTNet class
    """

    def __init__(self, pyramid=False, pyramid_dim=32,img_size=[32, 32, 32], trans_type='performer', down_ratio=[1, 1, 2, 4],
                 channels=[1, 32, 64, 128],
                 patch=[7, 3, 3], embed_dim=512, depth=12, num_heads=8, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm,
                 skip_connection=True, individual_use=False):
        super().__init__()
        self.individual = individual_use
        self.down_blocks = nn.ModuleList(
            [SWT4_3D(in_channel=1, out_channel=32, patch=3,stride=1,padding=1), # 32
             SWT4_3D(in_channel=32, out_channel=64, patch=3, stride=2, padding=1), # 16
             SWT4_3DT(in_channel=64, out_channel=128, patch=3, stride=2, padding=1),  # 8
             SWT4_3DT(in_channel=128, out_channel=256, patch=3, stride=2, padding=1)  # 4
             ])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=nn.LeakyReLU(negative_slope=0.2))
            for i in range(depth)])

        self.up_blocks = nn.ModuleList([
            SWT_up4_3DT(in_channel=512,out_channel=128,up_scale=2), #16
            SWT_up4_3DT(in_channel=256, out_channel=64, up_scale=2),# 32
            SWT_up4_3D(in_channel=128, out_channel=32, up_scale=2),# 64
            SWT_up4_3D(in_channel=64, out_channel=32, up_scale=1),  # 64
            SWT_up4_3D(in_channel=33, out_channel=32, up_scale=1),  # 64
            # SWT_up4_3D(in_channel=17, out_channel=16, up_scale=1),  # 64*64*64*16
        ])

        # self.sw = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(
            n_position=(img_size[0] //8) * (img_size[1] // 8) * (img_size[2] // 8),
            d_hid=embed_dim), requires_grad=False)

        self.final_proj = nn.Linear(32, 1)
        # self.final_LN = nn.LayerNorm(channels[1])

        self.bot_proj = nn.Linear(256, embed_dim)
        # self.bot_LN = nn.LayerNorm(channels[-1])
        self.bot_proj2 = nn.Linear(embed_dim, 256)
        self.sw = UnfoldNd(kernel_size=3,stride=1,padding=1)
        self.tanh = nn.Tanh()
        self.sc = skip_connection
        self.size = img_size
        self.ratio = down_ratio

    def forward(self, x):
        sc = []
        sc.append(x)
        for i,down in enumerate(self.down_blocks):
            x = down(x)
            B,Size_cube, C = x.shape
            x = x.transpose(1,2).reshape(B,C, round(Size_cube**(1./3.)),round(Size_cube**(1./3.)),round(Size_cube**(1./3.)))
            sc.append(x)

        # x = self.sw(x).transpose(1,2)
        x = torch.flatten(x,start_dim=2).transpose(1,2)
        x = self.bot_proj(x)
        x += self.PE

        for blk in self.bottleneck:
            x = blk(x)

        x = self.bot_proj2(x).transpose(1, 2)
        B, C, HW = x.shape
        x = x.reshape(B, C, round(HW**(1/3)), round(HW**(1/3)), round(HW**(1/3)))

        for i, up in enumerate(self.up_blocks):

            x = up(torch.cat((x,sc[-(i+1)]),dim=1),reshape=True if i<len(self.up_blocks)-1 else False)
        B, HW, C = x.shape
        x_out = x.reshape(B, C, round(HW ** (1 / 3)), round(HW ** (1 / 3)), round(HW ** (1 / 3)))
        x = self.final_proj(x)
        B, HW, C = x.shape
        x = x.reshape(B, C, round(HW ** (1 / 3)), round(HW ** (1 / 3)), round(HW ** (1 / 3)))
        return self.tanh(x), x_out


class PTNet_pyramid(nn.Module):
    """
    PTNet class
    """

    def __init__(self, pyramid=False, pyramid_dim=32,img_size=[64, 64, 64], trans_type='performer', down_ratio=[1, 1, 2, 4],
                 channels=[1, 32, 64, 128],
                 patch=[7, 3, 3], embed_dim=128, depth=5, num_heads=2, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm,
                 skip_connection=True, individual_use=False):
        super().__init__()
        self.individual = individual_use
        self.down_blocks = nn.ModuleList(
            [SWT4_3D(in_channel=1, out_channel=8, patch=3,stride=1,padding=1), # 64
             SWT4_3D(in_channel=8, out_channel=16, patch=3, stride=2, padding=1), # 16
             SWT4_3D(in_channel=16, out_channel=32, patch=3, stride=2, padding=1),  # 8
             SWT4_3D(in_channel=32, out_channel=64, patch=3, stride=2, padding=1)  # 8*8*8*128  4*4*4
             ])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=nn.LeakyReLU(negative_slope=0.2))
            for i in range(depth)])

        self.up_blocks = nn.ModuleList([
            SWT_up4_3D(in_channel=128,out_channel=32,up_scale=2), #16
            SWT_up4_3D(in_channel=64, out_channel=16, up_scale=2),# 32
            SWT_up4_3D(in_channel=32+32, out_channel=32, up_scale=2),# 64
            SWT_up4_3D(in_channel=40, out_channel=16, up_scale=1),  # 64
            SWT_up4_3D(in_channel=17, out_channel=16, up_scale=1),  # 64
            # SWT_up4_3D(in_channel=17, out_channel=16, up_scale=1),  # 64*64*64*16
        ])

        # self.sw = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(
            n_position=(img_size[0] //8) * (img_size[1] // 8) * (img_size[2] // 8),
            d_hid=embed_dim), requires_grad=False)

        self.final_proj = nn.Linear(16, 1)
        # self.final_LN = nn.LayerNorm(channels[1])

        self.bot_proj = nn.Linear(64, embed_dim)
        # self.bot_LN = nn.LayerNorm(channels[-1])
        self.bot_proj2 = nn.Linear(embed_dim, 64)
        self.sw = UnfoldNd(kernel_size=3,stride=1,padding=1)
        self.tanh = nn.Tanh()
        self.sc = skip_connection
        self.size = img_size
        self.ratio = down_ratio
        self.pyramid = Pyramid_layer()
        self.down_sample = nn.AvgPool3d(kernel_size=3,stride=2,padding=1, count_include_pad=False)

    def forward(self, x):
        low_x = self.down_sample(x)
        low_res, low_feat = self.pyramid(low_x)


        sc = []
        sc.append(x)
        for i,down in enumerate(self.down_blocks):
            x = down(x)
            B,Size_cube, C = x.shape
            x = x.transpose(1,2).reshape(B,C, round(Size_cube**(1./3.)),round(Size_cube**(1./3.)),round(Size_cube**(1./3.)))
            sc.append(x)

        # x = self.sw(x).transpose(1,2)
        x = torch.flatten(x,start_dim=2).transpose(1,2)
        x = self.bot_proj(x)
        x += self.PE

        for blk in self.bottleneck:
            x = blk(x)

        x = self.bot_proj2(x).transpose(1, 2)
        B, C, HW = x.shape
        x = x.reshape(B, C, round(HW**(1/3)), round(HW**(1/3)), round(HW**(1/3)))

        for i, up in enumerate(self.up_blocks):
            if i == 2:
                x = up(torch.cat((x, sc[-(i + 1)], low_feat), dim=1), reshape=True if i < len(self.up_blocks) - 1 else False)
            else:
                x = up(torch.cat((x,sc[-(i+1)]),dim=1),reshape=True if i<len(self.up_blocks)-1 else False)
        x = self.final_proj(x)
        B, HW, C = x.shape
        x = x.reshape(B, C, round(HW ** (1 / 3)), round(HW ** (1 / 3)), round(HW ** (1 / 3)))
        return self.tanh(x), low_res

@register_model
def PTN(img_size=[64, 64, 64], **kwargs):
    model = PTNet(img_size=img_size, down_ratio=[1, 1, 2, 4, 8],
                  channels=[1, 16, 32, 64, 128],
                  patch=[3, 3, 3, 3,3], embed_dim=256, depth=9, num_heads=2, individual_use=True)

    return model


@register_model
def PTN_local(img_size=[64, 64, 64], **kwargs):
    model = PTNet_local(img_size=img_size, **kwargs)

    return model
