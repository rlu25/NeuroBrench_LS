# Copyright (c) 2021, Xuzhe Zhang, Xinzi He, Yun Wang
# MIT License

from timm.models.registry import register_model
from .transformer_block import Block, get_sinusoid_encoding, SWT4_3D, SWT_up4_3D, SWT4_3DT, SWT_up4_3DT
from torch import nn
import torch
import torch.nn.functional as F


class PTNet(nn.Module):
    """
    PTNet class
    """

    def __init__(self, img_size=[64, 64, 64], trans_type='performer', down_ratio=[1, 1, 2, 4,8],
                 channels=[1, 16, 32, 64,128],
                 patch=[7, 3, 3], embed_dim=512, depth=1, num_heads=4, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm,
                 skip_connection=True, individual_use=False):
        super().__init__()
        self.individual = individual_use
        self.down_blocks = nn.ModuleList(
            [SWT4_3D(in_channel=channels[0], out_channel=channels[1], patch=3, stride=1, padding=1),
             SWT4_3D(in_channel=channels[1], out_channel=channels[2], patch=3, stride=2, padding=1),
             SWT4_3D(in_channel=channels[2], out_channel=channels[3], patch=3, stride=2, padding=1),
             SWT4_3D(in_channel=channels[3], out_channel=channels[4], patch=3, stride=2, padding=1)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=nn.LeakyReLU(negative_slope=0.2))
            for i in range(depth)])
        if not skip_connection:
            self.up_blocks = nn.ModuleList(
                [SWT_up4_3D(in_channel=channels[-(i + 1)],
                            out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                            up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                            padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)])
            self.up_blocks.append(SWT_up4_3D(in_channel=channels[1],
                                             out_channel=channels[1], patch=patch[0],
                                             up_scale=int(down_ratio[1] / down_ratio[0]),
                                             padding=int((patch[0] - 1) / 2)))
        else:

            # self.up_blocks = (nn.ModuleList(
            #     [SWT_up4_3D(in_channel=2 * channels[-(i + 1)],
            #                 out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
            #                 up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
            #                 padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)]))
            # self.up_blocks.append(SWT_up4_3D(in_channel=2 * channels[1],
            #                                  out_channel=channels[1], patch=patch[0],
            #                                  up_scale=int(down_ratio[2] / down_ratio[1]),
            #                                  padding=int((patch[0] - 1) / 2)))
            # self.up_blocks.append(SWT_up4_3D(in_channel=channels[1] + 1,
            #                                  out_channel=channels[1], patch=patch[0],
            #                                  up_scale=int(down_ratio[1] / down_ratio[0]),
            #                                  padding=int((patch[0] - 1) / 2)))
            self.up_blocks = (nn.ModuleList(
                [SWT_up4_3D(in_channel=2*channels[-2], out_channel=channels[-3], patch = 3, up_scale=2, padding =1),
                 SWT_up4_3D(in_channel=2*channels[-3], out_channel=channels[-4], patch=3, up_scale=2, padding=1),
                 SWT_up4_3D(in_channel=2*channels[-4], out_channel=channels[-4], patch=3, up_scale=2, padding=1),
                 SWT_up4_3D(in_channel=channels[-4]+1, out_channel=channels[-4], patch=3, up_scale=1, padding=1)]))
        # self.sw = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(
            n_position=(img_size[0] // down_ratio[-1]) * (
                    img_size[1] // down_ratio[-1]) * (img_size[2] // down_ratio[-1]),
            d_hid=embed_dim), requires_grad=False)

        self.final_proj = nn.Linear(channels[1], 1)
        # self.final_LN = nn.LayerNorm(channels[1])

        self.bot_proj = nn.Linear(channels[-2] * patch[-1] * patch[-1] * patch[-1], embed_dim)
        # self.bot_LN = nn.LayerNorm(channels[-1])
        self.bot_proj2 = nn.Linear(embed_dim, channels[-2])
        # self.bot_LN2 = nn.LayerNorm(embed_dim)
        self.tanh = nn.Tanh()
        self.sc = skip_connection
        self.size = img_size
        self.ratio = down_ratio

    def forward(self, x):
        x0 = x
        if not self.sc:
            for i, down in enumerate(self.down_blocks[:-1]):
                x = down(x)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]),
                                              int(self.size[2] / self.ratio[i + 1]))

            x = self.down_blocks[-1](x)
            x = self.bot_proj(x)
            x = x + self.PE
            for blk in self.bottleneck:
                x = blk(x)
            x = self.bot_proj2(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, int(self.size[0] / self.ratio[-1]), int(self.size[1] / self.ratio[-1]),
                          int(self.size[2] / self.ratio[-1]))
            for up in self.up_blocks[:-1]:
                x = up(x, size=[x.shape[2], x.shape[3]])
            x = self.up_blocks[-1](x, reshape=False)
        else:
            SC = []
            global_SC = []
            for i, down in enumerate(self.down_blocks[:-1]):
                x = down(x)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]),
                                              int(self.size[2] / self.ratio[i + 1]))
                SC.append(x)
            x = self.down_blocks[-1](x, attention=False)
            x = self.bot_proj(x)
            x = x + self.PE
            for blk in self.bottleneck:
                x = blk(x)
            x = self.bot_proj2(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, int(self.size[0] / self.ratio[-1]), int(self.size[1] / self.ratio[-1]),
                          int(self.size[2] / self.ratio[-1]))
            for i, up in enumerate(self.up_blocks[:-1]):
                x = up(x, size=[x.shape[2], x.shape[3], x.shape[4]], SC=SC[-(i + 1)], reshape=True)
                global_SC.append(x)
            if not self.individual:
                x_out = self.up_blocks[-1](x, SC=x0, reshape=False)
                x = self.final_proj(x_out).transpose(1, 2)

            else:
                x = self.up_blocks[-1](x, SC=x0, reshape=False)
                x = self.final_proj(x).transpose(1, 2)

        B, C, HW = x.shape
        x = x.reshape(B, C, self.size[0], self.size[1], self.size[2])
        if not self.individual:
            x_out = x_out.transpose(1, 2)
            B, C, HW = x_out.shape
            global_SC.append(x_out.reshape(B, C, self.size[0], self.size[1], self.size[2]))
            SC = SC+global_SC
            return self.tanh(x), SC
        else:
            return self.tanh(x)


class PTNet_local(nn.Module):
    def __init__(self, img_size=[64, 64, 64], trans_type='performer', down_ratio=[1, 1, 2, 4, 8],
                 channels=[1, 16, 32, 64, 128],
                 patch=[3, 3, 3, 3, 3], embed_dim=256, depth=9, num_heads=4, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm,
                 skip_connection=True):
        super().__init__()
        self.GlobalGenerator = PTNet(img_size=[int(img_size[0] / 2), int(img_size[1] / 2), int(img_size[2] / 2)],
                                     down_ratio=[1, 1, 2, 4, 8],
                                     channels=[1, 8, 16, 32, 64],
                                     patch=[3, 3, 3, 3, 3], embed_dim=256, depth=12, individual_use=False,
                                     skip_connection=True)
        self.down_blocks = nn.ModuleList(
            [SWT4_3D(in_channel=1, out_channel=16, patch=3, stride=1, padding=1),
             SWT4_3D(in_channel=72, out_channel=64, patch=3, stride=2, padding=1),
             SWT4_3D(in_channel=64, out_channel=128, patch=3, stride=2, padding=1),
             SWT4_3D(in_channel=128, out_channel=128, patch=3, stride=2, padding=1)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=nn.LeakyReLU(negative_slope=0.2))
            for i in range(depth)])
        if not skip_connection:
            self.up_blocks = nn.ModuleList(
                [SWT_up4_3D(in_channel=channels[-(i + 1)],
                            out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                            up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                            padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)])
            self.up_blocks.append(SWT_up4_3D(in_channel=channels[1],
                                             out_channel=channels[1], patch=patch[0],
                                             up_scale=int(down_ratio[1] / down_ratio[0]),
                                             padding=int((patch[0] - 1) / 2)))
        else:
            self.up_blocks = (nn.ModuleList(
                [SWT_up4_3D(in_channel=256, out_channel=64, patch=3, up_scale=2, padding=1), # 16
                 SWT_up4_3D(in_channel=128, out_channel=32, patch=3, up_scale=2, padding=1), # 32
                 SWT_up4_3D(in_channel=48, out_channel=16, patch=3, up_scale=2, padding=1), # 64
                 SWT_up4_3D(in_channel=57, out_channel=16, patch=3, up_scale=1, padding=1)]))
        # self.sw = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(
            n_position=(img_size[0] // down_ratio[-1]) * (img_size[1] // down_ratio[-1]) * (
                    img_size[2] // down_ratio[-1]),
            d_hid=embed_dim), requires_grad=False)

        self.final_proj = nn.Linear(16, 1)
        # self.final_LN = nn.LayerNorm(channels[1])

        self.bot_proj = nn.Linear(128 * patch[-1] * patch[-1] * patch[-1], embed_dim)
        # self.bot_LN = nn.LayerNorm(channels[-1])
        self.bot_proj2 = nn.Linear(embed_dim,128)
        # self.bot_LN2 = nn.LayerNorm(embed_dim)
        self.tanh = nn.Tanh()
        self.sc = skip_connection
        self.size = img_size
        self.ratio = down_ratio
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=False)
    def forward(self, x):
        x0 = x
        Global_res, global_SC = self.GlobalGenerator(
            F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=False))
        if not self.sc:
            for i, down in enumerate(self.down_blocks[:-1]):
                if i == 1:
                    x = down(torch.cat((x, self.up2(global_SC[0]), self.up4(global_SC[1]), self.up8(global_SC[2])), dim=1))
                else:
                    x = down(x)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]),
                                              int(self.size[2] / self.ratio[i + 1]))

            x = self.down_blocks[-1](x)
            x = self.bot_proj(x)
            x = x + self.PE
            for blk in self.bottleneck:
                x = blk(x)
            x = self.bot_proj2(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, int(self.size[0] / self.ratio[-1]), int(self.size[1] / self.ratio[-1]),
                          int(self.size[2] / self.ratio[-1]))
            for up in self.up_blocks[:-1]:
                x = up(x, size=[x.shape[2], x.shape[3]])
            x = self.up_blocks[-1](x, reshape=False)
        else:
            SC = []

            for i, down in enumerate(self.down_blocks[:-1]):
                if i == 1:
                    x = down(
                        torch.cat((x, self.up2(global_SC[0]), self.up4(global_SC[1]), self.up8(global_SC[2])), dim=1))
                else:
                    x = down(x)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]),
                                              int(self.size[2] / self.ratio[i + 1]))
                SC.append(x)
            x = self.down_blocks[-1](x, attention=False)
            x = self.bot_proj(x)
            x = x + self.PE
            for blk in self.bottleneck:
                x = blk(x)
            x = self.bot_proj2(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, int(self.size[0] / self.ratio[-1]), int(self.size[1] / self.ratio[-1]),
                          int(self.size[2] / self.ratio[-1]))
            for i, up in enumerate(self.up_blocks[:-2]):
                x = up(x, SC=SC[-(i + 1)], reshape=True, size=[x.shape[2], x.shape[3], x.shape[4]])
            for up in self.up_blocks[-2:-1]:
                x = up(x, SC=SC[0], reshape=True,
                       size=[x.shape[2], x.shape[3], x.shape[4]])

            x = self.up_blocks[-1](torch.cat((x,self.up8(global_SC[3]), self.up4(global_SC[4]), self.up2(global_SC[5]), self.up2(global_SC[6])),dim=1), SC=x0, reshape=False)
        x = self.final_proj(x).transpose(1, 2)

        B, C, HW = x.shape

        x = x.reshape(B, C, self.size[0], self.size[1], self.size[2])

        # Global_res = torch.flatten(Global_res,start_dim=2).transpose(1,2)
        # x_low_res = self.final_proj(Global_res).transpose(1, 2)
        #
        # x_low_res = x_low_res.reshape(B, C, int(self.size[0] / 2), int(self.size[1] / 2), int(self.size[2] / 2))

        return self.tanh(x), Global_res  # , self.tanh(x_low_res)


@register_model
def PTN(img_size=[64, 64, 64], **kwargs):
    model = PTNet(img_size=img_size, down_ratio=[1, 1, 2, 4, 8],
                  channels=[1, 16, 32, 64, 128],
                  patch=[3, 3, 3, 3, 3], embed_dim=256, depth=9, num_heads=4, individual_use=True)

    return model


@register_model
def PTN_local(img_size=[64, 64, 64], **kwargs):
    model = PTNet_local(img_size=img_size, **kwargs)

    return model
