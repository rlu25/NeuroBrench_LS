# Copyright (c) 2021, Xuzhe Zhang, Xinzi He, Yun Wang
# MIT License

from timm.models.registry import register_model
from models.transformer_block import Block, get_sinusoid_encoding, SWT4_3D, SWT_up4_3D
from torch import nn
import torch
import torch.nn.functional as F
from unfoldNd import UnfoldNd
from models.mae_cnn import MAE_CNN
from models.cfg.default import get_cfg_defaults

class PTNet(nn.Module):
    """
    PTNet class
    """

    def __init__(self, img_size=[64, 64, 64], trans_type='performer', down_ratio=[1, 1, 2, 4],
                 channels=[1, 32, 64, 128],
                 patch=[7, 3, 3], embed_dim=512, depth=9, num_heads=4, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm,
                 skip_connection=True, individual_use=True):
        super().__init__()
        self.individual = individual_use
        self.down_blocks = nn.ModuleList(
            [SWT4_3D(in_channel=channels[i], out_channel=channels[i + 1], patch=patch[i],
                     stride=int(down_ratio[i + 1] / down_ratio[i]),
                     padding=int((patch[i] - 1) / 2)) for i in range(len(down_ratio) - 1)])
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
                [SWT_up4_3D(in_channel=2 * channels[-(i + 1)],
                            out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                            up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                            padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)]))
            self.up_blocks.append(SWT_up4_3D(in_channel=2 * channels[1],
                                             out_channel=channels[1], patch=patch[0],
                                             up_scale=int(down_ratio[2] / down_ratio[1]),
                                             padding=int((patch[0] - 1) / 2)))
            self.up_blocks.append(SWT_up4_3D(in_channel=channels[1] + 1,
                                             out_channel=channels[1], patch=patch[0],
                                             up_scale=int(down_ratio[1] / down_ratio[0]),
                                             padding=int((patch[0] - 1) / 2)))
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
            if not self.individual:
                x = self.up_blocks[-1](x, SC=x0, reshape=True, size=[x.shape[2], x.shape[3], x.shape[4]])
            else:
                x = self.up_blocks[-1](x, SC=x0, reshape=False)

        if not self.individual:
            return x
        else:
            x = self.final_proj(x).transpose(1, 2)
            B, C, HW = x.shape
            x = x.reshape(B, C, self.size[0], self.size[1], self.size[2])
            return self.tanh(x)


class PTNet_local(nn.Module):
    def __init__(self, img_size=[64, 64, 64], trans_type='performer', down_ratio=[1, 1, 2, 4, 8],
                 channels=[1, 16, 48, 96, 192],
                 patch=[3, 3, 3, 3, 3], embed_dim=256, depth=9, num_heads=4, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm,
                 skip_connection=True):
        super().__init__()
        self.GlobalGenerator = PTNet(img_size=[int(img_size[0] / 2), int(img_size[1] / 2), int(img_size[2] / 2)],
                                     down_ratio=[1, 1, 2, 4],
                                     channels=[1, 16, 32, 64],
                                     patch=[3, 3, 3, 3], embed_dim=256, depth=9, individual_use=False,
                                     skip_connection=True)
        self.down_blocks = nn.ModuleList(
            [SWT4_3D(in_channel=1, out_channel=16, patch=3, stride=1, padding=1),  # 64
             SWT4_3D(in_channel=16, out_channel=32, patch=3, stride=2, padding=1),  # 32
             SWT4_3D(in_channel=48, out_channel=96, patch=3, stride=2, padding=1),  # 16
             SWT4_3D(in_channel=96, out_channel=192, patch=3, stride=2, padding=1)])  # 8
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
                [SWT_up4_3D(in_channel=2 * channels[-(i + 1)],
                            out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                            up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                            padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)]))
            self.up_blocks.append(SWT_up4_3D(in_channel=2 * channels[1],
                                             out_channel=channels[1], patch=patch[0],
                                             up_scale=int(down_ratio[2] / down_ratio[1]),
                                             padding=int((patch[0] - 1) / 2)))
            self.up_blocks.append(SWT_up4_3D(in_channel=channels[1] + 1,
                                             out_channel=channels[1], patch=patch[0],
                                             up_scale=int(down_ratio[1] / down_ratio[0]),
                                             padding=int((patch[0] - 1) / 2)))
        # self.sw = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(
            n_position=(img_size[0] // down_ratio[-1]) * (img_size[1] // down_ratio[-1]) * (
                    img_size[2] // down_ratio[-1]),
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
        Global_res = self.GlobalGenerator(F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=True))
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
            for i, down in enumerate(self.down_blocks[:-1]):
                x = down(x)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]),
                                              int(self.size[2] / self.ratio[i + 1]))
                if i == 1:
                    x = torch.cat((x, Global_res), dim=1)
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

            x = self.up_blocks[-1](x, SC=x0, reshape=False)
        x = self.final_proj(x).transpose(1, 2)

        B, C, HW = x.shape

        x = x.reshape(B, C, self.size[0], self.size[1], self.size[2])

        # Global_res = torch.flatten(Global_res,start_dim=2).transpose(1,2)
        # x_low_res = self.final_proj(Global_res).transpose(1, 2)
        #
        # x_low_res = x_low_res.reshape(B, C, int(self.size[0] / 2), int(self.size[1] / 2), int(self.size[2] / 2))

        return self.tanh(x)  # , self.tanh(x_low_res)


class Trans_global(nn.Module):
    """
    PTNet class
    """

    def __init__(self, embed_dim=64, depth=9, num_heads=2, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=nn.LeakyReLU(negative_slope=0.2))
            for i in range(depth)])
        self.sw = UnfoldNd(kernel_size=3, stride=1, padding=1)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(n_position=16 ** 3, d_hid=embed_dim), requires_grad=False)

        # self.final_proj = nn.Linear(32, 1)
        # self.final_LN = nn.LayerNorm(channels[1])

        self.bot_proj = nn.Linear(3 ** 3, embed_dim)
        # self.bot_LN = nn.LayerNorm(channels[-1])
        self.bot_proj2 = nn.Linear(embed_dim, 32)
        # self.bot_LN2 = nn.LayerNorm(embed_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.sw(x).transpose(1, 2)
        x = self.bot_proj(x)
        x = x + self.PE
        for blk in self.bottleneck:
            x = blk(x)
        x = self.bot_proj2(x)
        x = x.transpose(1, 2)
        B, C, HW = x.shape
        x = x.reshape(B, C, 16, 16, 16)
        return x


class Trans_global2(nn.Module):
    """
    PTNet class
    """

    def __init__(self, embed_dim=64, depth=9, num_heads=2, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=nn.LeakyReLU(negative_slope=0.2))
            for i in range(depth)])
        self.sw = UnfoldNd(kernel_size=3, stride=1, padding=1)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(n_position=8 ** 3, d_hid=embed_dim), requires_grad=False)

        # self.final_proj = nn.Linear(32, 1)
        # self.final_LN = nn.LayerNorm(channels[1])

        self.bot_proj = nn.Linear(3 ** 3, embed_dim)
        # self.bot_LN = nn.LayerNorm(channels[-1])
        self.bot_proj2 = nn.Linear(embed_dim, 32)
        # self.bot_LN2 = nn.LayerNorm(embed_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.sw(x).transpose(1, 2)
        x = self.bot_proj(x)
        x = x + self.PE
        for blk in self.bottleneck:
            x = blk(x)
        x = self.bot_proj2(x)
        # x = x.transpose(1, 2)
        # B, C, HW = x.shape
        # x = x.reshape(B, C, 8, 8, 8)
        return x


class PTNet_local_trans(nn.Module):
    def __init__(self, img_size=[64, 64, 64], 
                 trans_type='performer', 
                 down_ratio=[1, 1, 2, 4, 8],
                 channels=[1, 16, 32, 96, 192],
                 patch=[3, 3, 3, 3, 3], embed_dim=256, depth=9, num_heads=4, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm,
                 skip_connection=True):
        super().__init__()
        cfg = get_cfg_defaults()
        self.enc = MAE_CNN(cfg)

        
        self.connector_1 = nn.Sequential( 
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.ConvTranspose3d(512, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.ReLU(True)
        )
        self.connector_2 = nn.Sequential( 
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.Conv3d(512, 32, kernel_size=3, stride=1,
                               padding=1, bias=True),
            nn.ReLU(True)
        )
        self.connector_3 = nn.Sequential( 
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.ReLU(True)
        )        

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
            # 3 upsampling SC 
            # down ratio: [1,1,2,4,8]
            self.up_blocks = (
                nn.ModuleList(
                [SWT_up4_3D(in_channel=2 * channels[-(i + 1)],
                            out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                            up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                            padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(2, len(down_ratio) - 2)]
                )
                )
            self.up_blocks.append(SWT_up4_3D(in_channel=2 * channels[1],
                                             out_channel=channels[1], patch=patch[0],
                                             up_scale=int(down_ratio[2] / down_ratio[1]),
                                             padding=int((patch[0] - 1) / 2)))
            self.up_blocks.append(SWT_up4_3D(in_channel=channels[1] + 1,
                                             out_channel=channels[1], patch=patch[0],
                                             up_scale=int(down_ratio[1] / down_ratio[0]),
                                             padding=int((patch[0] - 1) / 2)))
        # self.sw = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(
            n_position=(img_size[0] // down_ratio[-1]) * (img_size[1] // down_ratio[-1]) * (
                    img_size[2] // down_ratio[-1]),
            d_hid=embed_dim), requires_grad=False)

        self.final_proj = nn.Linear(channels[1], 1)

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
        # Global_feat = self.GlobalGenerator(F.interpolate(x, scale_factor=0.25, mode='trilinear', align_corners=True))
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
            skip_, x = self.enc(x)
            skip_ = self.connector_1(skip_)
            # up  = nn.Upsample(scale_factor=2,mode="trilinear")
            SC.append(self.connector_3(skip_))
            SC.append(skip_)
            x = self.connector_2(x)
            
            
            
                        
            # TODO: Convert block below to our pretrained 3dmae encoder, 
            # refer to U-Net https://github.com/rlu25/NeuroBrench_LS/commit/ecadbdfd18a700be936cd558f01a9449118483ce#diff-803df9312e5e87a4eb477dc235f7a247d18e91880f20aa28c44623d2b92f82f4
            # 3 Skip connections + 2 upsampling to the desired size
            # comment out components used below in initialization
            # add CONNECTOR

        
            for i, up in enumerate(self.up_blocks[:-2]):
                # torch.Size([1, 512, 24, 24, 24])
                # torch.Size([1, 256, 48, 48, 48])
                # print(x.shape)
                # print(SC[-(i + 1)].shape)
                x = up(x, SC=SC[-(i + 1)], reshape=True, size=[x.shape[2], x.shape[3], x.shape[4]])
                
            for up in self.up_blocks[-2:-1]:
                #  torch.Size([1, 16, 48, 48, 48])
                # torch.Size([1, 32, 48, 48, 48])               
                # print(x.shape)
                # print(SC[0].shape)
                
                x = up(x, SC=SC[0], reshape=True,
                       size=[x.shape[2], x.shape[3], x.shape[4]])

            x = self.up_blocks[-1](x, SC=x0, reshape=False)
        x = self.final_proj(x).transpose(1, 2)

        B, C, HW = x.shape

        x = x.reshape(B, C, self.size[0], self.size[1], self.size[2])

        # Global_res = torch.flatten(Global_res,start_dim=2).transpose(1,2)
        # x_low_res = self.final_proj(Global_res).transpose(1, 2)
        # x_low_res = x_low_res.reshape(B, C, int(self.size[0] / 2), int(self.size[1] / 2), int(self.size[2] / 2))
        return self.tanh(x)  # , Global_res#, self.tanh(x_low_res)


class PTNet_local_trans_3layers(nn.Module):
    def __init__(self, img_size=[64, 64, 64], trans_type='performer', down_ratio=[1, 1, 2, 4, 8],
                 channels=[1, 16, 32, 96, 192],
                 patch=[3, 3, 3, 3, 3], embed_dim=256, depth=9, num_heads=4, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm,
                 skip_connection=True):
        super().__init__()
        self.GlobalGenerator = Trans_global()
        self.GlobalGenerator2 = Trans_global2()
        self.down_blocks = nn.ModuleList(
            [SWT4_3D(in_channel=1, out_channel=16, patch=3, stride=1, padding=1),  # 64
             SWT4_3D(in_channel=16, out_channel=32, patch=3, stride=2, padding=1),  # 32
             SWT4_3D(in_channel=32, out_channel=64, patch=3, stride=2, padding=1),  # 16
             SWT4_3D(in_channel=96, out_channel=192, patch=3, stride=2, padding=1)])  # 8
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim+32, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
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
                [SWT_up4_3D(in_channel=2 * channels[-(i + 1)],
                            out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                            up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                            padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)]))
            self.up_blocks.append(SWT_up4_3D(in_channel=2 * channels[1],
                                             out_channel=channels[1], patch=patch[0],
                                             up_scale=int(down_ratio[2] / down_ratio[1]),
                                             padding=int((patch[0] - 1) / 2)))
            self.up_blocks.append(SWT_up4_3D(in_channel=channels[1] + 1,
                                             out_channel=channels[1], patch=patch[0],
                                             up_scale=int(down_ratio[1] / down_ratio[0]),
                                             padding=int((patch[0] - 1) / 2)))
        # self.sw = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(
            n_position=(img_size[0] // down_ratio[-1]) * (img_size[1] // down_ratio[-1]) * (
                    img_size[2] // down_ratio[-1]),
            d_hid=embed_dim+32), requires_grad=False)

        self.final_proj = nn.Linear(channels[1], 1)
        # self.final_LN = nn.LayerNorm(channels[1])

        self.bot_proj = nn.Linear(channels[-2] * patch[-1] * patch[-1] * patch[-1], embed_dim)
        # self.bot_LN = nn.LayerNorm(channels[-1])
        self.bot_proj2 = nn.Linear(embed_dim+32, channels[-2])
        # self.bot_LN2 = nn.LayerNorm(embed_dim)
        self.tanh = nn.Tanh()
        self.sc = skip_connection
        self.size = img_size
        self.ratio = down_ratio

    def forward(self, x):
        x0 = x
        Global_feat = self.GlobalGenerator(F.interpolate(x, scale_factor=0.25, mode='trilinear', align_corners=True))
        Global_feat2 = self.GlobalGenerator2(F.interpolate(x, scale_factor=0.125, mode='trilinear', align_corners=True))

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
            for i, down in enumerate(self.down_blocks[:-1]):
                x = down(x)
                B, HW, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]),
                                              int(self.size[2] / self.ratio[i + 1]))
                # if i == 1:
                #     print(x.shape)
                #     print(Global_feat2.shape)
                #     x = torch.cat((x, Global_feat2), dim=1)
                if i == 2:
                    x = torch.cat((x, Global_feat), dim=1)
                SC.append(x)
            x = self.down_blocks[-1](x, attention=False)

            x = self.bot_proj(x)
            x = torch.cat((x,Global_feat2),dim=-1)
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

            x = self.up_blocks[-1](x, SC=x0, reshape=False)
        x = self.final_proj(x).transpose(1, 2)

        B, C, HW = x.shape

        x = x.reshape(B, C, self.size[0], self.size[1], self.size[2])

        # Global_res = torch.flatten(Global_res,start_dim=2).transpose(1,2)
        # x_low_res = self.final_proj(Global_res).transpose(1, 2)
        #
        # x_low_res = x_low_res.reshape(B, C, int(self.size[0] / 2), int(self.size[1] / 2), int(self.size[2] / 2))
        return self.tanh(x)  # , Global_res#, self.tanh(x_low_res)


class PTNet_local_trans_noSC(nn.Module):
    def __init__(self, img_size=[64, 64, 64], trans_type='performer', down_ratio=[1, 1, 2, 4, 8],
                 channels=[1, 16, 32, 96, 192],
                 patch=[3, 3, 3, 3, 3], embed_dim=256, depth=9, num_heads=4, mlp_ratio=2., drop_rate=0.,
                 qkv_bias=False, qk_scale=None, attn_drop_rate=0., drop_path_rate=0, norm_layer=nn.LayerNorm,
                 skip_connection=True):
        super().__init__()
        self.GlobalGenerator = Trans_global()
        self.down_blocks = nn.ModuleList(
            [SWT4_3D(in_channel=1, out_channel=16, patch=3, stride=1, padding=1),  # 64
             SWT4_3D(in_channel=16, out_channel=32, patch=3, stride=2, padding=1),  # 32
             SWT4_3D(in_channel=32, out_channel=64, patch=3, stride=2, padding=1),  # 16
             SWT4_3D(in_channel=96, out_channel=192, patch=3, stride=2, padding=1)])  # 8
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.bottleneck = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=nn.LeakyReLU(negative_slope=0.2))
            for i in range(depth)])


        self.up_blocks = (nn.ModuleList(
                [SWT_up4_3D(in_channel= channels[-(i + 1)],
                            out_channel=channels[-(i + 1) - 1], patch=patch[-(i + 1)],
                            up_scale=int(down_ratio[-(i + 1)] / down_ratio[-(i + 1) - 1]),
                            padding=int((patch[-(i + 1)] - 1) / 2)) for i in range(1, len(down_ratio) - 2)]))
        self.up_blocks.append(SWT_up4_3D(in_channel= channels[1],
                                             out_channel=channels[1], patch=patch[0],
                                             up_scale=int(down_ratio[2] / down_ratio[1]),
                                             padding=int((patch[0] - 1) / 2)))
        self.up_blocks.append(SWT_up4_3D(in_channel=channels[1],
                                             out_channel=channels[1], patch=patch[0],
                                             up_scale=int(down_ratio[1] / down_ratio[0]),
                                             padding=int((patch[0] - 1) / 2)))
        # self.sw = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.PE = nn.Parameter(data=get_sinusoid_encoding(
            n_position=(img_size[0] // down_ratio[-1]) * (img_size[1] // down_ratio[-1]) * (
                    img_size[2] // down_ratio[-1]),
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
        Global_feat = self.GlobalGenerator(F.interpolate(x, scale_factor=0.25, mode='trilinear', align_corners=True))

        for i, down in enumerate(self.down_blocks[:-1]):
            x = down(x)
            B, HW, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, int(self.size[0] / self.ratio[i + 1]),
                                              int(self.size[1] / self.ratio[i + 1]),
                                              int(self.size[2] / self.ratio[i + 1]))
            if i == 2:
                x = torch.cat((x, Global_feat), dim=1)
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

            x = up(x, SC=None, reshape=True, size=[x.shape[2], x.shape[3], x.shape[4]])
        for up in self.up_blocks[-2:-1]:
            x = up(x, SC=None, reshape=True,
                       size=[x.shape[2], x.shape[3], x.shape[4]])

        x = self.up_blocks[-1](x, SC=None, reshape=False)
        x = self.final_proj(x).transpose(1, 2)

        B, C, HW = x.shape

        x = x.reshape(B, C, self.size[0], self.size[1], self.size[2])

        # Global_res = torch.flatten(Global_res,start_dim=2).transpose(1,2)
        # x_low_res = self.final_proj(Global_res).transpose(1, 2)
        #
        # x_low_res = x_low_res.reshape(B, C, int(self.size[0] / 2), int(self.size[1] / 2), int(self.size[2] / 2))
        return self.tanh(x)  # , Global_res#, self.tanh(x_low_res)

@register_model
def PTN(img_size=[64, 64, 64], **kwargs):
    model = PTNet(img_size=img_size, down_ratio=[1, 1, 2, 4, 8],
                  channels=[1, 16, 32, 64, 128],
                  patch=[3, 3, 3, 3, 3], embed_dim=256, depth=9, individual_use=True)

    return model


@register_model
def PTN_local(img_size=[64, 64, 64], **kwargs):
    model = PTNet_local(img_size=img_size, **kwargs)

    return model


def PTN_local_trans(img_size=[64, 64, 64], **kwargs):
    model = PTNet_local_trans(img_size=img_size, **kwargs)

    return model

def PTN_local_trans2(img_size=[64,64,64],**kwargs):
    model = PTNet_local_trans_3layers(img_size=img_size,**kwargs)
    return model
def PTN_local_trans_noSC(img_size=[64, 64, 64], **kwargs):
    model = PTNet_local_trans_noSC(img_size=img_size, **kwargs)

    return model
import nibabel as nib
import os
import numpy as np
def sav_feat(feat,num,ext,cls):
    for i in range(0,feat.shape[1],int(feat.shape[1]/16)):
        img = feat[0,i,:,:].detach().cpu().numpy()
        img = nib.Nifti1Image(img,np.eye(4))
        nib.save(img,os.path.join('/home/xuzhe/Data/dHCP_VOI_64/feat_res_2D',ext+'_'+ str(i)+'_' +cls +'.nii.gz'))


