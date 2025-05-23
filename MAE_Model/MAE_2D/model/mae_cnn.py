import torch
import torch.nn as nn
from .blocks import create_encoders, ExtResNetBlock, _ntuple, res_decoders
import numpy as np
from .lsc_loss import BinaryKDLoss, ConsistencyLoss, KDLoss


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
        self.local_encoder = create_encoders(in_channels=7, f_maps=to_tuple(embed_dim), basic_module=ExtResNetBlock,
                                             conv_kernel_size=4, conv_stride_size=4, conv_padding=0, layer_order='gcr',
                                             num_groups=32)

        # upsample
        self.local_upsample = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=decoder_embed_dim, kernel_size=4,
                                                 stride=4)

        # decoder

        self.local_decoder = res_decoders(in_channels=decoder_embed_dim, f_maps=[16],
                                          basic_module=ExtResNetBlock, conv_kernel_size=3, conv_stride_size=1,
                                          conv_padding=0, layer_order='gcr', num_groups=8)

        # norm layers
        self.final_projection_local_recon = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.final_norm_local_recon = nn.GroupNorm(
            num_groups=8, num_channels=16)

        self.avgpool = nn.AdaptiveAvgPool2d((3, 1))  # 2D pooling
        self.kd_loss = BinaryKDLoss(kl_loss_factor=1.)
        # self.FCB_consist_loss = ConsistencyLoss(loss_factor=1.,feature_dim=1)
        self.consist_loss = ConsistencyLoss(loss_factor=1.,feature_dim=1)

    def patchify(self, imgs, p):
        """

        imgs: (N, C, H, W)
        Returns: (N, H*W/P^2, P^2)
        """
        N, C, H, W = imgs.shape
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0
        h, w = [i // p for i in self.cfg.data.patch_size[:2]]
        # print('before patchify')
        # print(imgs.shape) = torch.Size([4, 1, 96, 96])
        x = imgs.reshape(shape=(imgs.shape[0], C, h, p, w, p))
        x = torch.einsum('nchpwq->nchwpq', x)
        x = x.reshape(shape=(imgs.shape[0], C, h * w, p ** 2))

        return x


    def unpatchify(self, x, p):
        """

        x: (N, H*W/P^2, P^2)
        Returns: (N, C, H, W)
        """
        
        h, w = [i // p for i in self.cfg.data.patch_size[:2]]
        assert h * w == x.shape[2]

        x = x.reshape(shape=(x.shape[0], x.shape[1], h, w, p, p))
        x = torch.einsum('nchwpq->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], x.shape[1], h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio, p):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        x = self.patchify(x, p)

        N, C, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        ids_keep_exp = ids_keep.unsqueeze(1).unsqueeze(-1).repeat(1, C, 1, D)
        x_masked = torch.gather(x, dim=2, index=ids_keep_exp)

        # Pad missing (masked) patches with zeros
        x_empty = torch.zeros((N, C, L - len_keep, D), device=x.device)
        x_ = torch.cat([x_masked, x_empty], dim=2)

        # Restore original order
        ids_restore_exp = ids_restore.unsqueeze(1).unsqueeze(-1).repeat(1, C, 1, D)
        x_ = torch.gather(x_, dim=2, index=ids_restore_exp)

        # Binary mask: 0 = keep, 1 = masked
        mask = torch.ones((N, L), device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)  # [N, L]
        mask = mask.unsqueeze(1).repeat(1, C, 1)  # [N, C, L]

        # Expand mask for spatial (patch) dimension
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, D)  # [N, C, L, D]

        
        x_masked = self.unpatchify(x_, p)

        mask = self.unpatchify(mask, p)

        return x_masked, mask

    def forward_encoder(self, x, mask_ratio, p):

        # masking: length -> length * mask_ratio
        x, mask = self.random_masking(x, mask_ratio, p)

        # apply Transformer blocks
        for blk in self.local_encoder:
            x = blk(x)

        return x, mask

    def forward_local_decoder(self, x):
        x = self.local_upsample(x)
        # apply Transformer blocks
        for blk in self.local_decoder:
            x = blk(x)

        x = self.final_norm_local_recon(x)

        x = self.final_projection_local_recon(x)
        x = torch.sigmoid(x)

        return x

    def recon_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        loss = (pred - imgs) ** 2

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_train(self,
                      local_patch,
                    #   global_img,
                      local_patch2=None, global_img2=None):
        # reconstruction for view 1
        
        local_latent, local_mask = self.forward_encoder(
            local_patch, self.cfg.train.mask_ratio, self.cfg.train.local_mae_patch)
        local_pred = self.forward_local_decoder(local_latent)  # [N, L, p*p*3]
        idx = local_patch.shape[1]//2
        target_patch = local_patch[:, idx, :, :].unsqueeze(1)
        target_mask = local_mask[:, idx, :, :].unsqueeze(1)
        local_loss = self.recon_loss(target_patch, local_pred, target_mask)

        


        # if no second view, return zeros for all consistency terms
        if local_patch2 is None or global_img2 is None:
            return (
                local_loss, 
                # global_loss,
                local_pred, 
                # global_pred,
                local_mask, 
                # global_mask,
                # zero, zero, zero, zero, zero, zero
            )

        # View 2 reconstruction
        local_latent2, local_mask2 = self.forward_encoder(
            local_patch2, self.cfg.train.mask_ratio, self.cfg.train.local_mae_patch)
        local_pred2 = self.forward_local_decoder(local_latent2)
        local_loss2 = self.recon_loss(local_patch2, local_pred2, local_mask2)

        
        # Pool and flatten features for consistency 
        f_ll_1 = self.avgpool(local_latent).flatten(1)
        f_ll_2 = self.avgpool(local_latent2).flatten(1)
        
        f_lp_1 = local_pred.flatten(1)
        f_lp_2 = local_pred2.flatten(1)
        
        # Consistency losses
        # latent
        loss_kd_ll = self.kd_loss(f_ll_1, f_ll_2)
        loss_fcb_ll = self.FCB_consist_loss(f_ll_1, f_ll_2)
        loss_tb_ll = self.TB_consist_loss(f_ll_1, f_ll_2)
        
        # decoder
        loss_kd_lp = self.kd_loss(f_lp_1, f_lp_2)
        loss_fcb_lp = self.FCB_consist_loss(f_lp_1, f_lp_2)
        loss_tb_lp = self.TB_consist_loss(f_lp_1, f_lp_2)
        
        # aggregate reconstruction losses
        local_loss  = local_loss + local_loss2
        
        # aggregate consistency losses
        

        return (
            local_loss, 
            local_pred, 
            local_mask, 
            
        )


    def forward(self, local_patch, 
                # global_img
                ):
        local_latent, local_mask = self.forward_encoder(
            local_patch, self.cfg.train.mask_ratio, self.cfg.train.local_mae_patch)
        local_pred = self.forward_local_decoder(local_latent)  # [N, L, p*p*3]
        return local_pred, local_mask
    # global_pred, 
    
    # global_mask
