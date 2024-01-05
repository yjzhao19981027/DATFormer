import torch.nn as nn
import torch
from .token_performer import Token_performer
from .Transformer import token_TransformerEncoder
from .DistortionMapping import DistortionMapping
import numpy as np

class token_trans(nn.Module):
    def __init__(self, in_dim=64, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.,  attn_block=True):
        super(token_trans, self).__init__()

        self.attn_block = attn_block
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        if self.attn_block:
            self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=0.5)
        else:
            self.encoderlayer = DistortionMapping(input_channel=embed_dim, output_channel=embed_dim)

    def forward(self, fea):
        B, _, _ = fea.shape
        # fea [B, H*W, 64]
        # project to 384 dim
        fea = self.mlp(self.norm(fea))
        # fea [B, H*W, 384]
        if self.attn_block:
            fea = self.encoderlayer(fea)
        else:
            B, HW, C = fea.shape
            fea = fea.reshape(B, int(np.sqrt(HW // 2)), int(np.sqrt(HW // 2)) * 2, C).permute(0, 3, 1, 2)
            fea = self.encoderlayer(fea)
            fea = fea.permute(0, 2, 3, 1).reshape(B, HW, C)
            

        return fea


class reverse_t2t_module(nn.Module):
    def __init__(self, dim=384, token_dim=64, img_size=224, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True):
        super(reverse_t2t_module, self).__init__()

        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size // ratio,  2 * img_size // ratio), kernel_size=kernel_size, stride=stride, padding=padding)
        self.fuse = fuse
        if self.fuse:
            self.concatFuse = nn.Sequential(
                nn.Linear(token_dim*2, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )
            self.att = Token_performer(dim=token_dim, in_dim=token_dim, kernel_ratio=0.5)

            # project input feature to 64 dim
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, dec_fea, enc_fea=None):
        # from 384 to 64
        dec_fea = self.mlp(self.norm(dec_fea))

        # [1] token upsampling by the proposed reverse T2T module
        dec_fea = self.project(dec_fea)
        # [B, H*W, token_dim*kernel_size*kernel_size]
        dec_fea = self.upsample(dec_fea.transpose(1, 2))
        B, C, _, _ = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2)
        # [B, HW, C]

        if self.fuse:
            # [2] fuse encoder fea and decoder fea
            dec_fea = self.concatFuse(torch.cat([dec_fea, enc_fea], dim=2))
            dec_fea = self.att(dec_fea)
        return dec_fea


class Reverse_t2t(nn.Module):
    def __init__(self, embed_dim=384, token_dim=64, depth=2, img_size=224):

        super(Reverse_t2t, self).__init__()

        self.img_size = img_size
        # token upsampling and multi-level token fusion
        self.reverse_t2t_1_16 = reverse_t2t_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.reverse_t2t_1_8 = reverse_t2t_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.reverse_t2t_1_4 = reverse_t2t_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)

        # token based multi-task predictions
        self.token_pre_1_16 = token_trans(in_dim=embed_dim, embed_dim=embed_dim, depth=4, num_heads=6, attn_block=False)
        self.token_pre_1_8 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1, attn_block=True)
        self.token_pre_1_4 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1, attn_block=True)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 and classname.find('DW') == -1 and classname.find('DConvMLP') == -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4):
        # rgb_fea_1_16 [B, 14*14, 384] 
        # rgb_fea_1_8 [B, 28*28, 64]
        # rgb_fea_1_4 [B, 56*56, 64]
        B, _, _, = rgb_fea_1_16.size()

        token_fea_1_16 = self.token_pre_1_16(rgb_fea_1_16)
        # 1/16 -> 1/8
        # reverse T2T and fuse low-level feature
        fea_1_8 = self.reverse_t2t_1_16(token_fea_1_16, rgb_fea_1_8)

        # fea_1_8 = self.reverse_t2t_1_16(rgb_fea_1_16, rgb_fea_1_8)

        token_fea_1_8 = self.token_pre_1_8(fea_1_8)
        # 1/8 -> 1/4
        fea_1_4 = self.reverse_t2t_1_8(token_fea_1_8, rgb_fea_1_4)

        token_fea_1_4 = self.token_pre_1_4(fea_1_4)
        # 1/4 -> 1
        fea1_1 = self.reverse_t2t_1_4(token_fea_1_4)

        fea1_1 = fea1_1.transpose(1, 2).reshape(B, -1, self.img_size, self.img_size * 2)
        fea1_4 = fea_1_4.transpose(1, 2).reshape(B, -1, self.img_size // 4, self.img_size // 2)
        fea1_8 = fea_1_8.transpose(1, 2).reshape(B, -1, self.img_size // 8, self.img_size // 4)

        return fea1_1, fea1_4, fea1_8

