from core.config import cfg
import os
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from functools import partial
from funcs_utils import load_checkpoint
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp, Attention

import os
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class  GraphormerNet(nn.Module):
    def __init__(self, num_frames=cfg.DATASET.seqlen, num_joints=16, embed_dim=256, depth=3, num_heads=8, mlp_ratio=2., 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, pretrained=False):
        super().__init__()

        in_dim = 2
        out_dim = 3    
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.joint_embed = nn.Linear(in_dim, embed_dim)
        self.imgfeat_embed = nn.Linear(2048, embed_dim)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.depth = depth

        self.SpatialBlocks1 = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TemporalBlocks1 = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.SpatialBlocks2 = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TemporalBlocks2 = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm_s = norm_layer(embed_dim)
        self.norm_t = norm_layer(embed_dim)

        self.regression = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )
        self.fusion = torch.nn.Conv2d(in_channels=num_frames, out_channels=1, kernel_size=1)
        
        if pretrained:
            self._load_pretrained_model()
        
    def _load_pretrained_model(self):
        print("Loading pretrained posenet...")
        checkpoint = load_checkpoint(load_dir=cfg.MODEL.posenet_path, pick_best=True)
        self.load_state_dict(checkpoint['model_state_dict'])

    def SpaTemHead(self, x, img_feat):
        b, t, j, c = x.shape
        x = rearrange(x, 'b t j c  -> (b t) j c')
        x = self.joint_embed(x)
        x = x + rearrange(self.imgfeat_embed(img_feat), 'b t c  -> (b t) 1 c')
        x += self.spatial_pos_embed
        x = self.pos_drop(x)
        spablock = self.SpatialBlocks1[0]
        x = spablock(x)
        x = self.norm_s(x)
        
        x = rearrange(x, '(b t) j c -> (b j) t c', t=t)
        x += self.temporal_pos_embed
        x = self.pos_drop(x)
        temblock = self.TemporalBlocks1[0]
        x = temblock(x)
        x = self.norm_t(x)
        return x

    def forward(self, x, img_feat):
        b, t, j, c = x.shape
        # sh torch.Size([64, 16, 16, 2])
        # import pdb; pdb.set_trace()
        x = self.SpaTemHead(x, img_feat) # bj t c
        y = x
        
        for i in range(1, self.depth):
            SpaAtten1 = self.SpatialBlocks1[i]
            TemAtten1 = self.TemporalBlocks1[i]
            x = rearrange(x, '(b j) t c -> (b t) j c', j=j)
            x = SpaAtten1(x)
            x = self.norm_s(x)
            x = rearrange(x, '(b t) j c -> (b j) t c', t=t)
            x = TemAtten1(x)
            x = self.norm_t(x)

            SpaAtten2 = self.SpatialBlocks2[i]
            TemAtten2 = self.TemporalBlocks2[i]
            y = TemAtten2(y)
            y = self.norm_t(y)
            y = rearrange(y, '(b j) t c -> (b t) j c', j=j)
            y = SpaAtten2(y)
            y = self.norm_s(y)
            y = rearrange(y, '(b t) j c -> (b j) t c', t=t)
            x = 0.5 * x + 0.5 * y
            y = 0.5 * x + 0.5 * y
 
        x = rearrange(x, '(b j) t c -> b t j c', j=j)
        x = self.regression(x) # (b t (j * 3))
        x = x.view(b, t, j, -1)
        # xout = self.fusion(x)
        # xout = xout.squeeze(1)
        return x


def get_model(num_joint=16, embed_dim=256, depth=3, pretrained=False): 
    # num_joint = 16
    model = GraphormerNet(num_frames=cfg.DATASET.seqlen, num_joints=num_joint, embed_dim=embed_dim, depth=depth, pretrained=pretrained)
    return model