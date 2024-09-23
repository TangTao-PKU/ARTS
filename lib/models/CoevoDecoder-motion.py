import os, sys
sys.path.append('/home/tangt/PMCE/Seq2Seq/lib')

import numpy as np
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg, Mlp

from core.config import cfg
from graph_utils import build_verts_joints_relation
from models.backbones.mesh import Mesh
from functools import partial

# from models.update import UpdateBlock, Regressor
from models.spin import RegressorSpin
from models.NIKITS import NIKITS

# from models.HSCR import HSCR
from models.Residual import Residual
from math import sqrt

BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0
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


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, depth=4, embed_dim=512, mlp_hidden_dim=2048, length=16, h=8):
        super().__init__()
        drop_rate = 0.1
        drop_path_rate = 0.2
        attn_drop_rate = 0.

        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=True, qk_scale=None,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)]) 
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, v_dim, kv_num, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.kv_num = kv_num
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(v_dim, v_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):

        B, N, C = xq.shape
        v_dim = xv.shape[-1]
        q = self.wq(xq).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B,N1,C] -> [B,N1,H,(C/H)] -> [B,H,N1,(C/H)]
        k = self.wk(xk).reshape(B, self.kv_num, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B,N2,C] -> [B,N2,H,(C/H)] -> [B,H,N2,(C/H)]
        v = self.wv(xv).reshape(B, self.kv_num, self.num_heads, v_dim // self.num_heads).permute(0, 2, 1, 3)  # [B,N2,C] -> [B,N2,H,(C/H)] -> [B,H,N2,(C/H)]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,N1,(C/H)] @ [B,H,(C/H),N2] -> [B,H,N1,N2]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, v_dim)   # [B,H,N1,N2] @ [B,H,N2,(C/H)] -> [B,H,N1,(C/H)] -> [B,N1,H,(C/H)] -> [B,N1,C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, kv_num, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.2, 
                 attn_drop=0.2, drop_path=0.2, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.normq = norm_layer(q_dim)
        self.normk = norm_layer(k_dim)
        self.normv = norm_layer(v_dim)
        self.kv_num = kv_num
        self.attn = CrossAttention(q_dim, v_dim, kv_num = kv_num, num_heads=num_heads, qkv_bias=qkv_bias, 
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(q_dim)
            mlp_hidden_dim = int(q_dim * mlp_ratio)
            self.mlp = Mlp(in_features=q_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, xq, xk, xv):
        xq = xq + self.drop_path(self.attn(self.normq(xq), self.normk(xk), self.normv(xv)))
        if self.has_mlp:
            xq = xq + self.drop_path(self.mlp(self.norm2(xq)))

        return xq

class Pose2Mesh(nn.Module):
    def __init__(self, num_joint, embed_dim=256, SMPL_MEAN_vertices=osp.join(BASE_DATA_DIR, 'smpl_mean_vertices.npy')):
        super(Pose2Mesh, self).__init__()
        self.mesh = Mesh()

        self.projoint = nn.Linear(num_joint*3, 512)
        self.proj = nn.Linear(2048, 512)
        self.trans1 = Transformer(4, 512, 512*4,length=cfg.DATASET.seqlen)
        self.out_proj = nn.Linear(512, 2048)

        self.regressorspin = RegressorSpin()
        # self.regressorhscr = HSCR(drop=0.5)
        pretrained_dict = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))['model']
        self.regressorspin.load_state_dict(pretrained_dict, strict=False)
        
        self.init_pose = nn.Linear(num_joint*3,24*6)
        self.nikits = NIKITS(seq_len=cfg.DATASET.seqlen, num_joints=num_joint)

        self.residual = Residual(num_joint=num_joint)

        self.mcca = CrossAttentionBlock(q_dim=512, k_dim=512, v_dim=512, kv_num = cfg.DATASET.seqlen, num_heads=8, mlp_ratio=4., qkv_bias=True,
                                                drop=0., attn_drop=0., drop_path=0.2, has_mlp=True)
        
    def forward(self, joints, img_feats, is_train=True, J_regressor=None):
        # torch.Size([30, 16, 2048])
        batch_size = img_feats.shape[0]

        # print(joints.shape)
        # torch.Size([30, 16, 19, 3])
        joints_seq = joints

        # 1: Motion-centric Cross-attention
        # motion torch.Size([30, 15, 19, 3])
        img_feats_trans = self.proj(img_feats)              # B 16 512
        joints_seq_trans = self.projoint(joints_seq.view(batch_size, cfg.DATASET.seqlen, -1))
        first_repeat = joints_seq_trans[:, 0]
        first_repeat = first_repeat[:, None]
        joints_seq_trans = torch.cat([first_repeat, joints_seq_trans], dim=1)

        motion = joints_seq_trans[:,1:] - joints_seq_trans[:,:-1]
        img_feats_cross = self.mcca(motion, img_feats_trans, img_feats_trans)
        img_feats_trans = self.out_proj(img_feats_cross)

        # img_feats_trans = self.proj(img_feats)
        # img_feats_trans = self.trans1(img_feats_trans)      # B 16 512
        # img_feats_trans = self.out_proj(img_feats_trans)    # B 16 2048

        # 2: rotation init
        output_pose2SMPL = self.nikits(joints_seq, img_feats_trans)
        inv_pred2rot6d = output_pose2SMPL['inv_pred2rot6d'].reshape(batch_size, cfg.DATASET.seqlen, -1)
        # 时序IK网络设计
        # batchsize*seqlen*24*6
        # inv_pred2rot6d_mid, _ = self.fusion(inv_pred2rot6d.permute(1,0,2))
        # inv_pred2rot6d_mid = self.linear_fusion(F.relu(inv_pred2rot6d_mid[cfg.DATASET.seqlen//2]))

        # 3: bone-guided shape fitting
        
        
        # SMPL Regressor
        output = self.regressorspin(img_feats_trans, init_pose=inv_pred2rot6d, is_train=is_train, J_regressor=J_regressor)
        # output = self.regressorhscr(img_feats_trans,init_pose=inv_pred2rot6d, init_shape=smpl_output[-1]['theta'][:,75:], init_cam=smpl_output[-1]['theta'][:,:3], is_train=is_train, J_regressor=None)

        # attentive addtion
        smpl_vertices_mid = output[-1]['verts'][:,cfg.DATASET.seqlen//2]
        residual_joint, residual_mesh = self.residual(joints[:,8], img_feats[:,8])
        smpl_vertices_mid = 0.5 * smpl_vertices_mid + 0.5 * residual_mesh
        # import pdb; pdb.set_trace()

        return residual_joint, inv_pred2rot6d, smpl_vertices_mid, output  # B x 6890 x 3

def get_model(num_joint, embed_dim):
    model = Pose2Mesh(num_joint, embed_dim)
    return model

if __name__ == '__main__':
    
    model = get_model(17,64) # 模型初始化
    model = model.cuda()
    joint = torch.randn(32, 17, 3).cuda()
    img = torch.randn(32, 16, 2048).cuda()
    Pose2Mesh(joint,img)
    model.eval()
