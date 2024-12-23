import os, sys
sys.path.append('/data-home/tangt/models/ARTS/lib')

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
# from models.simple3dposeBaseSMPL import Simple3DPoseBaseSMPL
from models.TIK import TIK
# from models.layers.module import PoseNet, Pose2Feat, MeshNet, ParamRegressor

# from models.HSCR import HSCR
BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR

class CoevoBlock(nn.Module):
    def __init__(self, num_joint, num_vertx, joint_dim=64, vertx_dim=64):
        super(CoevoBlock, self).__init__()

        self.num_joint = num_joint
        self.num_vertx = num_vertx

        self.joint_proj = nn.Linear(3, joint_dim)
        self.vertx_proj = nn.Linear(3, vertx_dim)

        self.proj_v2j_dim = nn.Linear(num_vertx+num_joint+32, num_joint)
        self.proj_j2v_dim = nn.Linear(num_vertx+num_joint+32, num_vertx)
       
        self.proj_joint_feat2coor = nn.Linear(joint_dim, 3)
        self.proj_vertx_feat2coor = nn.Linear(vertx_dim, 3)

    def forward(self, joint, vertx, img_feat):

        joint_feat, vertx_feat = self.joint_proj(joint), self.vertx_proj(vertx) # [B,17,3] -> [B,17,64], [B,431,3] -> [B,431,64]
        
        img_feat = img_feat.reshape(-1,32,64)

        concat_feat = torch.cat([joint_feat, vertx_feat, img_feat], dim=1)
        
        joint_out_feat = self.proj_v2j_dim(concat_feat.transpose(1,2)).transpose(1,2)
        vertx_out_feat = self.proj_j2v_dim(concat_feat.transpose(1,2)).transpose(1,2)
        
        joint, vertx = self.proj_joint_feat2coor(joint_out_feat) + joint[:,:,:3], self.proj_vertx_feat2coor(vertx_out_feat) + vertx[:,:,:3] # [B,17,3], [B,431,3]

        return joint, vertx

class Residual(nn.Module):
    def __init__(self, num_joint, embed_dim=256, SMPL_MEAN_vertices=osp.join(BASE_DATA_DIR, 'smpl_mean_vertices.npy')):
        super(Residual, self).__init__()

        self.mesh = Mesh()
        
        # downsample mesh vertices from 6890 to 431
        init_vertices = torch.from_numpy(np.load(SMPL_MEAN_vertices)).cuda()
        downsample_verts_1723 = self.mesh.downsample(init_vertices)                    # [1723, 3]
        downsample_verts_431 = self.mesh.downsample(downsample_verts_1723, n1=1, n2=2) # [431, 3]
        self.register_buffer('init_vertices', downsample_verts_431)
        self.num_verts = downsample_verts_431.shape[0]

        # calculate the nearest joint of each vertex
        J_regressor = torch.from_numpy(np.load('data/Human36M/J_regressor_h36m_correct.npy').astype(np.float32)).cuda()
        self.joints_template = torch.matmul(J_regressor, init_vertices)
        self.vj_relation, self.jv_sets = build_verts_joints_relation(self.joints_template.cpu().numpy(), downsample_verts_431.cpu().numpy())

        self.coevoblock1 = CoevoBlock(num_joint, num_vertx=self.num_verts, joint_dim=cfg.MODEL.joint_dim, vertx_dim=cfg.MODEL.vertx_dim)
        self.coevoblock2 = CoevoBlock(num_joint, num_vertx=self.num_verts, joint_dim=cfg.MODEL.joint_dim, vertx_dim=cfg.MODEL.vertx_dim)
        self.coevoblock3 = CoevoBlock(num_joint, num_vertx=self.num_verts, joint_dim=cfg.MODEL.joint_dim, vertx_dim=cfg.MODEL.vertx_dim)
        self.upsample_conv1 = nn.Conv1d(self.num_verts, 1732, kernel_size=3, padding=1)
        self.upsample_conv2 = nn.Conv1d(1732, 6890, kernel_size=3, padding=1)
    
        
    def forward(self, joint, img_feat):
        
        # reinitialize each vertex to its nearest joint
        vertxs = joint[:, self.vj_relation, :3]
        
        # co-evolution blocks
        joints1, vertxs = self.coevoblock1(joint, vertxs, img_feat)
        joints2, vertxs = self.coevoblock2(joint, vertxs, img_feat)
        joints3, vertxs = self.coevoblock3(joint, vertxs, img_feat)
        vertxs1 = self.upsample_conv1(vertxs)
        vertxs_w_res = self.upsample_conv2(vertxs1)
        
        return joints3, vertxs_w_res  # B x 6890 x 3