import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix

from .layers.real_nvp import RealNVP
from .layers.shape_real_nvp import ShapeCondRealNVP
from .layers.smpl.SMPL import SMPL_layer
from easydict import EasyDict as edict

from models.spin import RegressorSpin
import os.path as osp
from core.config import cfg
BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR
joints_name_17 = (
    'pelvis', 'left_hip', 'right_hip',      # 2
    'spine1', 'left_knee', 'right_knee',    # 5
    'spine2', 'left_ankle', 'right_ankle',  # 8
    'spine3', 'left_foot', 'right_foot',    # 11
    'neck', 'left_collar', 'right_collar',  # 14
    'jaw',                                  # 15
    'left_shoulder', 'right_shoulder',      # 17
    'left_elbow', 'right_elbow',            # 19
    'left_wrist', 'right_wrist',            # 21
    'left_thumb', 'right_thumb',            # 23
    'head', 'left_middle', 'right_middle',  # 26
    'left_bigtoe', 'right_bigtoe'           # 28
)
joints_name_17 = (
    'Pelvis',                               # 0
    'L_Hip', 'L_Knee', 'L_Ankle',           # 3
    'R_Hip', 'R_Knee', 'R_Ankle',           # 6
    'Torso', 'Neck',                        # 8
    'Nose', 'Head',                         # 10
    'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
    'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
)
joints_name_24 = (
    'pelvis', 'left_hip', 'right_hip',      # 2
    'spine1', 'left_knee', 'right_knee',    # 5
    'spine2', 'left_ankle', 'right_ankle',  # 8
    'spine3', 'left_foot', 'right_foot',    # 11
    'neck', 'left_collar', 'right_collar',  # 14
    'jaw',                                  # 15
    'left_shoulder', 'right_shoulder',      # 17
    'left_elbow', 'right_elbow',            # 19
    'left_wrist', 'right_wrist',            # 21
    'left_thumb', 'right_thumb'             # 23
)

def get_nets(inp_dim):
    def nets():
        return nn.Sequential(nn.Linear(inp_dim, 512), nn.Dropout(), nn.LeakyReLU(), nn.Linear(512, 512), nn.Dropout(), nn.LeakyReLU(), nn.Linear(512, inp_dim), nn.Tanh())
    return nets


def get_nett(inp_dim):
    def nett():
        return nn.Sequential(nn.Linear(inp_dim, 512), nn.Dropout(), nn.LeakyReLU(), nn.Linear(512, 512), nn.Dropout(), nn.LeakyReLU(), nn.Linear(512, inp_dim))
    return nett


class TIK(nn.Module):
    def __init__(
        self,
        seq_len,
        num_joints,
        **cfg
    ):

        super(TIK, self).__init__()

        self.seqlen = seq_len

        self.regressor = FlowRegressor(num_joints)

    def forward(self, pose, img_feat, **kwargs):

        smpl_output = self.regressor(pose, img_feat, **kwargs)

        return smpl_output




class FlowRegressor(nn.Module):
    def __init__(self, num_joints,dtype=torch.float32):
        super(FlowRegressor, self).__init__()
        self.num_joints = num_joints

        h36m_jregressor = np.load('./data/base_data/J_regressor_h36m.npy')
        self.smpl_layer = SMPL_layer(
            './data/base_data/SMPL_NEUTRAL.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=torch.float32
        )
 

        self.feature_channel = 2048
        # self.jts_inp_dim = 17 * 3 + 17 * 1
        self.jts_inp_dim = self.num_joints * 3 + self.num_joints * 1
        self.jts_out_dim = 24 * 6
        self.phi_tot_dim = 23 * 2

        self.shape_tot_dim = 10

        self.zes_dim = 32
        self.jts_tot_dim_1 = self.jts_out_dim + self.zes_dim
        # self.zx_dim = self.jts_tot_dim_1 - self.jts_inp_dim - self.shape_tot_dim
        self.zx_dim = self.jts_tot_dim_1 - self.jts_inp_dim

        self.jts_tot_dim_2 = self.jts_out_dim + self.phi_tot_dim

        self.zet_dim = self.phi_tot_dim

        num_stack1 = 16
        num_stack2 = 16

        masks_2 = torch.from_numpy(np.array([
            [0, 1] * (self.jts_tot_dim_2 // 2),
            [1, 0] * (self.jts_tot_dim_2 // 2)
        ] * (num_stack2 // 2)).astype(np.float32))

        self.flow_j2s = ShapeCondRealNVP(jts_tot_dim = self.jts_tot_dim_1, shape_tot_dim = self.shape_tot_dim, num_stack=num_stack1, use_shape=False)
        self.flow_s2r = RealNVP(get_nets(self.jts_tot_dim_2), get_nett(self.jts_tot_dim_2), masks_2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.feature_channel, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)  # [cos(phi), sin(phi)]
        
        self.regressorspin = RegressorSpin()
        pretrained_dict = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))['model']
        self.regressorspin.load_state_dict(pretrained_dict, strict=False)

    def forward(self, pose, img_feat, target=None, **kwargs):
        # input size NTF
        batch_size, seqlen = pose.shape[:2]
        
        output = self.regressorspin(img_feat, is_train=True, J_regressor=None)
        pred_betas = output[-1]['theta'][:,:,75:85]
        
        xc = img_feat
        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)   
        xc = self.drop2(xc)
        input_phis = self.decphi(xc)
        input_joints = pose

        input_joints = input_joints.reshape(batch_size * seqlen, self.num_joints * 3)
        input_joints = align_root(input_joints, self.num_joints)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_sigma = torch.ones(batch_size * seqlen, self.num_joints * 1).to(device)

        input_phis = input_phis.reshape(batch_size, seqlen, 23, 2)
        input_phis = input_phis / (torch.norm(input_phis, dim=3, keepdim=True) + 1e-8)
        input_phis = input_phis.reshape(batch_size * seqlen, 23 * 2)

        input_shape = pred_betas.reshape(batch_size * seqlen, 10)

        zx = torch.zeros((batch_size * seqlen, self.zx_dim), device=input_joints.device, dtype=input_joints.dtype)

        input1 = torch.cat((
            input_joints, input_sigma,
            # input_betas,
            zx), dim=1)

        # 1. inverse pass: jts -> swing: [-1, 24 * 6]
        pose_out, _, _ = self.flow_j2s.inverse_p(input1, input_shape)
        pose_out = pose_out.reshape(batch_size * seqlen, self.jts_tot_dim_1)

        inv_pred2swingrot6d = pose_out[:, :24 * 6].contiguous().reshape(batch_size * seqlen, 24, 6)

        assert pose_out.shape[1] == self.zes_dim + 24 * 6
        
        # 2. inverse pass: swing, twist -> rot
        input2 = torch.cat((inv_pred2swingrot6d.reshape(-1, 24 * 6), input_phis), dim=1)
        pose_out, _ = self.flow_s2r.inverse_p(input2)
        inv_pred2rot6d = pose_out[:, :24 * 6].contiguous().reshape(batch_size * seqlen, 24, 6)
        assert pose_out.shape[1] == self.zet_dim + 24 * 6

        inv_pred_rot = rotation_6d_to_matrix(inv_pred2rot6d).reshape(batch_size * seqlen, 24, 9)

        pred_beta = pred_betas.reshape(batch_size * seqlen, 10)

        inv_pred_output = self.smpl_layer(
            pose_axis_angle=inv_pred_rot.reshape(batch_size * seqlen, 24, 9),
            betas=pred_beta.reshape(batch_size * seqlen, 10),
            global_orient=None,
            pose2rot=False,
            return_29_jts=True
        )

        inv_pred_rot = inv_pred_output.rot_mats

        assert torch.isnan(inv_pred_rot).sum() == 0, inv_pred_rot


        output = edict(
            rotmat=inv_pred_output['rot_mats'].reshape(batch_size, seqlen, 24, 3, 3),
            inv_pred2swingrot6d=inv_pred2swingrot6d.reshape(batch_size, seqlen, 24, 6),
            inv_pred2rot6d=inv_pred2rot6d.reshape(batch_size, seqlen, 24, 6),
            inv_pred2rotmat=inv_pred_rot.reshape(batch_size, seqlen, 24, 3, 3),
        )

        return output

def align_root(xyz_17, num_joints=19):
    shape = xyz_17.shape
    xyz_17 = xyz_17.reshape(-1, num_joints, 3)
    xyz_17 = xyz_17 - xyz_17[:, [0], :]
    xyz_17 = xyz_17.reshape(*shape)
    return xyz_17
