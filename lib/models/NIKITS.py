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
        # return nn.Sequential(nn.Linear(inp_dim, 1024), nn.LeakyReLU(), nn.Linear(1024, 1024), nn.LeakyReLU(), nn.Linear(1024, inp_dim), nn.Tanh())
        # return nn.Sequential(nn.Linear(inp_dim, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, inp_dim), nn.Tanh())
        # return nn.Sequential(nn.Linear(inp_dim, 512), nn.LeakyReLU(), nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, inp_dim), nn.Tanh())
        return nn.Sequential(nn.Linear(inp_dim, 512), nn.Dropout(), nn.LeakyReLU(), nn.Linear(512, 512), nn.Dropout(), nn.LeakyReLU(), nn.Linear(512, inp_dim), nn.Tanh())
    return nets


def get_nett(inp_dim):
    def nett():
        # return nn.Sequential(nn.Linear(inp_dim, 1024), nn.LeakyReLU(), nn.Linear(1024, 1024), nn.LeakyReLU(), nn.Linear(1024, inp_dim))
        # return nn.Sequential(nn.Linear(inp_dim, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, inp_dim))
        return nn.Sequential(nn.Linear(inp_dim, 512), nn.Dropout(), nn.LeakyReLU(), nn.Linear(512, 512), nn.Dropout(), nn.LeakyReLU(), nn.Linear(512, inp_dim))
    return nett


class NIKITS(nn.Module):
    def __init__(
        self,
        seq_len,
        num_joints,
        **cfg
    ):

        super(NIKITS, self).__init__()

        self.seqlen = seq_len

        self.regressor = FlowRegressor(num_joints)

    def forward(self, pose, img_feat, **kwargs):

        smpl_output = self.regressor(pose, img_feat, **kwargs)

        return smpl_output




class FlowRegressor(nn.Module):
    def __init__(self, num_joints,dtype=torch.float32):
        super(FlowRegressor, self).__init__()
        self.num_joints = num_joints

        # self.deconv_layers = self._make_deconv_layer()
        # self.final_layer = nn.Conv2d(
        #     self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)
        h36m_jregressor = np.load('/data0/tangt/models/PMCE/data/base_data/J_regressor_h36m.npy')
        self.smpl_layer = SMPL_layer(
            '/data0/tangt/models/PMCE/data/base_data/SMPL_NEUTRAL.pkl',
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

        # jts_tot_dim_1：24*6 + 32
        # shape_tot_dim = 10
        # num_stack = 16
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
        
        # print(pred_betas.shape)
        # pred_betas = pred_betas[:,None,:]

        # x0 = self.avg_pool(img_feat)
        # x0 = x0.view(x0.size(0), -1)
        xc = img_feat
        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)   
        xc = self.drop2(xc)
        input_phis = self.decphi(xc)
        # input_phis = input_phis[:,None,:]

        # input_joints = inp['pred_xyz_17']
        # pred_betas = inp['pred_beta']
        # input_sigma = inp['pred_sigma']
        # input_phis = inp['pred_phi']
        # input_joints = pose[:,None,:]
        input_joints = pose

        input_joints = input_joints.reshape(batch_size * seqlen, self.num_joints * 3)
        input_joints = align_root(input_joints, self.num_joints)
        # input_sigma = input_sigma.reshape(batch_size * seqlen, 17 * 1)
        # 置信度置为1

        # 检查GPU是否可用，若可用则设置默认设备为GPU，否则保持在CPU上
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_sigma = torch.ones(batch_size * seqlen, self.num_joints * 1).to(device)

        input_phis = input_phis.reshape(batch_size, seqlen, 23, 2)
        input_phis = input_phis / (torch.norm(input_phis, dim=3, keepdim=True) + 1e-8)
        input_phis = input_phis.reshape(batch_size * seqlen, 23 * 2)

        input_shape = pred_betas.reshape(batch_size * seqlen, 10)
        # input_shape = self.smpl_layer.betas2bones(pred_betas.reshape(batch_size * seqlen, 10))
        # print(input_shape.shape)

        # 随机噪音
        # zx = 1e-3 * torch.randn((batch_size * seqlen, self.zx_dim), device=input_joints.device, dtype=input_joints.dtype)
        zx = torch.zeros((batch_size * seqlen, self.zx_dim), device=input_joints.device, dtype=input_joints.dtype)

        # print("Tensor device:", input_joints.device)
        # print("Tensor device:", input_sigma.device)
        # print("Tensor device:", zx.device)

        input1 = torch.cat((
            input_joints, input_sigma,
            # input_betas,
            zx), dim=1)

        # 1. inverse pass: jts, beta -> swing: [-1, 24 * 6]
        # pose_out, _ = self.flow_j2s.inverse_p(input1)
        pose_out, _, _ = self.flow_j2s.inverse_p(input1, input_shape)
        pose_out = pose_out.reshape(batch_size * seqlen, self.jts_tot_dim_1)

        inv_pred2swingrot6d = pose_out[:, :24 * 6].contiguous().reshape(batch_size * seqlen, 24, 6)

        # inv_pred2beta = beta_out.reshape(batch_size * seqlen, 10)
        assert pose_out.shape[1] == self.zes_dim + 24 * 6
        inv_pred2zes = pose_out[:, -self.zes_dim:]
        
        # 2. inverse pass: swing, twist -> rot
        input2 = torch.cat((inv_pred2swingrot6d.reshape(-1, 24 * 6), input_phis), dim=1)
        
        pose_out, _ = self.flow_s2r.inverse_p(input2)

        inv_pred2rot6d = pose_out[:, :24 * 6].contiguous().reshape(batch_size * seqlen, 24, 6)
        assert pose_out.shape[1] == self.zet_dim + 24 * 6
        inv_pred2zet = pose_out[:, -self.zet_dim:]

        if target is not None:
            gt_jts17 = target['gt_xyz_17'].reshape(batch_size * seqlen, self.num_joints * 3)
            gt_jts17 = align_root(gt_jts17, self.num_joints)
            gt_betas = target['gt_betas'].reshape(batch_size * seqlen, 10)
            gt_phis = target['gt_phi'].reshape(batch_size * seqlen, 23 * 2)

            gt_rot_swing, gt_rot_twist = self.smpl_layer.twist_swing_decompose(
                pose_skeleton=gt_jts17.reshape(-1, self.num_joints, 3),
                betas=gt_betas.reshape(batch_size * seqlen, 10),
                phis=gt_phis.reshape(batch_size * seqlen, 23, 2),
            )
            gt_rot_swing_6d = matrix_to_rotation_6d(gt_rot_swing).reshape(batch_size * seqlen, 24 * 6)
            target['gt_rot_swing'] = gt_rot_swing
            target['gt_rot_swing_6d'] = gt_rot_swing_6d

            # ====== inverse pass: gt jts, beta -> gt swing ======
            gt_sigma17 = 1e-3 * torch.randn((batch_size * seqlen, 17 * 1), device=input_joints.device, dtype=input_joints.dtype)
            zx = 1e-3 * torch.randn((batch_size * seqlen, self.zx_dim), device=input_joints.device, dtype=input_joints.dtype)

            inp_gt1 = torch.cat((
                gt_jts17, gt_sigma17,
                # gt_betas,
                # gt_phis,
                zx), dim=1)

            gt_shape = gt_betas

            # 1. inverse pass, jts -> swing: [-1, 17 * 6]
            # inv_gt_out, _ = self.flow_j2s.inverse_p(inp_gt1)
            inv_gt_out, _, _ = self.flow_j2s.inverse_p(inp_gt1, gt_shape)
            inv_gt2swingrot6d = inv_gt_out[:, :24 * 6].contiguous().reshape(batch_size * seqlen, 24, 6)

            inv_gt2zes = inv_gt_out[:, -self.zes_dim:]

            # 2. inverse pass: swing, twist -> rot
            inp_gt2 = torch.cat((
                inv_gt2swingrot6d.reshape(-1, 24 * 6), gt_phis), dim=1)

            inv_gt_out, _ = self.flow_s2r.inverse_p(inp_gt2)
            inv_gt2rot6d = inv_gt_out[:, :24 * 6].contiguous().reshape(batch_size * seqlen, 24, 6)
            inv_gt2zet = inv_gt_out[:, -self.zet_dim:]

            inv_gt2rotmat = rotation_6d_to_matrix(inv_gt2rot6d).reshape(batch_size * seqlen, 24, 3, 3)

            inv_gt_output = self.smpl_layer(
                pose_axis_angle=inv_gt2rotmat.reshape(batch_size * seqlen, 24, 9),
                betas=gt_betas.reshape(batch_size * seqlen, 10),
                global_orient=None,
                pose2rot=False,
                return_17_jts=True
            )
            inv_gt_17joints = inv_gt_output['joints']

            # ====== forward pass: gt rotmat ======
            gt_rot_aa = target['gt_theta'].reshape(batch_size * seqlen, 24, 3)
            gt_rot_mat = axis_angle_to_matrix(gt_rot_aa)
            gt_rot_6d = matrix_to_rotation_6d(gt_rot_mat).reshape(batch_size * seqlen, 24 * 6)

            zero_zet = torch.zeros_like(inv_gt2zet)
            zero_zes = torch.zeros_like(inv_gt2zes)
            # [B, 17 * 6], [B,]
            # 1. forward pass: rot -> swing, twist
            latent = torch.cat((gt_rot_6d, zero_zet), dim=1)
            fwd_output, _ = self.flow_s2r.forward_p(latent)
            fwd_gt2swing6d = fwd_output[:, :24 * 6]
            fwd_gt2phis = fwd_output[:, -self.phi_tot_dim:]

            # 2. forward pass: swing -> jts, beta
            latent = torch.cat((fwd_gt2swing6d, zero_zes), dim=1)
            fwd_output, _, _ = self.flow_j2s.forward_p(latent, gt_shape)

            fwd_gt2jts = fwd_output[:, :17 * 3]
            # fwd_gt2betas = fwd_output[:, 17 * 4:17 * 4 + self.shape_tot_dim]
            fwd_gt2zx = fwd_output[:, -self.zx_dim:]

        assert torch.isnan(inv_pred2swingrot6d).sum() == 0, inv_pred2swingrot6d

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

        pred_shape = pred_beta.reshape(batch_size * seqlen, 10)

        inv_pred_vertices = inv_pred_output['vertices']

        inv_pred_29joints = inv_pred_output['joints']
        pred_xyz_17 = inv_pred_output['joints_from_verts']

        output = edict(
            rotmat=inv_pred_output['rot_mats'].reshape(batch_size, seqlen, 24, 3, 3),
            inv_pred2swingrot6d=inv_pred2swingrot6d.reshape(batch_size, seqlen, 24, 6),
            inv_pred2rot6d=inv_pred2rot6d.reshape(batch_size, seqlen, 24, 6),
            inv_pred2rotmat=inv_pred_rot.reshape(batch_size, seqlen, 24, 3, 3),
            # inv_pred2beta=inv_pred2beta,
            pred_shape=pred_shape,
            verts=inv_pred_vertices.reshape(batch_size, seqlen, -1, 3),
            inv_pred_29joints=inv_pred_29joints.reshape(batch_size, seqlen, 29, 3),
            pred_xyz_29=inv_pred_29joints.reshape(batch_size, seqlen, 29, 3),
            pred_xyz_29_struct=inv_pred_29joints.reshape(batch_size, seqlen, 29, 3),
            pred_xyz_17=pred_xyz_17.reshape(batch_size, seqlen, 17, 3),
            pred_beta=pred_beta,
            # pred_phi=inp['pred_phi'],
            # pred_cam=inp['pred_cam'],
            # pred_24joints=input_joints.reshape(batch_size, seqlen, 17, 3)[:, :, :24, :],
            inv_pred2zes=inv_pred2zes.reshape(batch_size, seqlen, self.zes_dim),
            inv_pred2zet=inv_pred2zet.reshape(batch_size, seqlen, self.zet_dim),
        )
        if target is not None:
            output['fwd_gt_2_17joints'] = fwd_gt2jts.reshape(batch_size, seqlen, self.num_joints, 3)
            output['fwd_gt_2_phis'] = fwd_gt2phis.reshape(batch_size, seqlen, 23, 2)
            # output['fwd_gt_2_betas'] = fwd_gt2betas.reshape(batch_size, seqlen, 10)
            output['fwd_gt_2_swing6d'] = fwd_gt2swing6d.reshape(batch_size, seqlen, 24, 6)
            output['fwd_gt_2_zx'] = fwd_gt2zx.reshape(batch_size, seqlen, self.zx_dim)
            # output['log_det_J'] = log_det_J.reshape(batch_size, seqlen, 1)
            output['inv_gt2swingrot6d'] = inv_gt2swingrot6d.reshape(batch_size, seqlen, 24, 6)
            output['inv_gt2rot6d'] = inv_gt2rot6d.reshape(batch_size, seqlen, 24, 6)
            output['inv_gt2rotmat'] = inv_gt2rotmat.reshape(batch_size, seqlen, 24, 3, 3)
            # output['inv_gt2phi'] = inv_gt2phi.reshape(batch_size, seqlen, 23, 2)
            # output['inv_gt2beta'] = inv_gt2beta.reshape(batch_size, seqlen, 10)
            output['inv_gt2zes'] = inv_gt2zes.reshape(batch_size, seqlen, self.zes_dim)
            output['inv_gt2zet'] = inv_gt2zet.reshape(batch_size, seqlen, self.zet_dim)
            output['inv_gt2joints'] = inv_gt_17joints.reshape(batch_size, seqlen, self.num_joints, 3)

        return output


def align_root(xyz_17, num_joints=19):
    shape = xyz_17.shape
    xyz_17 = xyz_17.reshape(-1, num_joints, 3)
    xyz_17 = xyz_17 - xyz_17[:, [0], :]
    xyz_17 = xyz_17.reshape(*shape)
    return xyz_17
