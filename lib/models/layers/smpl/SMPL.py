from easydict import EasyDict as edict

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix

from .lbs import lbs, hybrik, quat_to_rotmat, ts_decompose_rot, ts_compose, ts_decompose, betas2bones

try:
    import cPickle as pk
except ImportError:
    import pickle as pk


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class SMPL_layer(nn.Module):
    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10
    JOINT_NAMES = [
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
    ]
    LEAF_NAMES = [
        'head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe'
    ]
    root_idx_17 = 0
    root_idx_smpl = 0

    def __init__(self,
                 model_path,
                 h36m_jregressor,
                 gender='neutral',
                 dtype=torch.float32,
                 num_joints=29):

        super(SMPL_layer, self).__init__()

        self.ROOT_IDX = self.JOINT_NAMES.index('pelvis')
        self.LEAF_IDX = [self.JOINT_NAMES.index(
            name) for name in self.LEAF_NAMES]
        self.SPINE3_IDX = 9

        with open(model_path, 'rb') as smpl_file:
            self.smpl_data = Struct(**pk.load(smpl_file, encoding='latin1'))

        self.gender = gender

        self.dtype = dtype

        self.faces = self.smpl_data.f

        # Faces
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.smpl_data.f, dtype=np.int64), dtype=torch.long))

        # The vertices of the template model, (6890, 3)
        self.register_buffer('v_template',
                             to_tensor(to_np(self.smpl_data.v_template), dtype=dtype))

        # The shape components
        # Shape blend shapes basis, (6890, 3, 10)
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(self.smpl_data.shapedirs), dtype=dtype))

        # Pose blend shape basis: 6890 x 3 x 23*9, reshaped to 6890*3 x 23*9
        num_pose_basis = self.smpl_data.posedirs.shape[-1]
        # 23*9 x 6890*3
        posedirs = np.reshape(self.smpl_data.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # Vertices to Joints location (23 + 1, 6890)
        self.register_buffer(
            'J_regressor',
            to_tensor(to_np(self.smpl_data.J_regressor), dtype=dtype))
        # Vertices to Human3.6M Joints location (17, 6890)
        self.register_buffer(
            'J_regressor_h36m',
            to_tensor(to_np(h36m_jregressor), dtype=dtype))

        self.num_joints = num_joints

        # indices of parents for each joints
        parents = torch.zeros(len(self.JOINT_NAMES), dtype=torch.long)
        parents[:(self.NUM_JOINTS + 1)
                ] = to_tensor(to_np(self.smpl_data.kintree_table[0])).long()
        parents[0] = -1
        # extend kinematic tree
        parents[24] = 15
        parents[25] = 22
        parents[26] = 23
        parents[27] = 10
        parents[28] = 11
        if parents.shape[0] > self.num_joints:
            parents = parents[:24]

        self.register_buffer(
            'children_map',
            self._parents_to_children(parents))
        # (24,)
        self.register_buffer('parents', parents)

        # (6890, 23 + 1)
        self.register_buffer('lbs_weights',
                             to_tensor(to_np(self.smpl_data.weights), dtype=dtype))

    def _parents_to_children(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1

        children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')

        return children

    def forward(self,
                pose_axis_angle,
                betas,
                global_orient,
                transl=None,
                return_29_jts=False,
                pose2rot=True,
                return_verts=True):
        if global_orient is not None:
            full_pose = torch.cat([global_orient, pose_axis_angle], dim=1)
        else:
            full_pose = pose_axis_angle

        # vertices: (B, N, 3), joints: (B, K, 3)
        vertices, joints, rot_mats, joints_from_verts_h36m = lbs(betas, full_pose, self.v_template,
                                                                 self.shapedirs, self.posedirs,
                                                                 self.J_regressor, self.J_regressor_h36m, self.parents,
                                                                 self.lbs_weights, pose2rot=pose2rot, dtype=self.dtype)

        if return_29_jts:
            joints = get_jts_29(vertices, joints)

        if transl is not None:
            # apply translations
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts_h36m += transl.unsqueeze(dim=1)

        output = edict(
            vertices=vertices, joints=joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts_h36m)
        return output

    def hybrik(self,
               pose_skeleton,
               betas,
               phis,
               global_orient,
               transl=None,
               return_verts=True,
               return_29_jts=False,
               leaf_thetas=None):

        batch_size = pose_skeleton.shape[0]

        if leaf_thetas is not None:
            leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
            leaf_thetas = quat_to_rotmat(leaf_thetas)

        vertices, new_joints, rot_mats, joints_from_verts = hybrik(
            betas, global_orient, pose_skeleton, phis,
            self.v_template, self.shapedirs, self.posedirs,
            self.J_regressor, self.J_regressor_h36m, self.parents, self.children_map,
            self.lbs_weights, dtype=self.dtype, train=self.training,
            leaf_thetas=leaf_thetas)

        rot_mats = rot_mats.reshape(batch_size, 24, 3, 3)

        if return_29_jts:
            new_joints = get_jts_29(vertices, new_joints)

        if transl is not None:
            new_joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)

        output = edict(
            vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts)
        return output

    def twist_swing_decompose(
            self,
            pose_skeleton,
            betas,
            phis):

        batch_size = pose_skeleton.shape[0]

        rotmat_swing, rotmat_twist = ts_decompose(
            betas, pose_skeleton, phis,
            self.v_template, self.shapedirs, self.posedirs,
            self.J_regressor, self.parents, self.children_map,
            self.lbs_weights, dtype=self.dtype)

        return rotmat_swing.reshape(batch_size, 24, 3, 3), rotmat_twist.reshape(batch_size, 23, 3, 3)

    def twist_swing_decompose_rot(
            self,
            rotmats,
            betas):

        batch_size = rotmats.shape[0]

        rotmat_swing, rotmat_twist, angle_twist = ts_decompose_rot(
            betas, rotmats,
            self.v_template, self.shapedirs, self.posedirs,
            self.J_regressor, self.parents, self.children_map,
            self.lbs_weights, dtype=self.dtype)

        return rotmat_swing.reshape(batch_size, 24, 3, 3), rotmat_twist.reshape(batch_size, 23, 3, 3), angle_twist.reshape(batch_size, 23, 1)

    def twist_swing_composite(
            self,
            swing_rotmats,
            betas,
            phis,
            return_jts29=False):
        batch_size = swing_rotmats.shape[0]

        if swing_rotmats.shape[-1] == 6:
            swing_rotmats = rotation_6d_to_matrix(
                swing_rotmats.reshape(-1, 6)).reshape(-1, 24, 3, 3)

        vertices, new_joints, J_from_verts_h36m, rot_mats = ts_compose(
            betas, swing_rotmats, phis,
            self.v_template, self.shapedirs, self.posedirs,
            self.J_regressor, self.J_regressor_h36m, self.parents, self.children_map,
            self.lbs_weights, dtype=self.dtype)

        rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)

        if return_jts29:
            new_joints = get_jts_29(vertices, new_joints)

        output = edict(
            vertices=vertices,
            joints=new_joints,
            joints_from_verts=J_from_verts_h36m,
            rot_mats=rot_mats.reshape(batch_size, 24, 3, 3))
        return output

    def betas2bones(self, betas):

        template_J = betas2bones(
            betas,
            self.v_template, self.shapedirs,
            self.J_regressor
        )
        return template_J


def get_jts_29(vertices, jts):
    # head, left hand, right hand, left foot, right foot
    leaf_indices = [411, 2445, 5905, 3216, 6617]
    leaf_jts = vertices[:, leaf_indices]
    new_jts = torch.cat([jts, leaf_jts], dim=1)
    return new_jts
