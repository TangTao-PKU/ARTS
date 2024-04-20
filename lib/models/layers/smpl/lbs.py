

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import math

import torch
import torch.nn.functional as F


def rot_mat_to_euler(rot_mats):

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def find_dynamic_lmk_idx_and_bcoords(vertices, pose, dynamic_lmk_faces_idx,
                                     dynamic_lmk_b_coords,
                                     neck_kin_chain, dtype=torch.float32):

    batch_size = vertices.shape[0]

    aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                 neck_kin_chain)
    rot_mats = batch_rodrigues(
        aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

    rel_rot_mat = torch.eye(
        3, device=vertices.device, dtype=dtype).unsqueeze_(dim=0).repeat(
        batch_size, 1, 1)
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

    y_rot_angle = torch.round(
        torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                    max=39)).to(dtype=torch.long)
    neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
    mask = y_rot_angle.lt(-39).to(dtype=torch.long)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = (neg_mask * neg_vals +
                   (1 - neg_mask) * y_rot_angle)

    dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                           0, y_rot_angle)
    dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                          0, y_rot_angle)

    return dyn_lmk_faces_idx, dyn_lmk_b_coords


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):

    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        batch_size, -1, 3)

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def joints2bones(joints, parents):

    assert joints.shape[1] == parents.shape[0]
    bone_dirs = torch.zeros_like(joints)
    bone_lens = torch.zeros_like(joints[:, :, :1])

    for c_id in range(parents.shape[0]):
        p_id = parents[c_id]
        if p_id == -1:
            # Parent node
            bone_dirs[:, c_id] = joints[:, c_id]
        else:
            # Child node
            # (B, 3)
            diff = joints[:, c_id] - joints[:, p_id]
            length = torch.norm(diff, dim=1, keepdim=True) + 1e-8
            direct = diff / length

            bone_dirs[:, c_id] = direct
            bone_lens[:, c_id] = length

    return bone_dirs, bone_lens


def bones2joints(bone_dirs, bone_lens, parents):

    batch_size = bone_lens.shape[0]
    joints = torch.zeros_like(bone_dirs).expand(batch_size, 24, 3)

    for c_id in range(parents.shape[0]):
        p_id = parents[c_id]
        if p_id == -1:
            # Parent node
            joints[:, c_id] = bone_dirs[:, c_id]
        else:
            # Child node
            joints[:, c_id] = joints[:, p_id] + bone_dirs[:, c_id] * bone_lens[:, c_id]

    return joints


def betas2bones(betas, v_template, shapedirs, J_regressor):

    v_shaped = v_template + blend_shapes(betas, shapedirs)

    J = vertices2joints(J_regressor, v_shaped)
    return J


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, J_regressor_h36m, parents,
        lbs_weights, pose2rot=True, dtype=torch.float32):

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)


    J = vertices2joints(J_regressor, v_shaped)

    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        if pose.numel() == batch_size * 24 * 4:
            rot_mats = quat_to_rotmat(pose.reshape(batch_size * 24, 4)).reshape(batch_size, 24, 3, 3)
        elif pose.numel() == batch_size * 24 * 3:
            rot_mats = batch_rodrigues(
                pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])
        else:
            rot_mats = pose.reshape(batch_size, 24, 3, 3)

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs) \
            .view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents[:24], dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    J_from_verts = vertices2joints(J_regressor_h36m, verts)

    return verts, J_transformed, rot_mats, J_from_verts


def hybrik(betas, global_orient, pose_skeleton, phis,
           v_template, shapedirs, posedirs, J_regressor, J_regressor_h36m, parents, children,
           lbs_weights, dtype=torch.float32, train=False, leaf_thetas=None):
    
    batch_size = max(betas.shape[0], pose_skeleton.shape[0])
    device = betas.device

    # 1. Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # 2. Get the rest joints
    # NxJx3 array
    if leaf_thetas is not None:
        rest_J = vertices2joints(J_regressor, v_shaped)
    else:
        rest_J = torch.zeros((v_shaped.shape[0], 29, 3), dtype=dtype, device=device)
        rest_J[:, :24] = vertices2joints(J_regressor, v_shaped)

        leaf_number = [411, 2445, 5905, 3216, 6617]
        leaf_vertices = v_shaped[:, leaf_number].clone()
        rest_J[:, 24:] = leaf_vertices

    # 3. Get the rotation matrics
    if train:
        rot_mats, rotate_rest_pose = batch_inverse_kinematics_transform_naive(
            pose_skeleton, global_orient, phis,
            rest_J.clone(), children, parents, dtype=dtype, train=train,
            leaf_thetas=leaf_thetas)
    else:
        rot_mats, rotate_rest_pose = batch_inverse_kinematics_transform(
            pose_skeleton, global_orient, phis,
            rest_J.clone(), children, parents, dtype=dtype, train=train,
            leaf_thetas=leaf_thetas)

    test_joints = True
    if test_joints:
        J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), parents[:24], dtype=dtype)
    else:
        J_transformed = None

    # assert torch.mean(torch.abs(rotate_rest_pose - J_transformed)) < 1e-5
    # 4. Add pose blend shapes
    # rot_mats: N x (J + 1) x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
    pose_offsets = torch.matmul(pose_feature, posedirs) \
        .view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]
    J_from_verts_h36m = vertices2joints(J_regressor_h36m, verts)

    return verts, J_transformed, rot_mats, J_from_verts_h36m


def ts_decompose(
        betas, pose_skeleton, phis,
        v_template, shapedirs, posedirs, J_regressor, parents, children,
        lbs_weights, dtype=torch.float32):
    device = betas.device

    # 1. Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # 2. Get the rest joints
    # NxJx3 array
    rest_J = vertices2joints(J_regressor, v_shaped)

    rest_J = torch.zeros((v_shaped.shape[0], 29, 3), dtype=dtype, device=device)
    rest_J[:, :24] = vertices2joints(J_regressor, v_shaped)

    leaf_number = [411, 2445, 5905, 3216, 6617]
    leaf_vertices = v_shaped[:, leaf_number].clone()
    rest_J[:, 24:] = leaf_vertices

    # 1. Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    rotmat_swing, rotmat_twist = batch_inverse_kinematics_decompose(
        pose_skeleton.clone(), phis.clone(),
        rest_J.clone(), children, parents, dtype=dtype)

    return rotmat_swing, rotmat_twist


def ts_decompose_rot(
        betas, rotmats,
        v_template, shapedirs, posedirs, J_regressor, parents, children,
        lbs_weights, dtype=torch.float32):
    device = betas.device

    # 1. Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # 2. Get the rest joints
    # NxJx3 array
    rest_J = vertices2joints(J_regressor, v_shaped)

    rest_J = torch.zeros((v_shaped.shape[0], 29, 3), dtype=dtype, device=device)
    rest_J[:, :24] = vertices2joints(J_regressor, v_shaped)

    leaf_number = [411, 2445, 5905, 3216, 6617]
    leaf_vertices = v_shaped[:, leaf_number].clone()
    rest_J[:, 24:] = leaf_vertices

    # 1. Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    rotmat_swing, rotmat_twist, angle_twist = batch_inverse_kinematics_decompose_rot(
        rotmats.clone(), rest_J.clone(),
        children, parents, dtype=dtype)

    return rotmat_swing, rotmat_twist, angle_twist


def ts_compose(betas, swing_rotmats, phis,
               v_template, shapedirs, posedirs, J_regressor, J_regressor_h36m, parents, children,
               lbs_weights, dtype=torch.float32):
    batch_size = max(betas.shape[0], swing_rotmats.shape[0])
    device = betas.device

    # 1. Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # 2. Get the rest joints
    # NxJx3 array
    rest_J = torch.zeros((v_shaped.shape[0], 29, 3), dtype=dtype, device=device)
    rest_J[:, :24] = vertices2joints(J_regressor, v_shaped)

    leaf_number = [411, 2445, 5905, 3216, 6617]
    leaf_vertices = v_shaped[:, leaf_number].clone()
    rest_J[:, 24:] = leaf_vertices

    # 3. Get the rotation matrics
    rot_mats = batch_rot_composite(
        swing_rotmats.clone(), phis.clone(),
        rest_J.clone(), children, parents, dtype=dtype)

    test_joints = True
    if test_joints:
        J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), parents[:24], dtype=dtype)
    else:
        J_transformed = None

    # assert torch.mean(torch.abs(rotate_rest_pose - J_transformed)) < 1e-5
    # 4. Add pose blend shapes
    # rot_mats: N x (J + 1) x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
    pose_offsets = torch.matmul(pose_feature, posedirs) \
        .view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped.detach()

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]
    J_from_verts_h36m = vertices2joints(J_regressor_h36m, verts)

    return verts, J_transformed, J_from_verts_h36m, rot_mats


def vertices2joints(J_regressor, vertices):

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas, shape_disps):

    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):

    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):

    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]].clone()

    # (B, K + 1, 4, 4)
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        # (B, 4, 4) x (B, 4, 4)
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    # (B, K + 1, 4, 4)
    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def batch_inverse_kinematics_transform(
        pose_skeleton, global_orient,
        phis,
        rest_pose,
        children, parents, dtype=torch.float32, train=False,
        leaf_thetas=None):
    batch_size = pose_skeleton.shape[0]
    device = pose_skeleton.device

    rel_rest_pose = rest_pose.clone()
    rel_rest_pose[:, 1:] -= rest_pose[:, parents[1:]].clone()
    rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

    # rotate the T pose
    rotate_rest_pose = torch.zeros_like(rel_rest_pose)
    # set up the root
    rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

    rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
    rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, parents[1:]].clone()
    rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

    # the predicted final pose
    final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
    final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, 0:1] + rel_rest_pose[:, 0:1]

    rel_rest_pose = rel_rest_pose
    rel_pose_skeleton = rel_pose_skeleton
    final_pose_skeleton = final_pose_skeleton
    rotate_rest_pose = rotate_rest_pose

    assert phis.dim() == 3
    phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

    # TODO
    if train:
        global_orient_mat = batch_get_pelvis_orient(
            rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)
    else:
        global_orient_mat = batch_get_pelvis_orient_svd(
            rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)

    rot_mat_chain = [global_orient_mat]
    rot_mat_local = [global_orient_mat]
    # leaf nodes rot_mats
    if leaf_thetas is not None:
        leaf_cnt = 0
        leaf_rot_mats = leaf_thetas.view([batch_size, 5, 3, 3])

    for i in range(1, parents.shape[0]):
        if children[i] == -1:
            # leaf nodes
            if leaf_thetas is not None:
                rot_mat = leaf_rot_mats[:, leaf_cnt, :, :]
                leaf_cnt += 1

                rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
                    rot_mat_chain[parents[i]],
                    rel_rest_pose[:, i]
                )

                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents[i]],
                    rot_mat))
                rot_mat_local.append(rot_mat)
        elif children[i] == -3:
            # three children
            rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
                rot_mat_chain[parents[i]],
                rel_rest_pose[:, i]
            )

            spine_child = []
            for c in range(1, parents.shape[0]):
                if parents[c] == i and c not in spine_child:
                    spine_child.append(c)

            # original
            spine_child = []
            for c in range(1, parents.shape[0]):
                if parents[c] == i and c not in spine_child:
                    spine_child.append(c)

            children_final_loc = []
            children_rest_loc = []
            for c in spine_child:
                temp = final_pose_skeleton[:, c] - rotate_rest_pose[:, i]
                children_final_loc.append(temp)

                children_rest_loc.append(rel_rest_pose[:, c].clone())

            rot_mat = batch_get_3children_orient_svd(
                children_final_loc, children_rest_loc,
                rot_mat_chain[parents[i]], spine_child, dtype)

            rot_mat_chain.append(
                torch.matmul(
                    rot_mat_chain[parents[i]],
                    rot_mat)
            )
            rot_mat_local.append(rot_mat)
        else:
            # (B, 3, 1)
            rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[i]] + torch.matmul(
                rot_mat_chain[parents[i]],
                rel_rest_pose[:, i]
            )
            # (B, 3, 1)
            child_final_loc = final_pose_skeleton[:, children[i]] - rotate_rest_pose[:, i]

            if not train:
                orig_vec = rel_pose_skeleton[:, children[i]]
                template_vec = rel_rest_pose[:, children[i]]
                norm_t = torch.norm(template_vec, dim=1, keepdim=True)
                orig_vec = orig_vec * norm_t / torch.norm(orig_vec, dim=1, keepdim=True)

                diff = torch.norm(child_final_loc - orig_vec, dim=1, keepdim=True)
                big_diff_idx = torch.where(diff > 15 / 1000)[0]

                child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]

            child_final_loc = torch.matmul(
                rot_mat_chain[parents[i]].transpose(1, 2),
                child_final_loc)

            child_rest_loc = rel_rest_pose[:, children[i]]
            # (B, 1, 1)
            child_final_norm = torch.norm(child_final_loc, dim=1, keepdim=True)
            child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)

            child_final_norm = torch.norm(child_final_loc, dim=1, keepdim=True)

            # (B, 3, 1)
            axis = torch.cross(child_rest_loc, child_final_loc, dim=1)
            axis_norm = torch.norm(axis, dim=1, keepdim=True)

            # (B, 1, 1)
            cos = torch.sum(child_rest_loc * child_final_loc, dim=1, keepdim=True) / (child_rest_norm * child_final_norm + 1e-8)
            sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)

            # (B, 3, 1)
            axis = axis / (axis_norm + 1e-8)

            # Convert location revolve to rot_mat by rodrigues
            # (B, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

            # Convert spin to rot_mat
            # (B, 3, 1)
            spin_axis = child_rest_loc / child_rest_norm
            # (B, 1, 1)
            rx, ry, rz = torch.split(spin_axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
            # (B, 1, 1)
            cos, sin = torch.split(phis[:, i - 1], 1, dim=1)
            cos = torch.unsqueeze(cos, dim=2)
            sin = torch.unsqueeze(sin, dim=2)
            rot_mat_spin = ident + sin * K + (1 - cos) * torch.bmm(K, K)
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

            rot_mat_chain.append(torch.matmul(
                rot_mat_chain[parents[i]],
                rot_mat))
            rot_mat_local.append(rot_mat)

    # (B, K + 1, 3, 3)
    rot_mats = torch.stack(rot_mat_local, dim=1)

    return rot_mats, rotate_rest_pose.squeeze(-1)


def batch_inverse_kinematics_transform_naive(
        pose_skeleton, global_orient,
        phis,
        rest_pose,
        children, parents, dtype=torch.float32, train=False,
        leaf_thetas=None, need_detach=True):
    batch_size = pose_skeleton.shape[0]
    device = pose_skeleton.device

    rel_rest_pose = rest_pose.clone()
    rel_rest_pose[:, 1:] -= rest_pose[:, parents[1:]].clone()
    rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)
    unit_rel_rest_pose = rel_rest_pose / torch.norm(rel_rest_pose, dim=2, keepdim=True)

    # rotate the T pose
    rotate_rest_pose = torch.zeros_like(rel_rest_pose)
    # set up the root
    rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

    if need_detach:
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
    else:
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
    rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, parents[1:]].clone()
    rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

    # the predicted final pose
    final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
    final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, 0:1] + rel_rest_pose[:, 0:1]

    # assert phis.dim() == 3
    phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

    # TODO
    if global_orient is None:
        global_orient_mat = batch_get_pelvis_orient(
            rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)
    else:
        global_orient_mat = global_orient


    rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=pose_skeleton.device)
    rot_mat_local = torch.zeros_like(rot_mat_chain)
    rot_mat_chain[:, 0] = global_orient_mat
    rot_mat_local[:, 0] = global_orient_mat

    assert leaf_thetas is None

    idx_levs = [
        [0],
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17],
        [18, 19],
        [20, 21],
        [22, 23],
        [24, 25, 26, 27, 28]
    ]
    if leaf_thetas is not None:
        idx_levs = idx_levs[:-1]

    all_child_rest_loc = rel_rest_pose[:, children[1:24]]
    all_child_rest_norm = torch.norm(all_child_rest_loc, dim=2, keepdim=True)

    # Convert twist to rot_mat
    # (B, K, 3, 1)
    twist_axis = all_child_rest_loc / (all_child_rest_norm + 1e-8)
    # (B, K, 1, 1)
    rx, ry, rz = torch.split(twist_axis, 1, dim=2)
    zeros = torch.zeros((batch_size, 23, 1, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
        .view((batch_size, 23, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).reshape(1, 1, 3, 3)
    # (B, K, 1, 1)
    cos, sin = torch.split(phis, 1, dim=2)
    cos = torch.unsqueeze(cos, dim=3)
    sin = torch.unsqueeze(sin, dim=3)
    all_rot_mat_twist = ident + sin * K + (1 - cos) * torch.matmul(K, K)
    for idx_lev in range(1, len(idx_levs)):

        indices = idx_levs[idx_lev]
        if idx_lev == len(idx_levs) - 1:
            # leaf nodes
            assert NotImplementedError
        else:
            len_indices = len(indices)
            # (B, K, 3, 1)
            child_final_loc = torch.matmul(
                rot_mat_chain[:, parents[indices]].transpose(2, 3),
                rel_pose_skeleton[:, children[indices]])  # rotate back

            unit_child_rest = unit_rel_rest_pose[:, children[indices]]
            # child_rest_loc = rel_rest_pose[:, children[indices]]
            # (B, K, 1, 1)
            child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)

            unit_child_final = child_final_loc / (child_final_norm + 1e-8)
            # unit_child_rest = child_rest_loc / (child_rest_norm + 1e-8)

            axis = torch.cross(unit_child_rest, unit_child_final, dim=2)
            cos = torch.sum(unit_child_rest * unit_child_final, dim=2, keepdim=True)
            sin = torch.norm(axis, dim=2, keepdim=True)

            # (B, K, 3, 1)
            axis = axis / (sin + 1e-8)

            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=dtype, device=device)

            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).reshape(1, 1, 3, 3)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            rot_mat_twist = all_rot_mat_twist[:, [i - 1 for i in indices]]

            rot_mat = torch.matmul(rot_mat_loc, rot_mat_twist)

            rot_mat_chain[:, indices] = torch.matmul(
                rot_mat_chain[:, parents[indices]],
                rot_mat
            )
            rot_mat_local[:, indices] = rot_mat
    # (B, K + 1, 3, 3)
    # rot_mats = torch.stack(rot_mat_local, dim=1)
    rot_mats = rot_mat_local

    return rot_mats, rotate_rest_pose.squeeze(-1)


def batch_inverse_kinematics_decompose(
        pose_skeleton,
        phis,
        rest_pose,
        children, parents, dtype=torch.float32,
        leaf_thetas=None, need_detach=True):

    batch_size = pose_skeleton.shape[0]
    device = pose_skeleton.device

    rel_rest_pose = rest_pose.clone()
    rel_rest_pose[:, 1:] -= rest_pose[:, parents[1:]].clone()
    rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)
    unit_rel_rest_pose = rel_rest_pose / torch.norm(rel_rest_pose, dim=2, keepdim=True)

    # rotate the T pose
    rotate_rest_pose = torch.zeros_like(rel_rest_pose)
    # set up the root
    rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

    if need_detach:
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
    else:
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
    rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, parents[1:]].clone()
    rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

    # the predicted final pose
    final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
    final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, 0:1] + rel_rest_pose[:, 0:1]

    # assert phis.dim() == 3
    phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

    global_orient_mat = batch_get_pelvis_orient(
        rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents, children, dtype)


    rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=pose_skeleton.device)
    rot_mat_local = torch.zeros_like(rot_mat_chain)
    rot_mat_chain[:, 0] = global_orient_mat
    rot_mat_local[:, 0] = global_orient_mat

    rotmat_twist_list = torch.zeros((batch_size, 23, 3, 3), dtype=torch.float32, device=pose_skeleton.device)
    rotmat_swing_list = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=pose_skeleton.device)
    rotmat_swing_list[:, 0] = global_orient_mat

    idx_levs = [
        [0],
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17],
        [18, 19],
        [20, 21],
        [22, 23],
        [24, 25, 26, 27, 28]
    ]
    if leaf_thetas is not None:
        idx_levs = idx_levs[:-1]

    all_child_rest_loc = rel_rest_pose[:, children[1:24]]
    all_child_rest_norm = torch.norm(all_child_rest_loc, dim=2, keepdim=True)

    # Convert twist to rot_mat
    # (B, K, 3, 1)
    twist_axis = all_child_rest_loc / (all_child_rest_norm + 1e-8)
    # (B, K, 1, 1)
    rx, ry, rz = torch.split(twist_axis, 1, dim=2)
    zeros = torch.zeros((batch_size, 23, 1, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
        .view((batch_size, 23, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).reshape(1, 1, 3, 3)
    # (B, K, 1, 1)
    cos, sin = torch.split(phis, 1, dim=2)
    cos = torch.unsqueeze(cos, dim=3)
    sin = torch.unsqueeze(sin, dim=3)
    all_rot_mat_twist = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    for idx_lev in range(1, len(idx_levs)):

        indices = idx_levs[idx_lev]
        if idx_lev == len(idx_levs) - 1:
            # leaf nodes
            assert NotImplementedError
        else:
            len_indices = len(indices)
            # (B, K, 3, 1)
            child_final_loc = torch.matmul(
                rot_mat_chain[:, parents[indices]].transpose(2, 3),
                rel_pose_skeleton[:, children[indices]])  # rotate back

            unit_child_rest = unit_rel_rest_pose[:, children[indices]]
            # child_rest_loc = rel_rest_pose[:, children[indices]]
            # (B, K, 1, 1)
            child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)
            # child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)

            # # (B, K, 3, 1)
            # axis = axis / (axis_norm + 1e-8)
            unit_child_final = child_final_loc / (child_final_norm + 1e-8)
            # unit_child_rest = child_rest_loc / (child_rest_norm + 1e-8)

            axis = torch.cross(unit_child_rest, unit_child_final, dim=2)
            cos = torch.sum(unit_child_rest * unit_child_final, dim=2, keepdim=True)
            sin = torch.norm(axis, dim=2, keepdim=True)

            # (B, K, 3, 1)
            axis = axis / (sin + 1e-8)

            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=dtype, device=device)

            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).reshape(1, 1, 3, 3)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            rot_mat_twist = all_rot_mat_twist[:, [i - 1 for i in indices]]

            rot_mat = torch.matmul(rot_mat_loc, rot_mat_twist)

            rot_mat_chain[:, indices] = torch.matmul(
                rot_mat_chain[:, parents[indices]],
                rot_mat
            )
            rot_mat_local[:, indices] = rot_mat
            rotmat_swing_list[:, indices] = rot_mat_loc
            rotmat_twist_list[:, [i - 1 for i in indices]] = rot_mat_twist

    return rotmat_swing_list, rotmat_twist_list

def batch_rot_composite(swing_rotmats, phis, rest_pose, children, parents, dtype=torch.float32):

    batch_size = swing_rotmats.shape[0]
    device = swing_rotmats.device

    rel_rest_pose = rest_pose.clone()
    rel_rest_pose[:, 1:] -= rest_pose[:, parents[1:]].clone()
    rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

    assert phis.dim() == 3
    phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

    # torch.cuda.synchronize()
    # start_t = time.time()

    global_orient_mat = swing_rotmats[:, 0]

    rot_mat_chain = torch.zeros(
        (batch_size, 24, 3, 3), dtype=swing_rotmats.dtype, device=swing_rotmats.device)
    rot_mat_local = torch.zeros_like(rot_mat_chain)
    rot_mat_chain[:, 0] = global_orient_mat
    rot_mat_local[:, 0] = global_orient_mat

    idx_levs = [
        [0],
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17],
        [18, 19],
        [20, 21],
        [22, 23],
        [24, 25, 26, 27, 28]
    ]

    for idx_lev in range(1, len(idx_levs)):

        indices = idx_levs[idx_lev]
        if idx_lev == len(idx_levs) - 1:
            # leaf nodes
            assert NotImplementedError
        else:
            len_indices = len(indices)

            child_rest_loc = rel_rest_pose[:, children[indices]]  # need rotation back ?
            # (B, K, 1, 1)
            child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)

            rot_mat_swing = swing_rotmats[:, indices]
            # Convert spin to rot_mat
            # (B, K, 3, 1)
            spin_axis = child_rest_loc / (child_rest_norm + 1e-8)
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(spin_axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).reshape(1, 1, 3, 3)
            # (B, K, 1, 1)
            phi_indices = [item - 1 for item in indices]

            cos, sin = torch.split(phis[:, phi_indices], 1, dim=2)
            cos = torch.unsqueeze(cos, dim=3)
            sin = torch.unsqueeze(sin, dim=3)
            rot_mat_twist = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            # rot_mat_twist = phis[:, phi_indices]
            rot_mat = torch.matmul(rot_mat_swing, rot_mat_twist)

            rot_mat_chain[:, indices] = torch.matmul(
                rot_mat_chain[:, parents[indices]],
                rot_mat
            )
            rot_mat_local[:, indices] = rot_mat

            if not (torch.det(rot_mat) > 0).all():
                idx = ~(torch.det(rot_mat) > 0)  # all 0 except the error batch
                idx = (idx.reshape(batch_size, -1).sum(dim=1) > 0)
                print(
                    'composite',
                    -1, idx_lev, rot_mat_swing[idx], rot_mat_twist[idx],
                    (torch.det(rot_mat_swing) > 0).all(),
                    (torch.det(rot_mat_twist) > 0).all(),
                    sin[idx], cos[idx], K[idx]
                )

                raise NotImplementedError

    # (B, K + 1, 3, 3)
    rot_mats = rot_mat_local

    return rot_mats


def calc_swing(
        parents: torch.Tensor, children: torch.Tensor,
        rot_mat_chain: torch.Tensor, rel_pose_skeleton: torch.Tensor, unit_rel_rest_pose: torch.Tensor,
        all_rot_mat_twist: torch.Tensor, rot_mat_local: torch.Tensor) -> torch.Tensor:

    idx_levs = [
        [0],
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17],
        [18, 19],
        [20, 21],
        [22, 23],
    ]

    for idx_lev in range(1, len(idx_levs)):

        indices = idx_levs[idx_lev]

        len_indices = len(indices)
        # (B, K, 3, 1)
        child_final_loc = torch.matmul(
            rot_mat_chain[:, parents[indices]].transpose(2, 3),
            rel_pose_skeleton[:, children[indices]])  # rotate back

        unit_child_rest = unit_rel_rest_pose[:, children[indices]]
        # child_rest_loc = rel_rest_pose[:, children[indices]]
        # (B, K, 1, 1)
        child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)
        # child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)

        unit_child_final = child_final_loc / (child_final_norm + 1e-8)
        # unit_child_rest = child_rest_loc / (child_rest_norm + 1e-8)

        axis = torch.cross(unit_child_rest, unit_child_final, dim=2)
        cos = torch.sum(unit_child_rest * unit_child_final, dim=2, keepdim=True)
        sin = torch.norm(axis, dim=2, keepdim=True)

        # (B, K, 3, 1)
        axis = axis / (sin + 1e-8)

        # Convert location revolve to rot_mat by rodrigues
        # (B, K, 1, 1)
        rx, ry, rz = torch.split(axis, 1, dim=2)
        # zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=dtype, device=device)
        zeros = torch.zeros_like(rx)

        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2).view((-1, len_indices, 3, 3))
        ident = torch.eye(3, dtype=rx.dtype, device=rx.device).reshape(1, 1, 3, 3)
        rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
        rot_mat_twist = all_rot_mat_twist[:, [i - 1 for i in indices]]

        rot_mat = torch.matmul(rot_mat_loc, rot_mat_twist)

        rot_mat_chain[:, indices] = torch.matmul(
            rot_mat_chain[:, parents[indices]],
            rot_mat
        )
        rot_mat_local[:, indices] = rot_mat

    return rot_mat_local


# scripted_calc_swing = torch.jit.script(calc_swing)


def batch_get_pelvis_orient_svd(rel_pose_skeleton, rel_rest_pose, parents, children, dtype):
    pelvis_child = [int(children[0])]
    for i in range(1, parents.shape[0]):
        if parents[i] == 0 and i not in pelvis_child:
            pelvis_child.append(i)


    rest_mat = rel_rest_pose[:, pelvis_child].squeeze(3)
    target_mat = rel_pose_skeleton[:, pelvis_child].squeeze(3)
    S = rest_mat.transpose(1, 2).bmm(target_mat)

    mask_zero = S.sum(dim=(1, 2))

    S_non_zero = S[mask_zero != 0].reshape(-1, 3, 3)

    # U, _, V = torch.svd(S_non_zero)
    device = S_non_zero.device
    U, _, V = torch.svd(S_non_zero.cpu())
    U = U.to(device=device)
    V = V.to(device=device)

    rot_mat = torch.zeros_like(S)
    rot_mat[mask_zero == 0] = torch.eye(3, device=S.device)

    # rot_mat_non_zero = torch.bmm(V, U.transpose(1, 2))
    det_u_v = torch.det(torch.bmm(V, U.transpose(1, 2)))
    det_modify_mat = torch.eye(3, device=U.device).unsqueeze(0).expand(U.shape[0], -1, -1).clone()
    det_modify_mat[:, 2, 2] = det_u_v
    rot_mat_non_zero = torch.bmm(torch.bmm(V, det_modify_mat), U.transpose(1, 2))

    rot_mat[mask_zero != 0] = rot_mat_non_zero

    assert torch.sum(torch.isnan(rot_mat)) == 0, ('rot_mat', rot_mat)

    return rot_mat


def batch_get_pelvis_orient(rel_pose_skeleton, rel_rest_pose, parents, children, dtype):
    batch_size = rel_pose_skeleton.shape[0]
    device = rel_pose_skeleton.device

    assert children[0] == 3
    pelvis_child = [int(children[0])]
    for i in range(1, parents.shape[0]):
        if parents[i] == 0 and i not in pelvis_child:
            pelvis_child.append(i)

    spine_final_loc = rel_pose_skeleton[:, int(children[0])].clone()
    spine_rest_loc = rel_rest_pose[:, int(children[0])].clone()
    spine_norm = torch.norm(spine_final_loc, dim=1, keepdim=True)
    spine_norm = spine_final_loc / (spine_norm + 1e-8)

    rot_mat_spine = vectors2rotmat(spine_rest_loc, spine_final_loc, dtype)

    assert torch.sum(torch.isnan(rot_mat_spine)
                     ) == 0, ('rot_mat_spine', rot_mat_spine)
    center_final_loc = 0
    center_rest_loc = 0
    for child in pelvis_child:
        if child == int(children[0]):
            continue
        center_final_loc = center_final_loc + rel_pose_skeleton[:, child].clone()
        center_rest_loc = center_rest_loc + rel_rest_pose[:, child].clone()
    center_final_loc = center_final_loc / (len(pelvis_child) - 1)
    center_rest_loc = center_rest_loc / (len(pelvis_child) - 1)

    center_rest_loc = torch.matmul(rot_mat_spine, center_rest_loc)

    center_final_loc = center_final_loc - torch.sum(center_final_loc * spine_norm, dim=1, keepdim=True) * spine_norm
    center_rest_loc = center_rest_loc - torch.sum(center_rest_loc * spine_norm, dim=1, keepdim=True) * spine_norm

    center_final_loc_norm = torch.norm(center_final_loc, dim=1, keepdim=True)
    center_rest_loc_norm = torch.norm(center_rest_loc, dim=1, keepdim=True)

    # (B, 3, 1)
    axis = torch.cross(center_rest_loc, center_final_loc, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(center_rest_loc * center_final_loc, dim=1, keepdim=True) / (center_rest_loc_norm * center_final_loc_norm + 1e-8)
    sin = axis_norm / (center_rest_loc_norm * center_final_loc_norm + 1e-8)

    assert torch.sum(torch.isnan(cos)
                     ) == 0, ('cos', cos)
    assert torch.sum(torch.isnan(sin)
                     ) == 0, ('sin', sin)
    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_center = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    rot_mat = torch.matmul(rot_mat_center, rot_mat_spine)

    return rot_mat


def batch_get_3children_orient_svd(rel_pose_skeleton, rel_rest_pose, rot_mat_chain_parent, children_list, dtype):
    rest_mat = []
    target_mat = []
    for c, child in enumerate(children_list):
        if isinstance(rel_pose_skeleton, list):
            target = rel_pose_skeleton[c].clone()
            template = rel_rest_pose[c].clone()
        else:
            target = rel_pose_skeleton[:, child].clone()
            template = rel_rest_pose[:, child].clone()

        target = torch.matmul(
            rot_mat_chain_parent.transpose(1, 2),
            target)

        target_mat.append(target)
        rest_mat.append(template)

    rest_mat = torch.cat(rest_mat, dim=2)
    target_mat = torch.cat(target_mat, dim=2)
    S = rest_mat.bmm(target_mat.transpose(1, 2))

    # U, _, V = torch.svd(S)
    device = S.device
    U, _, V = torch.svd(S.cpu())
    U = U.to(device=device)
    V = V.to(device=device)

    # rot_mat = torch.bmm(V, U.transpose(1, 2))
    det_u_v = torch.det(torch.bmm(V, U.transpose(1, 2)))
    det_modify_mat = torch.eye(3, device=U.device).unsqueeze(0).expand(U.shape[0], -1, -1).clone()
    det_modify_mat[:, 2, 2] = det_u_v
    rot_mat = torch.bmm(torch.bmm(V, det_modify_mat), U.transpose(1, 2))

    assert torch.sum(torch.isnan(rot_mat)) == 0, ('3children rot_mat', rot_mat)
    return rot_mat


def vectors2rotmat(vec_rest, vec_final, dtype):
    batch_size = vec_final.shape[0]
    device = vec_final.device

    # (B, 1, 1)
    vec_final_norm = torch.norm(vec_final, dim=1, keepdim=True)
    vec_rest_norm = torch.norm(vec_rest, dim=1, keepdim=True)

    # (B, 3, 1)
    axis = torch.cross(vec_rest, vec_final, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(vec_rest * vec_final, dim=1, keepdim=True) / (vec_rest_norm * vec_final_norm + 1e-8)
    sin = axis_norm / (vec_rest_norm * vec_final_norm + 1e-8)

    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    return rot_mat_loc


def rotmat_to_quat(rotation_matrix):
    assert rotation_matrix.shape[1:] == (3, 3)
    rot_mat = rotation_matrix.reshape(-1, 3, 3)
    hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                       device=rotation_matrix.device)
    hom = hom.reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
    rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def quat_to_rotmat(quat):

    norm_quat = quat
    norm_quat = norm_quat / (norm_quat.norm(p=2, dim=1, keepdim=True) + 1e-8)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def rotmat_to_aa(rotmat):
    if rotmat.shape[1:] == (3, 3):
        rot_mat = rotmat.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                           device=rotmat.device)
        hom = hom.reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotmat = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotmat)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:

    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis
