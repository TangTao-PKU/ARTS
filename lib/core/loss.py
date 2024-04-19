import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from geometry import batch_rodrigues

class CoordLoss(nn.Module):
    def __init__(self, has_valid=False):
        super(CoordLoss, self).__init__()

        self.has_valid = has_valid
        self.criterion = nn.L1Loss(reduction='mean')

    def forward(self, pred, target, target_valid):
        if self.has_valid:
            pred, target = pred * target_valid, target * target_valid

        loss = self.criterion(pred, target)

        return loss


class LaplacianLoss(nn.Module):
    def __init__(self, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = 6890  # SMPL
        self.nf = faces.shape[0]
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            laplacian[i, :] /= (laplacian[i, i] + 1e-8)

        self.register_buffer('laplacian', torch.from_numpy(laplacian).cuda().float())

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.cat([torch.matmul(self.laplacian, x[i])[None, :, :] for i in range(batch_size)], 0)

        x = x.pow(2).sum(2)
        if self.average:
            return x.sum() / batch_size
        else:
            return x.mean()


class NormalVectorLoss(nn.Module):
    def __init__(self, face):
        super(NormalVectorLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt):
        face = torch.LongTensor(self.face).cuda()

        v1_out = coord_out[:, face[:, 1], :] - coord_out[:, face[:, 0], :]
        v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
        v2_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 0], :]
        v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
        v3_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 1], :]
        v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 0], :]
        v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
        v2_gt = coord_gt[:, face[:, 2], :] - coord_gt[:, face[:, 0], :]
        v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))
        loss = torch.cat((cos1, cos2, cos3), 1)
        return loss.mean()


class EdgeLengthLoss(nn.Module):
    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt):
        face = torch.LongTensor(self.face).cuda()

        d1_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
 
        diff1 = torch.abs(d1_out - d1_gt)
        diff2 = torch.abs(d2_out - d2_gt)
        diff3 = torch.abs(d3_out - d3_gt)
        loss = torch.cat((diff1, diff2, diff3), 1)
        return loss.mean()
    
class SMPLLoss(nn.Module):
    def __init__(self):
        super(SMPLLoss, self).__init__()
        self.criterion_regr = nn.MSELoss().cuda()

    def forward(self, pred_rotmat, pred_betas, gt_pose, gt_betas, mask_3d=None):
        pred_rotmat_valid = batch_rodrigues(pred_rotmat.reshape(-1,3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1,3)).reshape(-1, 24, 3, 3)
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
            if mask_3d is None:
                loss_regr_pose = loss_regr_pose.mean()
                loss_regr_betas = loss_regr_betas.mean()
            else:
                loss_regr_pose = (loss_regr_pose * mask_3d.unsqueeze(-1).unsqueeze(-1)).mean()
                loss_regr_betas = (loss_regr_betas * mask_3d).mean()
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).cuda()
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).cuda()
        return loss_regr_pose, loss_regr_betas

class LimbLengthError(nn.Module):
    """ Limb Length Loss: to let the """
    def __init__(self):
        super(LimbLengthError, self).__init__()
        self.CONNECTIVITY_DICT = [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 16), (9, 16), (8, 12), (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)]

    def forward(self, keypoints_3d_pred, keypoints_3d_gt):
        # (b, 17, 3)

        error = 0
        for (joint0, joint1) in self.CONNECTIVITY_DICT:
            limb_pred = keypoints_3d_pred[:, joint0] - keypoints_3d_pred[:, joint1]
            limb_gt = keypoints_3d_gt[:, joint0] - keypoints_3d_gt[:, joint1]
            if isinstance(limb_pred, np.ndarray):
                limb_pred = torch.from_numpy(limb_pred)
                limb_gt = torch.from_numpy(limb_gt)
            limb_length_pred = torch.norm(limb_pred, dim = 1)
            limb_length_gt = torch.norm(limb_gt, dim = 1)
            error += torch.abs(limb_length_pred - limb_length_gt).mean().cpu()

        return float(error)/len(self.CONNECTIVITY_DICT)

def get_loss(faces):
    loss = CoordLoss(has_valid=True), NormalVectorLoss(faces), EdgeLengthLoss(faces), \
           CoordLoss(has_valid=True), CoordLoss(has_valid=True), CoordLoss(has_valid=True),\
           SMPLLoss()

    return loss