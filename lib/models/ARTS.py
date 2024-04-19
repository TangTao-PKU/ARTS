import torch
import torch.nn as nn
from core.config import cfg as cfg
from models import SemiAnalytical, PoseEstimation

import os
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"


class ARTS(nn.Module):
    def __init__(self, num_joint, embed_dim, depth):
        super(ARTS, self).__init__()

        self.num_joint = num_joint
        self.pose_lifter = PoseEstimation.get_model(num_joint, embed_dim, depth, pretrained=cfg.MODEL.posenet_pretrained)
        self.pose_mesh_coevo = SemiAnalytical.get_model(num_joint, embed_dim)

    def forward(self, pose2d, img_feat, is_train=True):
        pose3d = self.pose_lifter(pose2d, img_feat)
        # pose3d = pose3d.reshape(-1, self.num_joint, 3)
        pose3d = pose3d.reshape(-1, cfg.DATASET.seqlen, self.num_joint, 3)
        
        evo_pose, init_smpl_pose, init_smpl_shape, final_mesh, smploutput = self.pose_mesh_coevo(pose3d / 1000, img_feat, is_train=is_train)
        
        # 计算loss仅考虑中间帧pose
        pose3d = pose3d[:,cfg.DATASET.seqlen // 2]

        return pose3d, evo_pose, init_smpl_pose, init_smpl_shape, final_mesh, smploutput


def get_model(num_joint, embed_dim, depth):
    model = ARTS(num_joint, embed_dim, depth)

    return model


if __name__ == '__main__':
    
    model = get_model(17,64,2) # 模型初始化
    model = model.cuda()
    pose = torch.randn(1, 17, 3).cuda()
    img = torch.randn(1, 16, 2048).cuda()
    ARTS(pose,img,2)
    model.eval()

# export PYTHONPATH=/data-home/tangt/models/models/Pose-baseline/lib