import numpy as np
import torch
import wandb
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter

import models
import Human36M.dataset, COCO.dataset, PW3D.dataset, MPII3D.dataset, MPII.dataset
from core.config import cfg
from core.loss import get_loss
from multiple_datasets import MultipleDatasets
from funcs_utils import get_optimizer, load_checkpoint, get_scheduler, count_parameters, lr_check
from models.backbones.mesh import Mesh
import time
from coord_utils import rigid_align

from geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis
import math
from smpl import SMPL
import trimesh

mesh_model = SMPL()

def get_dataloader(args, dataset_names, is_train):
    dataset_split = 'TRAIN' if is_train else 'TEST'
    batch_per_dataset = cfg[dataset_split].batch_size // len(dataset_names)
    dataset_list, dataloader_list = [], []

    print(f"==> Preparing {dataset_split} Dataloader...")
    # dataset_names train_list: ['Human36M', 'PW3D', 'MPII3D', 'COCO', 'MPII']
    for name in dataset_names:
        # eval执行字符串运算，此处为创建PW3D.dataset, MPII3D.dataset等类的对象
        dataset = eval(f'{name}.dataset')(dataset_split.lower(), args=args)
        print("# of {} {} data: {}".format(dataset_split, name, len(dataset)))
        dataloader = DataLoader(dataset,
                                batch_size=batch_per_dataset,
                                shuffle=cfg[dataset_split].shuffle,
                                num_workers=cfg.DATASET.workers,
                                pin_memory=False)
        dataset_list.append(dataset)
        dataloader_list.append(dataloader)

    if not is_train:
        return dataset_list, dataloader_list
    else:
        trainset_loader = MultipleDatasets(dataset_list, make_same_len=True)
        batch_generator = DataLoader(dataset=trainset_loader, \
                          batch_size=batch_per_dataset * len(dataset_names), \
                          shuffle=cfg[dataset_split].shuffle, \
                          num_workers=cfg.DATASET.workers, pin_memory=False)
        return dataset_list, batch_generator

def prepare_network(args, load_dir='', is_train=True):
    dataset_names = cfg.DATASET.train_list if is_train else cfg.DATASET.test_list
    dataset_list, dataloader = get_dataloader(args, dataset_names, is_train)
    model, criterion, optimizer, lr_scheduler = None, None, None, None
    loss_history, test_error_history = [], {'surface': [], 'joint': []}

    main_dataset = dataset_list[0]
    J_regressor = eval(f'torch.Tensor(main_dataset.joint_regressor_{cfg.DATASET.input_joint_set})')
    if is_train or load_dir:
        print(f"==> Preparing {cfg.MODEL.name} MODEL...")
        if cfg.MODEL.name == 'PMCE':
            model = models.PMCE.get_model(num_joint=main_dataset.joint_num, embed_dim=cfg.MODEL.hpe_dim, depth=cfg.MODEL.hpe_dep)
        elif cfg.MODEL.name == 'PoseEst':
            model = models.PoseEstimation.get_model(num_joint=main_dataset.joint_num, embed_dim=cfg.MODEL.hpe_dim, depth=cfg.MODEL.hpe_dep, pretrained=False)
        print('# of model parameters: {}'.format(count_parameters(model)))

    if is_train:
        criterion = get_loss(faces=main_dataset.mesh_model.face)
        optimizer = get_optimizer(model=model)
        lr_scheduler = get_scheduler(optimizer=optimizer)

    if load_dir and (not is_train or args.resume_training):
        print('==> Loading checkpoint')
        checkpoint = load_checkpoint(load_dir=load_dir, pick_best=(cfg.MODEL.name == 'PoseEst'))
        model.load_state_dict(checkpoint['model_state_dict'])

        if is_train:
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            curr_lr = 0.0

            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']

            lr_state = checkpoint['scheduler_state_dict']
            # update lr_scheduler
            lr_state['milestones'], lr_state['gamma'] = Counter(cfg.TRAIN.lr_step), cfg.TRAIN.lr_factor
            lr_scheduler.load_state_dict(lr_state)

            loss_history = checkpoint['train_log']
            test_error_history = checkpoint['test_log']
            cfg.TRAIN.begin_epoch = checkpoint['epoch'] + 1
            print('===> resume from epoch {:d}, current lr: {:.0e}, milestones: {}, lr factor: {:.0e}'
                  .format(cfg.TRAIN.begin_epoch, curr_lr, lr_state['milestones'], lr_state['gamma']))

    return dataloader, dataset_list, model, criterion, optimizer, lr_scheduler, loss_history, test_error_history


class Trainer:
    def __init__(self, args, load_dir):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history\
            = prepare_network(args, load_dir=load_dir, is_train=True)

        self.main_dataset = self.dataset_list[0]
        self.print_freq = cfg.TRAIN.print_freq

        self.J_regressor = eval(f'torch.Tensor(self.main_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')

        self.model = self.model.cuda()

        self.mesh = Mesh()

        self.normal_weight = cfg.MODEL.normal_loss_weight
        self.edge_weight = cfg.MODEL.edge_loss_weight
        self.joint_weight = cfg.MODEL.joint_loss_weight
        self.edge_add_epoch = cfg.TRAIN.edge_loss_start
        self.shape_weight = cfg.MODEL.shape_loss_weight
        self.pose_weight = cfg.MODEL.pose_loss_weight
        # self.vertices_weight = cfg.MODEL.vertices_loss_weight

        if cfg.TRAIN.wandb:
            wandb.init(config=cfg,
                   project=cfg.MODEL.name,
                   name='PMCE/' + cfg.output_dir.split('/')[-1],
                   dir=cfg.output_dir,
                   job_type="training",
                   reinit=True)

    def train(self, epoch):
        self.model.train()

        lr_check(self.optimizer, epoch)

        running_loss = 0.0
        batch_generator = tqdm(self.batch_generator)
        for i, (inputs, targets, meta) in enumerate(batch_generator):
            # convert to cuda
            input_pose, input_feat = inputs['pose2d'].cuda(), inputs['img_feature'].cuda()
            gt_lift3dpose, gt_reg3dpose, gt_mesh = targets['lift_pose3d'].cuda(), targets['reg_pose3d'].cuda(), targets['mesh'].cuda()
            # gt_smplpose, gt_smplshape, gt_smplcam = targets['smpl_pose'].cuda(), targets['smpl_shape'].cuda(), targets['smpl_cam'].cuda()
            gt_smplpose, gt_smplshape = targets['smpl_pose'].cuda(), targets['smpl_shape'].cuda()
            val_lift3dpose, val_reg3dpose, val_mesh = meta['lift_pose3d_valid'].cuda(), meta['reg_pose3d_valid'].cuda(), meta['mesh_valid'].cuda()
            # print(gt_smplpose.shape)#torch.Size([32, 16, 72])
            # print(gt_smplshape.shape)#torch.Size([32, 16, 10])
            
            pose3d, evo_pose, init_smpl_pose, init_smpl_shape, pred_mesh,smploutput = self.model(input_pose, input_feat, is_train=True) 
            # pred_pose由mesh回归
            pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh * 1000)
            
            # root_coor = pred_pose[:,:1]
            # pred_pose = pred_pose - root_coor
            # pred_mesh = pred_mesh - root_coor
            
            # gt_pose = torch.matmul(self.J_regressor[None, :, :], gt_mesh[:,8])
            
            # evo_pose由HPE和HMR联合回归
            # evo_pose = evo_pose * 1000
            # pose3d直接由HPE回归

            # print(gt_smplpose.shape)                    torch.Size([32, 72])
            # print(len(smpl_output[-1]['theta']))        32
            # print(len(smpl_output[-1]['theta'][3:75]))  29 因为第一个维度事batch3-32共29个
            # print(smpl_output[-1]['theta'][3:75])
            # tensor([[ 2.0667,  0.0611,  0.5134,  ...,  1.0024,  0.0573,  2.7759],
            #         [ 2.0042,  0.1659,  0.4902,  ...,  0.2384, -0.3255,  1.1016],
            #         [ 1.7245, -0.1121,  0.1805,  ...,  2.3353, -0.0802,  3.1159],
            #         ...,
            #         [ 1.4690, -0.0663,  0.5126,  ...,  0.5414, -1.3929,  0.6215],
            #         [ 1.9265,  0.1234,  0.3918,  ...,  0.1955,  0.5917,  0.8385],
            #         [ 1.3474, -0.0083,  0.4560,  ...,  0.6183, -1.4196,  0.4668]],
            #     device='cuda:0', grad_fn=<SliceBackward0>)

            # loss
            # loss1 网格顶点损失
            # loss2 normal损失
            # loss4 网格回归关节点损失
            # loss5 解码器输出关节点损失
            # loss6 编码器输出关节点损失
            # print(pred_mesh.shape)torch.Size([32, 6890, 3])
            # print(gt_mesh.shape)torch.Size([32, 16, 6890, 3])
            # print(val_mesh.shape)torch.Size([32, 16, 6890, 1])
            # print(pred_pose.shape)torch.Size([32, 17, 3])
            # print(gt_reg3dpose.shape)torch.Size([32, 17, 3])
            # print(val_reg3dpose.shape)  #torch.Size([32, 272, 3])
            # print(evo_pose.shape)torch.Size([32, 19, 3])
            # print(gt_lift3dpose.shape)torch.Size([32, 19, 3])
            # print(val_lift3dpose.shape)torch.Size([32, 19, 1])
            # print(pose3d.shape)torch.Size([32, 19, 3])
            loss1, loss2, loss4, loss5, loss6 = self.loss[0](pred_mesh, gt_mesh, val_mesh),  \
                                         self.normal_weight * self.loss[1](pred_mesh, gt_mesh), \
                                         self.joint_weight * self.loss[3](pred_pose, gt_reg3dpose, val_reg3dpose), \
                                         self.joint_weight * self.loss[4](evo_pose, gt_lift3dpose, val_lift3dpose), \
                                         self.joint_weight * self.loss[5](pose3d, gt_lift3dpose, val_lift3dpose)
            
            pa_loss = 0
            for n in range(pred_pose.shape[0]):
                pose_aligned = rigid_align(pred_pose[n].detach().cpu().numpy(), gt_reg3dpose[n].detach().cpu().numpy()) # perform rigid 
                pose_aligned = torch.from_numpy(pose_aligned).cuda()
                pa_loss += self.loss[3](pose_aligned, gt_reg3dpose[n], val_reg3dpose)
            pa_loss = self.joint_weight * (pa_loss / pred_pose.shape[0])

            # loss1, loss2, loss4, loss6 = self.vertices_weight * self.loss[0](pred_mesh, gt_mesh[:,8], val_mesh[:,8]),  \
            #                              self.normal_weight * self.loss[1](pred_mesh, gt_mesh[:,8]), \
            #                              self.joint_weight * self.loss[3](pred_pose, gt_reg3dpose, val_reg3dpose), \
            #                              self.joint_weight * self.loss[5](pose3d, gt_lift3dpose, val_lift3dpose)
            
            # mid-frame accuracy loss
            # smpl_pose_loss = self.loss[6](smploutput[-1]['theta'][:,8,3:75],gt_smplpose[:,8],target_valid=None)
            # smpl_shape_loss = self.loss[6](smploutput[-1]['theta'][:,8,75:],gt_smplshape[:,8],target_valid=None)
            # init_smpl_pose_loss
            # (batch_size, seqlen, 24, 6)
            init_rotmat = rot6d_to_rotmat(init_smpl_pose[:,cfg.DATASET.seqlen // 2]).view(init_smpl_pose[:,cfg.DATASET.seqlen // 2].shape[0], 24, 3, 3)
            init_axis = rotation_matrix_to_angle_axis(init_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
            init_smpl_pose_loss, init_smpl_shape_loss = self.loss[6](init_axis,\
                                                                    init_smpl_shape[:,cfg.DATASET.seqlen // 2],\
                                                                    gt_smplpose,\
                                                                    gt_smplshape,\
                                                                    mask_3d=None)
            init_smpl_loss = self.shape_weight * init_smpl_shape_loss + self.pose_weight * init_smpl_pose_loss

            smpl_pose_loss, smpl_shape_loss = self.loss[6](smploutput[-1]['theta'][:,cfg.DATASET.seqlen // 2,3:75],\
                                                           smploutput[-1]['theta'][:,cfg.DATASET.seqlen // 2,75:],\
                                                            gt_smplpose,\
                                                            gt_smplshape,\
                                                            mask_3d=None)
            mid_smpl_loss = self.shape_weight * smpl_shape_loss + self.pose_weight * smpl_pose_loss

            # sequence pose velocity loss 不合理 应当使用pose_velocity_loss
            # pred_smpl_pose_motion = smploutput[-1]['theta'][:,1:,3:75] - smploutput[-1]['theta'][:,:-1,3:75]
            # gt_smpl_pose_motion = gt_smplpose[:,1:] - gt_smplpose[:,:-1]
            # smpl_velocity_loss = self.loss[5](pred_smpl_pose_motion,gt_smpl_pose_motion,target_valid=None) * 0.1


            loss = loss1 + loss2 + loss4 + loss5 + loss6 + mid_smpl_loss
            # loss = loss1 + loss4 + mid_smpl_loss + pa_loss
            # loss = loss1 + loss4 + loss5 + loss6 + mid_smpl_loss + pa_loss

            # import pdb; pdb.set_trace()

            # 从第2个epoch后加入edge损失 后面代码也要修改
            if epoch > self.edge_add_epoch:
                loss3 = self.edge_weight * self.loss[2](pred_mesh, gt_mesh)
                loss += loss3

            # update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log
            running_loss += float(loss.detach().item())
            if cfg.TRAIN.wandb:
                wandb_loss1, wandb_loss2, wandb_loss4, wandb_loss6 = loss1.detach(), loss2.detach(), loss4.detach(), loss6.detach()
                wandb_loss3 = loss3.detach() if epoch > self.edge_add_epoch else 0
                wandb.log(
                    {
                        'train_loss/vertex_loss': wandb_loss1,
                        'train_loss/normal_loss': wandb_loss2,
                        'train_loss/edge_loss': wandb_loss3,
                        'train_loss/mesh2joint3d_loss': wandb_loss4,
                        'train_loss/liftjoint3d_loss': wandb_loss6
                    }
                )

            if i % self.print_freq == 0:
                loss1, loss2, loss4, loss5 = loss1.detach(), loss2.detach(), loss4.detach(), loss5.detach()
                loss3 = loss3.detach() if epoch > self.edge_add_epoch else 0
                init_loss = init_smpl_loss.detach()
                smpl_loss = mid_smpl_loss.detach()
                loss6 = loss6.detach()
                pa_loss = pa_loss.detach()
                total_loss = loss.detach()
                batch_generator.set_description(f'Epoch{epoch}_({i}/{len(batch_generator)}) => '
                                                f'vl: {loss1:.3f} '
                                                f'nl: {loss2:.3f} '
                                                f'el: {loss3:.3f} '
                                                f'm->j: {loss4:.3f} '
                                                f'evl: {loss5:.3f} '
                                                f'ljl: {loss6:.3f} '
                                                f'il: {init_loss:.3f} '
                                                f'sl: {smpl_loss:.3f} '
                                                f'pal: {pa_loss:.3f} '
                                                f'tl: {total_loss:.3f}')


        self.loss_history.append(running_loss / len(batch_generator))

        print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')


class Tester:
    def __init__(self, args, load_dir=''):
        self.val_loader, self.val_dataset, self.model, _, _, _, _, _ = \
            prepare_network(args, load_dir=load_dir, is_train=False)

        self.val_loader, self.val_dataset = self.val_loader[0], self.val_dataset[0]
        self.print_freq = cfg.TRAIN.print_freq

        self.J_regressor = eval(f'torch.Tensor(self.val_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')

        if self.model:
            self.model = self.model.cuda()

        # initialize error value
        self.surface_error = 9999.9
        self.joint_error = 9999.9

    def test(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()

        surface_error = 0.0
        joint_error = 0.0

        result = []
        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        loader = tqdm(self.val_loader)
        with torch.no_grad():
            for i, (inputs, targets, meta) in enumerate(loader):
                input_pose, input_feat = inputs['pose2d'].cuda(), inputs['img_feature'].cuda()
                gt_pose3d, gt_mesh = targets['reg_pose3d'].cuda(), targets['mesh'].cuda()
                
                gt_lift_pose3d = targets['lift_pose3d'].cuda()
                # print(input_pose.shape) # torch.Size([64, 16, 19, 2])
                # print(input_feat.shape) # torch.Size([64, 16, 2048])
                # print(gt_pose3d.shape) # torch.Size([64, 17, 3])  torch.Size([64, 16, 17, 3])
                # print(gt_mesh.shape) # torch.Size([64, 16, 6890, 3])

                pose3d, evo_pose, init_smpl_pose, init_smpl_shape, pred_mesh, smploutput = self.model(input_pose, input_feat, is_train=False)
                # pred_mesh = smpl_output[-1]['verts']
                pred_mesh, gt_mesh = pred_mesh * 1000, gt_mesh * 1000
                
                # root_coor = pred_pose[:,:1]
                # pred_pose = pred_pose - root_coor
                # pred_mesh = pred_mesh - root_coor

                # save_obj
                if i == 50:
                    gt_obj = gt_mesh[16].cpu().numpy()
                    mesh = trimesh.Trimesh(vertices=gt_obj, faces=mesh_model.face, process=False)
                    mesh.export('/data-home/tangt/models/PMCE/output/gt_obj_3dpw.obj')

                    pred_obj = pred_mesh[16].cpu().numpy()
                    mesh = trimesh.Trimesh(vertices=pred_obj, faces=mesh_model.face, process=False)
                    mesh.export('/data-home/tangt/models/PMCE/output/pred_obj_3dpw.obj')
                    # import pdb; pdb.set_trace()
                # pred_mesh = pred_mesh[:,None,:,:]
                # pred_mesh = pred_mesh.repeat(1,16,1,1)
                # print(pred_mesh.shape) # torch.Size([64, 16, 6890, 3])

                pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh)
                # pred_pose结果y轴误差非常大
                # print(pred_pose[0]-gt_pose[0])
                # tensor([[  -1.6835, -215.2693,   30.7076],
                #         [  10.0937, -208.7321,   -2.2924],
                #         [  13.3569, -246.2525,  -17.8623],
                #         [  -7.1944, -284.0257,  -45.8250],
                #         [ -12.1289, -223.0302,   61.5660],
                #         [  18.3279, -251.8084,    0.9298],
                #         [  10.0254, -282.4044,  -52.9871],
                #         [  -7.0715, -166.7856,   21.2642],
                #         [  -4.8401, -111.4207,  -28.2019],
                #         [ -18.4676,  -55.0926,  -84.7725],
                #         [ -27.8560,  -49.3625, -152.8790],
                #         [ -28.9368, -132.8869,   37.1572],
                #         [  10.3073, -170.1200,  113.0389],
                #         [   3.8194, -197.6202,   80.6891],
                #         [  23.8410, -111.5228,  -46.1689],
                #         [  51.1461, -143.1642,   34.4624],
                #         [  55.0545, -173.7980,   56.9125]]
                # pred_pose = pred_pose - pred_pose[:,:1]
                # print(pred_mesh.shape) # torch.Size([64, 6890, 3])
                # print(pred_pose.shape) # torch.Size([64, 17, 3])
                # print(pred_pose.shape) # torch.Size([64, 16, 17, 3])
                # gt_pose = torch.matmul(self.J_regressor[None, :, :], gt_mesh[:,8])
                # gt_pose与gt_pose3d相近，但不完全相同
                # print(gt_pose[0]-gt_pose3d[0])
                # tensor([[ 1.3653e-02, -1.7592e-02,  3.5995e-01],
                #         [-8.3466e-03, -1.3507e-02,  3.4268e-01],
                #         [-7.4234e-03, -3.0334e-02,  3.0605e-01],
                #         [ 2.1202e-02, -1.2262e-01,  1.4458e+00],
                #         [-2.4307e-02, -1.3148e-02,  3.8345e-01],
                #         [ 2.6550e-02,  2.9144e-02,  2.5100e-01],
                #         [ 2.4719e-03, -1.0010e-01,  8.3786e-01],
                #         [-6.0024e-03,  3.6896e-02,  5.5578e-01],
                #         [ 6.2370e-03,  2.2888e-02,  1.6426e-01],
                #         [-4.2458e-03, -3.6133e-02, -2.0937e-01],
                #         [-1.3676e-02, -6.2378e-02,  1.5158e-01],
                #         [ 3.2349e-03, -3.3569e-04,  6.1203e-02],
                #         [-3.8773e-02, -3.5599e-02,  7.9323e-02],
                #         [-1.7334e-02, -3.0479e-02,  6.0696e-01],
                #         [ 8.2703e-03,  3.3020e-02,  1.9454e-01],
                #         [ 4.7836e-02, -1.8280e-02,  1.7985e-01],
                #         [ 1.3885e-02, -1.9897e-02,  9.0593e-01]]

                j_error, s_error = self.val_dataset.compute_both_err(pred_mesh, gt_mesh, pred_pose, gt_pose3d)
                
                if i % self.print_freq == 0:
                    loader.set_description(f'{eval_prefix}({i}/{len(self.val_loader)}) => surface error: {s_error:.4f}, joint error: {j_error:.4f}')

                joint_error += j_error
                surface_error += s_error

                # Final Evaluation
                if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
                    pred_mesh, target_mesh = pred_mesh.detach().cpu().numpy(), gt_mesh.detach().cpu().numpy()
                    pred_pose, gt_pose3d = pred_pose.detach().cpu().numpy(), gt_pose3d.detach().cpu().numpy()
                    # len(input_pose)=batchsize
                    for j in range(len(input_pose)):
                        out = {}
                        out['mesh_coord'], out['mesh_coord_target'] = pred_mesh[j], target_mesh[j]
                        out['joint_coord'], out['joint_coord_target'] = pred_pose[j], gt_pose3d[j]
                        result.append(out)

            self.surface_error = surface_error / len(self.val_loader)
            self.joint_error = joint_error / len(self.val_loader)
            
            print(f'{eval_prefix}MPVPE: {self.surface_error:.2f}, MPJPE: {self.joint_error:.2f}')

            if cfg.TRAIN.wandb:
                wandb_joint_error = self.joint_error
                wandb_verts_error = self.surface_error
                wandb.log(
                    {
                        'epoch': epoch,
                        'error/MPJPE': wandb_joint_error,
                        'error/MPVPE': wandb_verts_error,
                    }
                )

            # Final Evaluation
            if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
                self.val_dataset.evaluate(result)


class LiftTrainer:
    def __init__(self, args, load_dir):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history \
            = prepare_network(args, load_dir=load_dir, is_train=True)

        self.loss = self.loss[0]
        self.main_dataset = self.dataset_list[0]
        self.num_joint = self.main_dataset.joint_num
        # self.num_joint = 16
        self.print_freq = cfg.TRAIN.print_freq

        self.model = self.model.cuda()

        if cfg.TRAIN.wandb:
            wandb.init(config=cfg,
                   project=cfg.MODEL.name,
                   name='PoseEst/' + cfg.output_dir.split('/')[-1],
                   dir=cfg.output_dir,
                   job_type="training",
                   reinit=True)

    def train(self, epoch):
        self.model.train()

        lr_check(self.optimizer, epoch)

        running_loss = 0.0
        batch_generator = tqdm(self.batch_generator)
        for i, (img_joint, cam_joint, joint_valid, img_features) in enumerate(batch_generator):
            img_joint, cam_joint = img_joint.cuda().float(), cam_joint.cuda().float()
            joint_valid = joint_valid.cuda().float()
            img_features = img_features.cuda().float()

            pred_joint = self.model(img_joint, img_features)
            # pred_joint = pred_joint.view(-1, self.num_joint, 3)
            # cam_joint = cam_joint.view(-1, self.num_joint, 3)
            pred_joint = pred_joint.view(-1, cfg.DATASET.seqlen, self.num_joint, 3)
            cam_joint = cam_joint.view(-1, cfg.DATASET.seqlen, self.num_joint, 3)
            # print(img_joint.shape)
            # print(cam_joint.shape)
            # print(joint_valid.shape)
            # print(img_features.shape)
            # print(pred_joint.shape)
            # torch.Size([64, 16, 19, 2])
            # torch.Size([64, 16, 19, 3])
            # torch.Size([64, 16, 19, 1])
            # torch.Size([64, 16, 2048])
            # torch.Size([64, 16, 19, 3])
            
            # torch.Size([64, 16, 19, 2])
            # torch.Size([64, 19, 3])
            # torch.Size([64, 19, 1])
            # torch.Size([64, 16, 2048])
            # torch.Size([64, 19, 3])
            # import pdb; pdb.set_trace()
            mpjpe_loss = self.loss(pred_joint, cam_joint, joint_valid)

            # pa_loss = 0
            # for n in range(pred_joint.shape[0]):
            #     pose_aligned = rigid_align(pred_joint[n,8].detach().cpu().numpy(), cam_joint[n,8].detach().cpu().numpy()) # perform rigid 
            #     pose_aligned = torch.from_numpy(pose_aligned).cuda()
            #     pa_loss += self.loss(pose_aligned, cam_joint[n,8], joint_valid)

            # pred_joint_motion = pred_joint[:,1:] - pred_joint[:,:-1]
            # gt_joint_motion = cam_joint[:,1:] - cam_joint[:,:-1]
            # velocity_valid = joint_valid[:,:-1]
            # velocity_loss = self.loss(pred_joint_motion, gt_joint_motion, velocity_valid)
            # loss = mpjpe_loss*0.1 + velocity_loss*0.1 + pa_loss.to('cuda')*0.1
            
            loss = mpjpe_loss

            self.optimizer.zero_grad()
            loss.backward()   # RuntimeError: grad can be implicitly created only for scalar outputs
            # loss.sum().backward()
            self.optimizer.step()

            running_loss += float(loss.detach().item())
            if cfg.TRAIN.wandb:
                wandb_loss = loss.detach()
                wandb.log(
                    {
                        'train_loss/total_loss': wandb_loss
                    }
                )

            if i % self.print_freq == 0:
                batch_generator.set_description(f'Epoch{epoch}_({i}/{len(self.batch_generator)}) => '
                                                f'total loss: {loss.detach():.4f} ')

        self.loss_history.append(running_loss / len(self.batch_generator))

        print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')


class LiftTester:
    def __init__(self, args, load_dir=''):
        self.val_loader, self.val_dataset, self.model, _, _, _, _, _ = \
            prepare_network(args, load_dir=load_dir, is_train=False)
        self.val_dataset = self.val_dataset[0]
        self.val_loader = self.val_loader[0]

        self.num_joint = self.val_dataset.joint_num
        # self.num_joint = 16
        self.print_freq = cfg.TRAIN.print_freq

        if self.model:
            self.model = self.model.cuda()

        # initialize error value
        self.surface_error = 9999.9
        self.joint_error = 9999.9

    def test(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()
        

        result = []
        joint_error = 0.0
        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        loader = tqdm(self.val_loader)
        with torch.no_grad():
            for i, (img_joint, cam_joint, _, img_features) in enumerate(loader):
                img_joint, cam_joint = img_joint.cuda().float(), cam_joint.cuda().float()
                img_features = img_features.cuda().float()

                pred_joint = self.model(img_joint, img_features)
                # pred_joint = pred_joint.view(-1, self.num_joint, 3)
                pred_joint = pred_joint.view(-1, cfg.DATASET.seqlen, self.num_joint, 3)
                # print(cam_joint.shape)
                # print(pred_joint.shape)
                # torch.Size([64, 16, 17, 3])
                # torch.Size([64, 16, 17, 3])

                mpjpe = self.val_dataset.compute_joint_err(pred_joint, cam_joint)
                joint_error += mpjpe

                if i % self.print_freq == 0:
                    loader.set_description(f'{eval_prefix}({i}/{len(self.val_loader)}) => joint error: {mpjpe:.4f}')

                # Final Evaluation
                if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
                    pred_joint, target_joint = pred_joint.detach().cpu().numpy(), cam_joint.detach().cpu().numpy()
                    for j in range(len(pred_joint)):
                        out = {}
                        out['joint_coord'], out['joint_coord_target'] = pred_joint[j], target_joint[j]
                        # print(pred_joint[j].shape)
                        # print(target_joint[j].shape)
                        # (16, 19, 3)
                        # (16, 19, 3)
                        result.append(out)

        self.joint_error = joint_error / len(self.val_loader)
        print(f'{eval_prefix}MPJPE: {self.joint_error:.4f}')

        if cfg.TRAIN.wandb:
                wandb_error = self.joint_error
                wandb.log(
                    {
                        'epoch': epoch,
                        'error/MPJPE': wandb_error
                    }
                )

        # Final Evaluation
        if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
            # print(len(result))
            # print(len(result[0]['joint_coord']))
            # print(len(result[1]['joint_coord_target']))
            # 34677 所有帧结果同时输入计算acc
            # 16
            # 16
            self.val_dataset.evaluate_joint(result)