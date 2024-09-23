import os, sys
sys.path.append('/data0/tangt/models/PMCE/lib')
import argparse
from core.config import cfg, update_config

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Train Pose2Mesh')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--resume_training', action='store_true', help='Resume Training')
parser.add_argument('--debug', action='store_true', help='reduce dataset items')
parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat')
parser.add_argument('--cfg', type=str, help='experiment configure file name')

args = parser.parse_args()
if args.cfg:
    update_config(args.cfg)
print('Seed = ', args.seed)

os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

import torch
# torch.backends.cudnn.enabled = False
import __init_path
import shutil
import random
import numpy as np
from funcs_utils import save_checkpoint, check_data_pararell, count_parameters
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

output_model_dir = os.path.join(cfg.checkpoint_dir, 'CoevoDecoder.py')
shutil.copyfile(src='/data0/tangt/models/PMCE/lib/models/CoevoDecoder.py', dst=output_model_dir)

output_model_dir = os.path.join(cfg.checkpoint_dir, 'PoseEstimation.py')
shutil.copyfile(src='/data0/tangt/models/PMCE/lib/models/PoseEstimation.py', dst=output_model_dir)

output_model_dir = os.path.join(cfg.checkpoint_dir, 'NIKITS.py')
shutil.copyfile(src='/data0/tangt/models/PMCE/lib/models/NIKITS.py', dst=output_model_dir)

output_model_dir = os.path.join(cfg.checkpoint_dir, 'PMCE.py')
shutil.copyfile(src='/data0/tangt/models/PMCE/lib/models/PMCE.py', dst=output_model_dir)

output_model_dir = os.path.join(cfg.checkpoint_dir, 'base.py')
shutil.copyfile(src='/data0/tangt/models/PMCE/lib/core/base.py', dst=output_model_dir)

from core.base import Trainer, Tester, LiftTrainer, LiftTester

if cfg.MODEL.name == 'PMCE':
    trainer = Trainer(args, load_dir='')
    tester = Tester(args)  # if not args.debug else None
elif cfg.MODEL.name == 'PoseEst':
    trainer = LiftTrainer(args, load_dir='')
    tester = LiftTester(args)  # if not args.debug else None

print("===> Start training...")

for epoch in range(cfg.TRAIN.begin_epoch, cfg.TRAIN.end_epoch + 1):
    trainer.train(epoch)
    trainer.lr_scheduler.step()

    tester.test(epoch, current_model=trainer.model)

    if epoch > 1:
        is_best = tester.joint_error < min(trainer.error_history['joint']) or tester.surface_error < min(trainer.error_history['surface'])
    else:
        is_best = None

    trainer.error_history['surface'].append(tester.surface_error)
    trainer.error_history['joint'].append(tester.joint_error)

    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': check_data_pararell(trainer.model.state_dict()),  # 
        'optim_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
        'train_log': trainer.loss_history,
        'test_log': trainer.error_history
    }, epoch, is_best)

print('Training Finished! All logs were saved in ', cfg.checkpoint_dir)