import os
import argparse
import shutil
import __init_path

from core.config import update_config, cfg

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Test ARTS')

parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--cfg', type=str, help='experiment configure file name')
parser.add_argument('--debug', action='store_true', help='reduce dataset items')
parser.add_argument('--gpu', type=str, default='0,', help='assign multi-gpus by comma concat')

args = parser.parse_args()
if args.cfg:
    update_config(args.cfg)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])



import torch
torch.manual_seed(args.seed)
from core.base import Tester, LiftTester
if cfg.MODEL.name == 'ARTS':
    tester = Tester(args, load_dir=cfg.TEST.weight_path)  # if not args.debug else None
elif cfg.MODEL.name == 'PoseEst':
    tester = LiftTester(args, load_dir=cfg.TEST.weight_path)  # if not args.debug else None

print("===> Start testing...")
tester.test(0)