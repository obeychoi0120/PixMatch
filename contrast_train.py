import os
import shutil
import torch
import argparse
from torch.backends import cudnn
import wandb

import numpy as np
import random

from util import pyutils, torchutils
import importlib

from module.dataloader import get_dataloader
from module.model import get_model
from module.optimizer import get_optimizer
from module.train import train_base, train_ssl

dataset_list = ['voc12', 'coco']

def get_arguments():
    parser = argparse.ArgumentParser()
    # Session
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--val_only', action='store_true')
    parser.add_argument('--session', default='wsss', type=str)
    parser.add_argument('--use_wandb', action='store_true')             # Use wandb Logging
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--val_freq', default=500, type=int)
    parser.add_argument('--seed', default=None, type=int)

    # Data
    parser.add_argument("--dataset", default='voc12', choices=dataset_list, type=str)
    parser.add_argument('--data_root', required=True, type=str)
    parser.add_argument('--saliency_root', type=str)
    parser.add_argument('--train_list', default='data/voc12/train_aug_id.txt', type=str)
    parser.add_argument('--val_list', default='data/voc12/train_id.txt', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--resize_size', default=(448, 768), type=int, nargs='*')
    parser.add_argument('--crop_size', default=448, type=int)        

    # Iteration & Optimizer
    parser.add_argument('--iter_size', default=1, type=int)
    parser.add_argument('--max_iters', default=10000, type=int)
    parser.add_argument('--max_epoches', default=None, type=int)        # default=15
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--wt_dec', default=5e-4, type=float)

    # Network
    parser.add_argument('--network', default='network.resnet38_contrast', type=str)
    parser.add_argument('--weights', required=True, type=str, default='pretrained/ilsvrc-cls_rna-a1_cls1000_ep-0001.params')
    
    # Hyperparameters for EPS
    parser.add_argument('--tau', default=0.5, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)

    ### Semi-supervised Learning ###
    parser.add_argument('--mode', required=True, choices=['base', 'ssl'])

    parser.add_argument('--use_ema',action='store_true') 
    parser.add_argument('--ema_m', default=0.999, type=float) 
    parser.add_argument('--p_cutoff', default=0.95, type=float)
    
    ### Augmentations ###
    parser.add_argument('--aug_type', default='strong', type=str)       # None / weak / strong : 'aug_type'
    parser.add_argument('--n_strong_augs', type=int)                    # number of RandAug
    parser.add_argument('--use_geom_augs', action='store_true')

    parser.add_argument('--bdry_size', default=0, type=int)
    parser.add_argument('--bdry_lambda', default=0.0, type=float)
    parser.add_argument('--bdry', action='store_true')
    parser.add_argument('--bdry_cutoff', default=0.0, type=float)

    args = parser.parse_args()

    # Dataset(Class Number)
    if args.dataset == 'voc12':
        args.num_classes = 21
    elif args.dataset == 'coco':
        args.num_classes = 81
    

    # Network type
    if 'contrast' in args.network:
        args.network_type = 'contrast'
    elif 'eps' in args.network:
        args.network_type = 'eps'
    else:
        raise Exception('No appropriate model type')
    
    return args


if __name__ == '__main__':
    # Get arguments
    args = get_arguments()

    # Set wandb Logger
    if args.use_wandb:
        wandb.init(name=args.session, project='WSSS')

    # Set Python Logger
    args.log_folder = os.path.join('train_log', args.session)
    os.makedirs(args.log_folder, exist_ok=True)

    # Control Randomness
    if args.seed:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Load dataset (train_ulb_loader=None where args.ssl==False)
    train_loader, val_loader = get_dataloader(args) ###

    # Max step
    num_data = len(open(args.train_list).read().splitlines())
    if args.max_epoches is None:
        args.max_epoches = int(args.max_iters * args.iter_size // (num_data // args.batch_size))
    max_step = len(train_loader) * args.max_epoches
    
    # Load (ImageNet) Pretrained Model and Set optimizer
    model = get_model(args)
    optimizer = get_optimizer(args, model, max_step)
        
    # DP
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.train()

    if args.use_wandb:
        wandb.config.update(args)
    if args.mode == 'ssl':
        train_ssl(train_loader, val_loader, model, optimizer, max_step, args)
    else:
        train_base(train_loader, val_loader, model, optimizer, max_step, args)
    print('Train Done.')