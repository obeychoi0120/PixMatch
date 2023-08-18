import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from chainercv.evaluations import calc_semantic_segmentation_confusion
import math
from util import pyutils
from PIL import Image

def get_masks_by_confidence(cam):
    '''
    input: CAM before softmax [B, C, H, W]
    output: list of masks [B, H, W] which belongs to confidence range of 
    [, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 0.95), [0.95, 0.99), [0.99, ]
    '''
    masks = []
    # cam = torch.softmax(cam, dim=1)
    _max_probs = cam.softmax(dim=1).max(dim=1).values
    masks.append(_max_probs.lt(0.4).float())
    masks.append(torch.logical_and(_max_probs.ge(0.4), _max_probs.lt(0.6)).float())
    masks.append(torch.logical_and(_max_probs.ge(0.6), _max_probs.lt(0.8)).float())
    masks.append(torch.logical_and(_max_probs.ge(0.8), _max_probs.lt(0.95)).float())
    masks.append(torch.logical_and(_max_probs.ge(0.95), _max_probs.lt(0.99)).float())
    masks.append(_max_probs.ge(0.99).float())
    return masks

def get_avg_meter(args):
    log_keys = ['loss_cls', 'loss_sup', 'mask_ratio', 'fg_mask_ratio', 'exp_mask_ratio']
    if args.network_type == 'contrast':
        log_keys.extend(['loss_er', 'loss_ecr', 'loss_nce', 'loss_sal',
                         ])
    if args.mode == 'ssl':
        log_keys.extend([
            'loss', 'loss_semcon', 'loss_bdry', 'bdry_ratio', \
            'mask_1', 'mask_2', 'mask_3', 'mask_4', 'mask_5', 'mask_6'
            ])
    avg_meter = pyutils.AverageMeter(*log_keys)
    return avg_meter