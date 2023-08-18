import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import functional as tvf
import wandb
from chainercv.evaluations import calc_semantic_segmentation_confusion
from chainercv.datasets import VOCSemanticSegmentationDataset
import os
import re 
import random
import numpy as np
from glob import glob
from copy import deepcopy
import natsort as nsort
import pdb
from util import pyutils
from data.augmentation.randaugment import tensor_augment_list
from module.loss import adaptive_min_pooling_loss, get_er_loss, get_eps_loss, get_contrast_loss, max_norm, balanced_cross_entropy
from module.validate import *
from module.ema import EMA
from module.helper import get_avg_meter, get_masks_by_confidence
import cv2

def train_base(train_dataloader, val_dataloader, model, optimizer, max_step, args):
    avg_meter = get_avg_meter(args=args)
    timer = pyutils.Timer("Session started: ")
    print("# of train data: ", len(train_dataloader.dataset))
    loader_iter = iter(train_dataloader)
    # Wandb logging
    if args.use_wandb:
        wandb.watch(model, log_freq=args.log_freq * args.iter_size)
    ### Train Scalars, Histograms, Images ###
    tscalar = {}
    ### validation logging
    val_freq = args.val_freq ### validation logging
    gamma = 0.10
    print(args)

    if args.val_only:
        print("Val-only mode.")
        ### Model Watch (log_freq=val_freq)
        if args.use_wandb:
            wandb.watch(model, log_freq=args.log_freq * args.iter_size)
        weight_path = glob(os.path.join(args.log_folder, '*.pth'))
        weight_path = nsort.natsorted(weight_path) 
        for weight in weight_path:
            print(f'Loading {weight}')
            model.module.load_state_dict(torch.load(weight), strict=True)
            model.eval()
            tmp = weight[-10:]
            try:
                iteration = int(re.sub(r'[^0-9]', '', tmp))
            except:
                iteration = args.max_iters
            validate_voc_ppc(args, model, val_dataloader, iteration, tag='val')
    
    else:
        for iteration in range(args.max_iters):
            for _ in range(args.iter_size):
                if args.network_type == 'contrast':
                    try:
                        img_id, img, sal, label = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(train_dataloader)
                        img_id, img, sal, label = next(loader_iter)
                    img = img.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)
                    sal = sal.cuda(non_blocking=True)

                    img_w2 = F.interpolate(img, size=(128, 128), mode='bilinear', align_corners=True)
                    sal2 = F.interpolate(sal, size=(128, 128), mode='bilinear', align_corners=True)
                    pred1, cam1, pred_rv1, cam_rv1, feat1 = model(img)
                    pred2, cam2, pred_rv2, cam_rv2, feat2 = model(img_w2)                        
                    loss_cls = F.multilabel_soft_margin_loss(pred1[:, :-1], label)
                    loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)
                    bg_score = torch.ones((B, 1)).cuda()
                    label_append_bg = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)  # (N, 21, 1, 1)
                    loss_cls_rv1 = adaptive_min_pooling_loss((cam_rv1 * label_append_bg)[:, :-1, :, :])
                    loss_cls_rv2 = adaptive_min_pooling_loss((cam_rv2 * label_append_bg)[:, :-1, :, :])
                    loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.
                    
                    # SEAM Losses
                    loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)
                    
                    # EPS+PPC Losses
                    loss_sal, _, _, _ = get_eps_loss(cam1, sal, label, args.tau, args.alpha, intermediate=True, num_class=args.num_classes)
                    loss_sal2, _, _, _ = get_eps_loss(cam2, sal2, label, args.tau, args.alpha, intermediate=True, num_class=args.num_classes)
                    loss_sal_rv, _, _, _ = get_eps_loss(cam_rv1, sal, label, args.tau, args.alpha, intermediate=True, num_class=args.num_classes)
                    loss_sal_rv2, _, _, _ = get_eps_loss(cam_rv2, sal2, label, args.tau, args.alpha, intermediate=True, num_class=args.num_classes)
                    loss_sal = (loss_sal + loss_sal2) / 2. + (loss_sal_rv + loss_sal_rv2) / 2.
                    loss_nce = get_contrast_loss(cam_rv1, cam_rv2, feat1, feat2, label, gamma=gamma, bg_thres=0.10)
                    loss_sup = loss_cls + loss_er + loss_ecr + loss_sal + loss_nce

                    avg_meter.add({
                        'loss_sup': loss_sup.item(),
                        'loss_cls': loss_cls.item(),
                        'loss_er': loss_er.item(),
                        'loss_ecr': loss_ecr.item(),
                        'loss_nce': loss_nce.item(),
                        'loss_sal': loss_sal.item(),
                        })
                    

                with torch.no_grad():
                    class_mask = label.unsqueeze(2).unsqueeze(3)
                    cam_w = model.module.forward_cam(img)   # r50
                    cam_w[:, :-1, :, :] = cam_w[:, :-1, :, :] * class_mask  # ignore activation of non-existing classes
                    conf_value, _ = cam_w.softmax(dim=1).max(dim=1)
                    pl_mask = conf_value.ge(args.p_cutoff).float()
                    pl_mask_fg = cam_w.softmax(dim=1)[:, :-1].max(dim=1).values.ge(args.p_cutoff).float()
                    avg_meter.add({'mask_ratio': pl_mask.mean().item(),
                                   'fg_mask_ratio':pl_mask_fg.mean().item(),
                                   })

                optimizer.zero_grad()
                loss_sup.backward()
                optimizer.step()
                
                # Logging 
                if (optimizer.global_step-1) % (args.log_freq * args.iter_size) == 0:
                    timer.update_progress(optimizer.global_step / max_step)
                    tscalar['train/lr'] = optimizer.param_groups[0]['lr']

                    if args.network_type == 'contrast':
                        print('Iter:%5d/%5d' % (iteration, args.max_iters),
                              'Loss_SUP:%.4f' % (avg_meter.get('loss_sup')),
                              'Loss_Cls:%.4f' % (avg_meter.get('loss_cls')),
                              'Loss_ER:%.4f' % (avg_meter.get('loss_er')),
                              'Loss_ECR:%.4f' % (avg_meter.get('loss_ecr')),
                              'Loss_NCE:%.4f' % (avg_meter.get('loss_nce')),
                              'Loss_Sal:%.4f' % (avg_meter.get('loss_sal')),
                              'mask_ratio:%.4f' % (avg_meter.get('mask_ratio')),
                              'fg_mask_ratio:%.4f' % (avg_meter.get('fg_mask_ratio')),
                              'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                              'ETA: %s' % (timer.get_est_remain()), 
                              flush=True)

                    ### wandb logging Scalars, Histograms, Images ###
                    if args.use_wandb:
                        wandb.log({'train/'+k: avg_meter.pop(k) for k in avg_meter.get_keys()}, step=iteration)
                        wandb.log({k: v for k, v in tscalar.items()}, step=iteration)
                    tscalar.clear()

                # Validate K times
                if (optimizer.global_step-1)!=0 and (optimizer.global_step-1) % (val_freq * args.iter_size) == 0:
                    model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                    torch.save(model.module.state_dict(), model_path)
                    print(f'Model {model_path} Saved.')
                    # Validation
                    validate_voc_ppc(args, model, val_dataloader, iteration, tag='val')
                
                timer.reset_stage()

        torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_last.pth'))


def train_ssl(train_dataloader, val_dataloader, model, optimizer, max_step, args):        
    torch.autograd.set_detect_anomaly(True)
    loader_iter = iter(train_dataloader)
    strong_transforms = tensor_augment_list(args.use_geom_augs)

    print(args)
    print('Using Geometric Transforms') if args.use_geom_augs else print("NOT using Geometric Transforms")
    print(f'Using {args.n_strong_augs} RandAugs.')

    # EMA
    if args.use_ema:
        print('EMA enabled')
        ema = EMA(model, args.ema_m)
        ema.register()
    else:
        print('EMA disabled')
        ema = None

    if args.bdry_lambda != 0.0:
        print(f"Boundary loss ratio: {args.bdry_lambda}")

    ### Train Scalars, Histograms, Images ###
    avg_meter = get_avg_meter(args=args)
    timer = pyutils.Timer("Session started: ")
    print("# of train data: ", len(train_dataloader.dataset))
    # Wandb logging
    if args.use_wandb:
        wandb.watch(model, log_freq=args.log_freq * args.iter_size)
    
    tscalar = {}
    val_freq = args.val_freq
    gamma = 0.10

    if args.val_only:
        print("Val-only mode.")
        if args.use_wandb:
            wandb.watch(model, log_freq=args.log_freq * args.iter_size)
        weight_path = glob(os.path.join(args.log_folder, '*.pth'))
        weight_path = nsort.natsorted(weight_path)
        for weight in weight_path:
            print(f'Loading {weight}')
            model.module.load_state_dict(torch.load(weight), strict=True)
            model.eval()
            tmp = weight[-10:]
            try:
                iteration = int(re.sub(r'[^0-9]', '', tmp))
            except:
                iteration = args.max_iters
            # Validation
            validate_voc_ppc(args, model, val_dataloader, iteration, tag='val')

    else:
        for iteration in range(args.max_iters):
            for _ in range(args.iter_size):
                if args.network_type == 'contrast':
                    try:
                        img_id, img_w, sal, img_s, randaug_ops, label = next(loader_iter)
                    except:
                        loader_iter = iter(train_dataloader)
                        img_id, img_w, sal, img_s, randaug_ops, label = next(loader_iter)

                    img_w = img_w.cuda(non_blocking=True)
                    img_s = img_s.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)
                    sal = sal.cuda(non_blocking=True)

                    img_w2 = F.interpolate(img_w, size=(128, 128), mode='bilinear', align_corners=True)
                    pred1, cam1, _, cam_rv1, feat1 = model(img_w)     # 112, 56, 56, 56
                    pred2, cam2, _, cam_rv2, feat2 = model(img_w2)    # 32, 16, 16, 16
                    loss_cls = F.multilabel_soft_margin_loss(pred1[:, :-1], label)
                    loss_cls2 = F.multilabel_soft_margin_loss(pred2[:, :-1], label)
                    bg_score = torch.ones((img_w.shape[0], 1)).cuda()
                    label_append_bg = torch.cat((label, bg_score), dim=1).unsqueeze(2).unsqueeze(3)  # (N, 21, 1, 1)
                    loss_cls_rv1 = adaptive_min_pooling_loss((cam_rv1 * label_append_bg)[:, :-1, :, :])
                    loss_cls_rv2 = adaptive_min_pooling_loss((cam_rv2 * label_append_bg)[:, :-1, :, :])
                    loss_cls = (loss_cls + loss_cls2) / 2. + (loss_cls_rv1 + loss_cls_rv2) / 2.
                    
                    # SEAM Losses
                    loss_er, loss_ecr = get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label_append_bg)
                    
                    # EPS+PPC Losses
                    loss_sal, _, _, _ = get_eps_loss(cam1, sal, label, args.tau, args.alpha, intermediate=True, num_class=args.num_classes)
                    loss_sal2, _, _, _ = get_eps_loss(cam2, sal, label, args.tau, args.alpha, intermediate=True, num_class=args.num_classes)
                    loss_sal_rv, _, _, _ = get_eps_loss(cam_rv1, sal, label, args.tau, args.alpha, intermediate=True, num_class=args.num_classes)
                    loss_sal_rv2, _, _, _ = get_eps_loss(cam_rv2, sal, label, args.tau, args.alpha, intermediate=True, num_class=args.num_classes)
                    loss_nce = get_contrast_loss(cam_rv1, cam_rv2, feat1, feat2, label, gamma=gamma, bg_thres=0.10)[0]
                    loss_sal = (loss_sal + loss_sal2) / 2. + (loss_sal_rv + loss_sal_rv2) / 2. 

                    loss_sup = loss_cls + loss_er + loss_ecr + loss_sal + loss_nce
                    avg_meter.add({
                        'loss_cls': loss_cls.item(),
                        'loss_er': loss_er.item(),
                        'loss_ecr': loss_ecr.item(),
                        'loss_nce': loss_nce.item(),
                        'loss_sal': loss_sal.item(),
                        'loss_sup': loss_sup.item()
                        })
                    
                elif args.network_type == 'eps':
                    try:
                        img_id, img_w, sal, img_s, randaug_ops, label = next(loader_iter)
                    except:
                        loader_iter = iter(train_dataloader)
                        img_id, img_w, sal, img_s, randaug_ops, label = next(loader_iter)
                    
                    img_w = img_w.cuda(non_blocking=True)
                    img_s = img_s.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)
                    sal = sal.cuda(non_blocking=True)

                    pred, cam = model(img_w)

                    # Classification loss
                    loss_cls = F.multilabel_soft_margin_loss(pred[:, :-1], label)
                    loss_sal, _, _, _ = get_eps_loss(cam, sal, label, args.tau, args.alpha, intermediate=True, num_class=args.num_classes)
                    loss_sup = loss_cls + loss_sal

                    avg_meter.add({
                        'loss_cls': loss_cls.item(),
                        'loss_sal': loss_sal.item(),
                        'loss_sup': loss_sup.item()
                        })

                #########################################################################
                ################################ Teacher ################################
                #########################################################################

                if ema is not None:
                    ema.apply_shadow()
                
                with torch.no_grad():
                    class_mask = label.unsqueeze(2).unsqueeze(3)
                    cam_w = model.module.forward_cam(img_w)
                    cam_w[:, :-1, :, :] = cam_w[:, :-1, :, :] * class_mask  # ignore activation of non-existing classes

                    conf_value, pl_w = cam_w.softmax(dim=1).max(dim=1)
                    pl_mask = conf_value.ge(args.p_cutoff).float()
                    avg_meter.add({'mask_ratio': pl_mask.mean().item()})
                    
                    pl_mask_fg = cam_w.softmax(dim=1)[:, :-1].max(dim=1).values.ge(args.p_cutoff).float() # PL 영역 중 foreground 영역
                    
                    avg_meter.add({'fg_mask_ratio':pl_mask_fg.mean().item()})

                    if args.bdry:
                        kernel = np.ones((args.bdry_size, args.bdry_size), np.int8)
                        pl_mask_bdry = np.zeros_like(pl_mask.cpu().numpy())
                        
                        for i in range(len(pl_mask)):
                            pl_mask_bdry[i] = cv2.dilate(pl_mask_fg[i].cpu().numpy(), kernel, iterations=1) - pl_mask_fg[i].cpu().numpy()
                        
                        pl_mask_bdry = torch.tensor(pl_mask_bdry).cuda()
                        pl_mask_bdry = torch.logical_and(pl_mask_bdry, conf_value.ge(args.bdry_cutoff)).float()
                        pl_mask_exp  = torch.logical_or(pl_mask, pl_mask_bdry).float()
                        avg_meter.add({'bdry_ratio': pl_mask_bdry.mean().item(),
                                       'exp_mask_ratio': pl_mask_exp.mean().item(),
                                       })

                if ema is not None:
                    ema.restore()

                ################################ Student ################################
                semcon_criterion = nn.CrossEntropyLoss(reduction='none')

                cam_s = model.module.forward_cam(img_s)
                cam_s[:, :-1, :, :] = cam_s[:, :-1, :, :] * class_mask

                # Hard Label CE Loss on PL region
                if args.bdry:
                    loss_semcon = (semcon_criterion(cam_s, pl_w) * pl_mask_exp).mean()
                else:
                    loss_semcon = (semcon_criterion(cam_s, pl_w) * pl_mask).mean()
                
                loss = loss_sup + loss_semcon 
                masks = get_masks_by_confidence(cam=cam_w)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if ema is not None:
                    ema.update()
                
                # Logging
                avg_meter.add({
                    'loss': loss.item(),
                    'loss_semcon': loss_semcon.item(),
                    'mask_1' : masks[0].mean().item(),
                    'mask_2' : masks[1].mean().item(),
                    'mask_3' : masks[2].mean().item(),
                    'mask_4' : masks[3].mean().item(),
                    'mask_5' : masks[4].mean().item(),
                    'mask_6' : masks[5].mean().item(),
                })          

                if (optimizer.global_step-1) % (args.log_freq * args.iter_size) == 0:
                    timer.update_progress(optimizer.global_step / max_step)
                    tscalar['train/lr'] = optimizer.param_groups[0]['lr']

                    if args.network_type == 'contrast':
                        print('Iter:%5d/%5d' % (iteration, args.max_iters), 
                            'Loss:%.4f' % (avg_meter.get('loss')),
                            'Loss_CLS:%.4f' % (avg_meter.get('loss_cls')),
                            'Loss_ER:%.4f' % (avg_meter.get('loss_er')),
                            'Loss_ECR:%.4f' % (avg_meter.get('loss_ecr')),
                            'Loss_NCE:%.4f' % (avg_meter.get('loss_nce')),
                            'Loss_Sal:%.4f' % (avg_meter.get('loss_sal')),
                            'Loss_SUP: %.4f' % (avg_meter.get('loss_sup')),
                            'Loss_SemCon: %.4f' % (avg_meter.get('loss_semcon')),
                            'conf_1:%.4f' % (avg_meter.get('mask_1')),
                            'conf_2:%.4f' % (avg_meter.get('mask_2')),
                            'conf_3:%.4f' % (avg_meter.get('mask_3')),
                            'conf_4:%.4f' % (avg_meter.get('mask_4')),
                            'conf_5:%.4f' % (avg_meter.get('mask_5')),
                            'conf_6:%.4f' % (avg_meter.get('mask_6')),
                            'mask_ratio:%.4f' % (avg_meter.get('mask_ratio')),
                            'exp_mask_ratio:%.4f' % (avg_meter.get('exp_mask_ratio')),
                            'fg_mask_ratio:%.4f' % (avg_meter.get('fg_mask_ratio')),
                            'bdry_ratio:%.4f' % (avg_meter.get('bdry_ratio')),
                            'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                            'ETA: %s' % (timer.get_est_remain()), 
                            flush=True)
                        
                    elif args.network_type == 'eps':
                        print('Iter:%5d/%5d' % (iteration, args.max_iters), 
                            'Loss:%.4f' % (avg_meter.get('loss')),
                            'Loss_CLS:%.4f' % (avg_meter.get('loss_cls')),
                            'Loss_Sal:%.4f' % (avg_meter.get('loss_sal')),
                            'Loss_SUP: %.4f' % (avg_meter.get('loss_sup')),
                            'Loss_SemCon: %.4f' % (avg_meter.get('loss_semcon')),
                            'Loss_Bdry: %.4f' % (avg_meter.get('loss_bdry')),
                            'conf_1:%.4f' % (avg_meter.get('mask_1')),
                            'conf_2:%.4f' % (avg_meter.get('mask_2')),
                            'conf_3:%.4f' % (avg_meter.get('mask_3')),
                            'conf_4:%.4f' % (avg_meter.get('mask_4')),
                            'conf_5:%.4f' % (avg_meter.get('mask_5')),
                            'conf_6:%.4f' % (avg_meter.get('mask_6')),
                            'mask_ratio:%.4f' % (avg_meter.get('mask_ratio')),
                            'exp_mask_ratio:%.4f' % (avg_meter.get('exp_mask_ratio')),
                            'fg_mask_ratio:%.4f' % (avg_meter.get('fg_mask_ratio')),
                            'bdry_ratio:%.4f' % (avg_meter.get('bdry_ratio')),
                            'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                            'ETA: %s' % (timer.get_est_remain()), 
                            flush=True)
                        
                    ### wandb logging Scalars, Histograms, Images ###
                    if args.use_wandb:
                        wandb.log({'train/'+k: avg_meter.pop(k) for k in avg_meter.get_keys()}, step=iteration)
                        wandb.log({k: v for k, v in tscalar.items()}, step=iteration)
                    tscalar.clear()
                    
                # Validate K times
                if (optimizer.global_step-1)!=0 and (optimizer.global_step-1) % (val_freq * args.iter_size) == 0:
                    # Save intermediate model
                    model_path = os.path.join(args.log_folder, f'checkpoint_{iteration}.pth')
                    torch.save(model.module.state_dict(), model_path)
                    print(f'Model {model_path} Saved.')
                    validate_voc_ppc(args, model, val_dataloader, iteration, tag='val')

                timer.reset_stage()
                
        torch.save(model.module.state_dict(), os.path.join(args.log_folder, 'checkpoint_last.pth'))