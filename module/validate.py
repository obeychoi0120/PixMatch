import torch
import numpy as np
from torch.nn import functional as F
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from data.dataset import get_categories
import cv2
from util import pyutils
import wandb
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm

def combine_img_cam(img, cam, p=0.5): # img: (3, H, W), cam: (H, W)
    heatmap = cm.jet(cam)[:,:,:-1]
    heatmap = heatmap.transpose(2,0,1)
    img = img / np.max(img)
    mix_img = p * img + (1-p) * heatmap
    # mix_img /= np.max(mix_img)
    return mix_img

def validate_voc_ppc(args, model, data_loader, iter, tag='val'):    
    timg = {}
    heatmap = {}
    heatmap_all = {}
    idx2class = get_categories(args.num_classes, bg_last=False, get_dict=True)
    gt_dataset = VOCSemanticSegmentationDataset(split='train', data_dir=args.data_root+'/../')
    labels = [gt_dataset.get_example_by_keys(i, (1,))[0] for i in range(len(gt_dataset))]
    model.eval()
    with torch.no_grad():
        hmaps = []
        hmaps_all = []
        preds = []
        confs = []

        conf_px = 0
        crt = 0
        conf_fg_px = 0
        crt_fg = 0
        bdry_px = 0
        crt_bdry = 0    
        exp_px = 0
        crt_exp = 0
        tot_px = 0
    
        for i, (img_id, img, label) in tqdm(enumerate(data_loader)):
            img = img.cuda()    # 1, 3, H, W
            label = label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)    # 1, 20, 1, 1
            gt = labels[i]
            valid_mask = (gt>=0)

            # Softmax -> Interpolation -> label filtering
            logit = model.module.forward_cam(img)
            logit_ = logit.clone()
            logit = logit.softmax(dim=1)                                # 1, 21, H', W'
            logit = F.interpolate(logit, gt.shape, mode='bilinear', align_corners=False)
            logit[:, :-1, :, :] = logit[:, :-1, :, :] * label
            max_probs, pred = logit[0].max(dim=0)
            max_probs, pred = max_probs.cpu().numpy(), pred.cpu().numpy()                        # H, W
            max_probs_fg = logit[0, :-1].max(dim=0).values.cpu().numpy()
            pred += 1              
            pred[pred==args.num_classes] = 0

            #bdry
            logit_[:, :-1, :, :] = logit_[:, :-1, :, :] * label
            pl_mask_fg = logit_.softmax(dim=1)[0, :-1].max(dim=0).values.ge(args.p_cutoff).float()
            kernel = np.ones((args.bdry_size, args.bdry_size), np.int8)
            bdry_mask = torch.tensor(cv2.dilate(pl_mask_fg.cpu().numpy(), kernel, iterations=1) - pl_mask_fg.cpu().numpy())
            bdry_mask = F.interpolate(bdry_mask.unsqueeze(0).unsqueeze(0), gt.shape, mode='nearest').squeeze().numpy()

            # Minmaxnorm
            cam = logit[0].clone().cpu().numpy()    # 21, H, W
            cam = cam / (np.max(cam, (1, 2), keepdims=True) + 1e-5) # class-wise (+bg) normalize
            hmap = combine_img_cam(img[0].cpu().numpy(), cam[:-1].sum(axis=0))
            hmap_all = combine_img_cam(img[0].cpu().numpy(), np.maximum(cam[:-1].sum(axis=0), cam[-1]))
            
            # logging confidents
            max_probs = logit[0].max(dim=0).values.cpu().numpy() # H, W
            max_probs_fg = logit[0, :-1].max(dim=0).values.cpu().numpy()                   
            
            conf_mask = (max_probs >= args.p_cutoff) * valid_mask
            conf_mask_fg = (max_probs_fg >= args.p_cutoff) * valid_mask
            
            conf_pred, conf_pred_fg = pred[conf_mask==1], pred[conf_mask_fg==1]             
            conf_gt, conf_gt_fg = gt[conf_mask==1], gt[conf_mask_fg==1]                    

            bdry_mask = bdry_mask * valid_mask
            bdry_gt, bdry_pred = gt[bdry_mask==1], pred[bdry_mask==1]
            
            exp_mask = np.maximum(conf_mask, bdry_mask)
            exp_gt, exp_pred = gt[exp_mask==1], pred[exp_mask==1]


            assert len(conf_pred) == len(conf_gt) and len(conf_pred_fg) == len(conf_gt_fg)
            
            conf_px += conf_mask.sum()                  # pixels with confident prediction
            crt += (conf_gt == conf_pred).sum()       
            
            conf_fg_px += conf_mask_fg.sum()            # pixels with confident prediction of FG
            crt_fg += (conf_gt_fg == conf_pred_fg).sum()
            
            bdry_px += bdry_mask.sum()
            crt_bdry += (bdry_gt == bdry_pred).sum()

            exp_px += exp_mask.sum()                    # pixels with expanded confidence mask
            crt_exp += (exp_gt == exp_pred).sum()

            tot_px += len(gt[gt>=0])

            preds.append(pred)
            confs.append((max_probs >= args.p_cutoff).astype(int))
            hmaps.append(hmap)
            hmaps_all.append(hmap_all)

        # Calculate confident pixels
        conf_acc = crt / (conf_px + 1e-10)        
        conf_fg_acc = crt_fg / (conf_fg_px + 1e-10)      
        bdry_acc = crt_bdry / (bdry_px + 1e-10)
        exp_acc = crt_exp / (exp_px + 1e-10)

        bdry_ratio = bdry_px / tot_px
        conf_ratio = conf_px / tot_px                   # ratio of confident pixel / all prediction
        conf_fg_ratio = conf_fg_px / tot_px             # ratio of confident fg pixel / all fg prediction
        exp_ratio = exp_px / tot_px

        # Calculate Metrics
        confusion = calc_semantic_segmentation_confusion(preds, labels)
        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        acc = gtjresj.sum() / confusion.sum()

        denominator = gtj + resj - gtjresj
        iou = gtjresj / denominator
        gtjresj.sum() / denominator.sum()

        fp = resj - gtjresj
        fn = gtj - gtjresj
        fpr_fg = fp[1:].sum() / denominator.sum()
        fpr_bg = fp[0].sum()  / denominator.sum()
        fnr    = fn[1:].sum() / denominator.sum()

        precision = gtjresj / (gtj + 1e-10)
        recall = gtjresj / (resj + 1e-10)

        # Logging Values
        log_hist= {'iou': iou, 'precision': precision, 'recall': recall}
        log_scalar = {
            'miou': np.nanmean(iou),
            'mprecision' : np.nanmean(precision),
            'mrecall'    : np.nanmean(recall),
            'accuracy'   : acc,
            'mFPR_fg'    : fpr_fg,
            'mFPR_bg'    : fpr_bg,
            'mFNR'       : fnr,
            'PL_acc'     : conf_acc,
            'Exp_PL_acc' : exp_acc,
            'PL_fg_acc'  : conf_fg_acc,
            'Bdry_acc'   : bdry_acc,
            'PL_ratio'   : conf_ratio,
            'Exp_PL_ratio': exp_ratio,
            'PL_fg_ratio': conf_fg_ratio,
            'Bdry_ratio' : bdry_ratio
            }
        
        print(f"mIoU       : {log_scalar['miou'] * 100:.2f}%")
        print(f"mPrec      : {log_scalar['mprecision'] * 100:.2f}%")
        print(f"mRecall    : {log_scalar['mrecall'] * 100:.2f}%")
        print(f"Accuracy   : {log_scalar['accuracy'] * 100:.2f}%")
        print(f"FPR(fg)    : {log_scalar['mFPR_fg'] * 100:.2f}%")
        print(f"FPR(bg)    : {log_scalar['mFPR_bg'] * 100:.2f}%")
        print(f"FNR(fg)    : {log_scalar['mFNR'] * 100:.2f}%")
        print('')
        print(f"PL_acc     : {log_scalar['PL_acc'] * 100:.2f}%")
        print(f"Exp_PL_acc : {log_scalar['Exp_PL_acc'] * 100:.2f}%")
        print(f"PL_fg_acc  : {log_scalar['PL_fg_acc'] * 100:.2f}%")
        print(f"Bdry_acc   : {log_scalar['Bdry_acc'] * 100:.2f}%")
        print('')
        print(f"PL_ratio   : {log_scalar['PL_ratio'] * 100:.2f}%")
        print(f"Exp_PL_ratio : {log_scalar['Exp_PL_ratio'] * 100:.2f}%")
        print(f"PL_fg_ratio: {log_scalar['PL_fg_ratio'] * 100:.2f}%")  # pred as fg
        print(f"Bdry_ratio : {log_scalar['Bdry_ratio'] * 100:.2f}%")  # pred as fg
        print('')
        print('IoU (%)    :', ' '.join([f'{v*100:0>4.1f}' for v in iou]))
        print('Prec (%)   :', ' '.join([f'{v*100:0>4.1f}' for v in precision]))
        print('Recall (%) :', ' '.join([f'{v*100:0>4.1f}' for v in recall])) 

        if args.use_wandb:
            ### Logging Images
            N_val = 50
            for i, (hmap, hmap_all, pred, conf, (img, _)) in enumerate(zip(hmaps[:N_val], hmaps_all[:N_val], preds[:N_val], confs[:N_val], gt_dataset)):
                timg[gt_dataset.ids[i]] = wandb.Image(np.transpose(img, axes=(1,2,0)),
                                                        masks={
                                                            'Pred': {
                                                                'mask_data'   : pred,
                                                                'class_labels': idx2class
                                                            },
                                                            'Ground_truth': {
                                                                'mask_data': np.where(labels[i]==-1, 255, labels[i]),
                                                                'class_labels': idx2class
                                                            },
                                                            'Confident': {
                                                                'mask_data': conf,
                                                                'class_labels': {0: 'unconf', 1: 'conf'},
                                                            }
                                                            })
                heatmap[gt_dataset.ids[i]] = wandb.Image(np.transpose(hmap, axes=(1,2,0)))
                heatmap_all[gt_dataset.ids[i]] = wandb.Image(np.transpose(hmap_all, axes=(1,2,0)))

            # Wandb logging
            wandb.log({tag+'/'+k: v for k, v in log_scalar.items()}, step=iter)
            wandb.log({tag+'/'+k: wandb.Histogram(v) for k, v in log_hist.items()}, step=iter)
            wandb.log({'img/'+k: img for k, img in timg.items()}, step=iter)
            wandb.log({'hmap/'+k: hmap for k, hmap in heatmap.items()}, step=iter)
            wandb.log({'hmap_all/'+k: hmap_all for k, hmap_all in heatmap_all.items()}, step=iter)

    model.train()

