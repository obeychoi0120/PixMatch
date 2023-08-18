from re import I
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import ImageDataset, ClassificationDataset, ClassificationDatasetWithSaliency, TorchvisionNormalize
import os.path as osp

from util import imutils

def get_dataloader(args):

    if args.mode != 'base':
        ssl_params = {
            'aug_type'      : args.aug_type, 
            'n_strong_augs' : args.n_strong_augs,
            'use_geom_augs' : args.use_geom_augs
            }
    else:
        ssl_params = {}
    
    # VOC12 dataset
    if args.dataset == 'voc12':
        if args.network_type in ['cls38', 'cls50', 'seam']:
            train_dataset = ClassificationDataset(
                dataset             = args.dataset,
                img_id_list_file    = args.train_list,
                img_root            = args.data_root,
                crop_size           = args.crop_size,
                resize_size         = args.resize_size,
                **ssl_params
            )
            
        elif args.network_type in ['eps', 'contrast']:
            train_dataset = ClassificationDatasetWithSaliency(
                dataset             = args.dataset,
                img_id_list_file    = args.train_list,
                img_root            = args.data_root,
                saliency_root       = args.saliency_root,
                crop_size           = args.crop_size,
                resize_size         = args.resize_size,
                **ssl_params
            )
        else:
            raise Exception("No appropriate train type")
        
        val_dataset = ClassificationDataset(
            dataset             = args.dataset,
            img_id_list_file    = args.val_list,
            img_root            = args.data_root,
            tv_transform        = transforms.Compose([
                                    np.asarray,
                                    TorchvisionNormalize(),
                                    imutils.HWC_to_CHW,
                                    torch.from_numpy
                                    ])
                )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True, 
            drop_last=True
        )

        # Currently avaliable batch size 1
        val_loader = DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False,
            num_workers=args.num_workers, 
            pin_memory=True, 
            drop_last=True
            )

    return train_loader, val_loader
