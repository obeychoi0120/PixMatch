import random
import os.path
import PIL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchvision import transforms
import torchvision.transforms.functional as vision_tf
from util.imutils import RandomResizeLong, random_crop_with_saliency, HWC_to_CHW
from data.augmentation.randaugment import RandAugment

def get_categories(num_classes=None, bg_last=False, get_dict=False):
    # VOC
    if num_classes == 21:
        categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    # COCO
    elif num_classes == 81:
        categories =  ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                       'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                       'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                       'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                       'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
                       'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                       'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                       'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']   
    if bg_last:
        categories.pop(0)
        categories.append('background')
    if get_dict:
        return {i:c for i, c in enumerate(categories)}
    else:
        return categories


def load_img_id_list(img_id_file):
    return open(img_id_file).read().splitlines()


def load_img_label_list_from_npy(img_name_list, dataset):
    cls_labels_dict = np.load(f'data/{dataset}/cls_labels.npy', allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_saliency_path(img_name, saliency_root='SALImages'):
    return os.path.join(saliency_root, img_name + '.png')


class ImageDataset(Dataset):
    """
    Base image dataset. This returns 'img_id' and 'image'
    Performs Weak or Strong Augmentations.
    """
    def __init__(self, dataset, img_id_list_file, img_root, tv_transform=None,
                 crop_size=448, resize_size=(448, 768), 
                 aug_type=None, use_geom_augs=False, n_strong_augs=5, patch_k=None):
        self.dataset = dataset
        self.img_id_list = load_img_id_list(img_id_list_file)
        self.img_root = img_root

        # Dataset use (1).self.transform(Torchvision Transformations)
        self.tv_transform = tv_transform
        # or (2).Weak & Strong Augmentations
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.aug_type = aug_type
        self.patch_k = patch_k
        self.use_geom_augs = use_geom_augs

        self.resizelong = RandomResizeLong(resize_size[0], resize_size[1])
        self.colorjitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        self.img_normal = TorchvisionNormalize()

        if self.aug_type == 'strong': ###
            blur_kernel_size = int(random.random() * 4.95)
            blur_kernel_size = blur_kernel_size + 1 if blur_kernel_size % 2 == 0 else blur_kernel_size
            self.blur = transforms.GaussianBlur(blur_kernel_size, sigma=(0.1, 2.0)) # non-geometric transformations
            self.randaug = RandAugment(self.use_geom_augs, n_strong_augs, 5)

    def __len__(self):
        return len(self.img_id_list)

    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]
        img = PIL.Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB")

        # Use torchvision Transforms
        if self.tv_transform:
            return img_id, self.tv_transform(img)

        # Use weak | strong augmentations
        else:
            ### Image 1. Weak Augmentation ###
            img_w, (weak_target_long, weak_hflip, weak_box) = self.__apply_transform(img,
                                                                                    get_transform=True, 
                                                                                    strong=False, 
                                                                                    target_long=None,
                                                                                    crop_size=self.crop_size, 
                                                                                    hflip=None, 
                                                                                    box=None)
            img_w = self.__totensor(img_w)    # (448, 448)
            if not self.aug_type:
                return img_id, img_w

            ### Image 2. Strong augmetation (for consistency regularization) ###
            elif self.aug_type == 'strong':
                img_s, _, ra_ops = self.__apply_transform(img,
                                                                get_transform=True,
                                                                strong=True,
                                                                target_long=weak_target_long,
                                                                crop_size=self.crop_size, 
                                                                hflip=weak_hflip,
                                                                box=weak_box,
                                                                patch_k = self.patch_k
                                                                ) 

                img_s = self.__totensor(img_s)
                return img_id, img_w, img_s, ra_ops

            else:
                raise Exception('No appropriate Augmentation type')
    
    def __apply_transform(self, img, get_transform=False, strong=False, target_long=None, crop_size=None, hflip=None, box=None, patch_k=None):
        # randomly resize
        if target_long is None:
            target_long = random.randint(self.resize_size[0], self.resize_size[1])
        img = self.resizelong(img, target_long)

        # Randomly flip
        if hflip is None:
            hflip = random.random() > 0.5
        if hflip == True:
            img = vision_tf.hflip(img)
        
        # Colorjitter
        img = self.colorjitter(img)

        # Random Crop
        img = np.asarray(img)
        img, _, tr_box = random_crop_with_saliency(imgarr=img, 
                                                   sal=None,
                                                   crop_size=crop_size,
                                                   get_box=True, 
                                                   box=box)

        # Strong Augmentation
        if strong:
            img = Image.fromarray(img)
            img = self.blur(img)
            img, ra_ops = self.randaug(img)
            img = np.asarray(img)

        # normalize
        img = self.img_normal(img)

        if get_transform: ###
            if strong:
                return img, (target_long, hflip, tr_box), ra_ops
            else:
                return img, (target_long, hflip, tr_box)
        else:
            return img
    
    def __totensor(self, img):
        # Image
        img = HWC_to_CHW(img)
        img = torch.from_numpy(img)

        return img

class ClassificationDataset(ImageDataset):
    """
    for SEAM, SIPE
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_list = load_img_label_list_from_npy(self.img_id_list, self.dataset)

    def __getitem__(self, idx):
        label = torch.from_numpy(self.label_list[idx])
        return super().__getitem__(idx) + (label,)


class ClassificationDatasetWithSaliency(ImageDataset):
    """
    for EPS, PPC
    """
    def __init__(self, saliency_root=None, **kwargs):
        super().__init__(**kwargs)
        # self.tv_transform is useless in ClassificationDatasetWithSaliency
        self.saliency_root = saliency_root
        self.label_list = load_img_label_list_from_npy(self.img_id_list, self.dataset)

    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]
        img = PIL.Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB")
        saliency = PIL.Image.open(get_saliency_path(img_id, self.saliency_root)).convert("RGB")

        label = torch.from_numpy(self.label_list[idx])

        ### Image 1 ###
        img_w, saliency1, (weak_target_long, weak_hflip, weak_box) = self.__apply_transform_with_sal(img, 
                                                                                                    saliency, 
                                                                                                    get_transform=True,
                                                                                                    strong=False,
                                                                                                    target_long=None,
                                                                                                    crop_size=self.crop_size,
                                                                                                    hflip=None,
                                                                                                    box=None)
        
        img_w, saliency1 = self.__totensor(img_w, saliency1)
        if not self.aug_type:
            return img_id, img_w, saliency1, label

        ### Image 2: Strong augmetation (for MT, FixMatch)
        elif self.aug_type == 'strong':
            ### TODO: mask transform, return aug information
            img_s, saliency2, tr_ops, ra_ops = self.__apply_transform_with_sal(img, 
                                                                                  saliency, 
                                                                                  get_transform=True, 
                                                                                  strong=True, 
                                                                                  target_long=weak_target_long,
                                                                                  crop_size=self.crop_size,
                                                                                  hflip=weak_hflip,
                                                                                  box=weak_box,
                                                                                  patch_k=self.patch_k
                                                                                  )
            # pdb.set_trace()
            img_s = self.__totensor(img_s)
            return img_id, img_w, saliency1, img_s, ra_ops, label
        
        else:
            raise Exception('No appropriate Augmentation type')

    def __apply_transform_with_sal(self, img, sal, get_transform=False, strong=False, target_long=None, crop_size=None, hflip=None, box=None, patch_k=None):
        # Randomly resize
        if target_long is None:
            target_long = random.randint(self.resize_size[0], self.resize_size[1])
        img = self.resizelong(img, target_long)
        sal = self.resizelong(sal, target_long)

        # Randomly flip
        if hflip is None:
            hflip = random.random() > 0.5
        if hflip == True:
            img = vision_tf.hflip(img)
            sal = vision_tf.hflip(sal)
        
        # Color jitter
        img = self.colorjitter(img)

        # Random Crop
        img = np.asarray(img)   # H, W, 3
        sal = np.asarray(sal)   # H, W, 3
        img, sal, tr_box = random_crop_with_saliency(imgarr=img,
                                                    sal=sal,
                                                    crop_size=crop_size,
                                                    get_box=True, 
                                                    box=box)

        # Strong Augmentation
        if strong:
            img = Image.fromarray(img)
            img = self.blur(img)
            img, ra_ops = self.randaug(img)
            img = np.asarray(img)
            
        # Normalize
        img = self.img_normal(img)
        sal = sal / 255.

        if get_transform:
            if strong:
                return img, sal, (target_long, hflip, tr_box), ra_ops
            else:
                return img, sal, (target_long, hflip, tr_box)
        else:
            return img, sal
    
    def __totensor(self, img, mask=None):
        # Permute to C, H, W
        img = HWC_to_CHW(img)

        # Make torch tensor
        img = torch.from_numpy(img)

        if mask is not None:
            mask = HWC_to_CHW(mask)
            mask = torch.from_numpy(mask)
            mask = torch.mean(mask, dim=0, keepdim=True)
            return img, mask
        else:
            return img
    

class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img