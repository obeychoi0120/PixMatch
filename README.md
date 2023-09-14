# PixMatch
The Pytorch implementation of _PixMatch for Weakly-Supervised Semantic Segmentation_, AAAI 2024, Under review.

![pixmatch](https://github.com/obeychoi0120/PixMatch/assets/75653891/37d52fe8-21e4-4ef7-97e6-57bd9a0d73e4)

## Abstract
Weakly-supervised Semantic Segmentation (WSSS) aims to provide a precise pixel-level classification results without pixel-level dense annotations. Image-level WSSS mainly relys on Class Activation Maps (CAMs) for object localization due to the absence of pixel-level annotations. However, CAMs tends to focus on narrow discriminative regions, and fails to cover entire object region. It is caused by the supervision gap between classification task and segmentation task. Existing WSSS methods have tried to boost feature representation learning or impose consistency regularization, but explicit methods to supervise CAMs were not considered. To tackle this issue, we propose a PixMatch framework which provides explicit supervision of non-discriminative area, encouraging model to learn various features of various objects. Specifically, we use strong perturbation to make challenging inference target, and focus on constructing confident pixel-wise supervision signal to supervise prediction of nondiscriminative regions. We use confident-based thresholding and simple boundary expansion strategy to encourage reliable prediction are used for pixel-level label propagation and enhance learning with reliable pixel-wise predictions. Extensive experiments on the PASCAL VOC 2012 show that our method boosts initial seed quality and segmentation performance by large margin, achieving new state-of-the-art performance.

## Environment
- Python 3.8.X
- Pytorch 1.10.0
- Torchvision 0.11.0
- Numpy 1.23.0
- pydensecrf https://github.com/lucasb-eyer/pydensecrf
- opencv-python
- wandb (for logging)
