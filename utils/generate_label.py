import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from typing import Union
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import cv2
import math 
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def build_semantic_segmentation_target(
    img: Tensor,  # HxW
    fire_thres_temp: float,
    upper_fire_thres_temp: float,
    not_fire_label: int = 0, 
    maybe_fire_label: int = 125,
    is_fire_label: int = 255,
) -> Tensor:
    """
    3-class segmentation task annotation, labels

    not_fire: (-inf, fire_thres_temp]

    maybe_fire: (fire_thres_temp, upper_fire_thres_temp)

    is_fire: [upper_fire_thres_temp, +inf)
    """
    tgt = img.new_full(img.shape, fill_value=not_fire_label, dtype=torch.uint8)
    tgt[img > fire_thres_temp] = maybe_fire_label
    tgt[img >= upper_fire_thres_temp] = is_fire_label
    return tgt


def dilation(x, ks=3, n_iter=5):
    x = x[None, None, ...]
    # max pooling
    for _ in range(n_iter):
        x = F.max_pool2d(x, ks, 1, int(ks / 2))
    return x[0, 0]


def closing(x, ks=3, n_iter=5):
    x = x[None, None, ...]
    # max pooling
    for _ in range(n_iter):
        x = F.max_pool2d(x, ks, 1, int(ks / 2))
    # min pooling
    x = -x
    for _ in range(n_iter):
        x = F.max_pool2d(x, ks, 1, int(ks / 2))
    return -x[0, 0]


def amb_temp_aug(
    img: Tensor,
    amb_temp: float,
    max_sun_temp_inc: float,
    new_amb_temp: float,
    alpha: float = .5,  # smooth transitions!
):
    # The difference between the new amb and the original amb which is 9 °C
    delta_amb = new_amb_temp - amb_temp
    new_img = img.clone()
    
    # fire threshold -> any temp greater than amb temp + max_sun (15 °C) is considered fire
    fire_thres_temp = amb_temp + max_sun_temp_inc
    fire_mask = img >= fire_thres_temp

    import matplotlib.pyplot as plt
    # plt.imshow(fire_mask)
    # plt.show()

    fire_mask = TF.gaussian_blur(fire_mask[None, None, ...].float(), 3)[0, 0]
    
    amb_mask = 1.0 - fire_mask
    y = new_img

    ########################################
    
    #w/o fire augmentation
    w = F.sigmoid(alpha * y) - F.sigmoid(alpha * (y - fire_thres_temp))
    x = w * delta_amb * amb_mask + y
    
    w2 = 4.8
    x = (w2 * fire_mask * x) + x
    
    # #with fire augmentation
    # w = F.sigmoid(alpha * y)
    # x = (w * delta_amb * amb_mask + y) 
    
    #########################################

    new_img = x

    if delta_amb > 0.0:
        fire_mask = img >= fire_thres_temp
        kernel_size = 3
        fm = fire_mask[None, None, :, :].float()  # 1x1xHxW

        
        larger_fire_mask = F.max_pool2d(fm, kernel_size, 1, int(kernel_size / 2))
        

        
        outer_fire_mask = larger_fire_mask - fm
        outer_fire_mask = outer_fire_mask[0, 0, :, :] > 0.0  # HxW


        avg_img = F.avg_pool2d(new_img[None, None, ...], kernel_size, 1, int(kernel_size / 2))[0, 0]
        outer_avg_img = dilation(new_img * outer_fire_mask)
        
        
        larger_fire_mask = larger_fire_mask[0, 0] > 0.0
        outer_avg_img = closing(new_img * outer_fire_mask)
        valley_mask = F.relu(outer_avg_img - (avg_img * larger_fire_mask)) > 0.0


        # fill valleys
        mask = torch.logical_and(valley_mask, larger_fire_mask)
        
        new_img[mask] = outer_avg_img[mask]
    else:
        fire_mask = img >= fire_thres_temp

    return new_img


def load_image_pil(path: str, grayscale: bool = True) -> torch.Tensor:
    """Load image with PIL, convert to grayscale if needed, and return a torch tensor."""
    img = Image.open(path)
    # if grayscale:
    #     img = img.convert('L')  # grayscale
    arr = np.array(img)  # HxW or HxWxC
    return torch.from_numpy(arr)


def to_tensor(image):
    return torch.from_numpy(image)  # dtype: uint8


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("datadir", type=Path, help="Directory to .TIF files")
    # parser.add_argument("targetdir", type=Path, help="Directory to save new targets")
    # parser.add_argument("--newdatadir", type=Path, default=None, help="Directory to save new .TIF files")
    # parser.add_argument("--amb_temp", type=float, default=9, help="Ambient temp. of original data")
    # parser.add_argument("--max_sun_temp_inc", type=float, default=15, help="How much hotter can bio mass be under sunlight")
    # parser.add_argument("--new_amb_temp", type=float, default=20, help="The new ambient temp. to simulate new data")
    # parser.add_argument("--max_amb_temp", type=float, default=30, help="Clip increase by new ambient temp. here")
    # parser.add_argument("--upper_fire_thres_temp", type=float, default=60, help="Above this temp. it is fire for sure")
    # parser.add_argument("--num_workers", type=int, default=4)
    # args = parser.parse_args()

    DIR = r'd:\Research\Wild Fire - Project\Data\Fire Images'
    mode = False #True means generate label, False means augment temp
    
    
    env_temp_folders = os.listdir(DIR)
    
    if mode:
    
        for env_temp_folder in tqdm(env_temp_folders, desc="Processing folders"):
            sub_folders = os.listdir(os.path.join(DIR, env_temp_folder))
            for sub_folder in sub_folders:
                GT_image = os.path.join(DIR, env_temp_folder, sub_folder, 'GT.tiff')
                corrected_image = load_image_pil(GT_image) + 0  
                
                new_amb_temp = int(env_temp_folder)
                # new_amb_temp = int(15)
                
                max_sun_temp_inc = 15
                upper_fire_thres_temp = 50
                new_fire_thres_temp = new_amb_temp + max_sun_temp_inc
                tgt = build_semantic_segmentation_target(
                    ((corrected_image) - 273.15), new_fire_thres_temp, upper_fire_thres_temp
                )
                Image.fromarray(tgt.cpu().numpy()).save(os.path.join(DIR, env_temp_folder, sub_folder, 'label_2.png'))
    else:
        
        for i in range(31):
            
            GT_image = os.path.join(r'd:\Research\Wild Fire - Project\Evaluation Metric\augmented_image\input_image\0000092.TIF')
            corrected_image = load_image_pil(GT_image)
            augmented_img = amb_temp_aug(corrected_image, amb_temp=9, max_sun_temp_inc=15, new_amb_temp=i)
            Image.fromarray(augmented_img.cpu().numpy()).save(os.path.join(
                                                                           r'd:\Research\Wild Fire - Project\Evaluation Metric\augmented_image\augmented_imgs_updated', 
                                                                           f'{str(i)}.tiff'))

        
  
