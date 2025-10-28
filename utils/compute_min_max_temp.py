from pathlib import Path
import torch
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class config:
    dataset_path: str = r'e:\Simulated_2D_grid\Augmented_fire_simulation'
    simulated: bool = True 
 

if __name__ == '__main__':
    
    conf = config()

    # # 2D subset stats.
    DIR = conf.dataset_path

    env_temp_folders = os.listdir(DIR)
    overlall_min, overall_max = [], []
    for env_temp_folder in tqdm(env_temp_folders, desc="Processing folders"):
        sub_folders = os.listdir(os.path.join(DIR, env_temp_folder))
        for sub_folder in sub_folders:
            for i in range(1):
                f = open(os.path.join(DIR, env_temp_folder, sub_folder, 'GT_pose_0_thermal.pngmin_max_temp.txt'))
                min_temp, max_temp = map(float, f.read().split(','))
                overlall_min.append(min_temp)
                overall_max.append(max_temp)
    
    print(min(overlall_min), max(overall_max))
                            
