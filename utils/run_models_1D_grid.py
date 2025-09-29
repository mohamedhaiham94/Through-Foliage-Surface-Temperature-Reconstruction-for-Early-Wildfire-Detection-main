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


def load_model(checkpoint: Path):
    """Load a traced model (gpu)."""
    model = torch.jit.load(checkpoint)
    return model

def load_image(uploaded_file, simulated) -> torch.Tensor:
    if simulated:
        return torch.from_numpy(
            np.array(Image.open(uploaded_file).convert('L'))
        )  # uint8; HxW
    else:
        return torch.from_numpy(
            np.array(Image.open(uploaded_file))
        )  # uint8; HxW


def to_Kelvin(img: torch.Tensor, min_temp, max_temp) -> torch.Tensor:
    return ((max_temp - min_temp) * (img / 255.0)) + min_temp  # in Kelvin

@dataclass
class config:
    model_path: str = r'D:\Research\Wild Fire - Project\Evaluation Metric\large\2025-08-11_11-05-38-1D(11x1)\checkpoints\scripted_model.pt'
    dataset_path: str = r'D:\Research\Wild Fire - Project\Evaluation Metric\real_data\sparse_data\1D_grid\integral\integrals'
    simulated: bool = True 
    mean: float = 293.90083216803276
    std: float = 11.017465352217659

if __name__ == '__main__':
    
    conf = config()

    model = load_model(conf.model_path)
    
    # # 2D subset stats.
    aos_mean = conf.mean
    aos_std = conf.std
    DIR = conf.dataset_path

    env_temp_folders = os.listdir(DIR)

    for env_temp_folder in tqdm(env_temp_folders, desc="Processing folders"):
        sub_folders = os.listdir(os.path.join(DIR, env_temp_folder))
        for sub_folder in sub_folders:
            f = open(os.path.join(DIR, env_temp_folder, sub_folder, 'global_min_max_temp.txt'))
            min_temp, max_temp = map(float, f.read().split(','))
            
            if conf.simulated:
                normalized_img_aos = to_Kelvin(load_image(os.path.join(DIR, env_temp_folder, sub_folder, 'integrall_normalized_0.png'), conf.simulated), min_temp, max_temp)
            else:
                normalized_img_aos = load_image(os.path.join(DIR, env_temp_folder, sub_folder, 'Layer_1.tiff'), conf.simulated) + 273.15
                            
            img1_norm = (normalized_img_aos - aos_mean) / aos_std
                    
            # env_temp = torch.tensor([int(env_temp_folder)]).to(torch.int64).cuda()
            env_temp = torch.tensor([int(12)]).to(torch.int64).cuda()

            # Model prediction
            with torch.inference_mode():
                img2_norm: torch.Tensor = model(
                    img1_norm[None, None, :, :].cuda(),
                    env_temp,
                    None,
                    True,
                )
                img2_k = aos_std * img2_norm.cpu() + aos_mean
                
                if conf.simulated:                    
                    Image.fromarray(img2_k[0, 0].cpu().numpy(), mode='F').save(os.path.join(DIR, env_temp_folder, sub_folder, f'corrected.tiff'))
                else:
                    Image.fromarray(img2_k[0, 0].cpu().numpy() - 273.15, mode='F').save(os.path.join(DIR, env_temp_folder, sub_folder, f'corrected.tiff'))
