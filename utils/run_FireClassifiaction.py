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
import torch.nn.functional as F


def load_model(checkpoint: Path):
    """Load a traced model (gpu)."""
    model = torch.jit.load(checkpoint)
    return model

def load_image(uploaded_file) -> torch.Tensor:
    return torch.from_numpy(
        np.array(Image.open(uploaded_file))
    )  # uint8; HxW


def to_Kelvin(img: torch.Tensor, min_temp, max_temp) -> torch.Tensor:
    return ((max_temp - min_temp) * (img / 255.0)) + min_temp  # in Kelvin


@dataclass
class config:
    model_name: str = 'segformer' #mambaout
    model_path: str = r'd:\Research\Wild Fire - Project\Evaluation Metric\large\scripted_model.pt'
    dataset_path: str = r'd:\Research\Wild Fire - Project\Evaluation Metric\2D\Fixed Temp\Env_Temp_15'
    simulated: bool = True 
    mean: float = 0
    std: float = 1
    normalize: bool = False
    label_name: str = 'corrected_label_dice+entropy.png'

if __name__ == '__main__':
    
    conf = config()
    
    # model = load_model(r'd:\Research\Wild Fire - Project\Evaluation Metric\large\2025-09-11_21-19-42_2D_Segformer(weighted_cross_loss)\checkpoints\scripted_model.pt')
    model = load_model(conf.model_path)


    # 2D subset stats.
    aos_mean = conf.mean
    aos_std = conf.std

    DIR = conf.dataset_path
    # DIR = r'd:\Research\Wild Fire - Project\Evaluation Metric\real_data\sparse_data\2D_grid\integral\integrals'

    env_temp_folders = os.listdir(DIR)


    for env_temp_folder in tqdm(env_temp_folders, desc="Processing folders"):
        sub_folders = os.listdir(os.path.join(DIR, env_temp_folder))
        for sub_folder in sub_folders:
            
            normalized_img_aos = load_image(os.path.join(DIR, env_temp_folder, sub_folder, 'corrected.tiff')) 
            
            if conf.normalize:
                normalized_img_aos = ( ( (normalized_img_aos - normalized_img_aos.min()) / (normalized_img_aos.max() - normalized_img_aos.min()) ) )

            img1_norm = (normalized_img_aos - aos_mean) / aos_std

            # env_temp = torch.tensor([int(env_temp_folder)]).to(torch.int64).cuda()
            env_temp = torch.tensor([int(20)]).to(torch.int64).cuda()


            # Model prediction
            with torch.inference_mode():
                if conf.model_name == 'mambaout':
                    img2_norm: torch.Tensor = model(
                        img1_norm[None, None, :, :].cuda(),
                        env_temp,
                        None,
                        False,
                    )
                else:
                    img2_norm: torch.Tensor = model(
                        img1_norm[None, None, :, :].cuda(),
                    )
                img2_k = aos_std * img2_norm.cpu() + aos_mean
                


                probs = F.softmax(img2_k, dim=1)   # normalize across channel dimension
                preds = torch.argmax(probs, dim=1) # pick most likely class per pixel

                # preds = preds.permute(1, 2, 0)
                preds = torch.where(preds == 0, torch.tensor(0, dtype=preds.dtype), preds)
                preds = torch.where(preds == 1, torch.tensor(125, dtype=preds.dtype), preds)
                preds = torch.where(preds == 2, torch.tensor(255, dtype=preds.dtype), preds)

                img = preds.squeeze(0).cpu().numpy()
                

                Image.fromarray(img.astype(np.uint8)).save(os.path.join(DIR, env_temp_folder, sub_folder, conf.label_name))
