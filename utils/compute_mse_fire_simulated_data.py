import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def read_image_pil(path, gray=False):
    """Read image with PIL and convert to float32 torch tensor."""
    img = Image.open(path)
    img = np.array(img, dtype=np.float32)
    return torch.from_numpy(img)


# def compute_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
#     """Compute Mean Squared Error between two float32 tensors."""
#     assert img1.shape == img2.shape, "Images must have the same shape"
#     return torch.mean((img1 - img2) ** 2).item(), torch.sqrt(torch.mean((img1 - img2) ** 2)).item()  # same unit as temperature

def compute_mse(img1: torch.Tensor, img2: torch.Tensor, img3: torch.Tensor):
    """Compute MSE and RMSE between two float32 tensors, ignoring zeros."""
    assert img1.shape == img2.shape, "Images must have the same shape"
    
    # Build mask: consider only pixels where neither img1 nor img2 is zero
    mask = (img1 != 0) & (img2 != 0)
    
    # print((img1)[mask])
    # print((img2)[mask])

    factor =  (img2 - img3)[mask]

    if factor.numel() > 0:
        factor = factor.max().item()

    else:
        factor = float('nan')  # or 0, depending on your use case
    # Apply mask
    diff = (img1 - img2)[mask]
    
    mse = torch.mean(diff ** 2).item()
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()
    return mse, rmse, factor


if __name__ == "__main__":
    
    row_list = []
    root = Path(r'D:\Research\Wild Fire - Project\Evaluation Metric\2D\Fixed Temp\Env_Temp_15')
    
    temperature_folders = [f for f in root.iterdir() if f.is_dir()]
    for temperature_folder in tqdm(temperature_folders, desc="Processing..."):
        aos_folders = [f for f in temperature_folder.iterdir() if f.is_dir()]
        for aos_folder in aos_folders:
            
            # Example usage
            path1 = aos_folder / "GT.tiff"
            path2 = aos_folder / "corrected.tiff"
            path3 = aos_folder / "integrall.tiff"
            
            # print(path1)
            # Read both images (choose PIL or cv2 function)
            img1 = read_image_pil(path1)
            img2 = read_image_pil(path2)
            img3 = read_image_pil(path3)

            # Ensure both tensors are float32
            img1 = img1.to(torch.float32)
            img2 = img2.to(torch.float32)
            img3 = img3.to(torch.float32)

            mask = img1 >= 323.15
            img1 = img1 * mask
            img2 = img2 * mask
            img3 = img3 * mask
            
            # print(img1)
            mse, rmse, factor = compute_mse(img1, img2, img3)
            
            # print(mse, mse_integrall)
            # fgdfg   
            row_list.append({
                "aos_folder": str(aos_folder.relative_to(root.parent)),
                "GT Path": path1,
                "Center Path": path2,
                "mse": mse,
                "rmse": rmse,
                "factor": factor
            })
            
            # print(f"MSE: {mse:.6f}")
            # print(f"RMSE: {rmse:.6f}")
            
            
    df = pd.DataFrame(
        row_list,
        columns=["aos_folder", "GT Path", "Center Path", "mse", "rmse", "factor"],
    )
    
    df.to_csv(r'D:\Research\Wild Fire - Project\Evaluation Metric\2D\Fixed Temp\corrected_fire_gt_results.csv')
    

