import pandas as pd
import os
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


# def compute_mse(GT: torch.Tensor, corrected: torch.Tensor) -> float:
#     """Compute Mean Squared Error between two float32 tensors."""
#     assert GT.shape == corrected.shape, "Images must have the same shape"
#     return torch.mean((GT - corrected) ** 2).item(), torch.sqrt(torch.mean((GT - corrected) ** 2)).item()  # same unit as temperature

def compute_mse(GT: torch.Tensor, corrected: torch.Tensor):
    """Compute MSE and RMSE between two float32 tensors, ignoring zeros."""
    assert GT.shape == corrected.shape, "Images must have the same shape"
    
    # Build mask: consider only pixels where neither GT nor corrected is zero
    mask = (GT != 0) & (corrected != 0)
    
    # print((GT)[mask])
    # print((corrected)[mask])

    # factor =  (corrected - aos)[mask]

    # if factor.numel() > 0:
    #     factor = factor.max().item()

    # else:
    #     factor = float('nan')  # or 0, depending on your use case
    
    # Apply mask
    diff = (GT - corrected)[mask]
    
    mse = torch.mean(diff ** 2).item()
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()
    return mse, rmse

# Building combined csv File

# df = pd.read_csv(r'd:\Research\Wild Fire - Project\Evaluation Metric\2D\Fixed Temp\Env_Temp_15.csv')
# df1 = pd.read_csv(r'd:\Research\Wild Fire - Project\Evaluation Metric\2D\Fixed Density\Density_80.csv')

# results = []

# # Iterate over rows
# for index, row in df.iterrows():

#     results.append({
#         "aos_folder": row["aos_folder"],
#         "env_temp": row["env_temp"],
#         "tree_density": row["aos_folder"].split('\\')[1]
#     })

# # Iterate over rows
# for index, row in df1.iterrows():

#     results.append({
#         "aos_folder": row["aos_folder"],
#         "env_temp": row["aos_folder"].split('\\')[1],
#         "tree_density": row["tree_density"]
#     })


# # Convert results list to a new DataFrame
# new_df = pd.DataFrame(results)

# # Save to a new CSV file (optional)
# new_df.to_csv(r"D:\Research\Wild Fire - Project\Evaluation Metric\2D\combined_testing_data.csv", index=False)

df = pd.read_csv(r'd:\Research\Wild Fire - Project\Evaluation Metric\2D\combined_testing_data.csv')
row_list = []

for temp in tqdm(range(0, 300, 5), desc="Processing..."):
    for index, row in df.iterrows():
        aos_folder = row["aos_folder"]
        
        prefix = r'd:\Research\Wild Fire - Project\Evaluation Metric\2D\Fixed Temp\\' if aos_folder.split('\\')[0] == 'Env_Temp_15' else r'd:\Research\Wild Fire - Project\Evaluation Metric\2D\Fixed Density\\'
        # Example usage
        path1 = prefix + aos_folder + "/GT.tiff"
        path2 = prefix + aos_folder + "/corrected_small.tiff"
        path3 = prefix + aos_folder + "/integrall.tiff"
        path4 = prefix + aos_folder + "/center.tiff"
        
        # print(path1)
        # Read both images (choose PIL or cv2 function)
        GT = read_image_pil(path1)
        corrected = read_image_pil(path2)
        aos = read_image_pil(path3)
        single = read_image_pil(path4)

        # Ensure both tensors are float32
        GT = GT.to(torch.float32)
        corrected = corrected.to(torch.float32)
        aos = aos.to(torch.float32)
        single = single.to(torch.float32)

        mask = (GT >= temp + 273.15) & (GT < temp + 5 + 273.15)
        GT = GT * mask
        corrected = corrected * mask
        aos = aos * mask
        single = single * mask
        
        # print(GT)
        mse_corrected, rmse_corrected = compute_mse(GT, corrected)
        mse_aos, rmse_aos = compute_mse(GT, aos)
        mse_single, rmse_single = compute_mse(GT, single)


        row_list.append({
            "aos_folder": row["aos_folder"],
            "env_temp": row["env_temp"],
            "tree_density": row["tree_density"],
            "result":"single_vs_GT",
            "mse": mse_single,
            "rmse": rmse_single,
            # "factor": factor
        })

        row_list.append({
            "aos_folder": row["aos_folder"],
            "env_temp": row["env_temp"],
            "tree_density": row["tree_density"],
            "result":"aos_vs_GT",
            "mse": mse_aos,
            "rmse": rmse_aos,
            # "factor": factor
        })       
       
        row_list.append({
            "aos_folder": row["aos_folder"],
            "env_temp": row["env_temp"],
            "tree_density": row["tree_density"],
            "result":"corrected_vs_GT",
            "mse": mse_corrected,
            "rmse": rmse_corrected,
            # "factor": factor
        })
         
    df = pd.DataFrame(
        row_list,
        columns=["aos_folder", "env_temp", "tree_density", "result", "mse", "rmse"],
    )

    df.to_csv(r'd:\Research\Wild Fire - Project\Evaluation Metric\2D\stats\\'+str(temp)+'.csv')
