from tqdm import tqdm
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.pylab as plt
import numpy as np


if __name__ == '__main__':
    
    saving_path = Path(r'D:\Research\Wild Fire - Project\Evaluation Metric\2D\Fixed Temp')
    
    center_gt_results = Path(r'd:\Research\Wild Fire - Project\Evaluation Metric\2D\Fixed Temp\corrected_fire_gt_results.csv')
    corrected_gt_results = Path(r'd:\Research\Wild Fire - Project\Evaluation Metric\2D\Fixed Temp\corrected_fire_gt_results.csv')
    integrall_gt_results = Path(r'd:\Research\Wild Fire - Project\Evaluation Metric\2D\Fixed Temp\corrected_fire_gt_results.csv')
    
    
    task = False # True for fixed_density, False for fixed temp
    
    x_axis = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30] if task else [220, 293, 366, 439, 512, 585, 658, 731, 804, 877, 950]
    
    
    # AOS vs GT
    results_aos_gt_mse = {"0": [], "3": [], "6": [], "9": [], "12": [], "15": [], "18": [], "21": [], "24": [], "27": [], "30": []} if task else {"30": [], "40": [], "50": [], "60": [], "70": [], "80": [], "90": [], "100": [], "110": [], "120": [], "130": []}
    results_aos_gt_rmse = {"0": [], "3": [], "6": [], "9": [], "12": [], "15": [], "18": [], "21": [], "24": [], "27": [], "30": []} if task else {"30": [], "40": [], "50": [], "60": [], "70": [], "80": [], "90": [], "100": [], "110": [], "120": [], "130": []}
    
    
    integrall_gt_results_df = pd.read_csv(integrall_gt_results)
    
    for i in range(len(integrall_gt_results_df)):
        results_aos_gt_mse[integrall_gt_results_df.iloc[i]["aos_folder"].split("\\")[1]].append(integrall_gt_results_df.iloc[i]["mse"])
        results_aos_gt_rmse[integrall_gt_results_df.iloc[i]["aos_folder"].split("\\")[1]].append(integrall_gt_results_df.iloc[i]["rmse"])
        
    means_aos_gt_mse = {k: float(np.mean(v)) for k, v in results_aos_gt_mse.items() if len(v) > 0}
    means_aos_gt_rmse = {k: float(np.mean(v)) for k, v in results_aos_gt_rmse.items() if len(v) > 0}


    # Center vs GT
    results_center_gt_mse = {"0": [], "3": [], "6": [], "9": [], "12": [], "15": [], "18": [], "21": [], "24": [], "27": [], "30": []} if task else {"30": [], "40": [], "50": [], "60": [], "70": [], "80": [], "90": [], "100": [], "110": [], "120": [], "130": []}
    results_center_gt_rmse = {"0": [], "3": [], "6": [], "9": [], "12": [], "15": [], "18": [], "21": [], "24": [], "27": [], "30": []} if task else {"30": [], "40": [], "50": [], "60": [], "70": [], "80": [], "90": [], "100": [], "110": [], "120": [], "130": []}
    
    
    center_gt_results_df = pd.read_csv(center_gt_results)
    
    for i in range(len(center_gt_results_df)):
        results_center_gt_mse[center_gt_results_df.iloc[i]["aos_folder"].split("\\")[1]].append(center_gt_results_df.iloc[i]["mse"])
        results_center_gt_rmse[center_gt_results_df.iloc[i]["aos_folder"].split("\\")[1]].append(center_gt_results_df.iloc[i]["rmse"])
        
    means_center_gt_mse = {k: float(np.nanmean(v)) for k, v in results_center_gt_mse.items() if len(v) > 0}
    means_center_gt_rmse = {k: float(np.nanmean(v)) for k, v in results_center_gt_rmse.items() if len(v) > 0}


    # corrected vs GT
    results_corrected_gt_mse = {"0": [], "3": [], "6": [], "9": [], "12": [], "15": [], "18": [], "21": [], "24": [], "27": [], "30": []} if task else {"30": [], "40": [], "50": [], "60": [], "70": [], "80": [], "90": [], "100": [], "110": [], "120": [], "130": []}
    results_corrected_gt_rmse = {"0": [], "3": [], "6": [], "9": [], "12": [], "15": [], "18": [], "21": [], "24": [], "27": [], "30": []} if task else {"30": [], "40": [], "50": [], "60": [], "70": [], "80": [], "90": [], "100": [], "110": [], "120": [], "130": []}
    
    
    corrected_gt_results_df = pd.read_csv(corrected_gt_results)
    
    for i in range(len(corrected_gt_results_df)):
        results_corrected_gt_mse[corrected_gt_results_df.iloc[i]["aos_folder"].split("\\")[1]].append(corrected_gt_results_df.iloc[i]["mse"])
        results_corrected_gt_rmse[corrected_gt_results_df.iloc[i]["aos_folder"].split("\\")[1]].append(corrected_gt_results_df.iloc[i]["rmse"])
        
    means_corrected_gt_mse = {k: float(np.mean(v)) for k, v in results_corrected_gt_mse.items() if len(v) > 0}
    means_corrected_gt_rmse = {k: float(np.mean(v)) for k, v in results_corrected_gt_rmse.items() if len(v) > 0}

    
  
    # Set plotting style
    sns.set_theme(style="whitegrid", context="talk")  # bigger fonts

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)

    # Plot with markers and thicker lines
    sns.lineplot(
        x=list(x_axis), 
        y=list(means_aos_gt_mse.values()), 
        color="blue", marker="o", linewidth=2, label="AOS integral"
    )
    sns.lineplot(
        x=list(x_axis), 
        y=list(means_center_gt_mse.values()), 
        color="green", marker="s", linewidth=2, label="single image"
    )
    sns.lineplot(
        x=list(x_axis), 
        y=list(means_corrected_gt_mse.values()), 
        color="red", marker="^", linewidth=2, label="corrected AOS integral"
    )

    # Title and labels
    plt.title("MSE", fontsize=16, weight="bold", pad=15)
    if task:
        plt.xlabel("temperature (°C)", fontsize=12)
    else:
        plt.xlabel("density (trees/ha)", fontsize=12)
    plt.ylabel("MSE", fontsize=12)

    # Grid, legend, and ticks
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12, title_fontsize=12, loc="upper left", frameon=True)
    plt.xticks(x_axis, fontsize=12, rotation=45)  # rotate labels if x-axis is categorical
    plt.yticks(fontsize = 12)
    plt.tight_layout()

    
    
    # plt.subplot(1, 1, 1)
    # sns.countplot(integrall_gt_results_df["direct_sunlight_effect"], color="blue")
    # plt.title("Environment Temperature Distribution")
    # plt.xlabel("Temperature")
    # plt.legend()

    # # Plot Tree Density
    plt.subplot(1, 2, 2)
    # Plot with markers and thicker lines
    sns.lineplot(
        x=list(x_axis), 
        y=list(means_aos_gt_rmse.values()), 
        color="blue", marker="o", linewidth=2, label="AOS integral"
    )
    sns.lineplot(
        x=list(x_axis), 
        y=list(means_center_gt_rmse.values()), 
        color="green", marker="s", linewidth=2, label="single image"
    )
    sns.lineplot(
        x=list(x_axis), 
        y=list(means_corrected_gt_rmse.values()), 
        color="red", marker="^", linewidth=2, label="corrected AOS integral"
    )

    # Title and labels
    plt.title("RMSE", fontsize=16, weight="bold", pad=15)
    if task:
        plt.xlabel("temperature (°C)", fontsize=12)
    else:
        plt.xlabel("density (trees/ha)", fontsize=12)
    
    plt.ylabel("RMSE", fontsize=12)

    # Grid, legend, and ticks
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12, title_fontsize=12, loc="upper left", frameon=True)
    plt.xticks(x_axis, fontsize=12, rotation=45)  # rotate labels if x-axis is categorical
    plt.yticks(fontsize = 12)
    plt.tight_layout()

    plt.tight_layout()
    plt.savefig(saving_path / "test_3.png")
    plt.close()
        