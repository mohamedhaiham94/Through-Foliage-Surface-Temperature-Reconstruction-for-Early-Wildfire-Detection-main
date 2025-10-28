from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_and_group(csv_path, filter_class=None, drop_dupes=True):
    """Load CSV, optionally filter by class, drop duplicates, and group by aos_folder."""
    df = pd.read_csv(csv_path)

    if filter_class is not None:
        df = df[df["Class label"] == filter_class]

    if drop_dupes:
        df = df.drop_duplicates(subset=["aos_folder"])

    grouped = {}
    for _, row in df.iterrows():
        folder = row["aos_folder"].split("\\")[1]
        grouped.setdefault(folder, []).append(row["miou"])

    means_sorted = {
        k: float(np.mean(v))
        for k, v in sorted(grouped.items(), key=lambda item: int(item[0]))
        if len(v) > 0
    }


    print(means_sorted)
    
    # take mean like your code
    return means_sorted


if __name__ == "__main__":

    # Paths
    saving_path = Path(r"E:\Simulated_2D_grid\IOU")

    paths = {
        "corrected AOS integral": Path(r"e:\Simulated_2D_grid\IOU\center.csv"),
        "single image": Path(r"e:\Simulated_2D_grid\IOU\center.csv"),
        "AOS integral": Path(r"e:\Simulated_2D_grid\IOU\integral.csv"),
        # "corrected_label_single image-resnet152": Path(r"D:\Research\Wild Fire - Project\Evaluation Metric\1D\Fixed Density\MIOU\corrected_label_single image_resnet152.csv"),
        # "corrected_label_AOS integral-resnet152": Path(r"D:\Research\Wild Fire - Project\Evaluation Metric\1D\Fixed Density\MIOU\corrected_label_AOS integral_resnet152.csv"),
    }

    # Toggle: True = fixed_density, False = fixed_temp
    task = False

    x_axis = (
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
        if task
        else [220, 293, 366, 439, 512, 585, 658, 731, 804, 877, 950]
    )

    # Load results
    results = {
        "corrected AOS integral": load_and_group(paths["corrected AOS integral"], filter_class=None, drop_dupes=True),
        "single image": load_and_group(paths["single image"], filter_class=None, drop_dupes=True),
        "AOS integral": load_and_group(paths["AOS integral"], filter_class=None, drop_dupes=True),
        # "single image-resnet152": load_and_group(paths["corrected_label_single image-resnet152"], filter_class=None, drop_dupes=True),
        # "AOS integral_resnet152": load_and_group(paths["corrected_label_AOS integral-resnet152"], filter_class=None, drop_dupes=True),
    }

    # Plotting
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(6, 6))

    markers = {"corrected AOS integral": "o", "single image": "^", "AOS integral": "s"}#, "single image-resnet152":"D", "AOS integral_resnet152": "^",} #,  "AOS integral dice_entropy": "D"}
    colors = {"corrected AOS integral": "red", "single image": "green", "AOS integral": "blue"}#, "single image-resnet152": "black", "AOS integral_resnet152": "purple"} #, "corrected_label_single image": "red", "AOS integral dice_entropy": "black"}

    for label, vals in results.items():
        sns.lineplot(
            x=x_axis[: len(vals)],
            y=list(vals.values()),
            marker=markers[label],
            color=colors[label],
            linewidth=2,
            label=label,
        )

    plt.title("jacard index", fontsize=16, weight="bold", pad=15)
    plt.xlabel("temperature (°C)" if task else "density (t/ha)", fontsize=18)
    plt.ylabel("mean iou", fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12, title_fontsize=12, loc="upper right", frameon=True)
    plt.xticks(x_axis, fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    plt.tight_layout()

    plt.savefig(saving_path / "overall.png", dpi=300)
    plt.close()
