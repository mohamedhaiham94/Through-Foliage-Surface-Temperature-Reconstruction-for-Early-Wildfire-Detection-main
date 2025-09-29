import os
import numpy as np
import cv2
import cmapy
import glob
import re

def numericalSort(value):
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def tone_mapping(
    x: np.ndarray,  # HxW
    min_val: float,
    max_val: float,
    colormap=cv2.COLORMAP_HOT,
) -> np.ndarray:
    """Return RGB numpy array with tone mapping applied."""
    assert x.ndim == 2, "Input must be grayscale (HxW)"
    x = np.clip((x - min_val) / (max_val - min_val), 0, 1)  # normalize [0,1]
    x = (255 * x).astype(np.uint8)  # to [0,255]
    x = cv2.applyColorMap(x, colormap)  # HxWx3 (BGR)
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)  # convert to RGB


if __name__ == "__main__":

    base_path = r'D:\Research\Wild Fire - Project\Evaluation Metric\generated_vs_real'
    real_images_path = os.path.join(base_path, "real_2")
    generated_images_path = os.path.join(base_path, "generated")
    results_path = os.path.join(base_path, "results-colorcoded")
    os.makedirs(results_path, exist_ok=True)

    real_files = sorted(glob.glob(real_images_path+ '/*.tiff'),key=numericalSort)
    generated_files = sorted(glob.glob(generated_images_path+ '/*.tif'),key=numericalSort)


    real_images, generated_images = [], []
    all_min, all_max = [], []

    # First loop: load + collect min/max
    for rfile, gfile in zip(real_files, generated_files):
        real_img = cv2.imread(rfile, -1)
        gen_img = cv2.imread(gfile, -1)

        real_images.append(real_img)
        generated_images.append(gen_img)

        all_min.extend([real_img.min(), gen_img.min()])
        all_max.extend([real_img.max(), gen_img.max()])

    global_min, global_max = min(all_min), max(all_max)
    print(f"Global min: {global_min}, Global max: {global_max}")

    # Second loop: save tone-mapped images
    for i, (real_img, gen_img) in enumerate(zip(real_images, generated_images), start=1):
        real_rgb = tone_mapping(real_img, global_min, global_max)
        gen_rgb = tone_mapping(gen_img, global_min, global_max)

        cv2.imwrite(os.path.join(results_path, f"real_{i-1}.png"), cv2.cvtColor(real_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(results_path, f"generated_{i-1}.png"), cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2BGR))
