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

    base_path = r'd:\Research\Wild Fire - Project\Evaluation Metric\higher_density'
    integral_2D_images_path = os.path.join(base_path, "integral_2D")
    integral_1D_images_path = os.path.join(base_path, "integral_1D")
    generated_2D_images_path = os.path.join(base_path, "2D")
    generated_1D_images_path = os.path.join(base_path, "1D")
    gt_images_path = os.path.join(base_path, "GT")
    
    results_path = os.path.join(base_path, "results")
    os.makedirs(results_path, exist_ok=True)

    integral_2D_files = sorted(glob.glob(integral_2D_images_path+ '/*.tiff'),key=numericalSort)
    integral_1D_files = sorted(glob.glob(integral_1D_images_path+ '/*.tiff'),key=numericalSort)
    generated_2D_files = sorted(glob.glob(generated_2D_images_path+ '/*.tiff'),key=numericalSort)
    generated_1D_files = sorted(glob.glob(generated_1D_images_path+ '/*.tiff'),key=numericalSort)
    gt_files = sorted(glob.glob(gt_images_path+ '/*.tiff'),key=numericalSort)

  

    integral_2D_images, integral_1D_images, generated_2D_images, generated_1D_images, gt_images = [], [], [], [], []
    all_min, all_max = [], []

    # First loop: load + collect min/max
    for integral_2D_file, integral_1D_file, twoD_file, oneD_file, gt_file in zip(integral_2D_files, integral_1D_files, generated_2D_files, generated_1D_files, gt_files):
        integral_2D_img = cv2.imread(integral_2D_file, -1)
        integral_1D_img = cv2.imread(integral_1D_file, -1)
        gen_2D_img = cv2.imread(twoD_file, -1)
        gen_1D_img = cv2.imread(oneD_file, -1)
        gt_img = cv2.imread(gt_file, -1)

        integral_2D_images.append(integral_2D_img)
        integral_1D_images.append(integral_1D_img)
        generated_2D_images.append(gen_2D_img)
        generated_1D_images.append(gen_1D_img)
        gt_images.append(gt_img)

        all_min.extend([integral_2D_img.min(), integral_1D_img.min(), gen_2D_img.min(), gen_1D_img.min(), gt_img.min()])
        all_max.extend([integral_2D_img.max(), integral_1D_img.max(), gen_2D_img.max(), gen_1D_img.max(), gt_img.max()])

    global_min, global_max = min(all_min), max(all_max)
    print(f"Global min: {global_min}, Global max: {global_max}")

    # Second loop: save tone-mapped images
    for i, (integral_2D_img, integral_1D_img, gen_2D_img, gen_1D_img, gt_image) in enumerate(zip(integral_2D_images, integral_1D_images, generated_2D_images, generated_1D_images, gt_images), start=1):
        integral_2D_rgb = tone_mapping(integral_2D_img, global_min, global_max)
        integral_1D_rgb = tone_mapping(integral_1D_img, global_min, global_max)
        gen_2d_rgb = tone_mapping(gen_2D_img, global_min, global_max)
        gen_1d_rgb = tone_mapping(gen_1D_img, global_min, global_max)
        gt_rgb = tone_mapping(gt_image, global_min, global_max)

        cv2.imwrite(os.path.join(results_path, f"integral_2D_{i-1}.png"), cv2.cvtColor(integral_2D_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(results_path, f"integral_1D_{i-1}.png"), cv2.cvtColor(integral_1D_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(results_path, f"generated_2D_{i-1}.png"), cv2.cvtColor(gen_2d_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(results_path, f"generated_1D_{i-1}.png"), cv2.cvtColor(gen_1d_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(results_path, f"gt_{i-1}.png"), cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR))
