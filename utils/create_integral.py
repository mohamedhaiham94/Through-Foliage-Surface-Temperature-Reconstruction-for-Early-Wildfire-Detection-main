import os
import argparse
import numpy as np
from PIL import Image
import cv2 

def load_images_from_folder(folder_path, folder_path1):
    """
    Load all images from a folder into a list of numpy arrays.
    Images must have the same dimensions and mode.
    """
    images = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            path = os.path.join(folder_path, fname)

            images.append(cv2.imread(path, -1))
            
              

    alpha = []
    for fname in sorted(os.listdir(folder_path1)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            path = os.path.join(folder_path1, fname)
            alpha.append(cv2.imread(path, 0))
            
               
    return images, alpha


def stack_and_average(images, alpha):
    """
    Stack a list of numpy arrays along a new axis (z) and compute their average.
    Returns an uint8 array suitable for saving as an image.
    """
    stack = np.stack(images, axis=0)  # shape: (N, H, W, C) or (N, H, W)

    stack_alpha = np.stack(alpha, axis=0)  # shape: (N, H, W, C) or (N, H, W)

    stack_2 = stack * stack_alpha

    # avg = np.mean(stack_2, axis=0)
    
    # Mask out zeros
    mask = stack_2 != 0

    # Sum only valid values
    sum_valid = np.sum(stack_2 * mask, axis=0)

    # Count only non-zero elements per pixel
    count_valid = np.sum(mask, axis=0)

    # Avoid division by zero
    avg = np.divide(sum_valid, count_valid, where=count_valid != 0)

    
    return avg.astype(np.float32)


def save_image(array, output_path):
    """
    Save a numpy array as an image.
    """
    img = Image.fromarray(array.astype(np.float32), mode='F')
    img.save(output_path)


def main():
    INPUT1 = r"d:\Research\Wild Fire - Project\Evaluation Metric\real_data\sparse_data\1D_grid\integral_vertical\variance\alpha_0"
    INPUT = r"d:\Research\Wild Fire - Project\Evaluation Metric\real_data\sparse_data\1D_grid\data\channels\recovered_temp_vertical"

    OUT = r"D:\Research\Wild Fire - Project\Evaluation Metric\real_data\sparse_data\1D_grid\integral_vertical\integrals\New folder\1\Layer_1.tiff"

    imgs, alpha = load_images_from_folder(INPUT, INPUT1)
    if not imgs:
        raise RuntimeError(f"No valid images found in {INPUT}")

    avg_img = stack_and_average(imgs, alpha)

    save_image(avg_img, OUT)
    print(f"Averaged image saved to {OUT}")

if __name__ == '__main__':
    main()
