from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def compute_miou(gt, pred, num_classes=3):
    # Flatten
    gt = gt.flatten()
    pred = pred.flatten()

    # Confusion matrix
    cm = confusion_matrix(gt, pred, labels=list(range(num_classes)))

    # IoU per class
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou = intersection / np.maximum(union, 1)  # avoid division by zero

    # Mean IoU
    miou = np.mean(iou)

    return iou, miou, cm

def read_img_cv2(path):
    # Read label images as grayscale (important!)
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Remap prediction
    image_new = np.copy(image)
    image_new[image == 0] = 0
    image_new[image == 125] = 1
    image_new[image == 255] = 2

    
    # Flatten for sklearn
    image_flatten = image_new.flatten()


    # # Confusion matrix
    # cm = confusion_matrix(gt_flat, pred_flat, labels=[0,1,2])
    # print("Confusion Matrix:\n", cm)

    # # Display confusion matrix
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fire (0)", "Maybe Fire (1)", "Fire (2)"])
    # disp.plot(cmap=plt.cm.Blues)
    # plt.show()

    # # Optional: classification report
    # print("\nClassification Report:\n", classification_report(gt_flat, pred_flat, labels=[0,1,2]))
    
    return image_flatten








if __name__ == "__main__":
    
    row_list = []
    root = Path(r'D:\Research\Wild Fire - Project\Evaluation Metric\1D\Fixed Temp\Env_Temp_15')
    saving_path = r'D:\Research\Wild Fire - Project\Evaluation Metric\1D\Fixed Temp\MIOU\integrall_label.csv'
    
    # root = Path(r'd:\Research\Wild Fire - Project\Evaluation Metric\1D\Fixed Temp\Env_Temp_15')
    # saving_path = r'd:\Research\Wild Fire - Project\Evaluation Metric\1D\Fixed Temp\MIOU\center_label.csv'
    
    temperature_folders = [f for f in root.iterdir() if f.is_dir()]
    for temperature_folder in tqdm(temperature_folders, desc="Processing..."):
        aos_folders = [f for f in temperature_folder.iterdir() if f.is_dir()]
        for aos_folder in aos_folders:
            
            # Example usage
            path1 = aos_folder / "label.png"
            path2 = aos_folder / 'integrall_label.png'

            # Read both images (choose PIL or cv2 function)
            gt = read_img_cv2(str(path1))
            generated = read_img_cv2(str(path2))

            # Example usage
            iou, miou, cm = compute_miou(gt, generated, num_classes=3)
            num_classes = 3

            precision = np.zeros(num_classes)
            recall = np.zeros(num_classes)
                
            
            for i, val in enumerate(iou):
                
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                row_list.append({
                    "aos_folder": str(aos_folder.relative_to(root.parent)),
                    "GT Path": path1,
                    "Generated Path": path2,
                    "Class label":i,
                    "iou": val,
                    "miou": miou,
                    "precision": precision[i],
                    "recall": recall[i],
                })

            row_list.append({
                "aos_folder": str(aos_folder.relative_to(root.parent)),
                "GT Path": path1,
                "Generated Path": path2,
                "Class label":None,
                "iou": None,
                "miou": miou,
                "precision": None,
                "recall": None,
            })
            
            
            
    df = pd.DataFrame(
        row_list,
        columns=["aos_folder", "GT Path", "Generated Path", "Class label", "iou", "miou", "precision", "recall"],
    )
    
    df.to_csv(saving_path)
    

