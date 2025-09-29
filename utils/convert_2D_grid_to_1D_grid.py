import os
from tqdm import tqdm
import re 
import glob

# Load the image
def numericalSort(value):
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

DIR = r'e:\Simulated_1D_grid\Augmented_fire_simulation_1D\Batch-1'


folders = os.listdir(DIR)


for folder in tqdm(folders, desc="Processing folders"):
    sub_folders = os.listdir(os.path.join(DIR, folder))
    for sub_folder in sub_folders:
        images= sorted(glob.glob(os.path.join(DIR, folder, sub_folder, 'images') + '/*.png'),key=numericalSort)
        temp_txt = sorted(glob.glob(os.path.join(DIR, folder, sub_folder, 'images') + '/*.txt'),key=numericalSort)
        
        output_file = os.path.join(DIR, folder, sub_folder, 'poses.txt') 
        

        data_lines = [
            "-10.0,0,35.0",
            "-8.0,0,35.0",
            "-6.0,0,35.0",
            "-4.0,0,35.0",
            "-2.0,0,35.0",
            "0.0,0,35.0",
            "2.0,0,35.0",
            "4.0,0,35.0",
            "6.0,0,35.0",
            "8.0,0,35.0",
            "10.0,0,35.0"
        ]

        with open(output_file, "w") as f:
            for line in data_lines:
                f.write(line + "\n")
    
        # Indices of images to keep
        keep_indices = {5, 16, 27, 38, 49, 60, 71, 82, 93, 104, 115}

        # Loop through images and delete those not in keep_indices
        for idx, filename in enumerate(images):
            if idx not in keep_indices:
                file_path = os.path.join(filename)
                os.remove(file_path)
                os.remove(temp_txt[idx])
        # print(os.path.join(DIR, folder, sub_folder))
        # print("Cleanup complete.")
