import numpy as np
import cv2
import os 
import re 
import json

def convert_RGB_to_32_bit(shifted_img_path: str, saving_path:str, rgb_path: str, json_path: str):
    """_summary_

    Args:
        shifted_img_path (str): _description_
        saving_path (str): _description_
    """

    images = os.listdir(shifted_img_path)

    # # Path to your folder
    # folder_path = rgb_path

    # # Get all .txt files
    # txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # # Sort them alphabetically or numerically if filenames contain numbers
    # txt_files.sort()  # simple alphabetical sort
    # # OR for numeric-aware sort:
    # txt_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f)) or 0))


    


    # Path to your JSON file
    json_path = json_path
    
    # Read the file
    with open(json_path, 'r') as f:
        data = json.load(f)

   
    for i, image in enumerate(images):

        RGB_image = cv2.imread(os.path.join(shifted_img_path, image), cv2.IMREAD_UNCHANGED).astype(np.float32)
        RGB_image = cv2.cvtColor(RGB_image, cv2.COLOR_RGB2BGR)


        # === Load in case of shifted images ===
        R_mask = (RGB_image[:, :, 0] != 0)
        G_mask = (RGB_image[:, :, 1] != 0)
        B_mask = (RGB_image[:, :, 2] != 0)

        numbers = re.findall(r"\d+", data["images"][i]["imagefile"])
        filename = f"rendered_image_{numbers[1]}.txt"

        # with open(os.path.join(rgb_path, image.replace('tiff', 'txt')), 'r') as f:
        with open(os.path.join(rgb_path, filename), 'r') as f:
            lines = f.readlines()
            
        # === Load in case to test if the conversion is correct or not ===
        # R_mask = np.load(os.path.join(shifted_img_path, image.replace('.png', '_red.npy')))
        # G_mask = np.load(os.path.join(shifted_img_path, image.replace('.png', '_green.npy')))
        # B_mask = np.load(os.path.join(shifted_img_path, image.replace('.png', '_blue.npy')))

        # with open(os.path.join(shifted_img_path, image.replace('png', 'txt')), 'r') as f:
        #     lines = f.readlines()
        
        min_r, max_r = float(lines[0].split(',')[0]), float(lines[0].split(',')[1])
        min_g, max_g = float(lines[1].split(',')[0]), float(lines[1].split(',')[1])
        min_b, max_b = float(lines[2].split(',')[0]), float(lines[2].split(',')[1])
        
        img_r = ( (RGB_image[:, :, 0] * (max_r - min_r) ) / (255) ) + min_r
        img_g = ( (RGB_image[:, :, 1] * (max_g - min_g) ) / (255) ) + min_g
        img_b = ( (RGB_image[:, :, 2] * (max_b - min_b) ) / (255) ) + min_b
        

          
        print(img_r.min(), img_r.max(), min_r, max_r)
        print(img_g.min(), img_g.max(), min_g, max_g)
        print(img_b.min(), img_b.max(), min_b, max_b)
        
        RGB_image = np.zeros((512, 512, 3))
        
        # RGB_image[R_mask, 0] = img_r
        # RGB_image[G_mask, 1] = img_g
        # RGB_image[B_mask, 2] = img_b
        
        RGB_image[:, :, 0][R_mask] = img_r[R_mask]
        RGB_image[:, :, 1][G_mask] = img_g[G_mask]
        RGB_image[:, :, 2][B_mask] = img_b[B_mask]

        cv2.imwrite(os.path.join(saving_path,str(image.replace('png', 'tiff'))), RGB_image.max(axis=2).astype(np.float32))
        
        
        # print(RGB_image[75, 369])
        
        
        # R_mask = (RGB_image[:, :, 0] != 0)
        # G_mask = (RGB_image[:, :, 1] != 0)
        # B_mask = (RGB_image[:, :, 2] != 0)

        
        # RGB_image[R_mask, 0] = RGB_image[R_mask, 0] 
        # RGB_image[G_mask, 1] = RGB_image[G_mask, 1] + 256
        # RGB_image[B_mask, 2] = RGB_image[B_mask, 2] + 512              
        
        # RGB_image = (((max_temp - min_temp) * (RGB_image / 767)))

        # RGB_image[R_mask, 0] += min_temp 
        
                   
        # print(RGB_image[76, 368], G_mask[76, 368])
        
        # sdf
        
    

if __name__ == '__main__':
    shifted_img_path = r'd:\Research\Wild Fire - Project\Evaluation Metric\real_data\person\1D Horizontal\integral\variance\shifted_images_0'
    saving_path = r'd:\Research\Wild Fire - Project\Evaluation Metric\real_data\person\1D Horizontal\channels\results_temp-20'
    rgb_path = r'd:\Research\Wild Fire - Project\Evaluation Metric\real_data\person\1D Horizontal\channels\RGB-20'
    json_path = r"d:\Research\Wild Fire - Project\Evaluation Metric\real_data\person\1D Horizontal\integral\poses\poses.json"
    
    
    #556.300, 7.800 min and max temp for first flight
    #397.20, 8.600 min and max temp for first flight
    #-5.50 - 16.30 min and max temp for person flight
    convert_RGB_to_32_bit(shifted_img_path, saving_path, rgb_path, json_path)
