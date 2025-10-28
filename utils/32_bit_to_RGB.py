import numpy as np
import cv2
import os 
import tifffile as tifffile
import re

def convert_32_bit_to_RGB(thermal_path: str, saving_path:str):
    """This function split the 32 bit temp data into 3 channels then save it in RGB image

    Args:
        thermal_path (str): _description_
    """
    images = os.listdir(thermal_path)

    for i, image in enumerate(images):

        min_temp, max_temp = 0, 0
        # img32 = cv2.imread(os.path.join(thermal_path, image), cv2.IMREAD_UNCHANGED).astype(np.float32)
        img32 = tifffile.imread(os.path.join(thermal_path, image))
        min_temp, max_temp = img32.min(), img32.max()
        
        
        numbers = re.findall(r"\d+", image)

        print(numbers)
        # sd
        # if i == 102:
        #     print(os.path.join(thermal_path, image))
        #     print(min_temp, max_temp)
        #     sdf       
        total_temp_distance = ((max_temp - abs(min_temp)) / 3)

        first_max_range = total_temp_distance + min_temp
        
        red_min, red_max = min_temp, first_max_range
        green_min, green_max = first_max_range, first_max_range + total_temp_distance
        blue_min, blue_max = green_max , max_temp


        filename = f"rendered_image_{numbers[1]}.txt"
        # with open(os.path.join(saving_path, image.replace('tiff', 'txt')), 'w') as f:
        with open(os.path.join(saving_path, filename), 'w') as f:
            f.writelines([
                f"{red_min}, {red_max}\n",
                f"{green_min}, {green_max}\n",
                f"{blue_min}, {blue_max}\n"
            ])
        
        # print(red_min, red_max)
        # print(green_min, green_max)
        # print(blue_min, blue_max)
        
        #First flight

        # img32 = ( (img32 - min_temp) / (max_temp - min_temp) ) * 767
                     
        

        

        RGB_image = np.zeros((512, 512, 3))

        R_mask = (img32 >= red_min) & (img32 <= red_max)
        G_mask = (img32 > green_min) & (img32 <= green_max)
        B_mask = (img32 > blue_min) & (img32 <= blue_max)


        # === Save ===
        np.save(os.path.join(saving_path,str(image.replace('.tiff', '_red.npy'))), R_mask)
        np.save(os.path.join(saving_path,str(image.replace('.tiff', '_green.npy'))), G_mask)
        np.save(os.path.join(saving_path,str(image.replace('.tiff', '_blue.npy'))), B_mask)




        # Check if the mask is empty
        if not np.any(R_mask):
            img_r = np.zeros_like(img32, dtype=np.uint8)
        else:
            img_r = ( (img32[R_mask] - red_min) / (red_max - red_min) ) * 255
            RGB_image[R_mask, 0] = img_r #np.clip(img32[R_mask], 0, 255)


        if not np.any(G_mask):
            img_g = np.zeros_like(img32, dtype=np.uint8)
        else:
            img_g = ( (img32[G_mask] - green_min) / (green_max - green_min) ) * 255
            RGB_image[G_mask, 1] = img_g #np.clip(img32[G_mask] - 256, 0, 255)
            
            
        if not np.any(B_mask):
            img_b = np.zeros_like(img32, dtype=np.uint8)
        else:
            img_b = ( (img32[B_mask] - blue_min) / (blue_max - blue_min) ) * 255
            RGB_image[B_mask, 2] = img_b #np.clip(img32[B_mask] - 512, 0, 255)
    
        
        # print(img_r.min(), img_r.max())
        # print(img_g.shape, img_b.max())
        # print(img_b.min(), img_g.max())

        # sdf
        



        RGB_image = cv2.cvtColor(RGB_image.astype(np.uint8), cv2.COLOR_RGB2BGR)

        
        cv2.imwrite(os.path.join(saving_path,str(image.replace('tiff', 'png'))), RGB_image) 
        
        
        # RGB_image = RGB_image.astype(np.float32)
        # RGB_image = cv2.cvtColor(RGB_image, cv2.COLOR_RGB2BGR)
        
        
        # # === Load in case to test if the conversion is correct or not ===
        # R_mask = np.load(os.path.join(saving_path, image.replace('.tiff', '_red.npy')))
        # G_mask = np.load(os.path.join(saving_path, image.replace('.tiff', '_green.npy')))
        # B_mask = np.load(os.path.join(saving_path, image.replace('.tiff', '_blue.npy')))

        # with open(os.path.join(saving_path, image.replace('tiff', 'txt')), 'r') as f:
        #     lines = f.readlines()
        
        # min_r, max_r = float(lines[0].split(',')[0]), float(lines[0].split(',')[1])
        # min_g, max_g = float(lines[1].split(',')[0]), float(lines[1].split(',')[1])
        # min_b, max_b = float(lines[2].split(',')[0]), float(lines[2].split(',')[1])
        
        # img_r = ( (RGB_image[:, :, 0] * (max_r - min_r) ) / (255) ) + min_r
        # img_g = ( (RGB_image[:, :, 1] * (max_g - min_g) ) / (255) ) + min_g
        # img_b = ( (RGB_image[:, :, 2] * (max_b - min_b) ) / (255) ) + min_b
        

          
        # # print(img_r.min(), img_r.max(), min_r, max_r)
        # # print(img_g.min(), img_g.max(), min_g, max_g)
        # # print(img_b.min(), img_b.max(), min_b, max_b)
        
        # RGB_image = np.zeros((512, 512, 3))
        
        # # RGB_image[R_mask, 0] = img_r
        # # RGB_image[G_mask, 1] = img_g
        # # RGB_image[B_mask, 2] = img_b
        
        # RGB_image[:, :, 0][R_mask] = img_r[R_mask]
        # RGB_image[:, :, 1][G_mask] = img_g[G_mask]
        # RGB_image[:, :, 2][B_mask] = img_b[B_mask]

        # cv2.imwrite(os.path.join(r'd:\Research\Wild Fire - Project\Evaluation Metric\real_data\person\data\channels\results_temp',
        #                          str(image)), RGB_image.sum(axis=2).astype(np.float32))
         

if __name__ == '__main__':
    thermal_path = r'd:\Research\Wild Fire - Project\Evaluation Metric\real_data\person\1D Horizontal\augmented_Thermal_temp_cropped-20'
    saving_path = r'd:\Research\Wild Fire - Project\Evaluation Metric\real_data\person\1D Horizontal\channels\RGB-20'
    
    #556.300, 7.800 min and max temp for first flight
    #397.20 - 8.600 min and max temp for second flight
    #-5.50 - 16.30 min and max temp for person flight
    convert_32_bit_to_RGB(thermal_path, saving_path)