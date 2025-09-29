import numpy as np
import cv2
import os 

def convert_RGB_to_32_bit(shifted_img_path: str, saving_path:str, max_temp: float, min_temp: float):
    """_summary_

    Args:
        shifted_img_path (str): _description_
        saving_path (str): _description_
    """

    images = os.listdir(shifted_img_path)

    for image in images:

        RGB_image = cv2.imread(os.path.join(shifted_img_path, image), cv2.IMREAD_UNCHANGED).astype(np.float32)
        RGB_image = cv2.cvtColor(RGB_image, cv2.COLOR_RGB2BGR)

        # print(RGB_image[75, 369])
        
        
        R_mask = (RGB_image[:, :, 0] != 0)
        G_mask = (RGB_image[:, :, 1] != 0)
        B_mask = (RGB_image[:, :, 2] != 0)

        
        RGB_image[R_mask, 0] = RGB_image[R_mask, 0] 
        RGB_image[G_mask, 1] = RGB_image[G_mask, 1] + 256
        RGB_image[B_mask, 2] = RGB_image[B_mask, 2] + 512              
        
        RGB_image = (((max_temp - min_temp) * (RGB_image / 767)))

        RGB_image[R_mask, 0] += min_temp 
        
                   
        # print(RGB_image[76, 368], G_mask[76, 368])
        
        # sdf
        cv2.imwrite(os.path.join(saving_path,str(image.replace('png', 'tiff'))), RGB_image.max(axis=2).astype(np.float32))
        
    

if __name__ == '__main__':
    shifted_img_path = r'd:\Research\Wild Fire - Project\Evaluation Metric\real_data\sparse_data\1D_grid\integral_vertical\variance\shifted_images_0'
    saving_path = r'd:\Research\Wild Fire - Project\Evaluation Metric\real_data\sparse_data\1D_grid\data\channels\recovered_temp_vertical'
    
    #556.300, 7.800 min and max temp for first flight
    #397.20, 8.600 min and max temp for first flight
    convert_RGB_to_32_bit(shifted_img_path, saving_path, 556.300, 7.800)