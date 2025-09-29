import numpy as np
import cv2
import os 


def convert_32_bit_to_RGB(thermal_path: str, saving_path:str, max_temp: float, min_temp: float):
    """This function split the 32 bit temp data into 3 channels then save it in RGB image

    Args:
        thermal_path (str): _description_
    """
    images = os.listdir(thermal_path)

    for image in images:

        img32 = cv2.imread(os.path.join(thermal_path, image), cv2.IMREAD_UNCHANGED).astype(np.float32)

        #First flight
        img32 = ( (img32 - min_temp) / (max_temp - min_temp) ) * 767
                     
        
        #Second flight
        # img32 = ((img32 - 8.600) / (397.20 - 8.600)) * 767

        RGB_image = np.zeros((512, 512, 3))

        R_mask = (img32 >= 0) & (img32 <= 255)
        G_mask = (img32 >= 256) & (img32 <= 511)
        B_mask = (img32 >= 512) & (img32 <= 767)


        RGB_image[R_mask, 0] = np.clip(img32[R_mask], 0, 255)
        RGB_image[G_mask, 1] = np.clip(img32[G_mask] - 256, 0, 255)
        RGB_image[B_mask, 2] = np.clip(img32[B_mask] - 512, 0, 255)


        RGB_image = cv2.cvtColor(RGB_image.astype(np.uint8), cv2.COLOR_RGB2BGR)

        
        cv2.imwrite(os.path.join(saving_path,str(image.replace('tiff', 'png'))), RGB_image) 
         

if __name__ == '__main__':
    thermal_path = r'd:\Research\Wild Fire - Project\Evaluation Metric\real_data\sparse_data\1D_grid\data\1D_thermal_crop_vertical'
    saving_path = r'd:\Research\Wild Fire - Project\Evaluation Metric\real_data\sparse_data\1D_grid\data\channels\RGB_vertical'
    
    #556.300, 7.800 min and max temp for first flight
    #397.20 - 8.600 min and max temp for second flight
    convert_32_bit_to_RGB(thermal_path, saving_path, 556.300, 7.800)