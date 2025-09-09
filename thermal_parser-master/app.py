import numpy as np
from thermal_parser import Thermal
import os 
import cv2

# Initialize the Thermal object
thermal = Thermal(dtype=np.float32)

DIR = r'd:\Research\Wild Fire - Project\Evaluation Metric\real_data\second\DJI_202508281913_004_AOS1JKU\Thermal'
OUT = r'd:\Research\Wild Fire - Project\Evaluation Metric\real_data\second\DJI_202508281913_004_AOS1JKU\Thermal_temp'
IMGS = os.listdir(DIR)

for img in IMGS:
    # Parse the image to get the temperature array
    temperature = thermal.parse(filepath_image=os.path.join(DIR, img))

    # Print the temperature array (optional, for debugging)
    print(temperature)

    # Check if the temperature is an instance of np.ndarray
    test = isinstance(temperature, np.ndarray)
    print(test)
    assert isinstance(temperature, np.ndarray)

    # Find and print the min and max temperatures
    min_temp = np.min(temperature)
    max_temp = np.max(temperature)

    # print(f"Min temperature: {min_temp}")
    # print(f"Max temperature: {max_temp}")
    # print(f"Shape: {temperature.shape}")

    cv2.imwrite(os.path.join(OUT, img.split('.')[0]+'.tiff'), temperature.astype(np.float32))


    # # Specify the pixel coordinates
    # x = 640  # replace with your desired x-coordinate
    # y = 0  # replace with your desired y-coordinate

    # # Check if the coordinates are within the valid range
    # if 0 <= y < 512 and 0 <= x < 640:
    #     # Get the temperature at the specified pixel
    #     pixel_temp = temperature[y, x]
    #     print(f"Temperature at pixel ({x}, {y}): {pixel_temp}")
    # else:
    #     print(f"Pixel coordinates ({x}, {y}) are out of bounds.")
