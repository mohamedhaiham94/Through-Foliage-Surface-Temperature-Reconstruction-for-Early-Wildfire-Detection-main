import logging
from PIL import Image
import os
import random
import numpy as np
from PIL import Image
import math
import shutil
import gz.math7 as gzm
from photo_shoot_config import PhotoShootConfig
from person_config import PersonConfig
from forest_config import ForestConfig
from world_config import WorldConfig
from uuid import uuid4
from launcher import Launcher
from typing import Tuple, Union
from math import sin, cos, atan2, sqrt
import time
from augmentation import *

Number = Union[int, float]
Vector = Tuple[Number, Number, Number]
logger = logging.getLogger(__name__)


# random.seed(seeed1)
class SimulationRunner:
    def __init__(self):
        
        # Initialize file paths and directories
        self.world_file_in = "../worlds/example_photo_shoot.sdf"
        self.world_file_out = "../../photo_shoot.sdf"
        self.output_directory = "/media/haitham/046C27CA6C27B578/Simulated_2D_grid/Batch-2" 
        
        self.database_dir = "../../Desktop/WildFire/env_temp_dataset" 

        # Load the initial world configuration
        self.world_config = WorldConfig()
        self.world_config.load(self.world_file_in)

        # Initialize counters and iteration settings
        self.PC_Num = 0
        self.iter_Number = 5000
        self.iteration = 0
        
        self.non_uniform = 1
        self.uniform = 1

        self.uniform_texture_count = 0
        self.normal_texture_count = 0
        # Path to the thermal texture database in PNG
        self.thermal_texture_dir = "/home/haitham/gazebo_sim/models/procedural-forest/materials/textures"
        
        # Path to the thermal texture database in TIF (for min and max temp)
        self.thermal_texture_dir_tif = "/media/haitham/046C27CA6C27B578/WildFire generated data/synthetic/Fire_images" 

        self.test_seed = 0
    def run(self):
        for i in range(self.iter_Number):
            random.seed(i + 599326232)
            self.test_seed = i + 599326232

            self.iteration = i
            # Reload world_config for each iteration
            self.world_config = WorldConfig()
            self.world_config.load(self.world_file_in)

            # Generate random parameters
            self.generate_random_parameters()

            self.convert_32bit_to_16bit(self.thermal_texture)

            self.get_min_max_temp()

            # Configure the simulation components
            self.configure_light_and_scene()
            self.configure_photo_shoot(i)
            
            # Config for Forest
            self.configure_forest()

            # Save the world configuration
            self.save_world_config()

            # Launch the simulation
            self.launch_simulation()

            # Generate ground truth image

            self.generate_ground_truth()
            self.convert_32bit_to_16bit_GT(self.thermal_texture)
            # Print iteration info
            print(f"\nIteration {i + 1} / {self.iter_Number} is running\n")


    def convert_32bit_to_16bit(self, image_path):
        """
        Convert a 32-bit image to 16-bit while normalizing values to fit the 16-bit range.

        Args:
            image_path (str): Path to the input 32-bit image.
            output_path (str): Path to save the 16-bit image.

        Returns:
            None
        """
        # Load the 32-bit image as a NumPy array
        image = Image.open(os.path.join(self.thermal_texture_dir_env_tif, image_path))
        if self.non_uniform <= 7:
            self.random_texture = 1
            # random.randint(0, 1)

            # self.non_uniform += 1 mona
        else:
            self.random_texture = 0
            self.uniform += 1

        if self.uniform == 4:
            self.uniform = 1
            self.non_uniform = 1




        print(self.random_texture)
        if self.random_texture:
            self.normal_texture_count += 1
            amb_temp = 9
            max_sun_temp_inc = 15

            new_img = amb_temp_aug(
            torch.from_numpy(np.array(image, dtype=np.float32)), #Input Image
            amb_temp, #our original amb temp which is 9
            max_sun_temp_inc, # 15 degree
            self.env_temperature, # choosen amb temp
            )

            image = Image.fromarray(new_img.cpu().numpy(), mode="F")
            self.thermal_image_referance = image

        else:
            self.uniform_texture_count += 1
            img = np.zeros((512,512)).astype(np.float32)
            temp_value = ((self.x_rand_Tree_C - 3))
            for i in range(512):
                for j in range(512):
                    img[i, j] = temp_value
            self.normal_texture = img

            image = Image.fromarray(img, mode="F")
            self.thermal_image_referance = image

        # Get the dimensions of the original image
        width, height = image.size

        # Calculate the center coordinates
        center_x, center_y = width // 2, height // 2

        # Define the cropping box
        crop_width, crop_height = 512, 512
        x_start = center_x - (crop_width // 2)
        y_start = center_y - (crop_height // 2)
        x_end = x_start + crop_width
        y_end = y_start + crop_height

        # Crop the image
        cropped_image = image.crop((x_start, y_start, x_end, y_end))
        image = cropped_image
        
        image_data = (np.array(image, dtype=np.float32) + 273.15) * 100  # 0 - 65535

        original_size = (512, 512)
        padded_size = (534, 534)

        # Calculate padding sizes
        pad_top = (padded_size[0] - original_size[0]) // 2
        pad_bottom = padded_size[0] - original_size[0] - pad_top
        pad_left = (padded_size[1] - original_size[1]) // 2
        pad_right = padded_size[1] - original_size[1] - pad_left

        # Pad the image with zeros (black)
        padded_image1 = np.pad(
            image_data.astype(np.uint16),
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )

        original_size = (534, 534)
        padded_size = (856, 856) # was 856 before

        # Calculate padding sizes
        pad_top = (padded_size[0] - original_size[0]) // 2
        pad_bottom = padded_size[0] - original_size[0] - pad_top
        pad_left = (padded_size[1] - original_size[1]) // 2
        pad_right = padded_size[1] - original_size[1] - pad_left

        # Pad the image with zeros (black)
        padded_image = np.pad(
            padded_image1.astype(np.uint16),
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )


        # Convert to 16-bit integers
        if self.random_texture:
            image_16bit = padded_image.astype(np.uint16) 
        else:
            image_16bit = padded_image.astype(np.uint16) 
        
        # Rotate the image 90 degrees to the right (clockwise)
        image_rotated = np.rot90(image_16bit, k=-1)  # k=-1 for clockwise rotation


        image_transformed = np.flipud(image_rotated)


        # Save the image
        Image.fromarray(image_transformed).save(self.thermal_texture_dir+"/ground_000_thermal.png")
        print(f"32-bit image converted to 16-bit and saved to {self.thermal_texture_dir}")

    def convert_32bit_to_16bit_GT(self, image_path):
        """
        Convert a 32-bit image to 16-bit while normalizing values to fit the 16-bit range.

        Args:
            image_path (str): Path to the input 32-bit image.
            output_path (str): Path to save the 16-bit image.

        Returns:
            None
        """
        # Load the 32-bit image as a NumPy array
        image = Image.open(os.path.join(self.thermal_texture_dir_env_tif, image_path))
        
        amb_temp = 9
        max_sun_temp_inc = 15
        upper_fire_thres_temp = 50
       
        if self.random_texture:
        
            new_img = amb_temp_aug(
            torch.from_numpy(np.array(image)), #Input Image
            amb_temp, #our original amb temp which is 9
            max_sun_temp_inc, # 15 degree
            self.env_temperature, # choosen amb temp
            )
            image = Image.fromarray(new_img.cpu().numpy()  , mode="F")
            self.thermal_image_referance = image
            new_fire_thres_temp = self.env_temperature + max_sun_temp_inc
            tgt = build_semantic_segmentation_target(
                torch.from_numpy(np.array(image) + round(self.direct_sun * math.cos(self.x_rand_Alpha))), new_fire_thres_temp, upper_fire_thres_temp
            )
            Image.fromarray(tgt.cpu().numpy()).save(self.patch_folder+"/label.png") 

        else:
           
            image = Image.fromarray(self.normal_texture + round(self.direct_sun * math.cos(self.x_rand_Alpha_rad)) , mode="F")
            self.thermal_image_referance = image

            new_fire_thres_temp = self.env_temperature + max_sun_temp_inc
            tgt = build_semantic_segmentation_target(
                torch.from_numpy(self.normal_texture + round(self.direct_sun * math.cos(self.x_rand_Alpha))) , new_fire_thres_temp, upper_fire_thres_temp
            )
            Image.fromarray(tgt.cpu().numpy()).save(self.patch_folder+"/label.png")
    
       
    def generate_random_parameters(self):
        
        # Random Env Temp and get texture from folder
        self.env_temperature = random.randint(0,30) # TODO: specify
        # self.env_temperature = random.randrange(0, 33, 3) # TODO: specify
        # self.thermal_texture_env_dir = f"{self.thermal_texture_dir}/{self.env_temperature}"
        
        self.thermal_texture_env_dir = f"{self.thermal_texture_dir}"
        self.thermal_texture_dir_env_tif = f"{self.thermal_texture_dir_tif}"


        # Randomly select a thermal texture
        thermal_textures = [
            f for f in os.listdir(self.thermal_texture_dir_env_tif) if f.endswith(".TIF")
        ]
        self.thermal_texture = random.choice(thermal_textures)


        # Number of trees per hectare (ha) 30 - 130
        # self.x_rand_treeNum = random.randrange(30, 131, 10)
        self.x_rand_treeNum = random.randint(30, 131)


        # Direct lightsun effect 0-15

        self.direct_sun = random.randint(0, 15)

        # Azimuth angle of sunlight direction (Alpha) -90 - +90
        self.x_rand_Alpha = random.uniform(-90, 90)
        self.x_rand_Alpha_rad = math.radians(self.x_rand_Alpha)
        # print("Azimuth angle (Alpha) in degrees =", self.x_rand_Alpha)

        # Compass direction of sunlight (Beta)
        self.x_rand_Beta = 90
        self.x_rand_Beta_rad = math.radians(self.x_rand_Beta)
        # print("Compass direction (Beta) in degrees =", self.x_rand_Beta)

        # Convert spherical coordinates to Cartesian for light direction
        self.x_1, self.x_2, self.x_3 = self.to_cartesian(
            1, self.x_rand_Alpha_rad, self.x_rand_Beta_rad
        )
        if self.x_3 > 0:
            self.x_3 = -self.x_3  # Ensure sunlight comes from above
        print(self.x_1, self.x_2, self.x_3)

        # Tree top temperature in degrees Celsius and convert to Kelvin
        self.x_rand_Tree_C = self.env_temperature + 0
        self.x_Tree_temp = self.x_rand_Tree_C + 273.15

    def get_min_max_temp(self):
        thermal_image = np.array(self.thermal_image_referance)
        thermal_image_C = thermal_image
        thermal_image_K = thermal_image + 273.15

        self.min_ground_temp_C = thermal_image_C.min()
        self.max_ground_temp_C = thermal_image_C.max()
        self.min_ground_temp_K = thermal_image_K.min()
        self.max_ground_temp_K = thermal_image_K.max()

        print(f"Min Ground Temp: {self.min_ground_temp_C} C / {self.min_ground_temp_K} K")
        print(f"Max Ground Temp: {self.max_ground_temp_C} C / {self.max_ground_temp_K} K ")


    def configure_light_and_scene(self):
        # Configure the sun as the light source
        light = self.world_config.get_light("sun")
        light.set_direction(gzm.Vector3d(self.x_1, self.x_2, self.x_3))
        light.set_cast_shadows(False)


    def configure_photo_shoot(self, i: int):
        person_config = PersonConfig()
        person_config.set_model_pose("idle")                 # Must match a .dae mesh file                                # in the respective model!
        person_config.set_temperature(293)                      # In Kelvin
        person_config.set_pose(gzm.Pose3d(0, 0, -10000, 0, 0, 0))    # First three values are x, y, z coordinates 
        self.world_config.add_plugin(person_config)

        photo_shoot_config = PhotoShootConfig()
        # Define the environment temperature as a string to use in the folder path
        env_temp_str = str(self.env_temperature)
        
        # Construct the path for the environment temperature folder
        env_temp_folder = os.path.join(self.output_directory, env_temp_str)
        # env_temp_folder = os.path.join(self.output_directory, str(self.x_rand_treeNum))
        
        
        # Ensure the environment temperature folder exists; create it if it doesn't
        os.makedirs(env_temp_folder, exist_ok=True)
        
        # Construct the path for the current iteration's patch folder within the env_temp folder
        self.seed = str(uuid4().hex)

        # self.seed = seeed2
        patch_folder = os.path.join(env_temp_folder, self.seed)
        
        # Remove the patch folder if it already exists to ensure a clean setup
        if not os.path.exists(patch_folder):
            os.makedirs(patch_folder)
            os.makedirs(patch_folder+"/images")
        else:
            print("already exists")
            patch_folder = os.path.join(env_temp_folder, str(uuid4))
            os.makedirs(patch_folder)
            os.makedirs(patch_folder+"/images")


        # Update the instance variable to point to the new patch folder
        self.patch_folder = patch_folder

        photo_shoot_config.set_directory(self.patch_folder+"/images")

        img_Name = f"{self.PC_Num}_{i}"
        photo_shoot_config.set_prefix(img_Name)

        # Set camera properties
        photo_shoot_config.set_direct_thermal_factor(self.direct_sun)  # direct sunlight - TODO (0 - 64, will be discussed) 
        photo_shoot_config.set_save_rgb(False)
        photo_shoot_config.set_save_thermal(True)
        photo_shoot_config.set_save_depth(False)


        # # # Define drone poses along a straight line with 0.5m spacing
        # num_images = 11
        # spacing = 2

        # inverse_x = False

        # if inverse_x:
        #     self.x_positions = [
        #         (i * spacing - ((num_images - 1) / 2) * spacing) * -1
        #         for i in range(num_images)
        #     ]
        # else:
        #     self.x_positions = [
        #         i * spacing - ((num_images - 1) / 2) * spacing
        #         for i in range(num_images)
        #     ]
        # self.poses = [
        #     gzm.Pose3d(x, 0, 35, 0, 1.57079632679, 0) for x in self.x_positions
        # ]  # x, y, z, roll, tilt angle, +rotation

        area_size = 20
        grid_resolution = 11

        grid_size= int((grid_resolution-1)/2)

        spacing = area_size / (grid_resolution - 1)
        self.posess  = []
        for i in range( -grid_size,grid_size+1,1):
            for j in range(-grid_size,grid_size+1):
                x = i * spacing
                y = j * spacing
                self.posess.append((x, y))

                
        self.poses = [
            gzm.Pose3d(p[0], p[1], 35, 0.0, 1.57079632679, 0.0) for p in self.posess
        ]  # x, y, z, -rotation, tilt angle, +rotation

        photo_shoot_config.add_poses(self.poses)

        self.world_config.add_plugin(photo_shoot_config)
        self.write_poses()

    def configure_forest(self):
        self.forest_config = ForestConfig()
        forest_config = self.forest_config

        forest_config.set_generate(True)
        forest_config.set_direct_spawning(True)
     
        # Use the selected thermal texture
        # map to minimum an maximum in the thermal texture
        forest_config.set_ground_thermal_texture(
            os.path.join(self.thermal_texture_env_dir, self.thermal_texture),
            # 0,
            0,#self.min_ground_temp_K, # Minimal  temperature in Kelvin
            655,#self.max_ground_temp_K # Minimal # Maximal temperature in Kelvin
        )

        # still in discussion (3 or 4 degress less then env. temperature)
        forest_config.set_trunk_temperature(self.x_Tree_temp)  # In Kelvin
        forest_config.set_twigs_temperature(self.x_Tree_temp)  # In Kelvin

        # 23 to 23 Meters in Reallife
        forest_config.set_size(37)  # fits exact to the 31 drone images
        forest_config.set_texture_size(37)

        forest_config.set_trees(self.x_rand_treeNum)
        
        ###################### mona
        # forest_config.set_seed(self.seed)
        forest_config.set_seed(self.test_seed)

        

        # Define tree species and properties (adjust as needed)
        forest_config.set_species(
            "Birch",
            {
                "percentage": 1.0,
                "homogeneity": 0.95,
                "trunk_texture": 0,
                "twigs_texture": 0,
                "tree_properties": {
                    "clump_max": 0.45 / 4,#
                    "clump_min": 0.4 / 4,#
                    "length_falloff_factor": 0.65 / 4,#
                    "length_falloff_power": 0.75 / 4,#
                    "branch_factor": 2.45,
                    "radius_falloff_rate": 0.7,
                    "climb_rate": 0.55,
                    "taper_rate": 0.8,
                    "twist_rate": 8.0,
                    "segments": 4,
                    "levels": 4,
                    "sweep_amount": 0.0,
                    "initial_branch_length": 0.7,
                    "trunk_length": 1.0,
                    "drop_amount": 0.0,
                    "grow_amount": 0.4,
                    "v_multiplier": 0.2,
                    "twig_scale": 0.2,
                },
            },
        )

        self.world_config.add_plugin(forest_config)


    def save_world_config(self):
        self.world_config.save(self.world_file_out)

    def launch_simulation(self):
        launcher = Launcher()
        launcher.set_launch_config("server_only", True)
        launcher.set_launch_config("running", True)
        launcher.set_launch_config("iterations", 2)
        launcher.set_launch_config("world", self.world_file_out)
        print(launcher.launch())

    def compute_integral_image(self):
        pass

    def write_poses(self):
        label_path = f"{self.patch_folder}/poses.txt"

        with open(label_path, "w+") as file:
            for coords in self.posess:
            # for coords in self.x_positions:
                file.write(f"{coords[0]},{coords[1]},35\n")
                # file.write(f"{coords},0,35.0\n")

        label_path = f"{self.patch_folder}/scene_parameters.txt"
        with open(label_path, "w+") as file:
            file.write(f"Selected Env temp: {self.env_temperature}\n"
                        f"Selected thermal texture: {self.thermal_texture}\n" 
                        f"Number of trees per (37x37): {self.x_rand_treeNum} / {round(self.x_rand_treeNum * 7.3)} \n"
                        f"Direct lightsun effect: {self.direct_sun}\n"
                        f"Added lightsun effect based on the Azimuth angle: {round(self.direct_sun * math.cos(self.x_rand_Alpha_rad))}\n"
                        f"Azimuth angle (Alpha) in degrees: {self.x_rand_Alpha}\n"
                        f"Tree top temperature: {self.x_rand_Tree_C}°C / {self.x_Tree_temp}K")


    @staticmethod
    def distance(a: Vector, b: Vector) -> Number:
        """Returns the distance between two cartesian points."""
        x = (b[0] - a[0]) ** 2
        y = (b[1] - a[1]) ** 2
        z = (b[2] - a[2]) ** 2
        return (x + y + z) ** 0.5

    @staticmethod
    def magnitude(x: Number, y: Number, z: Number) -> Number:
        """Returns the magnitude of the vector."""
        return sqrt(x * x + y * y + z * z)

    @staticmethod
    def to_spherical(x: Number, y: Number, z: Number) -> Vector:
        """Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi)."""
        radius = SimulationRunner.magnitude(x, y, z)
        theta = atan2(sqrt(x * x + y * y), z)
        phi = atan2(y, x)
        return (radius, theta, phi)

    @staticmethod
    def to_cartesian(radius: Number, theta: Number, phi: Number) -> Vector:
        """Converts a spherical coordinate (radius, theta, phi) into a cartesian one (x, y, z)."""
        x = radius * sin(theta) * cos(phi)
        y = radius * sin(theta) * sin(phi)
        z = radius * cos(theta)
        return (x, y, z)

    def generate_ground_truth(self):
        # Reload world_config
        self.world_config = WorldConfig()
        self.world_config.load(self.world_file_in)

        # Use the same random parameters generated before
        # Configure light and scene with the same parameters
        self.configure_light_and_scene()

        self.configure_photo_shoot_ground_truth()

        # Remove forest by setting trees to zero
        self.configure_forest_ground_truth()

        # Save the world configuration
        self.save_world_config()

        # Launch the simulation
        self.launch_simulation()

    def configure_photo_shoot_ground_truth(self):
        person_config = PersonConfig()
        person_config.set_model_pose("idle")                 # Must match a .dae mesh file                                # in the respective model!
        person_config.set_temperature(293)                      # In Kelvin
        person_config.set_pose(gzm.Pose3d(0, 0, -1000, 0, 0, 0))    # First three values are x, y, z coordinates 
        self.world_config.add_plugin(person_config)
    
        photo_shoot_config = PhotoShootConfig()

        # Use the same folder
        photo_shoot_config.set_directory(self.patch_folder)

        # Set a different prefix for the ground truth images
        img_Name = f"GT"
        photo_shoot_config.set_prefix(img_Name)

        # Set camera properties as before
        photo_shoot_config.set_direct_thermal_factor(self.direct_sun)  # direct sunlight TODO

        photo_shoot_config.set_save_rgb(False)
        photo_shoot_config.set_save_thermal(True)
        photo_shoot_config.set_save_depth(False)

        # Set only one pose at (0,0,35)
        pose = gzm.Pose3d(0, 0, 35.0, 0.0, 1.57079632679, 0.0)
        photo_shoot_config.add_pose(pose)

        self.world_config.add_plugin(photo_shoot_config)

    def configure_forest_ground_truth(self):
        self.forest_config = ForestConfig()
        forest_config = self.forest_config

        # Set forest generation but with zero trees
        forest_config.set_generate(True)
        forest_config.set_trees(0)
        forest_config.set_size(37)
        forest_config.set_direct_spawning(True)
        forest_config.set_texture_size(37)
        forest_config.set_seed(self.seed)                 # Change the seed for multiple runs!

        # Use the same ground thermal texture
        
        forest_config.set_ground_thermal_texture(
            os.path.join(self.thermal_texture_env_dir, self.thermal_texture),
            # 0,
            0, #self.min_ground_temp_K, # Minimal temperature in Kelvin
            655#self.max_ground_temp_K # Maximal temperature in Kelvin
        )

        self.world_config.add_plugin(forest_config)


if __name__ == "__main__":
    ttime = time.time()
    simulation_runner = SimulationRunner()
    simulation_runner.run()
    print(time.time() - ttime)
    print(f"Uniform texture count {simulation_runner.uniform_texture_count}")
    print(f"normal texture count {simulation_runner.normal_texture_count}")
