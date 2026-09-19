import os
import argparse
from PIL import Image
from rembg import remove
from utils.sam_utils import resize_image


def process_images(remove_bg):
    all_images = os.listdir(BASE_PATH)

    for image_name in all_images:
        base_name, _ = os.path.splitext(image_name)

        # Create directories for saving processed images
        image_dir = os.path.join(SAVE_BASE_PATH, base_name, 'images')
        os.makedirs(image_dir, exist_ok=True)

        # Open the image
        image_path = os.path.join(BASE_PATH, image_name)

        with Image.open(image_path) as img_pil:
            if remove_bg:
                # Use Rembg to remove the background and get the mask
                img_pil = remove(img_pil)
            img_pil = resize_image(img_pil, 1280)

            # Save the processed image
            mask_save_path = os.path.join(image_dir, '001.png')
            img_pil.save(mask_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images with optional background removal.")
    parser.add_argument('--remove', action='store_true',default=True, help="Remove background from images.")
    parser.add_argument('--folder_path', type=str, default='gp_cases', help="Path to the folder containing the images.")
    
    args = parser.parse_args()
    # Constants for directory paths
    BASE_PATH = args.folder_path
    SAVE_BASE_PATH = args.folder_path + '_dirs'

    process_images(args.remove)