import os
import cv2
import torch
import argparse
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from utils.sam_utils import create, seed_everything,save_gpt_input




def sam_image(sam, base_path):

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=300,
    )

    # Process each dataset
    for dataset_id in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset_id)
        img_folder = os.path.join(dataset_path, 'images')

        data_list = sorted(os.listdir(img_folder))

        img_list = []
        alpha_list = []

        for data_path in data_list:
            image_path = os.path.join(img_folder, data_path)
            image_rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
            alpha = image_rgba[:, :, 3]

            # Ensure alpha mask is binary
            alpha[alpha < 125] = 0
            alpha[alpha >= 125] = 255

            image = cv2.imread(image_path)
            image = torch.from_numpy(image)

            img_list.append(image)
            alpha_list.append(alpha[None, ...])

        # Prepare images and alphas for processing
        images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
        imgs = torch.cat(images)
        alphas = np.concatenate(alpha_list, 0)

        save_folder = os.path.join(dataset_path, 'seg')
        os.makedirs(save_folder, exist_ok=True)

        # Generate segmentation maps
        seg_map_vis = create(imgs, alphas, data_list, save_folder, mask_generator)
    
    return seg_map_vis



if __name__ == '__main__':
    seed_everything(42)

    parser = argparse.ArgumentParser(description = "Part-level Segmentation using SAM")
    parser.add_argument('--dataset_path', type=str, default="gp_cases_dirs")
    parser.add_argument('--sam_ckpt_path', type=str, default="./sam_vit_h_4b8939.pth")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    base_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    sam_image(sam, base_path)
    save_gpt_input(base_path)

    
