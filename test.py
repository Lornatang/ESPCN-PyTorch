# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

import cv2
import numpy as np
import torch
from natsort import natsorted

import imgproc
import model
import config
from image_quality_assessment import PSNR, SSIM
from utils import make_directory

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def main() -> None:
    # Initialize the super-resolution bsrgan_model
    g_model = model.__dict__[config.model_arch_name](in_channels=config.in_channels,
                                                     out_channels=config.out_channels,
                                                     channels=config.channels)
    g_model = g_model.to(device=config.device)
    print(f"Build `{config.model_arch_name}` model successfully.")

    # Load the super-resolution bsrgan_model weights
    checkpoint = torch.load(config.model_weights_path, map_location=lambda storage, loc: storage)
    g_model.load_state_dict(checkpoint["state_dict"])
    print(f"Load `{config.model_arch_name}` model weights "
          f"`{os.path.abspath(config.model_weights_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    make_directory(config.sr_dir)

    # Start the verification mode of the bsrgan_model.
    g_model.eval()

    # Initialize the sharpness evaluation function
    psnr = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim = SSIM(config.upscale_factor, config.only_test_y_channel)

    # Set the sharpness evaluation function calculation device to the specified model
    psnr = psnr.to(device=config.device, non_blocking=True)
    ssim = ssim.to(device=config.device, non_blocking=True)

    # Initialize IQA metrics
    psnr_metrics = 0.0
    ssim_metrics = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(config.lr_dir, file_names[index])
        sr_image_path = os.path.join(config.sr_dir, file_names[index])
        gt_image_path = os.path.join(config.gt_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        gt_y_tensor, gt_cb_image, gt_cr_image = imgproc.preprocess_one_image(gt_image_path, config.device)
        lr_y_tensor, lr_cb_image, lr_cr_image = imgproc.preprocess_one_image(lr_image_path, config.device)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_y_tensor = g_model(lr_y_tensor)

        # Save image
        sr_y_image = imgproc.tensor_to_image(sr_y_tensor, range_norm=False, half=True)
        sr_y_image = sr_y_image.astype(np.float32) / 255.0
        sr_ycbcr_image = cv2.merge([sr_y_image, gt_cb_image, gt_cr_image])
        sr_image = imgproc.ycbcr_to_bgr(sr_ycbcr_image)
        cv2.imwrite(sr_image_path, sr_image * 255.0)

        # Cal IQA metrics
        psnr_metrics += psnr(sr_y_tensor, gt_y_tensor).item()
        ssim_metrics += ssim(sr_y_tensor, gt_y_tensor).item()

    # Calculate the average value of the sharpness evaluation index,
    # and all index range values are cut according to the following values
    # PSNR range value is 0~100
    # SSIM range value is 0~1
    avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files
    avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files

    print(f"PSNR: {avg_psnr:4.2f} [dB]\n"
          f"SSIM: {avg_ssim:4.4f} [u]")


if __name__ == "__main__":
    main()
