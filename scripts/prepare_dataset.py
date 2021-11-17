# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
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
import argparse
import os
import shutil

from PIL import Image
from tqdm import tqdm


def main(args):
    image_dir = f"{args.output_dir}/x{args.upscale_factor}/train"
    image_size = int(args.image_size * args.upscale_factor)

    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)

    file_names = os.listdir(args.inputs_dir)
    for image_file_name in tqdm(file_names, total=len(file_names)):
        # Use PIL to read high-resolution image
        image = Image.open(f"{args.inputs_dir}/{image_file_name}")

        for pos_x in range(0, image.size[0] - image_size + 1, args.step):
            for pos_y in range(0, image.size[1] - image_size + 1, args.step):
                # crop box xywh
                crop_image = image.crop([pos_x, pos_y, pos_x + image_size, pos_y + image_size])
                # Save all images
                crop_image.save(f"{image_dir}/{image_file_name.split('.')[-2]}_{pos_x}_{pos_y}.{image_file_name.split('.')[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts (Use SRCNN functions).")
    parser.add_argument("--inputs_dir", type=str, default="T91/original", help="Path to input image directory. (Default: `T91/original`)")
    parser.add_argument("--output_dir", type=str, default="T91", help="Path to generator image directory. (Default: `T91`)")
    parser.add_argument("--image_size", type=int, default=17, help="Low-resolution image size from raw image. (Default: 17)")
    parser.add_argument("--step", type=int, default=13, help="Crop image similar to sliding window.  (Default: 13)")
    parser.add_argument("--upscale_factor", type=int, default=2, help="Image zoom scale. (Default: 2)")
    args = parser.parse_args()

    main(args)
