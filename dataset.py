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
"""Realize the function of dataset preparation."""
import os

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode as IMode

import imgproc

__all__ = ["ImageDataset"]


class ImageDataset(Dataset):
    """Customize the data set loading function and prepare low/high resolution image data in advance.
    Args:
        dataroot         (str): Training data set address
        image_size       (int): High resolution image size
        upscale_factor   (int): Image magnification
        mode             (str): Data set loading method, the training data set is for data enhancement,
                             and the verification data set is not for data enhancement
    """

    def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(ImageDataset, self).__init__()
        self.file_names = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]

        if mode == "train":
            self.hr_transforms = transforms.RandomCrop(image_size)
        else:
            self.hr_transforms = transforms.CenterCrop(image_size)

        self.lr_transforms = transforms.Resize(image_size // upscale_factor, interpolation=IMode.BICUBIC)

    def __getitem__(self, batch_index: int) -> [Tensor, Tensor]:
        # Read a batch of image data
        image = Image.open(self.file_names[batch_index])

        # Transform image
        hr_image = self.hr_transforms(image)
        lr_image = self.lr_transforms(hr_image)

        # Only extract the image data of the Y channel
        lr_image = np.array(lr_image).astype(np.float32)
        hr_image = np.array(hr_image).astype(np.float32)
        lr_ycbcr_image = imgproc.convert_rgb_to_ycbcr(lr_image)
        hr_ycbcr_image = imgproc.convert_rgb_to_ycbcr(hr_image)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_y_tensor = imgproc.image2tensor(lr_ycbcr_image, range_norm=False, half=False)
        hr_y_tensor = imgproc.image2tensor(hr_ycbcr_image, range_norm=False, half=False)

        return lr_y_tensor, hr_y_tensor

    def __len__(self) -> int:
        return len(self.file_names)
