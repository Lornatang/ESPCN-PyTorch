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
"""Realize the parameter configuration function of dataset, model, training and verification code."""
import torch
from torch.backends import cudnn

# ==============================================================================
# General configuration
# ==============================================================================
# Random seed to maintain reproducible results
torch.manual_seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Image magnification factor
upscale_factor = 2
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "espcn_x2"

if mode == "train":
    # Dataset
    train_image_dir = f"data/T91/ESPCN/train"
    valid_image_dir = f"data/T91/ESPCN/valid"

    image_size = int(upscale_factor * 17)
    batch_size = 16
    num_workers = 4

    # Incremental training and migration training
    resume = False
    strict = True
    start_epoch = 0
    resume_weight = ""

    # Total num epochs
    epochs = 2583

    # SGD optimizer parameter
    model_lr = 1e-2
    model_betas = (0.9, 0.999)

    # Optimizer scheduler parameter
    lr_scheduler_milestones = [int(epochs * 0.1), int(epochs * 0.8)]
    lr_scheduler_gamma = 0.1

    print_frequency = 50

if mode == "valid":
    # Test data address
    lr_dir = f"data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"results/test/{exp_name}"
    hr_dir = f"data/Set5/GTmod12"

    model_path = f"results/{exp_name}/last.pth"
