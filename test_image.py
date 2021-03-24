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

import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image

from espcn_pytorch import ESPCN

parser = argparse.ArgumentParser(description="Real-Time Single Image and Video Super-Resolution Using "
                                             "an Efficient Sub-Pixel Convolutional Neural Network.")
parser.add_argument("--file", type=str, default="./assets/baby.png",
                    help="Test low resolution image name. (default:`./assets/baby.png`)")
parser.add_argument("--scale-factor", default=4, type=int, choices=[2, 3, 4, 8],
                    help="Super resolution upscale factor. (default:4)")
parser.add_argument("--weights", type=str, default="weights/espcn_4x.pth",
                    help="Generator model name.  (default:`weights/espcn_4x.pth`)")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")

args = parser.parse_args()
print(args)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
model = ESPCN(scale_factor=args.scale_factor).to(device)

# Load state dicts
model.load_state_dict(torch.load(args.weights, map_location=device))

# Set eval mode
model.eval()

# Open image
image = Image.open(args.file).convert("YCbCr")
y, cb, cr = image.split()

inputs = transforms.ToTensor()(y).view(1, -1, y.size[1], y.size[0])
inputs = inputs.to(device)

with torch.no_grad():
    out = model(inputs)
out = out.cpu()
out_image_y = out[0].detach().numpy()
out_image_y *= 255.0
out_image_y = out_image_y.clip(0, 255)
out_image_y = Image.fromarray(np.uint8(out_image_y[0]), mode="L")

out_image_cb = cb.resize(out_image_y.size, Image.BICUBIC)
out_image_cr = cr.resize(out_image_y.size, Image.BICUBIC)
out_image = Image.merge("YCbCr", [out_image_y, out_image_cb, out_image_cr]).convert("RGB")
# before converting the result in RGB
out_image.save(f"espcn_{args.scale_factor}x.png")
