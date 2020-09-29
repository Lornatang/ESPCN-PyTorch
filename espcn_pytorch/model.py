# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
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
import torch.nn as nn
from torch.cuda import amp


class ESPCN(nn.Module):
    def __init__(self, scale_factor):
        super(ESPCN, self).__init__()

        # Feature mapping
        self.feature_maps = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        # Sub-pixel convolution layer
        self.sub_pixel = nn.Sequential(
            nn.Conv2d(32, 1 * (scale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Sigmoid()
        )

    @amp.autocast()
    def forward(self, inputs):
        out = self.feature_maps(inputs)
        out = self.sub_pixel(out)
        return out
