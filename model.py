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
from math import sqrt

import torch
from torch import nn

import config
from qnn.quantum_conv2d import QuantumConv2d

class ESPCN(nn.Module):
    """

    Args:
        upscale_factor (int): Image magnification factor.
    """

    def __init__(self) -> None:
        super(ESPCN, self).__init__()

        # Quantum feature mapping
        self.quantum_feature_map = nn.Sequential(
            QuantumConv2d(1, 1, (3, 3), (1, 1), config.backend, config.shots, config.shift),
        )

        # Feature mapping
        self.feature_maps = nn.Sequential(
            nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2)),
            nn.Tanh(),
            nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
            nn.Tanh(),
        )

        # Sub-pixel convolution layer
        self.sub_pixel = nn.Sequential(
            nn.Conv2d(32, config.upscale_factor ** 2, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(config.upscale_factor),
        )

        # Initial model weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantum_feature_map(x)
        x = self.feature_maps(x)
        x = self.sub_pixel(x)

        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == 32:
                    nn.init.normal_(module.weight.data, 0.0, 0.001)
                    nn.init.zeros_(module.bias.data)
                else:
                    nn.init.normal_(module.weight.data, 0.0, sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                    nn.init.zeros_(module.bias.data)
