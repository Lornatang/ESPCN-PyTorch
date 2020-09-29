#!/bin/bash

echo "Start downloading pre training model..."
wget https://github.com/Lornatang/ESPCN-PyTorch/releases/download/1.0/espcn_2x.pth
wget https://github.com/Lornatang/ESPCN-PyTorch/releases/download/1.0/espcn_3x.pth
wget https://github.com/Lornatang/ESPCN-PyTorch/releases/download/1.0/espcn_4x.pth
wget https://github.com/Lornatang/ESPCN-PyTorch/releases/download/1.0/espcn_8x.pth
echo "All pre training models have been downloaded!"
