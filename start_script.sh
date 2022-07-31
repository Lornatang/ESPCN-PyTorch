#!/bin/bash
# Pre-requisites for AWS ubuntu
apt-get update
apt-get install python3 python3-pip ffmpeg libsm6 libxext6 htop unzip vim cmake expect -y
pip3 install opencv-python numpy tqdm torch setuptools torchvision natsort tensorboard