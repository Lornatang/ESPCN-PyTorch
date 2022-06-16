#!/bin/bash
# Pre-requisites for AWS ubuntu
apt-get update
apt-get install python3 python3-pip software-properties-common ffmpeg libsm6 libxext6 htop unzip vim cmake  -y
pip install opencv-python numpy tqdm torch setuptools torchvision natsort tensorboard