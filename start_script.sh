#!/bin/bash
# Pre-requisites for ubuntu docker
apt-get update
apt-get install ffmpeg libsm6 libxext6 -y
pip install opencv-python numpy tqdm torch setuptools torchvision natsort tensorboard qiskit