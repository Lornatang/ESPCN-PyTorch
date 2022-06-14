#!/bin/bash
# Pre-requisites for ubuntu docker
apt-get update
apt-get install software-properties-common ffmpeg libsm6 libxext6 htop unzip vim nvidia-opencl-dev cmake -y
pip install opencv-python numpy tqdm torch setuptools torchvision natsort tensorboard
add-apt-repository ppa:ubuntu-toolchain-r/test -y
apt-get install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 10
ln -s /usr/bin/gcc /usr/bin/cc
ln -s /usr/bin/g++ /usr/bin/c++