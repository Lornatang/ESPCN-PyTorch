# ESPCN-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation of [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158v2).

## Table of contents

- [ESPCN-PyTorch](#espcn-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](#about-real-time-single-image-and-video-super-resolution-using-an-efficient-sub-pixel-convolutional-neural-network)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test ESPCN_x4](#test-espcn_x4)
        - [Train ESPCN_x4](#train-espcn_x4)
        - [Resume ESPCN_x4](#resume-train-espcn_x4)
    - [Result](#result)
    - [Credit](#credit)
        - [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](#real-time-single-image-and-video-super-resolution-using-an-efficient-sub-pixel-convolutional-neural-network)

## About Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network

If you're new to ESPCN, here's an abstract straight from the paper:

Recently, several models based on deep neural networks have achieved great success in terms of both reconstruction accuracy and computational
performance for single image super-resolution. In these methods, the low resolution (LR)
input image is upscaled to the high resolution (HR) space using a single filter, commonly bicubic interpolation, before reconstruction. This means
that the super-resolution (SR) operation is performed in HR space. We demonstrate that this is sub-optimal and adds computational complexity. In this
paper, we present the first convolutional neural network (CNN) capable of real-time SR of 1080p videos on a single K2 GPU. To achieve this, we propose
a novel CNN architecture where the feature maps are extracted in the LR space. In addition, we introduce an efficient sub-pixel convolution layer
which learns an array of upscaling filters to upscale the final LR feature maps into the HR output. By doing so, we effectively replace the
handcrafted bicubic filter in the SR pipeline with more complex upscaling filters specifically trained for each feature map, whilst also reducing the
computational complexity of the overall SR operation. We evaluate the proposed approach using images and videos from publicly available datasets and
show that it performs significantly better (+0.15dB on Images and +0.39dB on Videos) and is an order of magnitude faster than previous CNN-based
methods.

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test ESPCN_x4

Modify the `config.py` file.

- line 31: `model_arch_name` change to `espcn_x4`.
- line 36: `upscale_factor` change to `4`.
- line 38: `mode` change to `test`.
- line 40: `exp_name` change to `ESPCN_x4-Set5`.
- line 84: `lr_dir` change to `f"./data/Set5/LRbicx{upscale_factor}"`.
- line 86: `gt_dir` change to `f"./data/Set5/GTmod12"`.
- line 88: `model_weights_path` change to `./results/pretrained_models/ESPCN_x4-T91-64bf5ee4.pth.tar`.

```bash
python3 test.py
```

### Train ESPCN_x4

Modify the `config.py` file.

- line 31: `model_arch_name` change to `espcn_x4`.
- line 36: `upscale_factor` change to `4`.
- line 38: `mode` change to `test`.
- line 40: `exp_name` change to `ESPCN_x4-Set5`.
- line 84: `lr_dir` change to `f"./data/Set5/LRbicx{upscale_factor}"`.
- line 86: `gt_dir` change to `f"./data/Set5/GTmod12"`.

```bash
python3 train.py
```

### Resume train ESPCN_x4

Modify the `config.py` file.

- line 31: `model_arch_name` change to `espcn_x4`.
- line 36: `upscale_factor` change to `4`.
- line 38: `mode` change to `test`.
- line 40: `exp_name` change to `ESPCN_x4-Set5`.
- line 57: `resume_model_weights_path` change to `./samples/ESPCN_x4-Set5/epoch_xxx.pth.tar`.
- line 84: `lr_dir` change to `f"./data/Set5/LRbicx{upscale_factor}"`.
- line 86: `gt_dir` change to `f"./data/Set5/GTmod12"`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1609.05158v2.pdf](https://arxiv.org/pdf/1609.05158v2.pdf)

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

|  Method  | Scale |   Set5 (PSNR)    |   Set14 (PSNR)   |
|:--------:|:-----:|:----------------:|:----------------:|
| ESPCN_x4 |   2   |   -(**36.64**)   |   -(**32.35**)   |
| ESPCN_x3 |   3   | 32.55(**32.55**) | 29.08(**29.20**) |
| ESPCN_x4 |   4   | 30.90(**30.26**) | 27.73(**27.41**) |

```bash
# Download `ESPCN_x4-T91-64bf5ee4.pth.tar` weights to `./results/pretrained_models/ESPCN_x4-T91-64bf5ee4.pth.tar`
# More detail see `README.md<Download weights>`
python3 ./inference.py
```

Input:

<span align="center"><img width="240" height="360" src="figure/comic.png"/></span>

Output:

<span align="center"><img width="240" height="360" src="figure/sr_comic.png"/></span>

```text
Build `espcn_x4` model successfully.
Load `espcn_x4` model weights `./results/pretrained_models/ESPCN_x4-T91-64bf5ee4.pth.tar` successfully.
SR image save to `./figure/sr_comic.png`
```

### Credit

#### Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network

_Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, Zehan Wang_ <br>

**Abstract** <br>
Recently, several models based on deep neural networks have achieved great success in terms of both reconstruction accuracy and computational
performance for single image super-resolution. In these methods, the low resolution (LR)
input image is upscaled to the high resolution (HR) space using a single filter, commonly bicubic interpolation, before reconstruction. This means
that the super-resolution (SR) operation is performed in HR space. We demonstrate that this is sub-optimal and adds computational complexity. In this
paper, we present the first convolutional neural network (CNN) capable of real-time SR of 1080p videos on a single K2 GPU. To achieve this, we propose
a novel CNN architecture where the feature maps are extracted in the LR space. In addition, we introduce an efficient sub-pixel convolution layer
which learns an array of upscaling filters to upscale the final LR feature maps into the HR output. By doing so, we effectively replace the
handcrafted bicubic filter in the SR pipeline with more complex upscaling filters specifically trained for each feature map, whilst also reducing the
computational complexity of the overall SR operation. We evaluate the proposed approach using images and videos from publicly available datasets and
show that it performs significantly better (+0.15dB on Images and +0.39dB on Videos) and is an order of magnitude faster than previous CNN-based
methods.

[[Paper]](https://arxiv.org/pdf/1609.05158)

```
@article{DBLP:journals/corr/ShiCHTABRW16,
  author    = {Wenzhe Shi and
               Jose Caballero and
               Ferenc Husz{\'{a}}r and
               Johannes Totz and
               Andrew P. Aitken and
               Rob Bishop and
               Daniel Rueckert and
               Zehan Wang},
  title     = {Real-Time Single Image and Video Super-Resolution Using an Efficient
               Sub-Pixel Convolutional Neural Network},
  journal   = {CoRR},
  volume    = {abs/1609.05158},
  year      = {2016},
  url       = {http://arxiv.org/abs/1609.05158},
  archivePrefix = {arXiv},
  eprint    = {1609.05158},
  timestamp = {Mon, 13 Aug 2018 16:47:09 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/ShiCHTABRW16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
