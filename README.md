# ESPCN-PyTorch

### Overview
This repository contains an op-for-op PyTorch reimplementation of 
[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158).

### Table of contents
1. [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](#about-real-time-single-image-and-video-super-resolution-using-an-efficient-sub-pixel-convolutional-neural-network)
2. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights)
    * [Download dataset](#download-dataset)
3. [Test](#test)
4. [Train](#train-eg-voc2012)
    * [Example](#example-eg-voc2012)
5. [Contributing](#contributing) 
6. [Credit](#credit)

### About Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network.

If you're new to ESPCN, here's an abstract straight from the paper:

Recently, several models based on deep neural networks have achieved great success in terms of both reconstruction 
accuracy and computational performance for single image super-resolution. In these methods, the low resolution (LR) 
input image is upscaled to the high resolution (HR) space using a single filter, commonly bicubic interpolation, 
before reconstruction. This means that the super-resolution (SR) operation is performed in HR space. We demonstrate 
that this is sub-optimal and adds computational complexity. In this paper, we present the first convolutional neural 
network (CNN) capable of real-time SR of 1080p videos on a single K2 GPU. To achieve this, we propose a novel CNN 
architecture where the feature maps are extracted in the LR space. In addition, we introduce an efficient 
sub-pixel convolution layer which learns an array of upscaling filters to upscale the final LR feature maps into 
the HR output. By doing so, we effectively replace the handcrafted bicubic filter in the SR pipeline with more 
complex upscaling filters specifically trained for each feature map, whilst also reducing the computational complexity 
of the overall SR operation. We evaluate the proposed approach using images and videos from publicly available datasets 
and show that it performs significantly better (+0.15dB on Images and +0.39dB on Videos) and is an order of magnitude 
faster than previous CNN-based methods.

### Installation

#### Clone and install requirements

```bash
git clone https://github.com/Lornatang/ESPCN-PyTorch.git
cd ESPCN-PyTorch/
pip install -r requirements.txt
```

#### Download pretrained weights

```bash
cd weights/
bash download_weights.sh
```

#### Download dataset

```bash
cd data/
bash download_dataset.sh
```

### Test

Evaluate the overall performance of the network.
```bash
usage: test.py [-h] [--dataroot DATAROOT] [--scale-factor {2,3,4,8}]
               [--weights WEIGHTS] [--cuda]

Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel
Convolutional Neural Network..

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   The directory address where the image needs to be
                        processed. (default: `./data/Set5`).
  --scale-factor {2,3,4,8}
                        Image scaling ratio. (default: 4).
  --weights WEIGHTS     Generator model name. (default:`weights/espcn_4x.pth`)
  --cuda                Enables cuda


# Example
python test.py --dataroot ./data/Set5 --scale-factor 4 --weights ./weights/espcn_4x.pth --cuda
```

Evaluate the benchmark of validation data set in the network
```bash
usage: test_benchmark.py [-h] [--dataroot DATAROOT] [-j N]
                         [--image-size IMAGE_SIZE] --scale-factor {2,3,4,8}
                         --weights WEIGHTS [--cuda]

Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel
Convolutional Neural Network.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data/VOC2012`)
  -j N, --workers N     Number of data loading workers. (default:4)
  --image-size IMAGE_SIZE
                        Size of the data crop (squared assumed). (default:256)
  --scale-factor {2,3,4,8}
                        Low to high resolution scaling factor.
  --weights WEIGHTS     Path to weights.
  --cuda                Enables cuda

# Example
python test_benchmark.py --dataroot ./data/VOC2012 --scale-factor 4 --weights ./weights/espcn_4x.pth --cuda
```

Test single picture
```bash
usage: test_image.py [-h] [--file FILE] [--scale-factor {2,3,4,8}]
                     [--weights WEIGHTS] [--cuda]

Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel
Convolutional Neural Network.

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Test low resolution image name.
                        (default:`./assets/baby.png`)
  --scale-factor {2,3,4,8}
                        Super resolution upscale factor. (default:4)
  --weights WEIGHTS     Generator model name. (default:`weights/espcn_4x.pth`)
  --cuda                Enables cuda

# Example
python test_image.py --file ./assets/baby.png --scale-factor 4 ---weights ./weights/espcn_4x.pth -cuda
```

Test single video
```bash
usage: test_video.py [-h] --file FILE --weights WEIGHTS --scale-factor
                     {2,3,4,8} [--view] [--cuda]

Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel
Convolutional Neural Network.

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Test low resolution video name.
  --weights WEIGHTS     Generator model name.
  --scale-factor {2,3,4,8}
                        Super resolution upscale factor. (default:4)
  --view                Super resolution real time to show.
  --cuda                Enables cuda

# Example
python test_video.py --file ./data/1.mp4 --scale-factor 4 --weights ./weights/espcn_4x.pth --view --cuda
```

Low resolution / Recovered High Resolution / Ground Truth

<span align="center"><img src="assets/result.png" alt="">
</span>

### Train (e.g VOC2012)

```bash
usage: train.py [-h] [--dataroot DATAROOT] [-j N] [--epochs N]
                [--image-size IMAGE_SIZE] [-b N] [--lr LR]
                [--scale-factor {2,3,4,8}] [--weights WEIGHTS] [-p N]
                [--manualSeed MANUALSEED] [--cuda]

Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel
Convolutional Neural Network.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data/VOC2012`)
  -j N, --workers N     Number of data loading workers. (default:4)
  --epochs N            Number of total epochs to run. (default:100)
  --image-size IMAGE_SIZE
                        Size of the data crop (squared assumed). (default:256)
  -b N, --batch-size N  mini-batch size (default: 64), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --lr LR               Learning rate. (default:0.01)
  --scale-factor {2,3,4,8}
                        Low to high resolution scaling factor. (default:4).
  --weights WEIGHTS     Path to weights (to continue training).
  -p N, --print-freq N  Print frequency. (default:5)
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:0)
  --cuda                Enables cuda
```

#### Example (e.g VOC2012)

```bash
python train.py --dataroot ./data/VOC2012 --scale-factor 4 --cuda
```

If you want to load weights that you've trained before, run the following command.

```bash
python train.py --dataroot ./data/VOC2012 --scale-factor 4 --weights ./weights/espcn_4x_epoch_100.pth --cuda
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 

### Credit

#### Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
_Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, Zehan Wang_ <br>

**Abstract** <br>
Recently, several models based on deep neural networks have achieved great success in terms of both reconstruction 
accuracy and computational performance for single image super-resolution. In these methods, the low resolution (LR) 
input image is upscaled to the high resolution (HR) space using a single filter, commonly bicubic interpolation, 
before reconstruction. This means that the super-resolution (SR) operation is performed in HR space. We demonstrate 
that this is sub-optimal and adds computational complexity. In this paper, we present the first convolutional neural 
network (CNN) capable of real-time SR of 1080p videos on a single K2 GPU. To achieve this, we propose a novel CNN 
architecture where the feature maps are extracted in the LR space. In addition, we introduce an efficient 
sub-pixel convolution layer which learns an array of upscaling filters to upscale the final LR feature maps into 
the HR output. By doing so, we effectively replace the handcrafted bicubic filter in the SR pipeline with more 
complex upscaling filters specifically trained for each feature map, whilst also reducing the computational complexity 
of the overall SR operation. We evaluate the proposed approach using images and videos from publicly available datasets 
and show that it performs significantly better (+0.15dB on Images and +0.39dB on Videos) and is an order of magnitude 
faster than previous CNN-based methods.

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
