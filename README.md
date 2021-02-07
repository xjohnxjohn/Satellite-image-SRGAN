# Satellite-image-SRGAN using PyTorch
 Using GAN to do super resolution of satellite images.

![index_18_psnr_19.6009_ssim_0.6538](http://imghost.cx0512.com/images/2021/02/07/20210207202646.png)

<p style="text-align:center;">Fig.1. From left to right: ground truth; srgan result; bicubic result.</p>

This is a mini project fulfilled for ELEG5491: Introduction to Deep Learning, The Chinese University of Hong Kong. The course website is [here](http://dl.ee.cuhk.edu.hk/).  

The basic concept comes from the paper  [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

For the usage of this code, please walk through the below steps. 

## 0. System environment

| NO.  | Parameter        | Value       | NO.  | Parameter       | Value         |
| ---- | ---------------- | ----------- | ---- | --------------- | ------------- |
| 1    | Operating system | Ubuntu 18.4 | 6    | CPU version     | Intel Xeon(R) |
| 2    | GPU version      | Telsa K80   | 7    | CPU memory      | 47GB          |
| 3    | GPU memory       | 10GB        | 8    | CPU clock speed | 2.5GHz        |
| 4    | GPU number       | 3           | 9    | CPU core number | 12            |
| 5    | CUDA version     | 10.2        | 10   | CPU number      | 4             |

## 1. Create the dataset

First, go to the [source file folder](./My_dataset/source) to generate our own dataset. The source image is obtained by a satellite pushbroom scanning method with multiple stitches. Its original size is 1 x 1197 x 50500. We will use the[ crop_data script](./My_dataset/crop_data.py) to randomly crop the source image and generate the training, validation and testing dataset. 

In this case, we will generate 96000 training images, 3000 validation images, and 1000 testing images, respectively, with the shape of 1 x 128 x 128.

## Reference

1. [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)
2. [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) 