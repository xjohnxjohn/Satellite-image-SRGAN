import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms



# def display_transform():
#     return transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize(128),
#         transforms.CenterCrop(128),
#         transforms.ToTensor()
#     ])



# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.5])
std = np.array([0.5])


class TrainImageDataset(Dataset):
    def __init__(self, root, hr_shape,scale_factor = 4):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // scale_factor, hr_height // scale_factor), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                # transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.RandomCrop((hr_height, hr_height)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}
        # return img_lr, img_hr

    def __len__(self):
        return len(self.files)


class ValImageDataset(Dataset):
    def __init__(self, root, hr_shape,scale_factor = 4):
        hr_height, hr_width = hr_shape

        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // scale_factor, hr_height // scale_factor), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.hr_transform = transforms.Compose(
            [
                # transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.RandomCrop((hr_height, hr_height)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.hr_restore_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // scale_factor, hr_height // scale_factor), Image.BICUBIC),
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        img_hr_restore = self.hr_restore_transform(img)

        return {"lr": img_lr, "hr": img_hr, "hr_restore": img_hr_restore}
        # return img_lr, img_hr, img_hr_restore
    
    def __len__(self):
        return len(self.files)




class TestImageDataset(Dataset):
    def __init__(self, root, hr_shape, scale_factor=4):
        hr_height, hr_width = hr_shape

        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // scale_factor, hr_height // scale_factor), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.hr_transform = transforms.Compose(
            [
                # transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.RandomCrop((hr_height, hr_height)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.hr_restore_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // scale_factor, hr_height // scale_factor), Image.BICUBIC),
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])

        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        img_hr_restore = self.hr_restore_transform(img)

        return {"lr": img_lr, "hr": img_hr, "hr_restore": img_hr_restore}
        # return img_lr, img_hr, img_hr_restore
    
    def __len__(self):
        return len(self.files)


