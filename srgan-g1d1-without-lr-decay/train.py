# import ipdb
import argparse
import os
import numpy as np
import math
import itertools
import sys
import time
import datetime
import glob
import random
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
import torchvision.transforms as transforms
#import torchvision.utils as utils
#import torchvision.transforms.functional as F
from torchvision.utils import save_image, make_grid
from torchvision.models import vgg19

from math import log10
from tqdm import tqdm
import pandas as pd

from PIL import Image
from visualize import Visualizer
from torchnet.meter import AverageValueMeter
from models import *
from datasets import *
import pytorch_ssim

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--train_dataset_name", type=str, default="train", help="name of the train dataset")
parser.add_argument("--val_dataset_name", type=str, default="val", help="name of the val dataset")
parser.add_argument("--train_batch_size", type=int, default=128, help="size of the train batches")
parser.add_argument("--val_batch_size", type=int, default=1, help="size of the val batches")
parser.add_argument('--generatorLR', type=float, default=0.0002, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.0002, help='learning rate for discriminator')
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
# parser.add_argument("--decay_epoch", type=int, default=50, help="start lr decay every decay_epoch epochs")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=128, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=128, help="high res. image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument('--scale_factor', default=4, type=int, choices=[2, 4, 8], help='super resolution scale factor')  
parser.add_argument("--g_every", type=int, default=1, help="train the generator every g_every batches")
parser.add_argument("--d_every", type=int, default=1, help="train the discriminator every d_every batches")
parser.add_argument("--plot_every", type=int, default=100, help="plot using visdom every plot_every samples")
parser.add_argument("--save_every", type=int, default=1, help="save the model every save_every epochs")
opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 5"
os.makedirs("saved_models", exist_ok=True)
os.makedirs("images", exist_ok=True)
vis = Visualizer('SRGAN_new')

# cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorResNet(in_channels=opt.channels, out_channels=opt.channels, n_residual_blocks=16)      # change
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()
print('# generator parameters:', sum(param.numel() for param in generator.parameters()))                    # change
print('# discriminator parameters:', sum(param.numel() for param in discriminator.parameters()))            # change
print('# feature_extractor parameters:', sum(param.numel() for param in feature_extractor.parameters()))    # change
# print (generator)
# print (discriminator)
# print (feature_extractor)

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss(reduction='none')
criterion_content = torch.nn.L1Loss(reduction='none')

# Configure model
generator = nn.DataParallel(generator, device_ids=[0, 1, 2])
generator.to(device)
discriminator = nn.DataParallel(discriminator, device_ids=[0, 1, 2])
discriminator.to(device)
feature_extractor = nn.DataParallel(feature_extractor, device_ids=[0, 1, 2])
feature_extractor.to(device)
# criterion_GAN = nn.DataParallel(criterion_GAN, device_ids=[0, 1, 2])
# criterion_GAN = criterion_GAN.to(device)
# criterion_content = nn.DataParallel(criterion_content, device_ids=[0, 1, 2])
# criterion_content = criterion_content.to(device)


if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d_%d.pth" % (opt.scale_factor,opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d_%d.pth" % (opt.scale_factor,opt.epoch)))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.generatorLR, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR, betas=(opt.b1, opt.b2))

# scheduler_G = MultiStepLR(optimizer_G, milestones=[opt.decay_epoch, 2 * opt.decay_epoch, 3 * opt.decay_epoch], gamma=0.1)
# scheduler_D = MultiStepLR(optimizer_D, milestones=[opt.decay_epoch, 2 * opt.decay_epoch, 3 * opt.decay_epoch], gamma=0.1)


# Configure data loader
train_dataloader = DataLoader(
    TrainImageDataset("../../Datasets/My_dataset/single_channel_100000/%s" % opt.train_dataset_name, hr_shape=hr_shape, scale_factor = opt.scale_factor),                # change
    batch_size=opt.train_batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    ValImageDataset("../../Datasets/My_dataset/single_channel_100000/%s" % opt.val_dataset_name, hr_shape=hr_shape, scale_factor = opt.scale_factor),                # change
    batch_size=opt.val_batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

loss_GAN_meter = AverageValueMeter()
loss_content_meter = AverageValueMeter()
loss_G_meter= AverageValueMeter()
loss_real_meter = AverageValueMeter()
loss_fake_meter = AverageValueMeter()
loss_D_meter = AverageValueMeter()

# ----------
#  Training
# ----------

results = {'loss_G': [], 'loss_D': [], 'loss_GAN': [],'loss_content': [], 'loss_real': [], 'loss_fake': [], 'psnr': [], 'ssim': []}   

epoch_start = time.time()
for epoch in range(opt.epoch, opt.n_epochs):

    training_results = {'batch_sizes': 0, 'loss_G': 0, 'loss_D': 0, 'loss_GAN': 0, 'loss_content': 0, 'loss_real': 0, 'loss_fake': 0}  

    generator.train()                                       
    discriminator.train()
    training_out_path = 'training_results/SR_factor_' + str(opt.scale_factor) + '/' + 'epoch_' + str(epoch) + '/'
    os.makedirs(training_out_path, exist_ok=True)

    for i, imgs in enumerate(train_dataloader):
        start = time.time()

        training_results['batch_sizes'] += opt.train_batch_size    

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.module.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.module.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------
        if i % opt.g_every == 0:
            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Adversarial loss
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
            loss_GAN = loss_GAN.mean()
            
            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())
            loss_content = loss_content.mean()
            
            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN
            loss_G = loss_G.mean()

            loss_G.backward(torch.ones_like(loss_G))
            optimizer_G.step()
            # scheduler_G.step()


            loss_GAN_meter.add(loss_GAN.item())
            loss_content_meter.add(loss_content.item())
            loss_G_meter.add(loss_G.item())

        # ---------------------
        #  Train Discriminator
        # ---------------------
        if i % opt.d_every == 0:
            optimizer_D.zero_grad()

            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
            loss_real = loss_real.mean()
            loss_fake = loss_fake.mean()

            # Total loss
            loss_D = (loss_real + loss_fake) / 2
            loss_D = loss_D.mean()

            loss_D.backward(torch.ones_like(loss_D))
            optimizer_D.step()
            # scheduler_D.step()

            loss_real_meter.add(loss_real.item())
            loss_fake_meter.add(loss_fake.item())
            loss_D_meter.add(loss_D.item())


        # --------------
        #  Log Progress
        # --------------

        # loss for current batch before optimization 
        training_results['loss_G'] +=  loss_G.item() * opt.train_batch_size                    
        training_results['loss_D'] +=  loss_D.item() * opt.train_batch_size                    
        training_results['loss_GAN'] +=  loss_GAN.item() * opt.train_batch_size                
        training_results['loss_content'] +=  loss_content.item() * opt.train_batch_size        
        training_results['loss_real'] +=  loss_real.item() * opt.train_batch_size              
        training_results['loss_fake'] +=  loss_fake.item() * opt.train_batch_size

        batch_time = time.time() - start              
        print('[Epoch %d/%d] [Batch %d/%d] [loss_G: %.4f] [loss_D: %.4f] [loss_GAN: %.4f] [loss_content: %.4f] [loss_real: %.4f] [loss_fake: %.4f] [batch time: %.4fs]' % (
                epoch, opt.n_epochs, i, len(train_dataloader), training_results['loss_G'] / training_results['batch_sizes'],
                training_results['loss_D'] / training_results['batch_sizes'],
                training_results['loss_GAN'] / training_results['batch_sizes'],
                training_results['loss_real'] / training_results['batch_sizes'],
                training_results['loss_content'] / training_results['batch_sizes'],
                training_results['loss_fake'] / training_results['batch_sizes'],
                batch_time))


        # Save training image and plot loss
        batches_done = epoch * len(train_dataloader) + i
        if batches_done % opt.plot_every == 0:            
#            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=opt.scale_factor)
#            gen_hr = make_grid(gen_hr, nrow=8, normalize=True)
#            imgs_lr = make_grid(imgs_lr, nrow=8, normalize=True)
#            img_grid = torch.cat((imgs_lr, gen_hr), -1)
#            save_image(img_grid, "images/%d.png" % batches_done, normalize=True)
            
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=opt.scale_factor)
            training_out_imgs_lr_path = training_out_path + "imgs_lr/"
            training_out_imgs_hr_path = training_out_path + "imgs_hr/"
            training_out_gen_hr_path = training_out_path + "gen_hr/"
            os.makedirs(training_out_imgs_lr_path, exist_ok=True)
            os.makedirs(training_out_imgs_hr_path, exist_ok=True)
            os.makedirs(training_out_gen_hr_path, exist_ok=True)
#            save_image(imgs_lr.detach()[:1], training_out_imgs_lr_path + "imgs_lr_%d.png" % batches_done, normalize=True)
#            save_image(imgs_hr.data[:1], training_out_imgs_hr_path + "imgs_hr_%d.png" % batches_done, normalize=True)
#            save_image(gen_hr.data[:1], training_out_gen_hr_path + "gen_hr_%d.png" % batches_done, normalize=True)
            save_image(imgs_lr[:1], training_out_imgs_lr_path + "imgs_lr_%d.png" % batches_done, normalize=True)
            save_image(imgs_hr[:1], training_out_imgs_hr_path + "imgs_hr_%d.png" % batches_done, normalize=True)
            save_image(gen_hr[:1], training_out_gen_hr_path + "gen_hr_%d.png" % batches_done, normalize=True)

            gen_hr = make_grid(gen_hr, nrow=8, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=8, normalize=True)
            imgs_hr = make_grid(imgs_hr, nrow=8, normalize=True)
            img_grid_gl = torch.cat((gen_hr, imgs_lr), -1)
            img_grid_hg = torch.cat((imgs_hr, gen_hr), -1)
            save_image(img_grid_hg, "images/%d_hg.png" % batches_done, normalize=True)
            save_image(img_grid_gl, "images/%d_gl.png" % batches_done, normalize=True)

            # vis.images(imgs_lr.detach().cpu().numpy()[:1] * 0.5 + 0.5, win='imgs_lr_train')
            # vis.images(gen_hr.data.cpu().numpy()[:1] * 0.5 + 0.5, win='img_gen_train')
            # vis.images(imgs_hr.data.cpu().numpy()[:1] * 0.5 + 0.5, win='img_hr_train')
            vis.plot('loss_G_train', loss_G_meter.value()[0])
            vis.plot('loss_D_train', loss_D_meter.value()[0])
            vis.plot('loss_GAN_train', loss_GAN_meter.value()[0])
            vis.plot('loss_content_train', loss_content_meter.value()[0])           
            vis.plot('loss_real_train', loss_real_meter.value()[0])
            vis.plot('loss_fake_train', loss_fake_meter.value()[0])
   
    loss_GAN_meter.reset()
    loss_content_meter.reset()
    loss_G_meter.reset()
    loss_real_meter.reset()
    loss_fake_meter.reset()
    loss_D_meter.reset()


    # validate the generator model
    generator.eval()
    valing_out_path = 'valing_results/SR_factor_' + str(opt.scale_factor) + '/' + 'epoch_' + str(epoch) + '/'
    os.makedirs(valing_out_path, exist_ok=True)
        
    with torch.no_grad():
        # val_bar = tqdm(val_dataloader)
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        val_images = []
        for i, imgs in enumerate(val_dataloader):
            start = time.time()

            valing_results['batch_sizes'] += opt.val_batch_size    

            # Configure model input
            #img_lr, img_hr, img_hr_restore = imgs
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))
            img_hr_restore = Variable(imgs["hr_restore"].type(Tensor))
            gen_hr = generator(imgs_lr)

            batch_mse = ((gen_hr - imgs_hr) ** 2).data.mean()
            valing_results['mse'] += batch_mse * opt.val_batch_size
            batch_ssim = pytorch_ssim.ssim(gen_hr, imgs_hr).item()
            valing_results['ssims'] += batch_ssim * opt.val_batch_size
            valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

            # val_bar.set_description(desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (valing_results['psnr'], valing_results['ssim']), refresh=True)            
            print('[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (valing_results['psnr'], valing_results['ssim']))
            val_images.extend(
                [imgs_hr.data.cpu().squeeze(0), gen_hr.data.cpu().squeeze(0),
                 img_hr_restore.data.cpu().squeeze(0)])
        
        val_images = torch.stack(val_images) # 将list重新堆成４维张量
        # val_images = torch.chunk(val_images, val_images.size(0) // 15) # 若验证集大小为3000，则3000=15*200,15=3*5,生成的每张图片中共有15张子图
        val_images = torch.chunk(val_images, val_images.size(0) // 3)
        val_save_bar = tqdm(val_images, desc='[saving training results]')
        index = 1
        for image in val_save_bar:
            image = make_grid(image, nrow=3, padding=5, normalize=True)
            save_image(image, valing_out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5, normalize=True)
            index += 1
                

    # save loss\scores\psnr\ssim and visualize
    results['loss_G'].append(training_results['loss_G'] / training_results['batch_sizes'])
    results['loss_D'].append(training_results['loss_D'] / training_results['batch_sizes'])
    results['loss_GAN'].append(training_results['loss_GAN'] / training_results['batch_sizes'])
    results['loss_content'].append(training_results['loss_content'] / training_results['batch_sizes'])
    results['loss_real'].append(training_results['loss_real'] / training_results['batch_sizes'])
    results['loss_fake'].append(training_results['loss_fake'] / training_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])     
    
    vis.plot('loss_G_epoch', results['loss_G'][epoch])
    vis.plot('loss_D_epoch', results['loss_D'][epoch])
    vis.plot('loss_GAN_epoch', results['loss_GAN'][epoch])
    vis.plot('loss_content_epoch', results['loss_content'][epoch])
    vis.plot('loss_real_epoch', results['loss_real'][epoch])
    vis.plot('loss_fake_epoch', results['loss_fake'][epoch])
    vis.plot('psnr_epoch', results['psnr'][epoch])
    vis.plot('ssim_epoch', results['ssim'][epoch])



    # save model parameters
    data_out_path = './statistics/'
    os.makedirs(data_out_path, exist_ok=True)
    if epoch % opt.save_every == 0:
        # save_image(gen_hr.data[:16], 'images/%s.png' % epoch, normalize=True,range=(-1, 1))
        torch.save(generator.state_dict(), "saved_models/generator_%d_%d.pth" % (opt.scale_factor,epoch))
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d_%d.pth" % (opt.scale_factor,epoch))

        data_frame = pd.DataFrame(
            data={'loss_G': results['loss_G'], 'loss_D': results['loss_D'], 
                    'loss_GAN': results['loss_GAN'], 'loss_content': results['loss_content'], 
                    'loss_real': results['loss_real'], 'loss_fake': results['loss_fake'],
                    'PSNR': results['psnr'], 'SSIM': results['ssim']},
            # index=range(0, epoch)
            index=None
            )
        data_frame.to_csv(data_out_path + 'SR_factor_' + str(opt.scale_factor) + '_train_results.csv', index_label='Epoch')

elapse_time = time.time() - epoch_start
elapse_time = datetime.timedelta(seconds=elapse_time)
print("Training and validating time {}".format(elapse_time))