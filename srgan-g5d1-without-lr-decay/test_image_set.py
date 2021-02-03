import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

import pytorch_ssim
from datasets import TestImageDataset
from models import GeneratorResNet

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 5"

parser = argparse.ArgumentParser(description='Test Datasets')
parser.add_argument('--scale_factor', default=4, type=int, help='super resolution scale factor')
parser.add_argument('--model_name', default='generator_4_99.pth', type=str, help='GeneratorResNet generator epoch name')
parser.add_argument("--test_dataset_name", type=str, default="test", help="name of the test dataset")
parser.add_argument("--hr_height", type=int, default=128, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=128, help="high res. image width")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
opt = parser.parse_args()

SCALE_FACTOR = opt.scale_factor
MODEL_NAME = opt.model_name
hr_shape = (opt.hr_height, opt.hr_width)

results = {'Test': {'psnr': [], 'ssim': []}}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator = GeneratorResNet()
generator = nn.DataParallel(generator, device_ids=[0, 1, 2])
generator.to(device)

# generator.load_state_dict(torch.load("saved_models/generator_%d_%d.pth" % (4,99)))
generator.load_state_dict(torch.load("saved_models/" + MODEL_NAME))
generator.eval()


test_dataloader = DataLoader(
    TestImageDataset("../../Datasets/My_dataset/single_channel_100000/%s" % opt.test_dataset_name, hr_shape=hr_shape, scale_factor = opt.scale_factor),                # change
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu,
)

test_bar = tqdm(test_dataloader, desc='[testing datasets]')

test_out_path = 'testing_results/SRF_' + str(SCALE_FACTOR) + '/'
if not os.path.exists(test_out_path):
    os.makedirs(test_out_path)


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
with torch.no_grad():
    test_images=[]
    for imgs in test_bar:

        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))
        imgs_hr_restore = Variable(imgs["hr_restore"].type(Tensor))

        gen_hr = generator(imgs_lr)
        mse = ((imgs_hr - gen_hr) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
        # ssim = pytorch_ssim.ssim(gen_hr, imgs_hr).data[0]
        ssim = pytorch_ssim.ssim(gen_hr, imgs_hr).item()
        # save psnr\ssim
        results['Test']['psnr'].append(psnr)
        results['Test']['ssim'].append(ssim)

        test_images.extend(
            [imgs_hr.data.cpu().squeeze(0), gen_hr.data.cpu().squeeze(0),
            imgs_hr_restore.data.cpu().squeeze(0)])

    test_images = torch.stack(test_images)    
    test_images = torch.chunk(test_images, test_images.size(0) // 3)
    test_save_bar = tqdm(test_images, desc='[saving test results]')
    index = 1
    for image in test_save_bar:
        image = make_grid(image, nrow=3, padding=5, normalize=True)
        save_image(image, test_out_path + 'index_%d_psnr_%.4f_ssim_%.4f.png' % (index, psnr, ssim), padding=5, normalize=True)
        index += 1
        # image = utils.make_grid(test_images, nrow=3, padding=5)
        # utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
        #                  image_name.split('.')[-1], padding=5)



out_path = 'statistics/'
saved_results = {'psnr': [], 'ssim': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    if (len(psnr) == 0) or (len(ssim) == 0):
        psnr = 'No data'
        ssim = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(out_path + 'SR_factor_' + str(SCALE_FACTOR) + '_test_results.csv', index_label='DataSet')
