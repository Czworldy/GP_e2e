#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
sys.path.insert(0, join(dirname(__file__), '../../'))
sys.path.insert(0, join(dirname(__file__), '../../../'))
import os
import random
import argparse
import numpy as np
from datetime import datetime
import cv2
import seaborn as sns

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import torch
import torch.nn as nn
from torch.autograd import grad
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.model import Generator, Discriminator, Encoder
from robo_utils.oxford.oxford_dataset import DIVADataset
from robo_utils.kitti.torch_dataset import OurDataset as KittiDataset
from utils import write_params, check_shape, to_device, set_mute
from carla_utils import parse_yaml_file_unsafe
from torchvision.utils import save_image

random.seed(datetime.now())
torch.manual_seed(111)
torch.cuda.manual_seed(222)
torch.set_num_threads(16)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="train-GAN-6dim-01", help='name of the dataset')
# parser.add_argument('--dataset_name', type=str, default="train-latent-loss-05", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=30., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--vector_dim', type=int, default=64, help='vector dim')
parser.add_argument('--points_num', type=int, default=16, help='points number')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=1e-4, help='adam: learning rate')
parser.add_argument('--gamma', type=float, default=0.01, help='xy and vxy loss trade off')
parser.add_argument('--gamma2', type=float, default=0.01, help='xy and axy loss trade off')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=1000, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=5, help='interval between model test')
parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_speed', type=float, default=10., help='max speed')
parser.add_argument('--max_t', type=float, default=3., help='max time')
opt = parser.parse_args()
if opt.test_mode: opt.batch_size = 1
    
description = 'dropout'
log_path = 'result/log/'+opt.dataset_name+'/'
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)


# generator = Generatorv2(opt.vector_dim+2).to(device)

generator = Generator(input_dim=1+1+opt.vector_dim, output=2).to(device)
generator.load_state_dict(torch.load('result/saved_models/train-gan-costmap-03/generator_10000.pth'))
generator.eval()

param = parse_yaml_file_unsafe('./param_oxford.yaml')
train_loader = DataLoader(DIVADataset(param, mode='train', opt=opt), batch_size=1, shuffle=False, num_workers=1)
train_samples = iter(train_loader)

# param = parse_yaml_file_unsafe('./param_kitti.yaml')
# eval_loader = DataLoader(KittiDataset(param, mode='eval', opt=opt), batch_size=1, shuffle=False, num_workers=1)
# eval_samples = iter(eval_loader)
# train_samples = eval_samples


import time
for _ in range(100):
    batch  = next(train_samples)
    
    save_image(batch['img'][0][3:].data, 'test2/'+ str(time.time())+'.png', nrow=1, normalize=True)
    # import pdb; pdb.set_trace()
    # batch['img'] = batch['img'].to(device)
    """
    batch['t'] = batch['t'].view(-1,1).to(device)
    batch['v_0'] = batch['v_0'].view(-1,1).to(device)
    batch['v0_array'] = batch['v0_array'].view(-1,1).to(device)
    batch['xy'] = batch['xy'].view(-1,2).to(device)

    batch['t'].requires_grad = True
    real_traj = batch['xy'].view(-1, opt.points_num*2)
    condition = batch['v0_array']
    batch['domian'] = batch['domian'].to(device)

    x_list = []
    y_list = []
    for i in range(1000):
        # condition = torch.rand_like(condition).to(device)
        single_latent = torch.randn(opt.batch_size, opt.vector_dim)
        single_latent = single_latent.unsqueeze(1)
        latent = single_latent.expand(opt.batch_size, opt.points_num, opt.vector_dim)
        latent = latent.reshape(opt.batch_size * opt.points_num, opt.vector_dim)
        latent = latent.to(device)

        output_xy = generator(condition, latent, batch['t'])
        fake_traj = output_xy.data.cpu().numpy()*opt.max_dist

        fake_x = fake_traj[:,0]
        fake_y = fake_traj[:,1]

        x_list.append(fake_x)
        y_list.append(fake_y)



    fig = plt.figure(figsize=(11, 11))
    ax1 = fig.add_subplot(111)
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        ax1.plot(x, y,linewidth=3)


    ax1.set_xlabel('Forward/(m)')
    ax1.set_ylabel('Sideways/(m)')  
    ax1.set_xlim([0., 30])
    ax1.set_ylim([-30, 30])

    import time
    plt.savefig('test_result/'+ str(time.time())+'.png')
# plt.show()
"""