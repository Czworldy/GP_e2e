#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '../../../'))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import time
import random
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

from carla_utils import parse_yaml_file_unsafe
from robo_utils.oxford.oxford_dataset import DIMDataset, BoostDataset
from robo_utils.kitti.torch_dataset import OurDataset as KittiDataset
from utils import write_params, check_shape, to_device, set_mute
from dim_model import ImitativeModel

random.seed(datetime.now())
torch.manual_seed(233)
torch.cuda.manual_seed(233)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="test-DIM-dim6-01", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--points_num', type=int, default=16, help='points number')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--gamma', type=float, default=0.2, help='xy and vxy loss trade off')
parser.add_argument('--gamma2', type=float, default=0.0, help='xy and axy loss trade off')
parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=1000, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=25, help='interval between model test')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_speed', type=float, default=10., help='max distance')
parser.add_argument('--max_t', type=float, default=3., help='max time')
parser.add_argument('--model_num', type=int, default=5, help='model_num')
opt = parser.parse_args()

description = 'train DIM'
log_path = 'result/log/'+opt.dataset_name+'/'+str(round(time.time()))+'/'
os.makedirs(log_path, exist_ok=True)
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)

write_params(log_path, parser, description)


model = ImitativeModel(output_shape=(opt.points_num, 2)).to(device)
model.load_state_dict(torch.load('result/saved_models/%s/model_%d.pth'%('train-DIM-dim6-01', 74000)))

param = parse_yaml_file_unsafe('./param_oxford.yaml')

train_loader_cluster = [
    iter(DataLoader(
        DIMDataset(param, mode='train', opt=opt, data_index=i), 
            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)) for i in range(opt.model_num)
]


param = parse_yaml_file_unsafe('./param_kitti.yaml')
eval_loader = DataLoader(KittiDataset(param, mode='eval', opt=opt), batch_size=1, shuffle=False, num_workers=1)
eval_samples = iter(eval_loader)
# eval_loader = DataLoader(DIMDataset(param, mode='eval', opt=opt), batch_size=1, shuffle=False, num_workers=1)
# eval_samples = iter(eval_loader)

logger = SummaryWriter(log_dir=log_path)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

def interpolation(time_list, x_list, y_list, t):
    if t >= time_list[-1]:
        dt = time_list[-1] - time_list[-2]
        dx = x_list[-1] - x_list[-2]
        dy = y_list[-1] - y_list[-2]
        x = x_list[-1] + (dx/dt)*(t - time_list[-1])
        y = y_list[-1] + (dy/dt)*(t - time_list[-1])
    else:
        index = np.argmin(np.abs(time_list-t))
        if t >= time_list[index]:
            dt = time_list[index+1] - time_list[index]
            dx = x_list[index+1] - x_list[index]
            dy = y_list[index+1] - y_list[index]
            x = x_list[index] + (dx/dt)*(t - time_list[index])
            y = y_list[index] + (dy/dt)*(t - time_list[index])
        else:
            dt = time_list[index] - time_list[index-1]
            dx = x_list[index] - x_list[index-1]
            dy = y_list[index] - y_list[index-1]
            x = x_list[index-1] + (dx/dt)*(t - time_list[index-1])
            y = y_list[index-1] + (dy/dt)*(t - time_list[index-1])
    return x, y


def eval_model(total_steps):
    global model
    batch = next(eval_samples)
    to_device(batch, device)

    x = model._decoder._base_dist.mean.clone().detach().view(-1, opt.points_num, 2)
    x.requires_grad = True
    z = model._params(
        velocity=batch['v_0'].view(-1,1),
        visual_features=batch['img'],
    )

    optimizer = torch.optim.Adam(params=[x], lr=1e-1)

    x_best = x.clone()
    loss_best = torch.ones(()).to(x.device) * 1000.0
    for _ in range(50):
        optimizer.zero_grad()
        y, _ = model._decoder._forward(x=x, z=z)
        _, log_prob, logabsdet = model._decoder._inverse(y=y, z=z)
        imitation_prior = torch.mean(log_prob - logabsdet)
        loss= -imitation_prior
        loss.backward(retain_graph=True)
        optimizer.step()
        if loss < loss_best:
            x_best = x.clone()
            loss_best = loss.clone()

    plan, _ = model._decoder._forward(x=x_best, z=z)
    xy = plan.detach().cpu().numpy()[0]*opt.max_dist

    real_xy = batch['xy'].view(-1, 2).data.cpu().numpy()*opt.max_dist

    fake_x = xy[:,0]
    fake_y = xy[:,1]
    time_list = [0.0000, 0.1875, 0.3750, 0.5625, 0.7500, 0.9375, 1.1250, 1.3125, 1.5000, 1.6875, 1.8750, 2.0625, 2.2501, 2.4376, 2.6251, 2.8126]

    real_x = real_xy[:,0]
    real_y = real_xy[:,1]
    time = batch['t'].data.cpu().numpy()[0]*opt.max_t

    xs = []
    ys = []
    for t in time:
        x, y = interpolation(time_list, fake_x, fake_y, t)
        xs.append(x)
        ys.append(y)

    fake_x = np.array(xs)
    fake_y = np.array(ys)

    if total_steps % (4*opt.test_interval) == 0:
        max_x = 30.
        max_y = 30.

        fig = plt.figure(figsize=(7, 7))
        ax1 = fig.add_subplot(111)
        ax1.plot(real_x, real_y, label='real-trajectory', color = 'b', linewidth=3, linestyle='--')
        ax1.plot(fake_x, fake_y, label='fake-trajectory', color = 'r', linewidth=3)
        ax1.set_xlabel('Forward/(m)')
        ax1.set_ylabel('Sideways/(m)')  
        ax1.set_xlim([0., max_x])
        ax1.set_ylim([-max_y/2, max_y/2])
        plt.legend(loc='lower right')
        
        plt.legend(loc='lower left')
        plt.savefig('result/output/%s/' % opt.dataset_name+'/'+str(total_steps)+'.png')
        plt.close('all')

    ex = np.mean(np.abs(fake_x-real_x))
    ey = np.mean(np.abs(fake_y-real_y))
    fde = np.hypot(fake_x - real_x, fake_y - real_y)[-1]
    ade = np.mean(np.hypot(fake_x - real_x, fake_y - real_y))
    
    logger.add_scalar('eval/ex',  ex.item(),  total_steps)
    logger.add_scalar('eval/ey',  ey.item(),  total_steps)
    logger.add_scalar('eval/fde', fde.item(), total_steps)
    logger.add_scalar('eval/ade', ade.item(), total_steps)


for total_steps in range(1000000):
    eval_model(total_steps)
"""    
for total_steps in range(1000000):
    train_loader = train_loader_cluster[total_steps % 5]
    batch = next(train_loader)
    check_shape(batch)
    to_device(batch, device)
    
    y = batch['xy']#*opt.max_dist

    z = model._params(
        velocity=batch['v_0'].view(-1,1),
        visual_features=batch['img'],
    )
    _, log_prob, logabsdet = model._decoder._inverse(y=y, z=z)
    check_shape(log_prob, 'log_prob')
    check_shape(logabsdet, 'logabsdet')
    loss = -torch.mean(log_prob - logabsdet, dim=0)
    optimizer.zero_grad()
    loss.backward()
    logger.add_scalar('train/loss',  loss.item(),  total_steps)
    # torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
    optimizer.step()
    set_mute(True)
    if total_steps > 0 and total_steps % opt.test_interval == 0:
        eval_model(total_steps)

    if total_steps > 0 and total_steps % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), 'result/saved_models/%s/model_%d.pth'%(opt.dataset_name, total_steps))
"""