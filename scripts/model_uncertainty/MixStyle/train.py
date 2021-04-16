#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
sys.path.insert(0, join(dirname(__file__), '../../'))
sys.path.insert(0, join(dirname(__file__), '../../../'))
sys.path.insert(0, join(dirname(__file__), '../../../'))

import os
import time
import random
import argparse
import numpy as np
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.boost_model import ModelMixStyle, ModelMixStyleWithV
from carla_utils import parse_yaml_file_unsafe
from robo_utils.oxford.oxford_dataset import DIVADataset#BoostDataset, DIMDataset
from robo_utils.kitti.torch_dataset import BoostDataset as KittiDataset
from utils import write_params, check_shape, to_device, set_mute

random.seed(datetime.now())
torch.manual_seed(555)
torch.cuda.manual_seed(555)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="baseline-mixstyle-01", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--traj_steps', type=int, default=8, help='traj steps')
parser.add_argument('--points_num', type=int, default=16, help='points number')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--gamma', type=float, default=0.2, help='xy and vxy loss trade off')
parser.add_argument('--gamma2', type=float, default=0.0, help='xy and axy loss trade off')
parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=500, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=25, help='interval between model test')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_speed', type=float, default=10., help='max distance')
parser.add_argument('--max_t', type=float, default=3., help='max time')
parser.add_argument('--model_num', type=int, default=5, help='model_num')
opt = parser.parse_args()
if opt.test_mode: opt.batch_size = 1
    
description = 'train cluster, with v0 loss'
log_path = 'result/log/'+opt.dataset_name+'/'+str(round(time.time()))+'/'
os.makedirs(log_path, exist_ok=True)
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)

write_params(log_path, parser, description)

logger = SummaryWriter(log_dir=log_path)


model = ModelMixStyleWithV().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=opt.weight_decay)


param = parse_yaml_file_unsafe('./param_oxford.yaml')
train_loader = DataLoader(DIVADataset(param, mode='train', opt=opt), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
train_samples = iter(train_loader)

param = parse_yaml_file_unsafe('./param_kitti.yaml')
eval_trajectory_dataset = KittiDataset(param, 'eval', opt)#2
dataloader_eval = DataLoader(eval_trajectory_dataset, batch_size=1, shuffle=False, num_workers=1)
eval_samples = iter(dataloader_eval)

criterion = nn.MSELoss().to(device)
criterion_l1 = nn.SmoothL1Loss().to(device)
trajectory_criterion = torch.nn.MSELoss().to(device)


def show_traj_with_uncertainty(fake_traj, real_traj, step, model_index, logvar=None):
    fake_xy = fake_traj
    x = fake_xy[:,0]*opt.max_dist
    y = fake_xy[:,1]*opt.max_dist

    real_xy = real_traj
    real_x = real_xy[:,0]*opt.max_dist
    real_y = real_xy[:,1]*opt.max_dist

    max_x = 30.
    max_y = 30.

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)

    ax1.plot(x, y, label='trajectory', color = 'r', linewidth=5)
    if logvar is not None:
        x_var = logvar[:,0]#*opt.max_dist
        y_var = logvar[:,1]#*opt.max_dist

        x_var = np.sqrt(np.exp(x_var))*opt.max_dist
        y_var = np.sqrt(np.exp(y_var))*opt.max_dist
        ax1.fill_betweenx(y, x-x_var, x+x_var, color="crimson", alpha=0.4)
        ax1.fill_between(x, y-y_var, y+y_var, color="cyan", alpha=0.4)

    ax1.plot(real_x, real_y, label='real-trajectory', color = 'b', linewidth=5, linestyle='--')
    ax1.set_xlabel('Forward/(m)')
    ax1.set_ylabel('Sideways/(m)')  
    ax1.set_xlim([0., max_x])
    ax1.set_ylim([-max_y/2, max_y/2])
    plt.legend(loc='lower right')
    
    plt.legend(loc='lower left')
    plt.savefig('result/output/%s/' % opt.dataset_name+str(step)+'_'+str(model_index)+'_curve.png')
    plt.close('all')

def eval_error(total_step):
    model.eval()

    batch = next(eval_samples)
    batch['t'] = batch['t'].view(-1,1)
    batch['v0_array'] = batch['v0_array'].view(-1,1)
    batch['xy'] = batch['xy'].view(-1,2)
    to_device(batch, device)

    feature = model.get_encoder(batch['img'], batch['v_0'])
    feature_dim = feature.shape[-1]
    feature = feature.unsqueeze(1)
    feature = feature.expand(1, batch['t'].shape[0], feature_dim)
    feature = feature.reshape(1 * batch['t'].shape[0], feature_dim)
    
    output = model.get_trajectory(feature, batch['v0_array'], batch['t'])

    output_xy = output[:,:2]
    # logvar = output[:,2:]

    real_traj = batch['xy'].data.cpu().numpy()*opt.max_dist
    fake_traj = output_xy.data.cpu().numpy()*opt.max_dist

    real_x = real_traj[:,0]
    real_y = real_traj[:,1]
    fake_x = fake_traj[:,0]
    fake_y = fake_traj[:,1]

    ex = np.mean(np.abs(fake_x-real_x))
    ey = np.mean(np.abs(fake_y-real_y))
    fde = np.hypot(fake_x - real_x, fake_y - real_y)[-1]
    ade = np.mean(np.hypot(fake_x - real_x, fake_y - real_y))
    
    logger.add_scalar('eval/ex',  ex.item(),  total_step)
    logger.add_scalar('eval/ey',  ey.item(),  total_step)
    logger.add_scalar('eval/fde', fde.item(), total_step)
    logger.add_scalar('eval/ade', ade.item(), total_step)

    model.train()

def eval_all_model(total_step):
    model.eval_models()
    batch = next(eval_samples)
    batch['t'] = batch['t'].view(-1,1)
    batch['v0_array'] = batch['v0_array'].view(-1,1)
    batch['xy'] = batch['xy'].view(-1,2)
    to_device(batch, device)

    real_traj = batch['xy'].data.cpu().numpy()*opt.max_dist
    real_x = real_traj[:,0]
    real_y = real_traj[:,1]

    fake_x_list = []
    fake_y_list = []
    x_var_list = []
    y_var_list = []
    for model_index in range(opt.model_num):
        feature = model.get_encoder(batch['img'])
        feature_dim = feature.shape[-1]
        feature = feature.unsqueeze(1)
        feature = feature.expand(1, opt.points_num, feature_dim)
        feature = feature.reshape(1 * opt.points_num, feature_dim)
        
        output = model.get_trajectory(feature, batch['v0_array'], batch['t'])

        output_xy = output[:,:2]
        fake_traj = output_xy.data.cpu().numpy()*opt.max_dist
        fake_x = fake_traj[:,0]
        fake_y = fake_traj[:,1]
        fake_x_list.append(fake_x)
        fake_y_list.append(fake_y)
        
        logvar = output[:,2:].data.cpu().numpy()
        x_var = logvar[:,0]
        y_var = logvar[:,1]
        x_var = np.sqrt(np.exp(x_var))*opt.max_dist
        y_var = np.sqrt(np.exp(y_var))*opt.max_dist
        x_var_list.append(x_var)
        y_var_list.append(y_var)

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    plt.axis('equal')

    for model_index in range(opt.model_num):
        x = fake_x_list[model_index]
        y = fake_y_list[model_index]
        x_var = x_var_list[model_index]
        y_var = y_var_list[model_index]
        ax1.plot(x, y, linewidth=2)
        ax1.fill_betweenx(y, x-x_var, x+x_var, color="crimson", alpha=0.4)
        ax1.fill_between(x, y-y_var, y+y_var, color="cyan", alpha=0.4)

    fake_x = np.mean(np.array(fake_x_list), axis=0)
    fake_y = np.mean(np.array(fake_y_list), axis=0)
    ax1.plot(fake_x, fake_y, linewidth=2)
    ax1.plot(real_x, real_y, label='real', color='black', linewidth=3)

    plt.legend()
    # plt.show()
    plt.savefig('result/output/%s/' % opt.dataset_name+str(total_step)+'_curve.png', dpi=400)
    plt.close('all')

    model.train_models()

# def train_adversarial():
#     pass

# def train_supervised():
#     pass

for total_steps in range(1000000):
    model_index = random.randint(0, opt.model_num-1)

    batch = next(train_samples)
    check_shape(batch)
    to_device(batch, device)

    batch['t'] = batch['t'].view(-1,1)
    batch['t'].requires_grad = True
    batch['v0_array'] = batch['v0_array'].view(-1,1)

    check_shape(batch)
    to_device(batch, device)

    feature = model.get_encoder(batch['img'], batch['v_0'])
    feature_dim = feature.shape[-1]

    feature = feature.unsqueeze(1)
    feature = feature.expand(opt.batch_size, opt.points_num, feature_dim)
    feature = feature.reshape(opt.batch_size * opt.points_num, feature_dim)
    check_shape(feature, 'feature')

    output = model.get_trajectory(feature, batch['v0_array'], batch['t'])

    output_xy = output[:,:2]
    # logvar = output[:,2:]
    # check_shape(logvar, 'logvar')

    vx = grad(output_xy[:,0].sum(), batch['t'], create_graph=True)[0]
    vy = grad(output_xy[:,1].sum(), batch['t'], create_graph=True)[0]
    output_vxy = (opt.max_dist/opt.max_t)*torch.cat([vx, vy], dim=1)
    check_shape(output_vxy, 'output_vxy')

    fake_traj = output_xy.reshape(-1, opt.points_num*2)
    real_traj = batch['xy'].view(-1, opt.points_num*2)
    # fake_logvar = logvar.reshape(-1, opt.points_num*2)

    check_shape(output_xy, 'output_xy')

    loss_trajectory_x = trajectory_criterion(real_traj.view(opt.batch_size, 16,2)[:,:,0]*opt.max_dist, fake_traj.view(opt.batch_size, 16,2)[:,:,0]*opt.max_dist)
    loss_trajectory_y = trajectory_criterion(real_traj.view(opt.batch_size, 16,2)[:,:,1]*opt.max_dist, fake_traj.view(opt.batch_size, 16,2)[:,:,1]*opt.max_dist)
    loss_trajectory = loss_trajectory_x + 5*loss_trajectory_y

    # l2_loss = torch.pow((output_xy - batch['xy'].view(-1, 2)), 2)
    # check_shape(l2_loss, 'l2_loss')
    # uncertainty_loss = torch.mean((torch.exp(-logvar) * l2_loss + logvar) * 0.5)

    loss_vxy = criterion(output_vxy, opt.max_speed*batch['vxy'].view(-1, 2))

    # v0 loss
    t0 = torch.zeros_like(batch['t']).to(device)
    t0.requires_grad = True
    output = model.get_trajectory(feature, batch['v0_array'], t0)
    output_xy = output[:,:2]

    vx = grad(output_xy[:,0].sum(), t0, create_graph=True)[0]
    vy = grad(output_xy[:,1].sum(), t0, create_graph=True)[0]
    output_vxy = (opt.max_dist/opt.max_t)*torch.cat([vx, vy], dim=1)

    output_v0 = torch.norm(output_vxy, dim=1).unsqueeze(1)
    real_v0 = batch['v0_array']*opt.max_speed
    check_shape(output_v0, 'output_v0')
    check_shape(real_v0, 'real_v0')

    loss_v0 = criterion_l1(output_v0, real_v0)

    # loss = uncertainty_loss + 10*loss_v0 + opt.gamma*loss_vxy
    loss = loss_trajectory + 10*loss_v0 + opt.gamma*loss_vxy

    check_shape(loss, 'loss')
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
    optimizer.step()
    set_mute(True)

    logger.add_scalar('train/loss_trajectory', loss_trajectory.item(), total_steps)
    logger.add_scalar('train/loss_v0', loss_v0.item(), total_steps)
    logger.add_scalar('train/loss_vxy', loss_vxy.item(), total_steps)
    # logger.add_scalar('train/var', opt.max_dist*torch.mean(torch.sqrt(torch.exp(logvar))).item(), total_steps)


    if total_steps > 0 and total_steps % opt.test_interval == 0:
        eval_error(total_steps)

    if total_steps > 0 and total_steps % opt.checkpoint_interval == 0:
        # cluster.save_models('result/saved_models/%s/'%(opt.dataset_name), total_steps)
        # os.makedirs('result/saved_models/%s/'%(opt.dataset_name), exist_ok=True)
        torch.save(model.state_dict(), 'result/saved_models/%s/'%(opt.dataset_name)+'model_'+str(total_steps)+'.pth')