#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
sys.path.insert(0, join(dirname(__file__), '../../'))
import time
import os
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
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.model import Generator, CNNNorm
from utils import write_params
from learning.dataset import CostMapDataset

random.seed(datetime.now())
torch.manual_seed(999)
torch.cuda.manual_seed(999)
torch.set_num_threads(16)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="il-cpstmap-01", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=30., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--vector_dim', type=int, default=2, help='vector dim')
parser.add_argument('--points_num', type=int, default=16, help='points number')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.2, help='xy and vxy loss trade off')
parser.add_argument('--gamma2', type=float, default=0.05, help='xy and axy loss trade off')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=1000, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=25, help='interval between model test')
parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_speed', type=float, default=10., help='max speed')
parser.add_argument('--max_t', type=float, default=3., help='max time')
opt = parser.parse_args()
if opt.test_mode: opt.batch_size = 1
    
description = 'pretrain encoder with fixed GAN generator'
log_path = 'result/log/'+opt.dataset_name+'/'
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output2/%s' % opt.dataset_name, exist_ok=True)
if not opt.test_mode:
    logger = SummaryWriter(log_dir=log_path)
    write_params(log_path, parser, description)


generator = Generator(opt.vector_dim+2).to(device)
#generator.load_state_dict(torch.load('../result/saved_models/train-gan-03/generator_20000.pth'))

encoder = CNNNorm(input_dim=1, out_dim=2).to(device)

criterion = torch.nn.MSELoss()
e_optimizer = torch.optim.Adam(encoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)


train_loader = DataLoader(CostMapDataset(data_index=[1,2,3,4,5,6,7,8,9], opt=opt, dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/'), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
test_loader = DataLoader(CostMapDataset(data_index=[10], opt=opt, dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/'), batch_size=1, shuffle=False, num_workers=1)
test_samples = iter(test_loader)
    
def test_traj_v(xs, ys, step):
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    for i in range(len(xs)):
        ax1.plot(xs[i], ys[i], label=str(round(0.8*i, 1)), linewidth=5)
    ax1.set_xlabel('Forward/(m)')
    ax1.set_ylabel('Sideways/(m)')  
    ax1.set_xlim([0., 40])
    ax1.set_ylim([-20, 20])
    #plt.legend(loc='lower right')
    #plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.))
    plt.legend(loc='center', bbox_to_anchor=(0.9, 0.5))
    plt.savefig('result/output2/%s/v_' % opt.dataset_name+str(step)+'_curve.png')
    plt.close('all')

    
def show_traj(fake_traj, real_traj, t, step):
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
    ax1.plot(real_x, real_y, label='real-trajectory', color = 'b', linewidth=5, linestyle='--')
    ax1.set_xlabel('Forward/(m)')
    ax1.set_ylabel('Sideways/(m)')  
    ax1.set_xlim([0., max_x+5])
    ax1.set_ylim([-max_y, max_y])
    plt.legend(loc='lower right')
    
    t = max_x*t
    plt.legend(loc='lower left')
    #plt.show()
    plt.savefig('result/output/%s/' % opt.dataset_name+str(step)+'_curve.png')
    plt.close('all')

def draw_two_dim_vector(total_step):
    encoder.eval()
    generator.eval()
    #xs = []
    #ys = []
    batch = next(test_samples)
    batch['t'] = batch['t'].view(-1,1).to(device)
    batch['v_0'] = batch['v_0'].view(-1,1).to(device)
    batch['v0_array'] = batch['v0_array'].view(-1,1).to(device)
    batch['xy'] = batch['xy'].view(-1,2).to(device)
    # batch['t'].requires_grad = True
    #input_img = torch.cat((batch['img'], batch['nav']), dim=1).to(device)
    input_img = batch['img'].to(device)
    latent_vector = encoder(input_img)
    noise = latent_vector.unsqueeze(1)
    #noise = torch.randn(1, opt.vector_dim).unsqueeze(1)
    noise = noise.expand(1, opt.points_num, opt.vector_dim)
    noise = noise.reshape(1*opt.points_num, opt.vector_dim).clone()
    #noise.requires_grad = True
    noise = noise.to(device)
    fig = plt.figure(figsize=(19, 19))
    ax1 = fig.add_subplot(111)
    n = 4
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            noise[:,0] = i/n
            noise[:,1] = j/n

            t_with_v = torch.cat([batch['t'], batch['v0_array']], dim=1)
            output_xy = generator(noise, t_with_v)
            fake_traj = output_xy.view(-1, opt.points_num*2)
            fake_traj = fake_traj.view(-1, 2)[:,:2].view(1, -1, 2).data.cpu().numpy()[0]
            x = fake_traj[:,0]*opt.max_dist
            y = fake_traj[:,1]*opt.max_dist
            
            new_x = [item+i*40 for item in x]
            new_y = [item+j*40 for item in y]
            #xs.append(new_x)
            #ys.append(new_y)
            
            ax1.plot(new_x, new_y, label='2', linewidth=5)

    plt.savefig('result/output2/%s/s_' % opt.dataset_name+str(total_step)+'_curve.png', dpi=400)
    #plt.show()
    generator.train()
    encoder.train()
    
def draw_diff_v(total_step):
    encoder.eval()
    generator.eval()
    xs = []
    ys = []
    batch = next(test_samples)
    #input_img = torch.cat((batch['img'], batch['nav']), dim=1).to(device)
    input_img = batch['img'].to(device)
    latent_vector = encoder(input_img)
    batch['t'] = batch['t'].view(-1,1).to(device)
    #batch['v0_array'] = batch['v0_array'].view(-1,1).to(device)
    batch['xy'] = batch['xy'].view(-1,2).to(device)
    # batch['t'].requires_grad = True
    #noise = torch.randn(1, opt.vector_dim).unsqueeze(1)
    noise = latent_vector.unsqueeze(1)
    noise = noise.expand(1, opt.points_num, opt.vector_dim)
    noise = noise.reshape(1*opt.points_num, opt.vector_dim).clone()
    #noise.requires_grad = True
    noise = noise.to(device)
    for i in range(11):
        #noise = torch.zeros(1, opt.vector_dim).unsqueeze(1)
        #noise[:,0] = -1.0 + i/10.
        v0 = torch.FloatTensor([[0.8*i/opt.max_speed]*opt.points_num]).view(-1,1).to(device)
        #t_with_v = torch.cat([batch['t'], batch['v0_array']], dim=1)
        t_with_v = torch.cat([batch['t'], v0], dim=1)
        output_xy = generator(noise, t_with_v)
        fake_traj = output_xy.view(-1, opt.points_num*2)
        fake_traj = fake_traj.view(-1, 2)[:,:2].view(1, -1, 2).data.cpu().numpy()[0]
        x = fake_traj[:,0]*opt.max_dist
        y = fake_traj[:,1]*opt.max_dist
        xs.append(x)
        ys.append(y)
        
    test_traj_v(xs, ys, total_step)
    generator.train()
    encoder.train()

def eval_error(total_step):
    encoder.eval()
    generator.eval()

    batch = next(test_samples)
    input_img = batch['img'].to(device)
    latent_vector = encoder(input_img)
    batch_size = batch['t'].shape[0]

    batch['t'] = batch['t'].view(-1,1).to(device)
    batch['v_0'] = batch['v_0'].view(-1,1).to(device)
    batch['v0_array'] = batch['v0_array'].view(-1,1).to(device)
    batch['xy'] = batch['xy'].view(-1,2).to(device)
    # batch['t'].requires_grad = True


    noise = latent_vector.unsqueeze(1)
    noise = noise.expand(batch_size, opt.points_num, opt.vector_dim)
    noise = noise.reshape(batch_size*opt.points_num, opt.vector_dim)
    noise = noise.to(device)
    
    t_with_v = torch.cat([batch['t'], batch['v0_array']], dim=1)
    
    output_xy = generator(noise, t_with_v)

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
    
    logger.add_scalar('train/ex',  ex.item(),  total_step)
    logger.add_scalar('train/ey',  ey.item(),  total_step)
    logger.add_scalar('train/fde', fde.item(), total_step)
    logger.add_scalar('train/ade', ade.item(), total_step)

    generator.train()
    encoder.train()

total_step = 0
for i, batch in enumerate(train_loader):
    total_step += 1
    if total_step == 1: continue
    #print(batch['img'].shape)
    #print(batch['nav'].shape)
    #input_img = torch.cat((batch['img'], batch['nav']), dim=1).to(device)
    input_img = batch['img'].to(device)
    latent_vector = encoder(input_img)

    batch['t'] = batch['t'].view(-1,1).to(device)
    batch['v_0'] = batch['v_0'].view(-1,1).to(device)
    batch['v0_array'] = batch['v0_array'].view(-1,1).to(device)
    batch['xy'] = batch['xy'].view(-1,2).to(device)
    #batch['vxy'] = batch['vxy'].view(-1,2).to(device)
    #batch['axy'] = batch['axy'].view(-1,2).to(device)
    batch['t'].requires_grad = True
    
    real_traj = batch['xy'].view(-1, opt.points_num*2)
    # label = Variable(torch.FloatTensor(opt.batch_size,1).fill_(1.0), requires_grad=False)
    # label = label.to(device)

    noise = latent_vector.unsqueeze(1)

    noise = noise.expand(opt.batch_size, opt.points_num, opt.vector_dim)
    noise = noise.reshape(opt.batch_size * opt.points_num, opt.vector_dim)
    noise = noise.to(device)
    
    t_with_v = torch.cat([batch['t'], batch['v0_array']], dim=1)
    
    output_xy = generator(noise, t_with_v)
    fake_traj = output_xy.view(-1, opt.points_num*2)

    il_loss = criterion(batch['xy'], output_xy)
    """
    vx = (opt.max_dist/opt.max_t)*grad(output_xy.view(-1, opt.points_num, 2)[:,0].sum(), batch['t'], create_graph=True)[0]
    vy = (opt.max_dist/opt.max_t)*grad(output_xy.view(-1, opt.points_num, 2)[:,1].sum(), batch['t'], create_graph=True)[0]
    vxy = torch.cat([vx, vy], dim=1)
    start_v = vxy.view(-1, opt.points_num, 2)[:,0]/opt.max_speed
    
    # start point loss
    start_points = output_xy.view(-1, opt.points_num, 2)[:,0]
    ideal_start_points = torch.zeros(opt.batch_size, 2).to(device)
    start_point_loss = criterion(start_points, ideal_start_points)
    start_v_loss = criterion(torch.norm(start_v, dim=1), batch['v_0'].squeeze(1))
    """
    generator.zero_grad()
    encoder.zero_grad()
    
    loss_E = il_loss#start_point_loss + start_v_loss + il_loss
    loss_E.backward()
    e_optimizer.step()
    g_optimizer.step()
    
    logger.add_scalar('train/il_loss', il_loss.item(), total_step)
    #logger.add_scalar('train/start_point_loss', start_point_loss.item(), total_step)
    #logger.add_scalar('train/start_v_loss', start_v_loss.item(), total_step)

    if total_step % opt.test_interval == 0:
        eval_error(total_step)
        # draw_diff_v(total_step)
        # draw_two_dim_vector(total_step)

    if total_step % opt.test_interval == 0:
        show_traj(fake_traj.view(-1, 2)[:,:2].view(opt.batch_size, -1, 2).data.cpu().numpy()[0], batch['xy'].view(opt.batch_size, -1, 2).data.cpu().numpy()[0], batch['t'].view(opt.batch_size, -1).data.cpu().numpy()[0], total_step)
    if total_step % opt.checkpoint_interval == 0:
        torch.save(encoder.state_dict(), 'result/saved_models/%s/encoder_%d.pth'%(opt.dataset_name, total_step))
        torch.save(generator.state_dict(), 'result/saved_models/%s/generator_%d.pth'%(opt.dataset_name, total_step))
