#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
from threading import Condition

from torch import pinverse
sys.path.insert(0, join(dirname(__file__), '..'))
sys.path.insert(0, join(dirname(__file__), '../../'))

import cv2
import os
import copy
import random
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import torch
import torch.nn as nn
from torch.autograd import grad
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.model import Generator, Discriminator, CNN
from utils import write_params
from learning.dataset import ImageDataset#CostMapDataset

from utils.collect_pm import CollectPerspectiveImage
from utils.carla_sensor import Sensor, CarlaSensorMaster
from robo_utils.basic.perspective_mapping import PerspectiveMapping


import simulator
simulator.load('/home/wang/CARLA_0.9.9.4')
from simulator import config

import carla_utils as cu
import carla

random.seed(datetime.now())
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.set_num_threads(16)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="train-cgan-14", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=30., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--vector_dim', type=int, default=128, help='vector dim')
parser.add_argument('--points_num', type=int, default=16, help='points number')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=1e-4, help='adam: learning rate')
parser.add_argument('--gamma', type=float, default=0.01, help='xy and vxy loss trade off')
parser.add_argument('--gamma2', type=float, default=0.01, help='xy and axy loss trade off')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=1000, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=200, help='interval between model test')
parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_speed', type=float, default=10., help='max speed')
parser.add_argument('--max_t', type=float, default=3., help='max time')
opt = parser.parse_args()
if opt.test_mode: opt.batch_size = 1

description = 'small learning rate for encoder, with v0'
log_path = 'result/log/'+opt.dataset_name+'/'
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output2/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output3/%s' % opt.dataset_name, exist_ok=True)
logger = SummaryWriter(log_dir=log_path)
write_params(log_path, parser, description)

sensor_dict = {
    'camera':{
        'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
        # 'callback':image_callback,
    },
    'lidar':{
        'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
        # 'callback':lidar_callback,
    },
}

# param = Param()
param = cu.parse_yaml_file_unsafe('./param.yaml')
sensor = cu.PesudoSensor(sensor_dict['camera']['transform'], config['camera'])
sensor_master = CarlaSensorMaster(sensor, sensor_dict['camera']['transform'], binded=True)
# collect_perspective = CollectPerspectiveImage(param, sensor_master)
camera_param = cu.CameraParams(sensor)
# import pdb; pdb.set_trace()
pm = PerspectiveMapping(param, camera_param.K_augment, camera_param.T_img_imu)

generator = Generator(opt.vector_dim+256+1+1).to(device)
discriminator = Discriminator(opt.points_num*2+256+1).to(device)
encoder = CNN(input_dim=3, out_dim=256).to(device)
# discriminator.load_state_dict(torch.load('result/saved_models/train-cgan-12/discriminator_1000.pth'))
# generator.load_state_dict(torch.load('result/saved_models/train-cgan-12/generator_1000.pth'))
# encoder.load_state_dict(torch.load('result/saved_models/train-cgan-12/encoder_1000.pth'))
# discriminator.load_state_dict(torch.load('result/saved_models/train-cgan-01/discriminator_10000.pth'))
# generator.load_state_dict(torch.load('result/saved_models/train-cgan-01/generator_10000.pth'))
# encoder.load_state_dict(torch.load('result/saved_models/train-cgan-01/encoder_10000.pth'))
# discriminator.load_state_dict(torch.load('result/saved_models/train-cgan-09/discriminator_87000.pth'))
# generator.load_state_dict(torch.load('result/saved_models/train-cgan-09/generator_87000.pth'))
# encoder.load_state_dict(torch.load('result/saved_models/train-cgan-09/encoder_87000.pth'))


start_point_criterion = torch.nn.MSELoss()
trajectory_criterion = torch.nn.MSELoss()
g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
#g_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
#d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
e_optimizer = torch.optim.RMSprop(encoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

train_loader = DataLoader(ImageDataset(data_index=[item for item in range(2,11)], opt=opt, dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/'), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
test_loader = DataLoader(ImageDataset(data_index=[1], opt=opt, dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/'), batch_size=1, shuffle=False, num_workers=1)
test_samples = iter(test_loader)

def draw_pm(img, traj):
    pose_array = np.vstack((traj.T, np.zeros((4,traj.shape[0]))))
    empty_image = pm.trajectory_pose(pose_array, np.identity(4), None)
    empty_image = cv2.cvtColor(empty_image, cv2.COLOR_BGR2GRAY)
    mask = np.where(empty_image > 100)
    img[mask] = (random.randint(0,127)+127, random.randint(0,127)+127, random.randint(0,127)+127)
    return img

# def in_picture(point):
#     return (point[0] >= 0 and point[0] < opt.width) and \
#         (point[1] >= 0 and point[1] < opt.height)

# def traj_augmentation(traj, n=4):
#     points = []
#     for i in range(traj.shape[0] - 1):
#         if i < traj.shape[0]/2: num = 2*n
#         else: num = n
#         for j in range(num):
#             x = traj[i][0]*(j/num) + traj[i+1][0]*(1- j/num)
#             y = traj[i][1]*(j/num) + traj[i+1][1]*(1- j/num)
#             points.append([x, y])
#     return np.array(points)

# def traj_pm(img, org_traj):
#     x0 = 400
#     y0 = 200
#     fov = 120
#     f = x0 /(2 * np.tan(fov * np.pi / 360))

#     z = - sensor_dict['camera']['transform'].location.z
#     image_uv = []

#     traj = traj_augmentation(org_traj)
#     for i in range(traj.shape[0]):
#         if traj[i][0] < sensor_dict['camera']['transform'].location.x + 0.0001: continue
#         x = -traj[i][1]*f/(traj[i][0] - sensor_dict['camera']['transform'].location.x) + x0/2
#         y = -z*f / (traj[i][0] - sensor_dict['camera']['transform'].location.x) + y0/2
#         image_uv.append([int(x), int(y)])
#     image_uv = np.array(image_uv)

#     total = image_uv.shape[0]

#     color = (random.randint(0,200)+55, random.randint(0,200)+55, random.randint(0,200)+55)
#     for i in range(total-1):
#         point1 = (int(image_uv[i][0]), int(image_uv[i][1]))
#         point2 = (int(image_uv[i+1][0]), int(image_uv[i+1][1]))
#         if in_picture(point1) and in_picture(point2):
#             img = cv2.line(img, point1, point2, color, 4)                
#     return img
    

def draw_img(total_step):
    batch = next(test_samples)
    img = (batch['img'].numpy()[0]*127+128).transpose((1,2,0))
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    generator.eval()

    batch['img'] = batch['img'].to(device)
    batch['t'] = batch['t'].view(-1,1).to(device)
    batch['v_0'] = batch['v_0'].view(-1,1).to(device)
    batch['xy'] = batch['xy'].view(-1,2)#.to(device)

    noise = torch.randn(1, opt.vector_dim).unsqueeze(1)
    noise = noise.expand(1, opt.points_num, opt.vector_dim)
    noise = noise.reshape(1*opt.points_num, opt.vector_dim).clone()
    noise = noise.to(device)

    img_feature = encoder(batch['img'])
    #########################################
    img_feature = torch.cat([img_feature, batch['v_0']], dim=1)
    #########################################
    expand_img_feature = img_feature.unsqueeze(1)
    expand_img_feature = expand_img_feature.expand(1, opt.points_num, 256+1)
    expand_img_feature = expand_img_feature.reshape(1*opt.points_num, 256+1)

    input_feature = torch.cat([expand_img_feature, noise], dim=1)

    n = 4
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            noise[:,0] = i/n
            noise[:,1] = j/n
            input_feature = torch.cat([expand_img_feature, noise], dim=1)
            output_xy = generator(input_feature, batch['t'])
            fake_traj = output_xy.view(-1, opt.points_num*2)
            fake_traj = fake_traj.view(-1, 2)[:,:2].view(1, -1, 2).data.cpu().numpy()[0]
            # real_traj = batch['xy'].view(opt.points_num, 2).numpy()
            img = draw_pm(img, fake_traj*opt.max_dist)
            # img = traj_pm(img, fake_traj*opt.max_dist)
            
    cv2.imwrite('result/output3/'+str(opt.dataset_name)+'/'+str(total_step)+'.png', img)
    generator.train()

def test_traj(xs, ys, step):
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    for i in range(len(xs)):
        ax1.plot(xs[i], ys[i], label=str(round(-1.0 + i/10., 2)), linewidth=5)
    ax1.set_xlabel('Forward/(m)')
    ax1.set_ylabel('Sideways/(m)')  
    ax1.set_xlim([0., 50])
    ax1.set_ylim([-25, 25])
    #plt.legend(loc='lower right')
    #plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.))
    plt.legend(loc='center', bbox_to_anchor=(0.9, 0.5))
    plt.savefig('result/output2/%s/' % opt.dataset_name+str(step)+'_curve.png')
    plt.close('all')
    
def test_traj_v(xs, ys, step):
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    for i in range(len(xs)):
        ax1.plot(xs[i], ys[i], label=str(round(0.8*i, 1)), linewidth=5)
    ax1.set_xlabel('Forward/(m)')
    ax1.set_ylabel('Sideways/(m)')  
    ax1.set_xlim([0., 50])
    ax1.set_ylim([-25, 25])
    #plt.legend(loc='lower right')
    #plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.))
    plt.legend(loc='center', bbox_to_anchor=(0.9, 0.5))
    plt.savefig('result/output2/%s/v_' % opt.dataset_name+str(step)+'_curve.png')
    plt.close('all')
    
def test_traj2(xs, ys, step, j):
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    plt.xlim(0., 50)
    plt.ylim(-25, 25)
    for i in range(len(xs)):
        ax1.plot(xs[i], ys[i], label=str(round(-1.0 + i/10., 2)), linewidth=5)
    plt.legend(loc='center', bbox_to_anchor=(0.9, 0.5))
    plt.savefig('result/output2/%s/' % opt.dataset_name+str(step)+'_'+str(round(j, 1))+'_curve.png')
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
    generator.eval()
    #xs = []
    #ys = []
    batch = next(test_samples)
    batch['img'] = batch['img'].to(device)
    batch['t'] = batch['t'].view(-1,1).to(device)
    batch['v_0'] = batch['v_0'].view(-1,1).to(device)
    batch['v0_array'] = batch['v0_array'].view(-1,1).to(device)
    batch['xy'] = batch['xy'].view(-1,2).to(device)
    #batch['t'].requires_grad = True

    noise = torch.randn(1, opt.vector_dim).unsqueeze(1)
    noise = noise.expand(1, opt.points_num, opt.vector_dim)
    noise = noise.reshape(1*opt.points_num, opt.vector_dim).clone()
    noise = noise.to(device)

    img_feature = encoder(batch['img'])
    #########################################
    img_feature = torch.cat([img_feature, batch['v_0']], dim=1)
    #########################################
    expand_img_feature = img_feature.unsqueeze(1)
    expand_img_feature = expand_img_feature.expand(1, opt.points_num, 256+1)
    expand_img_feature = expand_img_feature.reshape(1*opt.points_num, 256+1)

    input_feature = torch.cat([expand_img_feature, noise], dim=1)

    fig = plt.figure(figsize=(19, 19))
    ax1 = fig.add_subplot(111)
    n = 4
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            # noise[:,0] = i/n
            # noise[:,1] = j/n
            noise = torch.randn(1, opt.vector_dim).unsqueeze(1)
            noise = noise.expand(1, opt.points_num, opt.vector_dim)
            noise = noise.reshape(1*opt.points_num, opt.vector_dim).clone()
            noise = noise.to(device)

            input_feature = torch.cat([expand_img_feature, noise], dim=1)
            output_xy = generator(input_feature, batch['t'])
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
    
def draw_diff_v(total_step):
    generator.eval()
    xs = []
    ys = []
    batch = next(test_samples)
    batch['t'] = batch['t'].view(-1,1).to(device)
    #batch['v0_array'] = batch['v0_array'].view(-1,1).to(device)

    batch['xy'] = batch['xy'].view(-1,2).to(device)
    batch['t'].requires_grad = True
    noise = torch.randn(1, opt.vector_dim).unsqueeze(1)
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

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class GradientPaneltyLoss(nn.Module):
    def __init__(self):
         super(GradientPaneltyLoss, self).__init__()

    def forward(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones_like(y)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

fn_GP = GradientPaneltyLoss().to(device)

total_step = 0
for i, batch in enumerate(train_loader):
    total_step += 1
    if total_step == 1: continue

    batch['img'] = batch['img'].to(device)
    batch['t'] = batch['t'].view(-1,1).to(device)
    batch['v_0'] = batch['v_0'].view(-1,1).to(device)
    batch['v0_array'] = batch['v0_array'].view(-1,1).to(device)
    batch['xy'] = batch['xy'].view(-1,2).to(device)
    #batch['vxy'] = batch['vxy'].view(-1,2).to(device)
    #batch['axy'] = batch['axy'].view(-1,2).to(device)
    batch['t'].requires_grad = True
    
    real_traj = batch['xy'].view(-1, opt.points_num*2)
    label = Variable(torch.FloatTensor(opt.batch_size,1).fill_(1.0), requires_grad=False)
    label = label.to(device)
    
    noise = torch.randn(opt.batch_size, opt.vector_dim).unsqueeze(1)
    noise = noise.expand(opt.batch_size, opt.points_num, opt.vector_dim)
    noise = noise.reshape(opt.batch_size*opt.points_num, opt.vector_dim)
    noise.requires_grad = True
    noise = noise.to(device)
    
    #t_with_v = torch.cat([batch['t'], batch['v0_array']], dim=1)
    t = batch['t']
    img_feature = encoder(batch['img'])
    #########################################
    img_feature = torch.cat([img_feature, batch['v_0']], dim=1)
    #########################################
    expand_img_feature = img_feature.unsqueeze(1)
    expand_img_feature = expand_img_feature.expand(opt.batch_size, opt.points_num, 256+1)
    expand_img_feature = expand_img_feature.reshape(opt.batch_size*opt.points_num, 256+1)

    input_feature = torch.cat([expand_img_feature, noise], dim=1)
    
    output_xy = generator(input_feature, t)
    
    grad0 = grad(output_xy.sum(), noise, create_graph=True)[0]
    grad_loss = grad0.norm()
    
    set_requires_grad(discriminator, True)
    discriminator.zero_grad()

    real_traj_with_feature = torch.cat([real_traj, img_feature.detach()], dim=1)
    
    pred_real = discriminator(real_traj_with_feature)
    
    fake_traj = output_xy.view(-1, opt.points_num*2)
    vx = (opt.max_dist/opt.max_t)*grad(output_xy.view(-1, opt.points_num, 2)[:,0].sum(), batch['t'], create_graph=True)[0]
    vy = (opt.max_dist/opt.max_t)*grad(output_xy.view(-1, opt.points_num, 2)[:,1].sum(), batch['t'], create_graph=True)[0]
    vxy = torch.cat([vx, vy], dim=1)
    start_v = vxy.view(-1, opt.points_num, 2)[:,0]/opt.max_speed
    
    # start point loss
    start_points = output_xy.view(-1, opt.points_num, 2)[:,0]
    ideal_start_points = torch.zeros(opt.batch_size, 2).to(device)
    start_point_loss = start_point_criterion(start_points, ideal_start_points)
    
    start_v_loss = start_point_criterion(torch.norm(start_v, dim=1), batch['v_0'].squeeze(1))

    fake_traj_with_feature = torch.cat([fake_traj.detach(), img_feature.detach()], dim=1)
    pred_fake = discriminator(fake_traj_with_feature)
    
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand_as(real_traj)
    alpha = alpha.to(device)
    interpolated = (alpha * real_traj.data + (1 - alpha) * fake_traj.detach().data).requires_grad_(True)
    
    output_ = torch.cat([interpolated, img_feature.detach()], dim=1)
    src_out_ = discriminator(output_)
    loss_D_real = torch.mean(pred_real)
    loss_D_fake = -torch.mean(pred_fake)
    # Gradient penalty Loss
    loss_D_gp = fn_GP(src_out_, output_)
    loss_D = 0.5 * (loss_D_real + loss_D_fake) + loss_D_gp
    loss_D.backward()
    torch.nn.utils.clip_grad_value_(discriminator.parameters(), clip_value=1)
    d_optimizer.step()
    
    set_requires_grad(discriminator, False)
    generator.zero_grad()
    encoder.zero_grad()

    fake_traj_with_feature = torch.cat([fake_traj, img_feature], dim=1)
    pred_fake = discriminator(fake_traj_with_feature)
    loss_G = torch.mean(pred_fake)# + 5*start_point_loss + start_v_loss
    loss_G.backward()
    torch.nn.utils.clip_grad_value_(generator.parameters(), clip_value=1)
    torch.nn.utils.clip_grad_value_(encoder.parameters(), clip_value=1)
    g_optimizer.step()
    e_optimizer.step()
    
    logger.add_scalar('train/loss_G', loss_G.item(), total_step)
    logger.add_scalar('train/grad_loss', grad_loss.item(), total_step)
    logger.add_scalar('train/loss_D_real', loss_D_real.item(), total_step)
    logger.add_scalar('train/loss_D_fake', loss_D_fake.item(), total_step)
    logger.add_scalar('train/loss_D_gp', loss_D_gp.item(), total_step)
    logger.add_scalar('train/start_point_loss', start_point_loss.item(), total_step)
    logger.add_scalar('train/start_v_loss', start_v_loss.item(), total_step)
    
    if total_step % opt.test_interval == 0:
        #draw_diff_v(total_step)
        draw_img(total_step)
        # draw_two_dim_vector(total_step)

    if total_step % opt.test_interval == 0:
        show_traj(fake_traj.view(-1, 2)[:,:2].view(opt.batch_size, -1, 2).data.cpu().numpy()[0], batch['xy'].view(opt.batch_size, -1, 2).data.cpu().numpy()[0], batch['t'].view(opt.batch_size, -1).data.cpu().numpy()[0], total_step)
    if total_step % opt.checkpoint_interval == 0:
        torch.save(encoder.state_dict(), 'result/saved_models/%s/encoder_%d.pth'%(opt.dataset_name, total_step))
        torch.save(generator.state_dict(), 'result/saved_models/%s/generator_%d.pth'%(opt.dataset_name, total_step))
        torch.save(discriminator.state_dict(), 'result/saved_models/%s/discriminator_%d.pth'%(opt.dataset_name, total_step))
