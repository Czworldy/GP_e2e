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

from learning.model import Generator, Discriminator
from robo_utils.oxford.oxford_dataset import GANDataset
from utils import write_params, check_shape, to_device, set_mute
from carla_utils import parse_yaml_file_unsafe

random.seed(datetime.now())
torch.manual_seed(233)
torch.cuda.manual_seed(233)
torch.set_num_threads(16)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="train-gan-new-01", help='name of the dataset')
# parser.add_argument('--dataset_name', type=str, default="train-gan-costmap-e2e-01", help='name of the dataset')
# parser.add_argument('--dataset_name', type=str, default="test-gan-costmap-04", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=30., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--vector_dim', type=int, default=64, help='vector dim')
parser.add_argument('--points_num', type=int, default=16, help='points number')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=1e-4, help='adam: learning rate')
parser.add_argument('--gamma', type=float, default=0.01, help='xy and vxy loss trade off')
parser.add_argument('--gamma2', type=float, default=0.01, help='xy and axy loss trade off')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=1000, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=50, help='interval between model test')
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
os.makedirs('result/output2/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output3/%s' % opt.dataset_name, exist_ok=True)
if not opt.test_mode:
    logger = SummaryWriter(log_dir=log_path)
    write_params(log_path, parser, description)


generator = Generator(input_dim=1+1+opt.vector_dim, output=2).to(device)
discriminator = Discriminator(opt.points_num*2+1).to(device)

generator.load_state_dict(torch.load('result/saved_models/train-gan-costmap-03/generator_10000.pth'))
# discriminator.load_state_dict(torch.load('result/saved_models/pretrain-gan-01/discriminator_50000.pth'))
# generator.load_state_dict(torch.load('result/saved_models/train-gan-costmap-vector2-01/generator_5000.pth'))
# discriminator.load_state_dict(torch.load('result/saved_models/train-gan-costmap-02/discriminator_46000.pth'))
# generator.load_state_dict(torch.load('result/saved_models/train-gan-costmap-02/generator_46000.pth'))
# discriminator.load_state_dict(torch.load('result/saved_models/train-gan-costmap-dropout-01/discriminator_35000.pth'))
# generator.load_state_dict(torch.load('result/saved_models/train-gan-costmap-dropout-01/generator_35000.pth'))


start_point_criterion = torch.nn.MSELoss()
criterion = torch.nn.BCELoss()#.to(device)
trajectory_criterion = torch.nn.MSELoss()
# e_optimizer = torch.optim.RMSprop(encoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
#g_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
#d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)


param = parse_yaml_file_unsafe('./param_oxford.yaml')
train_loader = DataLoader(GANDataset(param, mode='train', opt=opt), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
train_samples = iter(train_loader)

# test_loader = DataLoader(GANDataset(param, mode='eval', opt=opt), batch_size=1, shuffle=False, num_workers=1)
# test_samples = iter(test_loader)

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

def test_uncertainty(total_step):
    generator.eval()

    batch = next(test_samples)
    batch['t'] = batch['t'].view(-1,1).to(device)
    batch['v0_array'] = batch['v0_array'].view(-1,1).to(device)
    batch['xy'] = batch['xy'].view(-1,2)

    # t = torch.arange(0, batch['t'][-1][0], 0.02).unsqueeze(1).to(device)
    t = batch['t']

    v0 = torch.ones_like(t)*batch['v0_array'][0][0]
    v0 = v0.to(device)

    noise = torch.randn(1, opt.vector_dim).unsqueeze(1)
    noise = noise.expand(1, t.shape[0], opt.vector_dim)
    noise = noise.reshape(1*t.shape[0], opt.vector_dim).clone()
    #noise.requires_grad = True
    noise = noise.to(device)
    ####################################################
    # latent_vector = torch.randn_like(latent_vector)

    single_condition = batch['v_0'].to(device)

    condition = latent_vector.unsqueeze(1)
    condition = condition.expand(1, t.shape[0], condition.shape[-1])
    condition = condition.reshape(1*t.shape[0], condition.shape[-1])
    condition = torch.cat([condition, v0], dim=1)

    fig = plt.figure()
    # fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    ax1.set_xlim([0., 30])
    ax1.set_ylim([-15, 15])
    ax1.set_xlabel('Longitudinal/(m)')
    ax1.set_ylabel('Lateral/(m)')
    plt.title('Distribution of the Generated Trajectories')

    real_xy = batch['xy'].data.numpy()*opt.max_dist
    ax1.plot(real_xy[:,0], real_xy[:,1], linewidth=2, c='c')

    n = 5
    xs = [0, 0, 30, 30]
    ys = [-15, 15, -15, 15]

    avg_prediction = []
    avg_prediction_real = []
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            noise[:,0] = i/n
            noise[:,1] = j/n

            output_xy = generator(condition, noise, t)

            fake_traj = output_xy.view(-1, opt.points_num*2)
            fake_traj_with_condition = torch.cat([fake_traj, single_condition], dim=1)
            real_traj = batch['xy'].view(-1, opt.points_num*2).to(device)
            real_traj_with_condition = torch.cat([real_traj, single_condition], dim=1)
            pred_fake = discriminator(fake_traj_with_condition)
            pred_real = discriminator(real_traj_with_condition)
            # import pdb;pdb.set_trace()
            # print(pred_fake.data.cpu().numpy()[0][0])
            avg_prediction.append(pred_fake.data.cpu().numpy()[0][0])
            avg_prediction_real.append(pred_real.data.cpu().numpy()[0][0])
            
            # print(pred_fake)

            fake_traj = output_xy.view(-1, t.shape[0]*2)
            fake_traj = fake_traj.view(-1, 2)[:,:2].view(1, -1, 2).data.cpu().numpy()[0]
            x = fake_traj[:,0]*opt.max_dist
            y = fake_traj[:,1]*opt.max_dist
            
            # new_x = [item for item in x]
            # new_y = [item for item in y]
            # ax1.plot(new_x, new_y, label='2', linewidth=5)
            for i in range(len(x)):
                xs.append(x[i])
                ys.append(y[i])

    print('fake:', sum(avg_prediction)/len(avg_prediction), min(avg_prediction), max(avg_prediction))
    print('real', sum(avg_prediction_real)/len(avg_prediction_real), min(avg_prediction_real), max(avg_prediction_real))
    sns.kdeplot(xs, ys, shade=True, bw="silverman", gridsize=100, cmap="Purples", ax=ax1, cbar=True)

    plt.savefig('result/output3/%s/' % opt.dataset_name+str(total_step)+'.png', dpi=400)
    plt.cla()
    generator.train()

def draw_two_dim_vector(total_step):
    generator.eval()
    #xs = []
    #ys = []
    batch = next(test_samples)
    batch['t'] = batch['t'].view(-1,1).to(device)
    batch['v_0'] = batch['v_0'].view(-1,1).to(device)
    batch['v0_array'] = batch['v0_array'].view(-1,1).to(device)
    batch['xy'] = batch['xy'].view(-1,2).to(device)
    batch['t'].requires_grad = True

    real_condition = batch['v_0']
    fake_condition = torch.rand_like(real_condition)

    batch_fake_condition = fake_condition.unsqueeze(1).expand(1, opt.points_num, 1).reshape(1*opt.points_num, 1)#batch['v0_array']

    # real_traj_with_condition = torch.cat([real_traj, real_condition], dim=1)

    # for generator
    noise = torch.randn(1, opt.vector_dim).to(device)
    # noise = noise.unsqueeze(1)
    # noise = noise.expand(opt.batch_size, opt.points_num, noise.shape[-1])
    # noise = noise.reshape(opt.batch_size * opt.points_num, noise.shape[-1])

    # output_xy = generator(batch_fake_condition, noise, batch['t'])

    fig = plt.figure(figsize=(19, 19))
    ax1 = fig.add_subplot(111)
    n = 4
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            noise[:,0] = i/n
            noise[:,1] = j/n

            latent = noise.unsqueeze(1)
            latent = latent.expand(1, opt.points_num, 2)
            latent = latent.reshape(1 * opt.points_num, 2)

            # output_xy = generator(condition, noise, batch['t'])
            output_xy = generator(batch_fake_condition, latent, batch['t'])
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

    input_img = batch['img'].to(device)
    latent_vector = encoder(input_img)
    condition = latent_vector.unsqueeze(1)
    condition = condition.expand(1, opt.points_num, condition.shape[-1])
    condition = condition.reshape(1*opt.points_num, condition.shape[-1])

    noise = torch.randn(1, opt.vector_dim).unsqueeze(1)
    noise = noise.expand(1, opt.points_num, opt.vector_dim)
    noise = noise.reshape(1*opt.points_num, opt.vector_dim).clone()
    noise = noise.to(device)
    for i in range(11):
        v0 = torch.FloatTensor([[0.8*i/opt.max_speed]*opt.points_num]).view(-1,1).to(device)
        new_condition = torch.cat([condition, v0], dim=1)

        output_xy = generator(new_condition, noise, batch['t'])
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

    # for discriminator
    # single_condition = torch.rand_like(batch['v_0'])#batch['v_0']
    # condition = torch.rand_like(single_condition).unsqueeze(1).expand(opt.batch_size, opt.points_num, 1).reshape(opt.batch_size*opt.points_num, 1)#batch['v0_array']
    real_condition = batch['v_0']

    fake_condition = torch.rand_like(real_condition)
    batch_fake_condition = fake_condition.unsqueeze(1).expand(opt.batch_size, opt.points_num, 1).reshape(opt.batch_size*opt.points_num, 1)#batch['v0_array']

    real_traj_with_condition = torch.cat([real_traj, real_condition], dim=1)

    # for generator
    noise = torch.randn(opt.batch_size, opt.vector_dim).to(device)
    noise = noise.unsqueeze(1)
    noise = noise.expand(opt.batch_size, opt.points_num, noise.shape[-1])
    noise = noise.reshape(opt.batch_size * opt.points_num, noise.shape[-1])

    output_xy = generator(batch_fake_condition, noise, batch['t'])
    #grad0 = grad(output_xy.sum(), noise, create_graph=True)[0]
    #grad_loss = grad0.norm()
    
    set_requires_grad(discriminator, True)
    discriminator.zero_grad()
    pred_real = discriminator(real_traj_with_condition)
    
    fake_traj = output_xy.view(-1, opt.points_num*2)
    vx = (opt.max_dist/opt.max_t)*grad(output_xy.view(-1, opt.points_num, 2)[:,0].sum(), batch['t'], create_graph=True)[0]
    vy = (opt.max_dist/opt.max_t)*grad(output_xy.view(-1, opt.points_num, 2)[:,1].sum(), batch['t'], create_graph=True)[0]
    vxy = torch.cat([vx, vy], dim=1)
    start_v = vxy.view(-1, opt.points_num, 2)[:,0]/opt.max_speed
    
    # start point loss
    start_points = output_xy.view(-1, opt.points_num, 2)[:,0]
    ideal_start_points = torch.zeros(opt.batch_size, 2).to(device)
    start_point_loss = start_point_criterion(start_points, ideal_start_points)
    
    # start_v_loss = start_point_criterion(torch.norm(start_v, dim=1), batch['v_0'].squeeze(1))
    start_v_loss = start_point_criterion(torch.norm(start_v, dim=1), fake_condition.squeeze(1))

    fake_traj_with_condition = torch.cat([fake_traj.detach(), fake_condition], dim=1)
    pred_fake = discriminator(fake_traj_with_condition)
    
    alpha = torch.rand(opt.batch_size, 1)
    single_alpha = alpha.to(device)
    interpolated_condition = (single_alpha * real_condition.data + (1 - single_alpha) * fake_condition.data).requires_grad_(True)
    # interpolated_condition = fake_condition

    alpha = alpha.expand_as(real_traj)
    alpha = alpha.to(device)
    interpolated = (alpha * real_traj.data + (1 - alpha) * fake_traj.detach().data).requires_grad_(True)
    
    output_ = torch.cat([interpolated, interpolated_condition], dim=1)

    src_out_ = discriminator(output_)
    loss_D_real = torch.mean(pred_real)
    loss_D_fake = torch.mean(pred_fake)
    # Gradient penalty Loss
    loss_D_gp = fn_GP(src_out_, output_)
    loss_D = loss_D_fake - loss_D_real + 10*loss_D_gp
    # loss_D = 0.5 * (loss_D_real + loss_D_fake) + loss_D_gp
    loss_D.backward()
    torch.nn.utils.clip_grad_value_(discriminator.parameters(), clip_value=1)
    d_optimizer.step()
    
    set_requires_grad(discriminator, False)
    generator.zero_grad()
    # single_condition = torch.cat([latent_vector, batch['v_0']], dim=1)
    fake_traj_with_condition = torch.cat([fake_traj, fake_condition], dim=1)
    pred_fake = discriminator(fake_traj_with_condition)
    loss_G = -torch.mean(pred_fake) + 10*start_point_loss + 10*start_v_loss
    loss_G.backward()
    # torch.nn.utils.clip_grad_value_(encoder.parameters(), clip_value=1)
    torch.nn.utils.clip_grad_value_(generator.parameters(), clip_value=1)
    g_optimizer.step()
    
    logger.add_scalar('train/loss_G', loss_G.item(), total_step)
    logger.add_scalar('train/loss_D_real', loss_D_real.item(), total_step)
    logger.add_scalar('train/loss_D_fake', loss_D_fake.item(), total_step)
    logger.add_scalar('train/loss_D_gp', loss_D_gp.item(), total_step)
    #logger.add_scalar('train/grad_loss', grad_loss.item(), total_step)

    # if total_step % opt.test_interval == 0:
        # draw_diff_v(total_step)
        # draw_two_dim_vector(total_step)
        # test_uncertainty(total_step)

    if total_step % opt.test_interval == 0:
        show_traj(fake_traj.view(-1, 2)[:,:2].view(opt.batch_size, -1, 2).data.cpu().numpy()[0], batch['xy'].view(opt.batch_size, -1, 2).data.cpu().numpy()[0], batch['t'].view(opt.batch_size, -1).data.cpu().numpy()[0], total_step)
    
    if total_step % opt.checkpoint_interval == 0:
        torch.save(generator.state_dict(), 'result/saved_models/%s/generator_%d.pth'%(opt.dataset_name, total_step))
        torch.save(discriminator.state_dict(), 'result/saved_models/%s/discriminator_%d.pth'%(opt.dataset_name, total_step))
