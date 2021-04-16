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
from PIL import Image, ImageDraw
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.boost_model import Cluster
from utils import write_params, check_shape, to_device, set_mute

import carla_utils as cu
from robo_utils.kitti.torch_dataset import BoostDataset

random.seed(datetime.now())
torch.manual_seed(233)
torch.cuda.manual_seed(233)
torch.set_num_threads(16)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="avg-test-kitti-01", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=30., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--points_num', type=int, default=10, help='points number')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--gamma', type=float, default=0.2, help='xy and vxy loss trade off')
parser.add_argument('--gamma2', type=float, default=0.05, help='xy and axy loss trade off')
parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=2000, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=10, help='interval between model test')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_speed', type=float, default=10., help='max speed')
parser.add_argument('--max_t', type=float, default=3., help='max time')
parser.add_argument('--model_num', type=int, default=5, help='model_num')
opt = parser.parse_args()
if opt.test_mode: opt.batch_size = 1

description = 'change costmap'
log_path = 'result/log/'+opt.dataset_name+'/'+str(round(time.time()))+'/'
os.makedirs(log_path, exist_ok=True)
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)
write_params(log_path, parser, description)

logger_cluster = [
    SummaryWriter(log_dir=log_path+'model_'+str(i)) for i in range(opt.model_num)
]
    
cluster = Cluster(model_num=opt.model_num, device=device)
# cluster.load_models('result/saved_models/boost-05/', 40000)
# cluster.load_models('result/saved_models/boost-07/', 35000)
# cluster.load_models('result/saved_models/adversarial-01/', 28000)
cluster.load_models('result/saved_models/adversarial-02/', 43000)


param = cu.parse_yaml_file_unsafe('./param_kitti.yaml')
# trajectory_dataset = BoostDataset(param, 'train', opt)#7
# dataloader = DataLoader(trajectory_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
eval_trajectory_dataset = BoostDataset(param, 'eval', opt)#2
dataloader_eval = DataLoader(eval_trajectory_dataset, batch_size=1, shuffle=False, num_workers=opt.n_cpu)
eval_samples = iter(dataloader_eval)

# def eval_metric(step):
#     batch = next(eval_samples)
#     mask = [2, 5, 8, 10, 13, 17, 20, 23, 26, 29]
#     batch['ts_list'] = batch['ts_list'][:,mask]
#     batch['x_list'] = batch['x_list'][:,mask]
#     batch['y_list'] = batch['y_list'][:,mask]
#     batch['vx_list'] = batch['vx_list'][:,mask]
#     batch['vy_list'] = batch['vy_list'][:,mask]
    
#     t = batch['ts_list'].flatten().unsqueeze(1).to(device)
#     t.requires_grad = True
    
#     batch['img'] = batch['img'].expand(len(t),10,1,opt.height, opt.width)
#     batch['img'] = batch['img'].to(device)
#     batch['v_0'] = batch['v_0'].expand(len(t),1)
#     batch['v_0'] = batch['v_0'].to(device)
#     batch['xy'] = batch['xy'].to(device)
#     batch['vxy'] = batch['vxy'].to(device)
#     batch['img'].requires_grad = True
#     #print('batch[xy]', batch['xy'])
#     #print(t.shape, batch['ts_list'].flatten().unsqueeze(1).shape)
    
#     output = model(batch['img'], t, batch['v_0'])
#     vx = grad(output[:,0].sum(), t, create_graph=True)[0][:,0]*(opt.max_dist/opt.max_t)
#     vy = grad(output[:,1].sum(), t, create_graph=True)[0][:,0]*(opt.max_dist/opt.max_t)

#     x = output[:,0]*opt.max_dist
#     y = output[:,1]*opt.max_dist

#     ax = grad(vx.sum(), t, create_graph=True)[0]*(1./opt.max_t)
#     ay = grad(vy.sum(), t, create_graph=True)[0]*(1./opt.max_t)

#     jx = grad(ax.sum(), t, create_graph=True)[0]*(1./opt.max_t)
#     jy = grad(ay.sum(), t, create_graph=True)[0]*(1./opt.max_t)
    
#     vx = vx.data.cpu().numpy()
#     vy = vy.data.cpu().numpy()
#     x = x.data.cpu().numpy()
#     y = y.data.cpu().numpy()
    
#     real_x = batch['x_list'].data.cpu().numpy()[0]
#     real_y = batch['y_list'].data.cpu().numpy()[0]
#     real_vx = batch['vx_list'].data.cpu().numpy()[0]
#     real_vy = batch['vy_list'].data.cpu().numpy()[0]
#     ts_list = batch['ts_list'].data.cpu().numpy()[0]
    
#     # print('ts_list', ts_list, 'x_list', real_x, 'vx_list', real_vx)

#     ex = np.mean(np.abs(x-real_x))
#     ey = np.mean(np.abs(y-real_y))
#     evx = np.mean(np.abs(vx - real_vx))
#     evy = np.mean(np.abs(vy - real_vy))
#     fde = np.hypot(x - real_x, y - real_y)[-1]
#     ade = np.mean(np.hypot(x - real_x, y - real_y))
#     ev = np.mean(np.hypot(vx - real_vx, vy - real_vy))

#     jx = jx.data.cpu().numpy()
#     jy = jy.data.cpu().numpy()

#     # t = opt.max_t*ts_list#[1:]

#     smoothness = np.mean(np.hypot(jx, jy))
    
#     logger.add_scalar('metric/ex', ex, step)
#     logger.add_scalar('metric/ey', ey, step)
#     logger.add_scalar('metric/evx', evx, step)
#     logger.add_scalar('metric/evy', evy, step)
#     logger.add_scalar('metric/fde', fde, step)
#     logger.add_scalar('metric/ade', ade, step)
#     logger.add_scalar('metric/ev', ev, step)
#     logger.add_scalar('metric/smoothness', smoothness, step)
    
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
    cluster.eval_models()
    for model_index in range(opt.model_num):
        batch = next(eval_samples)
        batch['t'] = batch['t'].view(-1,1)
        batch['v0_array'] = batch['v0_array'].view(-1,1)
        batch['xy'] = batch['xy'].view(-1,2)
        to_device(batch, device)

        feature = cluster.get_encoder(batch['img'], model_index)
        feature_dim = feature.shape[-1]
        feature = feature.unsqueeze(1)
        feature = feature.expand(1, opt.points_num, feature_dim)
        feature = feature.reshape(1 * opt.points_num, feature_dim)
        
        output = cluster.get_trajectory(feature, batch['v0_array'], batch['t'], model_index)

        output_xy = output[:,:2]
        logvar = output[:,2:]

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
        
        logger = logger_cluster[model_index]
        logger.add_scalar('eval/ex',  ex.item(),  total_step)
        logger.add_scalar('eval/ey',  ey.item(),  total_step)
        logger.add_scalar('eval/fde', fde.item(), total_step)
        logger.add_scalar('eval/ade', ade.item(), total_step)

    cluster.train_models()

def eval_all_model(total_step):
    cluster.eval_models()
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
        feature = cluster.get_encoder(batch['img'], model_index)
        feature_dim = feature.shape[-1]
        feature = feature.unsqueeze(1)
        feature = feature.expand(1, opt.points_num, feature_dim)
        feature = feature.reshape(1 * opt.points_num, feature_dim)
        
        output = cluster.get_trajectory(feature, batch['v0_array'], batch['t'], model_index)

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

    cluster.train_models()

def eval_new_error(total_step, logger_avg, logger_min):
    cluster.eval_models()

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
    logvar_list = []
    for model_index in range(opt.model_num):
        feature = cluster.get_encoder(batch['img'], model_index)
        feature_dim = feature.shape[-1]
        feature = feature.unsqueeze(1)
        feature = feature.expand(1, opt.points_num, feature_dim)
        feature = feature.reshape(1 * opt.points_num, feature_dim)
        
        output = cluster.get_trajectory(feature, batch['v0_array'], batch['t'], model_index)

        output_xy = output[:,:2]
        logvar = output[:,2:]

        fake_traj = output_xy.data.cpu().numpy()*opt.max_dist

        fake_x = fake_traj[:,0]
        fake_y = fake_traj[:,1]
        fake_x_list.append(fake_x)
        fake_y_list.append(fake_y)

        logvar_list.append(opt.max_dist*torch.mean(torch.sqrt(torch.exp(logvar))).data.cpu().numpy())
        # logvar_list.append(logvar.cpu().numpy()*opt.max_dist)

    ################################################################
    # import pdb; pdb.set_trace()

    fake_x = np.mean(np.array(fake_x_list), 0)
    fake_y = np.mean(np.array(fake_y_list), 0)

    ex = np.mean(np.abs(fake_x-real_x))
    ey = np.mean(np.abs(fake_y-real_y))
    fde = np.hypot(fake_x - real_x, fake_y - real_y)[-1]
    ade = np.mean(np.hypot(fake_x - real_x, fake_y - real_y))
        
    logger = logger_avg
    logger.add_scalar('eval/ex',  ex.item(),  total_step)
    logger.add_scalar('eval/ey',  ey.item(),  total_step)
    logger.add_scalar('eval/fde', fde.item(), total_step)
    logger.add_scalar('eval/ade', ade.item(), total_step)
    ################################################################
    index = np.argmin(np.array(logvar_list))
    fake_x = fake_x_list[index]
    fake_y = fake_y_list[index]

    ex = np.mean(np.abs(fake_x-real_x))
    ey = np.mean(np.abs(fake_y-real_y))
    fde = np.hypot(fake_x - real_x, fake_y - real_y)[-1]
    ade = np.mean(np.hypot(fake_x - real_x, fake_y - real_y))

    logger = logger_min
    logger.add_scalar('eval/ex',  ex.item(),  total_step)
    logger.add_scalar('eval/ey',  ey.item(),  total_step)
    logger.add_scalar('eval/fde', fde.item(), total_step)
    logger.add_scalar('eval/ade', ade.item(), total_step)

    cluster.train_models()

logger_avg = SummaryWriter(log_dir=log_path+'avg_model')
logger_min = SummaryWriter(log_dir=log_path+'min_uncty')
for total_steps in range(1000000):
    eval_new_error(total_steps, logger_avg, logger_min)
    eval_error(total_steps)

# print('Start to train ...')
# for total_step in range(10000):
#     batch = next(eval_samples)
#     check_shape(batch)
#     to_device(batch, device)
#     batch['t'] = batch['t'].view(-1,1)
#     batch['t'].requires_grad = True
#     batch['v0_array'] = batch['v0_array'].view(-1,1)
#     # continue
#     model_index = 0
#     feature = cluster.get_encoder(batch['img'], model_index)
#     feature_dim = feature.shape[-1]
#     feature = feature.unsqueeze(1)
#     opt.batch_size = 1# eval
#     feature = feature.expand(opt.batch_size, opt.points_num, feature_dim)
#     feature = feature.reshape(opt.batch_size * opt.points_num, feature_dim)
#     check_shape(feature, 'feature')
#     check_shape(batch['v0_array'], 'v0_array')
#     check_shape(batch['t'], 't')
#     output = cluster.get_trajectory(feature, batch['v0_array'], batch['t'], model_index)

#     output_xy = output[:,:2]
#     logvar = output[:,2:]
#     check_shape(logvar, 'logvar')

#     vx = grad(output_xy[:,0].sum(), batch['t'], create_graph=True)[0]
#     vy = grad(output_xy[:,1].sum(), batch['t'], create_graph=True)[0]
#     output_vxy = (opt.max_dist/opt.max_t)*torch.cat([vx, vy], dim=1)
#     check_shape(output_vxy, 'output_vxy')

#     fake_traj = output_xy.reshape(-1, opt.points_num*2)
#     fake_logvar = logvar.reshape(-1, opt.points_num*2)

#     check_shape(output_xy, 'output_xy')
