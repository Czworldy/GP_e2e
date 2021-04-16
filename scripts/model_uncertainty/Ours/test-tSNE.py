
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
sys.path.insert(0, join(dirname(__file__), '../../'))
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
from sklearn import manifold
from tqdm import tqdm

import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.boost_model import Discriminator, Cluster
from carla_utils import parse_yaml_file_unsafe
from robo_utils.oxford.oxford_dataset import BoostDataset, DIMDataset
from utils import write_params, check_shape, to_device, set_mute

random.seed(datetime.now())
torch.manual_seed(233)
torch.cuda.manual_seed(233)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="test-t-SNE", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--traj_steps', type=int, default=8, help='traj steps')
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
if opt.test_mode: opt.batch_size = 1
    
description = 'train cluster, with v0 loss'
log_path = 'result/log/'+opt.dataset_name+'/'+str(round(time.time()))+'/'
os.makedirs(log_path, exist_ok=True)
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)

write_params(log_path, parser, description)

logger_cluster = [
    SummaryWriter(log_dir=log_path+'model_'+str(i)) for i in range(opt.model_num)
]

cluster = Cluster(model_num=1, device=device)
# cluster.load_models('result/saved_models/boost-05/', 40000)
cluster.load_models('result/saved_models/adversarial-single-model-02/', 6200)

param = parse_yaml_file_unsafe('./param_oxford.yaml')

train_loader_cluster = [
    iter(DataLoader(
        DIMDataset(param, mode='train', opt=opt, data_index=i), 
            batch_size=1, shuffle=False, num_workers=1)) for i in range(opt.model_num)
]

num_points = 20000
feature_list = np.random.randn(num_points*1, 256)
label = []

for total_step in tqdm(range(num_points)):
    train_loader = train_loader_cluster[total_step % opt.model_num]
    batch = next(train_loader)
    batch['t'] = batch['t'].view(-1,1)
    batch['v0_array'] = batch['v0_array'].view(-1,1)
    batch['xy'] = batch['xy'].view(-1,2)
    to_device(batch, device)

    real_traj = batch['xy'].data.cpu().numpy()*opt.max_dist
    real_x = real_traj[:,0]
    real_y = real_traj[:,1]
    model_index = 0

    feature = cluster.get_encoder(batch['img'], index=model_index)
    check_shape(feature, 'feature')
    check_shape(batch['xy'], 'xy')
    set_mute(True)
    feature = feature.data.cpu().numpy()[0]
    
    feature_list[total_step] = feature
    label.append(total_step % opt.model_num)


# num_points = 5000
# traj_list = np.random.randn(num_points*opt.model_num, opt.points_num*2)
# label = []
# for model_index in range(opt.model_num):
#     for total_step in tqdm(range(num_points)):
#         batch = next(train_loader_cluster[model_index])
#         traj_list[total_step+num_points*model_index] = batch['xy'].flatten().data.cpu().numpy()
#         label.append(model_index)


# eval_loader = DataLoader(BoostDataset(param, mode='eval', opt=opt), batch_size=1, shuffle=False, num_workers=1)
# eval_samples = iter(eval_loader)
# cluster.eval_models()
# num_points = 50000
# feature_list = np.random.randn(num_points*opt.model_num, 256)
# traj_list = np.random.randn(num_points*opt.model_num, 16*2)
# label = []

# for total_step in tqdm(range(num_points)):
#     batch = next(eval_samples)
#     batch['t'] = batch['t'].view(-1,1)
#     batch['v0_array'] = batch['v0_array'].view(-1,1)
#     batch['xy'] = batch['xy'].view(-1,2)
#     to_device(batch, device)

#     real_traj = batch['xy'].data.cpu().numpy()*opt.max_dist
#     real_x = real_traj[:,0]
#     real_y = real_traj[:,1]
#     for model_index in range(opt.model_num):
#         feature = cluster.get_encoder(batch['img'], index=model_index)
#         check_shape(feature, 'feature')
#         check_shape(batch['xy'], 'xy')
#         set_mute(True)
#         feature = feature.data.cpu().numpy()[0]
#         index = model_index + total_step * opt.model_num
#         feature_list[index] = feature
#         traj_list[index] = batch['xy'].flatten().data.cpu().numpy()
#         label.append(model_index)

        # final_angle = np.arctan2(real_y[-1], real_x[-1])
        # if real_x[-1] > 10:
        #     label.append(1)
        # elif real_y[-1] < -3:
        #     label.append(2)
        # elif real_y[-1] > 3:
        #     label.append(3)
        # else:
        #     label.append(4)
        # if final_angle > (5*np.pi/180):
        #     label.append(1)
        #     label_1.append(1)
        # elif final_angle < -5*np.pi/180.:
        #     label.append(2)
        #     label_2.append(1)
        # else:
        #     label.append(3)
        #     label_3.append(1)

X = feature_list
# X = traj_list
y = label
print('run t-SNE')
# '''t-SNE'''
tsne = manifold.TSNE(n_components=2, init='pca')
X_tsne = tsne.fit_transform(X)
print('finish t-SNE')
# print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

# '''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    # plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
    #          fontdict={'weight': 'bold', 'size': 4})
    plt.scatter(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set1(y[i]), s=10)
plt.xticks([])
plt.yticks([])
plt.savefig('single_traj_'+str(num_points)+'.pdf')
# plt.savefig('feature_'+str(num_points)+'.pdf')
# plt.show()