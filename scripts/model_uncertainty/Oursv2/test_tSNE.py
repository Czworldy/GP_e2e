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
import seaborn as sns

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.model import Generator, EncoderWithV, Encoder, Generatorv2, MobileNetV2, MixStyleEncoder, CNN
from robo_utils.oxford.oxford_dataset import DIVADataset
from robo_utils.kitti.torch_dataset import OurDataset as KittiDataset
from utils import write_params, check_shape, to_device, set_mute
from carla_utils import parse_yaml_file_unsafe

from tqdm import tqdm
from sklearn import manifold

random.seed(datetime.now())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="tSNE-01", help='name of the dataset')
# parser.add_argument('--dataset_name', type=str, default="train-latent-loss-05", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=30., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--vector_dim', type=int, default=64, help='vector dim')
parser.add_argument('--points_num', type=int, default=16, help='points number')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_speed', type=float, default=10., help='max speed')
parser.add_argument('--max_t', type=float, default=3., help='max time')
opt = parser.parse_args()

encoder = EncoderWithV(input_dim=6, out_dim=opt.vector_dim).to(device)

encoder.load_state_dict(torch.load('result/saved_models/main-IRM-two-models-09/encoder4_52000.pth'))
# encoder.load_state_dict(torch.load('result/saved_models/train-e2e-02/encoder_12000.pth'))
# encoder.load_state_dict(torch.load('result/saved_models/train-e2e-fix-decoder-01/encoder_189000.pth'))
# encoder = RNNEncoder(input_dim=3, out_dim=opt.vector_dim).to(device)

param = parse_yaml_file_unsafe('./param_oxford.yaml')
train_loader = DataLoader(DIVADataset(param, mode='train', opt=opt), batch_size=opt.batch_size, shuffle=False, num_workers=16)
train_samples = iter(train_loader)

param = parse_yaml_file_unsafe('./param_kitti.yaml')
eval_loader = DataLoader(KittiDataset(param, mode='eval', opt=opt), batch_size=opt.batch_size, shuffle=False, num_workers=16)
eval_samples = iter(eval_loader)


num_points = 400
file_dir = 'ours_fix_alpha_'+str(num_points)
os.makedirs(file_dir, exist_ok=True)



feature_list = np.random.randn(num_points*opt.batch_size*2, opt.vector_dim)
label = []

for total_step in tqdm(range(num_points)):
    batch = next(train_samples)
    to_device(batch, device)
    # feature = encoder(batch['img'])
    feature = encoder(batch['img'], batch['v_0'])
    check_shape(feature, 'feature')
    set_mute(True)
    feature_list[total_step*opt.batch_size: (total_step+1)*opt.batch_size] = feature.data.cpu().numpy()
label = [0]*num_points*opt.batch_size


for total_step in tqdm(range(num_points, 2*num_points)):
    batch = next(eval_samples)
    to_device(batch, device)
    feature = encoder(batch['img'], batch['v_0'])
    check_shape(feature, 'feature')
    set_mute(True)
    feature_list[total_step*opt.batch_size: (total_step+1)*opt.batch_size] = feature.data.cpu().numpy()
label.extend([1]*num_points*opt.batch_size)

np_feature_list = np.array(feature_list)
print(np_feature_list.shape)
np.save(file_dir+'/feature.npy', np_feature_list)


X = feature_list
y = label
print('run t-SNE')
# '''t-SNE'''
tsne = manifold.TSNE(n_components=2, init='pca')
X_tsne = tsne.fit_transform(X)

np.save(file_dir+'/t-SNE.npy', X_tsne)
# X_tsne = np.load('e2e_100_1616562155.9316053/t-SNE.npy')
# import pdb; pdb.set_trace()
print('finish t-SNE')

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)

plt.figure(figsize=(8, 8))
print(plt.cm.Set1(y[0]))
for i in tqdm(range(X_norm.shape[0])):
    # plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
    #          fontdict={'weight': 'bold', 'size': 4})
    plt.scatter(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set1(y[i]), marker='o', s=8, alpha=0.3)
plt.xticks([])
plt.yticks([])
# plt.show()
plt.savefig(file_dir+'/oxford_kitti_'+str(num_points)+'.pdf')
