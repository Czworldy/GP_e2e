#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '../../../'))

import torch

import time
import numpy as np

from agent import RIPAgent

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = RIPAgent(output_shape=(16, 2), model_num=5, opt_steps=10)

observation = {
    'visual_features': torch.randn(
        (32, 3, 200, 400)).to(device),  #[B, 3, H, W]
    'velocity': torch.randn((32, 1)).to(device),  #[B, 1]
}
t1 = time.time()
xy = model.step(observation)
t2 = time.time()
print(xy)
print('time:', t2 - t1)