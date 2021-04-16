#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import gym

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from .common import ReplayBuffer,GaussianExploration,soft_update
from .model import QNet, QNet2, QNet3
from learning.model import  EncoderWithV

from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3():
    def __init__(
            self,
            args = None,
            env_name = 'Pendulum-v0',
            load_dir = './ckpt',
            logger = None,
            buffer_size = 3e3,
            seed = 1,
            max_episode_steps = None,
            noise_decay_steps = 1e5,
            batch_size = 8,
            discount = 0.99,
            train_freq = 100,
            policy_freq = 2,
            learning_starts = 500,
            tau = 0.005,
            save_eps_num = 100,
            external_env = None,
            is_fix_policy_net = False
            ):
        self.args = args
        self.env_name = env_name
        self.load_dir = load_dir
        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.buffer_size = buffer_size
        self.noise_decay_steps = noise_decay_steps
        self.batch_size = batch_size
        self.discount = discount
        self.policy_freq = policy_freq
        self.learning_starts = learning_starts
        self.tau = tau
        self.save_eps_num = save_eps_num
        self.logger = logger
        self.is_fix_policy_net = is_fix_policy_net

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.noise = GaussianExploration(
            gym.spaces.Box(low=-np.ones(64), high=np.ones(64), dtype=np.float32),
            max_sigma=0.5, min_sigma=0.05,
            decay_period=self.noise_decay_steps)
        
        action_dim = 64
        self.value_net1 = QNet3(action_dim).to(device)
        self.value_net2 = QNet3(action_dim).to(device)
        self.policy_net = EncoderWithV(input_dim=6, out_dim=64).to(device)
        
        self.target_value_net1 = QNet3(action_dim).to(device)
        self.target_value_net2 = QNet3(action_dim).to(device)
        self.target_policy_net = EncoderWithV(input_dim=6, out_dim=64).to(device)
        """
        try:
            self.load(directory=self.load_dir, filename=self.env_name)
            print('Load model successfully !')
        except:
            print('WARNING: No model to load !')
        """
        soft_update(self.value_net1, self.target_value_net1, soft_tau=1.0)
        soft_update(self.value_net2, self.target_value_net2, soft_tau=1.0)
        soft_update(self.policy_net, self.target_policy_net, soft_tau=1.0)
        
        self.value_criterion = nn.MSELoss()
        
        policy_lr = 1e-5
        value_lr  = 1e-4
        
        # self.value_optimizer1 = optim.SGD(self.value_net1.parameters(), lr=value_lr, momentum=0.9, weight_decay=5e-4)
        # self.value_optimizer2 = optim.SGD(self.value_net2.parameters(), lr=value_lr, momentum=0.9, weight_decay=5e-4)
        # self.policy_optimizer = optim.SGD(self.policy_net.parameters(), lr=policy_lr, momentum=0.9, weight_decay=5e-4)
        self.value_optimizer1 = optim.Adam(self.value_net1.parameters(), lr=value_lr)
        self.value_optimizer2 = optim.Adam(self.value_net2.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        self.train_steps = 0
        
    def save(self, directory, filename):
        print("model saving......")
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.value_net1.state_dict(), '%s/%s_value_net1.pkl' % (directory, filename))
        torch.save(self.value_net2.state_dict(), '%s/%s_value_net2.pkl' % (directory, filename))
        if not self.is_fix_policy_net:
            torch.save(self.policy_net.state_dict(), '%s/%s_policy_net.pkl' % (directory, filename))

    def load(self, directory, filename):
        self.value_net1.load_state_dict(torch.load('%s/%s_value_net1.pkl' % (directory, filename)))
        self.value_net2.load_state_dict(torch.load('%s/%s_value_net2.pkl' % (directory, filename)))
        self.policy_net.load_state_dict(torch.load('%s/%s_policy_net.pkl' % (directory, filename)))
        
    def train_step(self,
           step,
           noise_std = 0.2,
           noise_clip=0.5
          ):
         
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        # print('state shape:', state.shape)

        # print('state type[0]:', type(state))
        # state_img      = torch.from_numpy(state['img_nav']).to(device)
        # lis = [item['v0']/self.args.max_speed for item in state]
        # print(lis)
        state_img      = torch.cat([item['img_nav'].unsqueeze(0) for item in state], dim=0).to(device)
        # print(state_img.shape)
        # state_img      = state_img.unsqueeze(0).to(device)
        # state_img      = state['img_nav'].unsqueeze(0).to(device)
        state_v0       = torch.FloatTensor([item['v0']/self.args.max_speed for item in state]).unsqueeze(1).to(device)
        # print("v0:",state_v0)
        # print("v0 shape",state_v0.shape)
        # next_state_img = torch.from_numpy(next_state['img_nav']).to(device)
        next_state_img = torch.cat([item['img_nav'].unsqueeze(0) for item in next_state], dim=0).to(device)
        # next_state_img = next_state_img.unsqueeze(0).to(device)
        # next_state_img = next_state['img_nav'].unsqueeze(0).to(device)
        next_state_v0  = torch.FloatTensor([item['v0']/self.args.max_speed for item in next_state]).unsqueeze(1).to(device)
        action         = torch.FloatTensor(action).to(device)
        reward         = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done           = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
    
        next_action = self.target_policy_net(next_state_img, next_state_v0)
        noise = torch.normal(torch.zeros(next_action.size()), noise_std).to(device)
        noise = torch.clamp(noise, -noise_clip, noise_clip)
        next_action += noise
        next_action = torch.clamp(next_action, -0.99, 0.99)
    
        target_q_value1  = self.target_value_net1(next_state_img, next_state_v0, next_action)
        target_q_value2  = self.target_value_net2(next_state_img, next_state_v0, next_action)
        target_q_value   = torch.min(target_q_value1, target_q_value2)
        expected_q_value = reward + (1.0 - done) * self.discount * target_q_value
        #print(reward.mean(), target_q_value.mean())
    
        q_value1 = self.value_net1(state_img, state_v0, action)
        q_value2 = self.value_net2(state_img, state_v0, action)
        
        value_loss1 = self.value_criterion(q_value1, expected_q_value.detach())
        value_loss2 = self.value_criterion(q_value2, expected_q_value.detach())

        if self.logger is not None:
            self.logger.add_scalar('q_value1', q_value1.mean().item(), self.train_steps)
            self.logger.add_scalar('q_value2', q_value2.mean().item(), self.train_steps)
            self.logger.add_scalar('value_loss1', value_loss1.item(), self.train_steps)
            self.logger.add_scalar('value_loss2', value_loss2.item(), self.train_steps)

        self.value_optimizer1.zero_grad()
        value_loss1.backward()
        # torch.nn.utils.clip_grad_value_(self.value_net1.parameters(), clip_value=1)
        self.value_optimizer1.step()
    
        self.value_optimizer2.zero_grad()
        value_loss2.backward()
        # torch.nn.utils.clip_grad_value_(self.value_net2.parameters(), clip_value=1)
        self.value_optimizer2.step()
    
        if step % self.policy_freq == 0 and self.is_fix_policy_net == False:
        # if step % self.policy_freq == 0:
            print("update policy net!")
            policy_loss = self.value_net1(state_img, state_v0, self.policy_net(state_img, state_v0))
            policy_loss = -policy_loss.mean()

            if self.logger is not None:
                self.logger.add_scalar('policy_loss', policy_loss.item(), self.train_steps)
    
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), clip_value=1)
            self.policy_optimizer.step()
    
            soft_update(self.value_net1, self.target_value_net1, soft_tau=self.tau)
            soft_update(self.value_net2, self.target_value_net2, soft_tau=self.tau)
            soft_update(self.policy_net, self.target_policy_net, soft_tau=self.tau)

        self.train_steps += 1

    # def predict(self, state):
    #     return self.policy_net.get_action(state)