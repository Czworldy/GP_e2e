#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import gym

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from .common import ReplayBuffer,GaussianExploration,soft_update
from .model import QNet,PolicyNet,CNNNorm, QNet2

from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3():
    def __init__(
            self,
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
            external_env = None
            ):
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

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.noise = GaussianExploration(
            gym.spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.]), dtype=np.float32),
            max_sigma=0.5, min_sigma=0.05,
            decay_period=self.noise_decay_steps)
        
        action_dim = 2
        self.value_net1 = QNet2(action_dim).to(device)
        self.value_net2 = QNet2(action_dim).to(device)
        #self.policy_net = PolicyNet(action_dim).to(device)
        self.policy_net = CNNNorm(input_dim=1, out_dim=2).to(device)
        
        self.target_value_net1 = QNet2(action_dim).to(device)
        self.target_value_net2 = QNet2(action_dim).to(device)
        #self.target_policy_net = PolicyNet(action_dim).to(device)
        self.target_policy_net = CNNNorm(input_dim=1, out_dim=2).to(device)
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
        value_lr  = 1e-5

        # yujiyu
        # policy_lr = 1e-6
        # value_lr  = 1e-6        
        # self.value_optimizer1 = optim.SGD(self.value_net1.parameters(), lr=value_lr, momentum=0.9, weight_decay=5e-4)
        # self.value_optimizer2 = optim.SGD(self.value_net2.parameters(), lr=value_lr, momentum=0.9, weight_decay=5e-4)
        # self.policy_optimizer = optim.SGD(self.policy_net.parameters(), lr=policy_lr, momentum=0.9, weight_decay=5e-4)
        
        # self.value_optimizer1 = optim.SGD(self.value_net1.parameters(), lr=value_lr, weight_decay=5e-4)
        # self.value_optimizer2 = optim.SGD(self.value_net2.parameters(), lr=value_lr, weight_decay=5e-4)
        # self.policy_optimizer = optim.SGD(self.policy_net.parameters(), lr=policy_lr, weight_decay=5e-4)

        # yujiyu 
        self.value_optimizer1 = optim.Adam(self.value_net1.parameters(), lr=value_lr, weight_decay=5e-4)
        self.value_optimizer2 = optim.Adam(self.value_net2.parameters(), lr=value_lr, weight_decay=5e-4)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr, weight_decay=5e-4)
        
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        self.train_steps = 0
        
    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.value_net1.state_dict(), '%s/%s_value_net1.pkl' % (directory, filename))
        torch.save(self.value_net2.state_dict(), '%s/%s_value_net2.pkl' % (directory, filename))
        torch.save(self.policy_net.state_dict(), '%s/%s_policy_net.pkl' % (directory, filename))

    def load(self, directory, filename):
        self.value_net1.load_state_dict(torch.load('%s/%s_value_net1.pkl' % (directory, filename)))
        self.value_net2.load_state_dict(torch.load('%s/%s_value_net2.pkl' % (directory, filename)))
        self.policy_net.load_state_dict(torch.load('%s/%s_policy_net.pkl' % (directory, filename)))
        
    def train_step(self,
           step,
           noise_std = 0.2,
           noise_clip = 0.5
          ):

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        # print('train state', type(state))
        # print('train type', state.shape)
        # print('train shape[0]', state[0].shape)
        # print('train type[0]', type(state[0]))
        state      = torch.from_numpy(state).to(device)
        next_state = torch.from_numpy(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
    
        next_action = self.target_policy_net(next_state)
        noise = torch.normal(torch.zeros(next_action.size()), noise_std).to(device)
        noise = torch.clamp(noise, -noise_clip, noise_clip)
        next_action += noise
        next_action = torch.clamp(next_action, -0.99, 0.99)
    
        target_q_value1  = self.target_value_net1(next_state, next_action)
        target_q_value2  = self.target_value_net2(next_state, next_action)
        target_q_value   = torch.min(target_q_value1, target_q_value2)
        expected_q_value = reward + (1.0 - done) * self.discount * target_q_value
        #print(reward.mean(), target_q_value.mean())
    
        q_value1 = self.value_net1(state, action)
        q_value2 = self.value_net2(state, action)
        
        value_loss1 = self.value_criterion(q_value1, expected_q_value.detach())
        value_loss2 = self.value_criterion(q_value2, expected_q_value.detach())

        if self.logger is not None:
            self.logger.add_scalar('q_value1', q_value1.mean().item(), self.train_steps)
            self.logger.add_scalar('q_value2', q_value2.mean().item(), self.train_steps)
            self.logger.add_scalar('value_loss1', value_loss1.item(), self.train_steps)
            self.logger.add_scalar('value_loss2', value_loss2.item(), self.train_steps)

        self.value_optimizer1.zero_grad()
        value_loss1.backward()
        torch.nn.utils.clip_grad_value_(self.value_net1.parameters(), clip_value=1)
        self.value_optimizer1.step()
    
        self.value_optimizer2.zero_grad()
        value_loss2.backward()
        torch.nn.utils.clip_grad_value_(self.value_net2.parameters(), clip_value=1)
        self.value_optimizer2.step()
    
        if step % self.policy_freq == 0:

            policy_loss = self.value_net1(state, self.policy_net(state))
            policy_loss = -policy_loss.mean() 

            # yujiyu error : only use value_net1
            # policy_loss_1 = self.value_net1(state, self.policy_net(state))
            # policy_loss_2 = self.value_net2(state, self.policy_net(state))
            # policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

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

    def predict(self, state):
        return self.policy_net.get_action(state)