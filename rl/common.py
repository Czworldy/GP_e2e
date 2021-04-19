import gym
import random
import numpy as np
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action

class GaussianExploration(object):
    def __init__(self, action_space, max_sigma=1.0, min_sigma=0.01, decay_period=1e4):
        self.low  = action_space.low
        self.high = action_space.high
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = int(decay_period)
    
    def get_action(self, action, t=0):
        #sigma  = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        self.sigma = self.sigma*1.#0.9999 #0.9999 #0.9997 #0.99998
        self.sigma = max(self.sigma, self.min_sigma)
        action = action + np.random.normal(loc=0, scale=0.2, size=len(action)) #* self.sigma #0.2
        return np.clip(action, self.low, self.high)


def soft_update(net, target_net, soft_tau=1e-2):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )