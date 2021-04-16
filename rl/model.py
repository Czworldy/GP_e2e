#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.1)

class CNN(nn.Module):
    def __init__(self,input_dim=1, out_dim=256):
        super(CNN, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, self.out_dim, 3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        #x = self.bn2(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)         
        x = self.conv3(x)
        #x = self.bn3(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = F.leaky_relu(x)
        x = x.view(-1, self.out_dim)
        return x
    
class QNet(nn.Module):
    def __init__(self, num_actions):
        super(QNet, self).__init__()
        hidden_dim = 256
        self.cnn_feature_dim = hidden_dim
        self.rnn_hidden_dim = hidden_dim
        self.cnn = CNN(input_dim=1, out_dim=self.cnn_feature_dim)
        self.gru = nn.GRU(
            input_size = self.cnn_feature_dim, 
            hidden_size = self.rnn_hidden_dim, 
            num_layers = 3,
            batch_first=True,
            dropout=0.2)
        self.mlp = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim + num_actions, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(weights_init)
        
    def forward(self, x, action):
        batch_size, timesteps, C, H, W = x.size()
        
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        
        x = x.view(batch_size, timesteps, -1)
        x, h_n, = self.gru(x)

        x = F.leaky_relu(x[:, -1, :])
        
        x = torch.cat([x, action], 1)
        x = self.mlp(x)
        return x
    
class QNet2(nn.Module):
    def __init__(self, num_actions):
        super(QNet2, self).__init__()
        hidden_dim = 256
        self.cnn_feature_dim = hidden_dim
        self.cnn = CNN(input_dim=1, out_dim=self.cnn_feature_dim) #yujiyu 6
        self.mlp = nn.Sequential(
            nn.Linear(self.cnn_feature_dim + num_actions, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(weights_init)
        
    def forward(self, x, action):
        x = self.cnn(x)
        x = F.leaky_relu(x)
        x = torch.cat([x, action], 1)
        x = self.mlp(x)
        return x

class QNet3(nn.Module):
    def __init__(self, num_actions):
        super(QNet3, self).__init__()
        hidden_dim = 256
        self.cnn_feature_dim = hidden_dim
        self.cnn = CNN(input_dim=6, out_dim=self.cnn_feature_dim) #yujiyu 6
        self.mlp = nn.Sequential(
            nn.Linear(self.cnn_feature_dim + num_actions + 1, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(weights_init)
        
    def forward(self, x, v, action):
        x = self.cnn(x)
        x = F.leaky_relu(x)
        # print("x shape", x.shape)
        # print("action shape", action.shape)
        x = torch.cat([x, action], 1)
        x = torch.cat([x, v], 1)
        x = self.mlp(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, num_actions=2):
        super(PolicyNet, self).__init__()
        hidden_dim = 256
        self.cnn_feature_dim = hidden_dim
        self.rnn_hidden_dim = hidden_dim
        self.cnn = CNN(input_dim=1, out_dim=self.cnn_feature_dim)
        self.gru = nn.GRU(
            input_size = self.cnn_feature_dim, 
            hidden_size = self.rnn_hidden_dim, 
            num_layers = 3,
            batch_first=True,
            dropout=0.2)
        self.mlp = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
            nn.Tanh()
        )

        self.apply(weights_init)
        
    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        
        x = x.view(batch_size, timesteps, -1)
        x, h_n, = self.gru(x)

        x = F.leaky_relu(x[:, -1, :])
        
        x = self.mlp(x)
        return x
    
    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]
    
class CNNNorm(nn.Module):
    def __init__(self, input_dim=1, out_dim=256):
        super(CNNNorm, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, self.out_dim),
            nn.Tanh()
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)         
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = F.leaky_relu(x, inplace=True)
        x = x.view(-1, 256)
        x = self.mlp(x)
        return x
    
    def get_action(self, state):
        state  = torch.stack([state]).to(device)
        #state  = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]