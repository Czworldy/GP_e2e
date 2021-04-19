#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

class EncoderWithV(nn.Module):
    def __init__(self, input_dim=6, out_dim=64):
        super(EncoderWithV, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, self.out_dim, 3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.linear1 = nn.Linear(self.out_dim + 1, 256)
        self.linear2 = nn.Linear(256, self.out_dim)
        
        

    def forward(self, x, v):
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
        x = x.view(-1, self.out_dim)
        # print(x)
        x = F.leaky_relu(x, inplace=True)
        x = torch.cat([x, v], 1)
        x = self.linear1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear2(x)
        return x

class PolicyNet(nn.Module):
    def __init__(self, input_dim=6, out_dim=1):
        super(PolicyNet, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 64, 3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.nlinear1 = nn.Linear(64, 1024)
        self.nlinear2 = nn.Linear(1024, 512)
        self.nlinear3 = nn.Linear(512, self.out_dim)
        
        

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
        x = x.view(-1, 64)
        # print(x.shape)
        x = F.leaky_relu(x, inplace=True)
        x = self.nlinear1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.nlinear2(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.nlinear3(x)
        # throttle = F.sigmoid(x[:,0])
        # steer = F.tanh(x[:,1])
        # torch.cat([throttle, steer], dim=0)
        ### action only care about steer ###
        x = F.tanh(x)
        return x

class ValueNet(nn.Module):
    def __init__(self, input_dim=6, out_dim=1, action_dim=1):
        super(ValueNet , self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 64, 3, stride=2, padding=1)
        
        self.nbn1 = nn.BatchNorm2d(64)
        self.nbn2 = nn.BatchNorm2d(128)
        self.nbn3 = nn.BatchNorm2d(256)

        self.nlinear1 = nn.Linear(64+action_dim, 1024)
        self.nlinear2 = nn.Linear(1024, 512)
        self.nlinear3 = nn.Linear(512, self.out_dim)
        
        

    def forward(self, x, action):
        x = self.conv1(x)
        x = self.nbn1(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.nbn2(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)         
        x = self.conv3(x)
        x = self.nbn3(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = x.view(-1, 64)
        x = F.leaky_relu(x, inplace=True)
        # print("x",x.shape)
        # print("a",action.shape)
        x = torch.cat([x, action], 1)
        # print("cat x",x.shape)
        x = self.nlinear1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.nlinear2(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.nlinear3(x)
        return x



if __name__ == "__main__":
    model = PolicyNet()
    val = ValueNet()
    pretrain_model = EncoderWithV()
    pretrain_model.load_state_dict(torch.load('/home/cz/Downloads/learning-uncertainty-master/scripts/encoder_e2e.pth'))
    
    pretrained_dict = torch.load('/home/cz/Downloads/learning-uncertainty-master/scripts/encoder_e2e.pth')
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print(pretrained_dict.keys())
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    model.conv1.requires_grad_(False)
    model.conv2.requires_grad_(False)
    model.conv3.requires_grad_(False)
    model.conv4.requires_grad_(False)
    model.bn1.requires_grad_(False)
    model.bn2.requires_grad_(False)
    model.bn3.requires_grad_(False)
    x = torch.ones([3,6,200,400])*2
    y = model(x)
    v = val(x,y)
    print(v.shape)
    print("y-------------------",y)
    print(y.shape)
    op = optim.Adam(model.parameters(),lr=1e-5)
    op.zero_grad()
    
    loss_f = torch.nn.MSELoss()
    loss = loss_f(y,torch.ones([3,1]))
    loss.backward()
    op.step()

    y = model(x)
    print("y-------------------",y)

    # layers = list(model.children())
    # print(layers)
    # x = torch.ones([1,6,200,400])
    # v = torch.ones([1,1])
    # y = model(x)
    # y = pretrain_model(x,v)



