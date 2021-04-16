#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        try:
            nn.init.constant_(m.bias, 0.01)
        except:
            pass
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0.01)


class ModelGRU(nn.Module):
    def __init__(self, hidden_dim=256):
        super(ModelGRU, self).__init__()
        self.cnn_feature_dim = hidden_dim
        self.rnn_hidden_dim = hidden_dim
        self.cnn = CNN(input_dim=1, out_dim=self.cnn_feature_dim)
        #self.cnn = ResidualNet()
        self.gru = nn.GRU(
            input_size = self.cnn_feature_dim, 
            hidden_size = self.rnn_hidden_dim, 
            num_layers = 3,
            batch_first=True,
            dropout=0.2)
        self.mlp = MLP_COS(input_dim=self.rnn_hidden_dim+2)

    def forward(self, x, t, v0):
        batch_size, timesteps, C, H, W = x.size()
        
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        
        x = x.view(batch_size, timesteps, -1)
        x, h_n, = self.gru(x)

        x = F.leaky_relu(x[:, -1, :])
        x = self.mlp(x, t, v0)
        return x

class ModelCurve(nn.Module):
    def __init__(self, output_dim=3, hidden_dim=256):
        super(ModelCurve, self).__init__()
        self.output_dim = output_dim
        self.cnn_feature_dim = hidden_dim
        self.cnn = CNN(input_dim=1, out_dim=self.cnn_feature_dim)
        self.gru = nn.GRU(
            input_size = self.cnn_feature_dim, 
            hidden_size = self.cnn_feature_dim, 
            num_layers = 3,
            batch_first=True,
            dropout=0.2)
        self.mlp = MLP(input_dim=self.cnn_feature_dim+1, output_dim=2*self.output_dim)

    def forward(self, x, v0):
        # batch_size, timesteps, C, H, W = x.size()
        
        # x = x.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        
        # x = x.view(batch_size, timesteps, -1)
        # x, h_n, = self.gru(x)

        # x = F.leaky_relu(x[:, -1, :])
        x = self.mlp(x, v0)
        return x

    def predict(self, x, v0, t):
        t = t.squeeze(2)
        batch_size, timesteps, C, H, W = x.size()
        
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        
        x = x.view(batch_size, timesteps, -1)
        x, h_n, = self.gru(x)

        x = F.leaky_relu(x[:, -1, :])
        x = self.mlp(x, v0)
        out_xs = []
        out_ys = []

        for i in range(self.output_dim):
            out_x = torch.mul(x[:,2*i].unsqueeze(1), torch.pow(t, i+1))
            out_y = torch.mul(x[:,2*i+1].unsqueeze(1), torch.pow(t, i+1))
            out_xs.append(out_x)
            out_ys.append(out_y)
    
        out_xs = torch.stack(out_xs)
        out_ys = torch.stack(out_ys)
        xs = torch.sum(out_xs, dim=0)
        ys = torch.sum(out_ys, dim=0)
        return xs, ys

    def predict_vel(self, x, v0, t):
        t = t.squeeze(2)
        batch_size, timesteps, C, H, W = x.size()
        
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        
        x = x.view(batch_size, timesteps, -1)
        x, h_n, = self.gru(x)

        x = F.leaky_relu(x[:, -1, :])
        x = self.mlp(x, v0)

        out_xs = []
        out_ys = []
        for i in range(self.output_dim):
            out_x = torch.mul(x[:,2*i].unsqueeze(1), torch.pow(t, i+1))
            out_y = torch.mul(x[:,2*i+1].unsqueeze(1), torch.pow(t, i+1))
            out_xs.append(out_x)
            out_ys.append(out_y)
    
        out_xs = torch.stack(out_xs)
        out_ys = torch.stack(out_ys)
        xs = torch.sum(out_xs, dim=0)
        ys = torch.sum(out_ys, dim=0)

        out_vxs = []
        out_vys = []
        for i in range(self.output_dim):
            out_vx = torch.mul(x[:,2*i].unsqueeze(1), torch.pow(t, i))
            out_vy = torch.mul(x[:,2*i+1].unsqueeze(1), torch.pow(t, i))
            out_vxs.append(out_vx)
            out_vys.append(out_vy)

        out_vxs = torch.stack(out_vxs)
        out_vys = torch.stack(out_vys)
        vxs = torch.sum(out_vxs, dim=0)
        vys = torch.sum(out_vys, dim=0)

        return xs, ys, vxs, vys

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

class MLP_COS(nn.Module):
    def __init__(self, input_dim=257, rate=1.0, output=2):
        super(MLP_COS, self).__init__()
        self.rate = rate
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, output)
        
        self.apply(weights_init)
        
    def forward(self, x, t, v0):
        x = torch.cat([x, t], dim=1)
        x = torch.cat([x, v0], dim=1)
        x = self.linear1(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.linear4(x)
        #x = F.leaky_relu(x)
        x = torch.cos(self.rate*x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear5(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim=257, output_dim=2):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, output_dim)
        
        self.apply(weights_init)
        
    def forward(self, x, v0):
        x = torch.cat([x, v0], dim=1)
        x = self.linear1(x)
        x = F.leaky_relu(x, inplace=True)
        # x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        x = F.leaky_relu(x, inplace=True)
        # x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        x = F.leaky_relu(x, inplace=True)
        # x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear4(x)
        return x


class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """
    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features, out_features*num_gaussians)
        self.mu = nn.Linear(in_features, out_features*num_gaussians)

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu

class RNN_MDN(nn.Module):
    def __init__(self, hidden_dim=256, input_dim=1, k=3):
        super(RNN_MDN, self).__init__()
        self.cnn_feature_dim = hidden_dim
        self.rnn_hidden_dim = hidden_dim
        self.k = k
        self.cnn = CNN(input_dim=input_dim, out_dim=self.cnn_feature_dim)
        self.gru = nn.GRU(
            input_size = self.cnn_feature_dim, 
            hidden_size = self.rnn_hidden_dim, 
            num_layers = 3,
            batch_first=True,
            dropout=0.2
            )
        self.mdn = MDN(in_features=self.rnn_hidden_dim, out_features=4*10, num_gaussians=10)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        
        x = x.view(batch_size, timesteps, -1)
        x, _ = self.gru(x)
        
        x = torch.tanh(x[:, -1, :])

        pi, mu, sigma = self.mdn(x)
        return pi, mu, sigma
class Generator(nn.Module):
    def __init__(self, input_dim=8, output=2):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, output)
        
        self.apply(weights_init)
        
    def forward(self, x, t):
        x = torch.cat([x, t], dim=1)
        x = self.linear1(x)
        x = F.leaky_relu(x)
        #x = torch.tanh(x)
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        #x = torch.tanh(x)
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.linear3(x)
        x = F.leaky_relu(x)
        #x = torch.tanh(x)
        #x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.linear4(x)
        #x = F.leaky_relu(x)
        x = torch.cos(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear5(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, input_dim=8*6*2, output=1):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, output)
        
        self.apply(weights_init)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        #x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        #x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        x = F.leaky_relu(x)
        #x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear4(x)
        #x = torch.sigmoid(x)
        return x
    
class Generator2(nn.Module):
    def __init__(self, input_dim=8, output=2):
        super(Generator2, self).__init__()
        self.affine_dim = 64
        self.linear_t = nn.Linear(1, self.affine_dim)
        self.linear_v = nn.Linear(1, self.affine_dim)
        self.linear1 = nn.Linear(input_dim+2*self.affine_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, output)
        
        self.apply(weights_init)
        
    def forward(self, x, v, t):
        affine_v = self.linear_v(v)
        affine_t = self.linear_t(t)
        x = torch.cat([x, affine_v], dim=1)
        x = torch.cat([x, affine_t], dim=1)
        x = self.linear1(x)
        x = F.leaky_relu(x)
        #x = torch.tanh(x)
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        #x = torch.tanh(x)
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.linear3(x)
        x = F.leaky_relu(x)
        #x = torch.tanh(x)
        #x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.linear4(x)
        #x = F.leaky_relu(x)
        x = torch.cos(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear5(x)
        return x
    
class Discriminator2(nn.Module):
    def __init__(self, input_dim=8*6*2, output=1):
        super(Discriminator2, self).__init__()
        self.affine_dim = 64
        self.linear_v = nn.Linear(1, self.affine_dim)
        self.linear1 = nn.Linear(input_dim+self.affine_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, output)
        
        self.apply(weights_init)
        
    def forward(self, x, v):
        affine_v = self.linear_v(v)
        x = torch.cat([x, affine_v], dim=1)
        x = self.linear1(x)
        x = F.leaky_relu(x)
        #x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        #x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        x = F.leaky_relu(x)
        #x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear4(x)
        #x = torch.sigmoid(x)
        return x