# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

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

class CNN(nn.Module):
    def __init__(self, input_dim=1, out_dim=256):
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
        x = x.view(-1, self.out_dim)
        return x

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
    
class Generatorv2(nn.Module):
    def __init__(self, input_dim=8, output=2):
        super(Generatorv2, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, output)
        
        self.apply(weights_init)
        
    def forward(self, noise, t_with_v):
        x = torch.cat([noise, t_with_v], dim=1)
        x = self.linear1(x)
        x = F.leaky_relu(x, inplace=True)
        #x = torch.tanh(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        x = F.leaky_relu(x, inplace=True)
        #x = torch.tanh(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        x = F.leaky_relu(x, inplace=True)
        #x = torch.tanh(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.linear4(x)
        #x = F.leaky_relu(x)
        x = torch.cos(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear5(x)
        return x

class Generator(nn.Module):
    def __init__(self, input_dim=8, output=2):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, output)
        
        self.apply(weights_init)
        
    def forward(self, condition, noise, t):
        x = torch.cat([condition, noise], dim=1)
        # print("cat1:", x.shape)
        x = torch.cat([x, t], dim=1)
        x = self.linear1(x)
        x = F.leaky_relu(x, inplace=True)
        #x = torch.tanh(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        x = F.leaky_relu(x, inplace=True)
        #x = torch.tanh(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        x = F.leaky_relu(x, inplace=True)
        #x = torch.tanh(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        
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
        x = F.leaky_relu(x, inplace=True)
        #x = torch.tanh(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        x = F.leaky_relu(x, inplace=True)
        #x = torch.tanh(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        x = F.leaky_relu(x, inplace=True)
        #x = torch.tanh(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear4(x)
        # x = torch.sigmoid(x)
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


class Encoder(nn.Module):
    def __init__(self, input_dim=3, out_dim=256):
        super(Encoder, self).__init__()
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
        return x

class RNNEncoder(nn.Module):
    def __init__(self, input_dim=3, out_dim=64):
        super(RNNEncoder, self).__init__()
        self.cnn = Encoder(input_dim=input_dim, out_dim=out_dim)

        self.gru = nn.GRU(
            input_size = out_dim, 
            hidden_size = out_dim, 
            num_layers = 3,
            batch_first=True,
            dropout=0.2
            )

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        x = F.leaky_relu(x, inplace=True)
        
        x = x.view(batch_size, timesteps, -1)
        x, h_n, = self.gru(x)

        # x = F.leaky_relu(x[:, -1, :])
        x = x[:, -1, :]
        return x


class MobileNetV2(nn.Module):
    def __init__(self,num_classes: int, in_channels: int = 3,):
        super(MobileNetV2, self).__init__()

        self._model = torch.hub.load(
            github="pytorch/vision:v0.6.0",
            model="mobilenet_v2",
            num_classes=num_classes,
        )

        # HACK(filangel): enables non-RGB visual features.
        _tmp = self._model.features._modules['0']._modules['0']
        self._model.features._modules['0']._modules['0'] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=_tmp.out_channels,
            kernel_size=_tmp.kernel_size,
            stride=_tmp.stride,
            padding=_tmp.padding,
            bias=_tmp.bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass from the MobileNetV2."""
        return self._model(x)


class MobileNetV2WithV(nn.Module):
    def __init__(self,num_classes: int, in_channels: int = 3,):
        super(MobileNetV2WithV, self).__init__()

        self._model = torch.hub.load(
            github="pytorch/vision:v0.6.0",
            model="mobilenet_v2",
            num_classes=num_classes,
        )

        # HACK(filangel): enables non-RGB visual features.
        _tmp = self._model.features._modules['0']._modules['0']
        self._model.features._modules['0']._modules['0'] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=_tmp.out_channels,
            kernel_size=_tmp.kernel_size,
            stride=_tmp.stride,
            padding=_tmp.padding,
            bias=_tmp.bias,
        )

        self.linear1 = nn.Linear(num_classes + 1, 256)
        self.linear2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor, v) -> torch.Tensor:
        """Forward pass from the MobileNetV2."""
        x = self._model(x)
        x = F.leaky_relu(x, inplace=True)
        x = torch.cat([x, v], 1)
        x = self.linear1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear2(x)
        return x


class EncoderWithV(nn.Module):
    def __init__(self, input_dim=3, out_dim=256):
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
        
        self.apply(weights_init)

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
        x = F.leaky_relu(x, inplace=True)
        x = torch.cat([x, v], 1)
        x = self.linear1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.linear2(x)
        return x
    # def get_action(self, state, v):
    #     state  = torch.stack([state]).to(device)
    #     action = self.forward(state,v)
    #     return action.detach().cpu().numpy()[0]

class MLPNormal(nn.Module):
    def __init__(self, input_dim=257, output_dim=2):
        super(MLPNormal, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, output_dim)
        
        self.apply(weights_init)
        
    def forward(self, x):
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

class ModelCurve(nn.Module):
    def __init__(self, input_dim=3, output_dim=16):
        super(ModelCurve, self).__init__()
        self.output_dim = output_dim
        self.mlp = MLP(input_dim=input_dim, output_dim=2*self.output_dim)

    def forward(self, condition, latent, t):
        x = torch.cat([condition, latent], dim=1)
        x = self.mlp(x)

        t = t.squeeze(2)

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

    def predict(self, condition, latent, t):
        x = torch.cat([condition, latent], dim=1)
        x = self.mlp(x)

        t = t.squeeze(2)

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

    def predict_vel(self, condition, latent, t):
        x = torch.cat([condition, latent], dim=1)
        x = self.mlp(x)

        t = t.squeeze(2)

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



class DoubleCNN(nn.Module):
    def __init__(self, out_dim=256):
        super(DoubleCNN, self).__init__()
        self.img_encoder = CNN(input_dim=3, out_dim=256)
        self.nav_encoder = CNN(input_dim=3, out_dim=256)
        self.mlp = MLPNormal(input_dim = 256*2, output_dim=out_dim)

    def forward(self, x):
        img_feature = self.img_encoder(x[:,:3,...])
        nav_feature = self.nav_encoder(x[:,3:,...])
        input_feature = torch.cat([img_feature, nav_feature], 1)
        # print(img_feature.shape, input_feature.shape)
        feature = self.mlp(input_feature)
        return feature

class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

        print('* MixStyle params')
        print(f'- p: {p}')
        print(f'- alpha: {alpha}')

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.randperm(B)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix




    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

        print('* MixStyle params')
        print(f'- p: {p}')
        print(f'- alpha: {alpha}')

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.randperm(B)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix


class MixStyleEncoder(nn.Module):
    def __init__(self, input_dim=3, out_dim=256):
        super(MixStyleEncoder, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, self.out_dim, 3, stride=2, padding=1)
        self.mixstyle = MixStyle(p=0.5, alpha=0.1)

        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.mixstyle(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.mixstyle(x)
        x = F.max_pool2d(x, 2, 2)         
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.mixstyle(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = F.leaky_relu(x)
        x = x.view(-1, self.out_dim)
        return x