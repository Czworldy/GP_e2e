#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import time
import numpy as np
import cv2
from PIL import Image

import torch
from torch.autograd import Variable
from torch.distributions import Categorical

def write_params(log_path, parser, description=None):
    opt = parser.parse_args()
    options = parser._optionals._actions
    with open(log_path+'params.md', 'w+') as file:
        file.write('# Params\n')
        file.write('********************************\n')
        file.write('Time: '+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n')
        if description is not None:
            file.write('**Description**: '+description+'\n')
        
        file.write('| Param | Value | Description |\n')
        file.write('| ----- | ----- | ----------- |\n')
        for i in range(len(parser._optionals._actions)):
            option = options[i]
            if option.dest != 'help':
                file.write('|**'+ option.dest+'**|'+str(opt.__dict__[option.dest])+'|'+option.help+'|\n')
        file.write('********************************\n\n')
        
def fig2data(fig):
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def gaussian_probability(sigma, mu, target):
    """Returns the probability of `data` given MoG parameters `sigma` and `mu`.
    
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        data (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    #data = torch.randn(32, 20, 7)
    data = target.unsqueeze(1)
    print(data.shape , mu.shape)
    ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((data - mu) / sigma)**2) / sigma
    return torch.prod(ret, 2)


def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target)
    nll = -torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)


def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
    for i, idx in enumerate(pis):
        sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
    return sample