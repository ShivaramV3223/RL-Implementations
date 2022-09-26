import gym
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Vanilla_Policy_Gradient(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.input_size = model_params['input_size']
        self.output_size = model_params['output_size']
        self.net_params = model_params['net_params']
        self.layers = nn.ModuleList()
        current_dim = self.input_size
        for hdim in self.net_params:
            self.layers.append(nn.Linear(current_dim, hdim))
            self.layers.append(nn.ReLU())
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, self.output_size))
        self.layers.append(nn.Linear(current_dim, self.output_size))
        
    def forward(self, x):
        for layers in self.layers[:-2]:
            x = layers(x)
        mean = self.layers[-2](x)
        #print("mean: ", mean)
        log_std = self.layers[-1](x)
        std = torch.exp(log_std)
        #print("std: ",std)
        distribution = torch.distributions.MultivariateNormal(mean, torch.diag(std))
        return distribution
        
class Value_function(nn.Module):
    def __init__(self, value_params):
        super().__init__()
        self.input_size = value_params['input_size']
        self.output_size = value_params['output_size']
        self.net_params = value_params['net_params']
        self.layers = nn.ModuleList()
        current_dim = self.input_size
        for hdim in self.net_params:
            self.layers.append(nn.Linear(current_dim, hdim))
            self.layers.append(nn.ReLU())
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, self.output_size))
                           
    def forward(self, x):
        for layers in self.layers:
            x = layers(x)
        return x