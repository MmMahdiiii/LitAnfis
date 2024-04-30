import torch
import numpy as np
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class FQ(nn.Module):
    def __init__(self, in_features: int, rules: int, out_features: int, task='classification', fuzzy_model='mamdani', device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.mean = nn.Parameter(torch.rand((1, in_features, rules), **factory_kwargs))
        self.std = nn.Parameter(torch.rand((1, in_features, rules), **factory_kwargs))
        self.tt = nn.Parameter(torch.randn((1, in_features, rules), **factory_kwargs) * 0.1)
        self.decoder_linear = nn.Linear(in_features=rules, out_features=in_features)  # Decoder
        if fuzzy_model == 'tsk':
            self.output_linear = nn.Linear(in_features=rules * (in_features + 1), out_features=out_features)
        elif fuzzy_model == 'mamdani':
            self.output_linear = nn.Linear(in_features=rules, out_features=out_features)
            
        self.task = task
        self.fuzzy_model = fuzzy_model

    def forward(self, X, phase='final'):
        y = self.encode(X)
        if phase == 'pretrain':
            reconstructed_X = self.decoder_linear(y)
            return reconstructed_X
        elif self.fuzzy_model == 'tsk':
            y = self.tsk(X, y)
        elif self.fuzzy_model == 'mamdani':
            y = self.mamdani(y)

        if self.task == 'classification':
            y = torch.softmax(y, dim=1)
        
        return y

    def encode(self, X):
        y = X.unsqueeze(2) - self.mean
        y = y / self.std
        y = torch.exp(-y**2)
        tt = torch.tanh(self.tt)
        y = (y * tt) + 0.5 * (1 - tt)
        y = torch.prod(y, dim=1)
        f_sum = torch.sum(y, dim=1).unsqueeze(dim=1)
        y = y / f_sum
        return y
    
    def mamdani(self, y):
        y = self.output_linear(y)
        return y
    
    def tsk(self, X, y):
        y = y.unsqueeze(1)
        X = torch.concatenate([X, torch.ones((X.shape[0], 1)).to(y.device)], dim=1)
        X = X.unsqueeze(2)
        y = X * y
        y = torch.flatten(y, start_dim=1)
        y = self.output_linear(y)
        return y