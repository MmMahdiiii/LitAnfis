import torch
import numpy as np
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class FQ(nn.Module):
    def __init__(self, in_features: int, rules: int, out_features: int, mid_layer_dim=None,  task='classification', fuzzy_model='mamdani', device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.cor = False
        self.rules_count = rules
        self.device = device
        self.mean = nn.Parameter(torch.rand((1, in_features, rules), **factory_kwargs))
        
        
        if mid_layer_dim == None:
            self.std = nn.Parameter(torch.rand((1, in_features, rules), **factory_kwargs))
            self.tt = nn.Parameter(torch.randn((1, in_features, rules), **factory_kwargs) * 0.1)
            if fuzzy_model == 'tsk':
                self.output_linear = nn.Linear(in_features=rules * (in_features + 1), out_features=out_features)

        else: 
            self.cor = True
            self.transition = nn.Parameter(torch.randn((rules, in_features, mid_layer_dim), **factory_kwargs) * 0.1)
            self.tt = nn.Parameter(torch.randn((1, mid_layer_dim, rules), **factory_kwargs) * 0.1)
            
            if fuzzy_model == 'tsk':
                self.output_linear = nn.Linear(in_features=rules * (mid_layer_dim + 1), out_features=out_features)
        
        if fuzzy_model == 'mamdani':
            self.output_linear = nn.Linear(in_features=rules, out_features=out_features)
        
        
        self.decoder_linear = nn.Linear(in_features=rules, out_features=in_features)  # Decoder
        
        
        self.task = task
        self.fuzzy_model = fuzzy_model

    def forward(self, X, phase='final'):
        z, y = self.encode(X)
        
        if self.cor:
            pow2 = torch.sqrt(torch.pow(self.transition, 2).sum(dim=1))
            hat = self.transition / pow2.unsqueeze(dim=1)
            cov = hat.permute([0, 2, 1]) @ hat
            # mask = torch.eye(cov.shape[1], cov.shape[1], dtype=torch.bool).unsqueeze(0).expand(*cov.shape)
            # mask = mask.to(self.device)
            # cov.masked_fill_(mask, 0)
            loss = cov.sum().item()
        
        if phase == 'pretrain':
            reconstructed_X = self.decoder_linear(y)
            if self.cor: 
                return reconstructed_X, loss
            return reconstructed_X
        elif self.fuzzy_model == 'tsk':
            y = self.tsk(z, y)
        elif self.fuzzy_model == 'mamdani':
            y = self.mamdani(y)

        if self.task == 'classification':
            y = torch.softmax(y, dim=1)
        
        if self.cor:
            return y, loss
        return y

    def encode(self, X):
        y = X.unsqueeze(2) - self.mean
        z = X
        if self.cor:
            y = y.permute(2, 0, 1)
            y = y @ self.transition
            y = y.permute(1, 2, 0)
            z = y
        else:    
            y = y / self.std
            
        y = torch.exp(-y**2)
        tt = torch.tanh(self.tt)
        y = (y * tt) + 0.5 * (1 - tt)
        y = torch.prod(y, dim=1)
        f_sum = torch.sum(y, dim=1).unsqueeze(dim=1)
        y = y / f_sum
        return z, y
    
    def mamdani(self, y):
        y = self.output_linear(y)
        return y
    
    def tsk(self, X, y):
        y = y.unsqueeze(1)
        if self.cor:
            X = torch.concatenate([X, torch.ones((X.shape[0], 1, self.rules_count)).to(y.device)], dim=1)
        else:
            X = torch.concatenate([X, torch.ones((X.shape[0], 1)).to(y.device)], dim=1)
            X = X.unsqueeze(2)
        y = X * y
        y = torch.flatten(y, start_dim=1)
        y = self.output_linear(y)
        return y
    