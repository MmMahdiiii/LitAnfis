import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class FQ_regression(nn.Module):
    def __init__(self, in_features: int, rules: int, out_features: int, alpha=0.1, device=None, dtype=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.mean = Parameter(torch.rand((1, in_features, rules), **factory_kwargs))
        self.std = Parameter(torch.rand((1, in_features, rules), **factory_kwargs))
        self.tt = Parameter(torch.randn((1, in_features, rules), **factory_kwargs) * 0.1)
        self.linear = nn.Linear(in_features=rules, out_features=out_features, bias=True)
        self.eye = torch.eye(rules, requires_grad=False).to(device)
        self.alpha = alpha

    def forward(self, X: Tensor):
        y = X.unsqueeze(2)
        tt = torch.tanh(self.tt)
        y = y - self.mean
        y = y / self.std
        y = torch.exp(-torch.pow(y, 2))
        y = (y * tt) + 0.5 * (1 - tt)
        y = torch.prod(y, dim=1)
        ####
        cov = torch.abs(torch.mm(y.t(), y))
        regularization_term = (torch.sum(cov) - torch.sum(torch.diagonal(cov, 0))) * self.alpha
        ####
        # f_sum = torch.sum(y, dim=1)
        # f_sum = torch.where(f_sum == 0, f_sum, 1).unsqueeze(dim=1)
        # y = y / f_sum
        y = self.linear(y)
        return y, regularization_term
