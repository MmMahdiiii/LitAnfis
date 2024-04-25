import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter

class FQ_classification(nn.Module):
    def __init__(self, in_features: int, rules: int, out_features: int, device=None, dtype=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.mean = Parameter(torch.rand((1, in_features, rules), **factory_kwargs))
        self.std = Parameter(torch.rand((1, in_features, rules), **factory_kwargs))
        self.tt = Parameter(torch.randn((1, in_features, rules), **factory_kwargs) * 0.1)
        self.linear = nn.Linear(in_features=rules, out_features=out_features)

    def forward(self, X: Tensor):
        y = X.unsqueeze(2)
        tt = torch.tanh(self.tt)
        y = y - self.mean
        y = y / self.std
        y = torch.exp(-y**2)
        y = (y * tt) + 0.5 * (1 - tt)
        y = torch.prod(y, dim=1)
        f_sum = torch.sum(y, dim=1).unsqueeze(dim=1)
        y = y / f_sum
        y = y.unsqueeze(1)
        X = torch.concatenate([X, torch.ones((X.shape[0], 1)).to(y.device)], dim=1)
        X = X.unsqueeze(2)
        y = X * y
        y = torch.flatten(y, start_dim=1)
        y = self.linear(y)
        y = torch.softmax(y, dim=1)
        return y