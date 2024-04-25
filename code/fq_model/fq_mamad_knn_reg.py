import torch
from torch import Tensor
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np

class FQ_regression(nn.Module):
    def __init__(self, in_features: int, rules: int, out_features: int, device=None, dtype=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.factory_kwargs = factory_kwargs = {'device': device, 'dtype': dtype}
        self.rules = rules
        self.in_features = in_features
        # self.mean = Parameter(torch.rand((1, in_features, rules), **factory_kwargs))
        # self.std = Parameter(torch.rand((1, in_features, rules), **factory_kwargs))
        self.tt = Parameter(torch.randn((1, in_features, rules), **factory_kwargs) * 0.1)
        self.linear = nn.Linear(in_features=rules, out_features=out_features, bias=True)

    def _rule_initialization(self, X, y):
        X = np.array(X)
        y = np.array(y)
        Z = np.concatenate([X, y], axis=1)
        nbrs = NearestNeighbors(n_neighbors=Z.shape[0] // self.rules, algorithm='ball_tree', n_jobs=-1).fit(Z)
        distances, indices = nbrs.kneighbors(Z)
        ek = np.mean(distances, axis=1)
        
        mean = torch.zeros((1, self.in_features, self.rules), **self.factory_kwargs)
        std = torch.zeros((1, self.in_features, self.rules), **self.factory_kwargs)

        for i in range(self.rules):
            vi = np.argmin(ek)
            knn = indices[vi]
            mean[0, :, i] = torch.from_numpy(X[vi, :])
            std[0, :, i] = torch.from_numpy(np.std(X[knn, :], axis=0))
            ek[knn] = np.inf
        
        self.mean = Parameter(mean)
        self.std = Parameter(std)
        print(mean)
        print(std)

    def forward(self, X: Tensor):
        y = X.unsqueeze(2)
        tt = torch.tanh(self.tt)
        y = y - self.mean
        y = y / self.std
        y = torch.exp(-torch.pow(y, 2))
        y = (y * tt) + 0.5 * (1 - tt)
        y = torch.prod(y, dim=1)
        f_sum = torch.sum(y, dim=1).unsqueeze(dim=1)
        y = y / f_sum
        y = self.linear(y)
        return y