import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score  # or any other metric
import pandas as pd


class LitAnfis(nn.Module):
    def __init__(self, in_features: int, rules: int, out_features: int, binary: bool, drop_out_p=0.5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.rules_count = rules
        self.in_features = in_features
        self.out_features = out_features

        self.binary = binary

        if binary:
            self.out_features = out_features = 1

        self.drop_out_p = drop_out_p

        self.device = device

        self.mean = nn.Parameter(torch.rand(
            (in_features, rules), **factory_kwargs))
        self.std = nn.Parameter(torch.rand(
            (in_features, rules), **factory_kwargs))
        self.literal = nn.Parameter(torch.randn(
            (in_features, rules), **factory_kwargs) * 0.1)

        self.tsk_linear = nn.Linear(
            in_features=in_features, out_features=rules * out_features, bias=True, **factory_kwargs)
        self.decoder_linear = nn.Linear(
            in_features=rules, out_features=in_features, bias=True, **factory_kwargs)

        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(p=drop_out_p)


    def forward(self, X):
        y = self.encode(X)
        entropy = - y * torch.log(y)

        if self.rules_count > 1:
            y = F.normalize(y, p=1, dim=1)  # Adjust the dim based on your needs

        y = self.drop_out(y)

        reconstructed_X = self.decoder_linear(y)

        y = self.tsk(X, y)

        return y, reconstructed_X, entropy

    def encode(self, X):
        mean = self.mean.view(1, *self.mean.shape)
        std = self.std.view(1, *self.std.shape)

        X = X.view(*X.shape, 1)

        def gaussmf(x, mu, sigma):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        y = gaussmf(X, mean, std)

        literal = self.sigmoid(self.literal)
        y = (y * literal) + (1 - y) * (1 - literal)

        epsilon = 1e-10  # A small value to prevent log(0)
        y = torch.log(y + epsilon)

        max_log_y= torch.max(y, dim=1, keepdim=True)[0]

        y = torch.sum(y - max_log_y, dim=1)  # Subtract max for numerical stability

        y = torch.exp(y) * torch.exp(max_log_y.squeeze(dim=1))
        
        return y

    def tsk(self, X, y):
        X = self.tsk_linear(X)

        X = X.reshape(-1, self.rules_count, self.out_features)
        y = y.reshape(-1, self.rules_count, 1)
        X = X * y

        return X.sum(dim=1)


class MamdaniLitAnfis(LitAnfis):
    def __init__(self, in_features: int, rules: int, out_features: int, binary: bool, drop_out_p=0.5, device=None, dtype=None):
        super().__init__(in_features, rules, out_features, binary, drop_out_p, device, dtype)
    
        factory_kwargs = {'device': device, 'dtype': dtype}

        if binary:
            self.out_features = out_features = 1

        self.mamdani_linear = nn.Linear(
            in_features=rules, out_features=out_features, bias=True, **factory_kwargs)

    def mamdani(self, y):
        return self.mamdani_linear(y)
    
    def forward(self, X):
        y = self.encode(X)
        entropy = - y * torch.log(y)

        if self.rules_count > 1:
            y = F.normalize(y, p=1, dim=1)  # Adjust the dim based on your needs

        reconstructed_X = self.decoder_linear(y)

        y = self.mamdani(y)

        return y, reconstructed_X, entropy



class SklearnLitAnfisWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, device=None, dtype=torch.float32):
        self.device = device if device else 'cpu'
        self.dtype = dtype
        # Initialize the model
        self.model = model.to(self.device)

    def fit(self, X, y):
        return self

    def predict(self, X):
        self._check_is_filiteraled()
        X = self._convert_to_tensor(X)

        # Use the model to get predictions
        with torch.no_grad():
            y_pred= self.model(X)[0]

        if self.model.binary:
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.cpu().numpy() > 0.5
        else:
            y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred.argmax(dim=1).cpu().numpy()
        return y_pred

    def predict_proba(self, X):
        self._check_is_filiteraled()
        X = self._convert_to_tensor(X)

        with torch.no_grad():
            y_pred = self.model(X)[0]
        
        if self.model.binary:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = torch.softmax(y_pred, dim=1)
    
        return y_pred.cpu().numpy()

    def score(self, X, y):
        y_pred = self.predict(X)[0]
        return accuracy_score(y, y_pred)

    def _convert_to_tensor(self, data):
        """ Helper function to convert numpy arrays to torch tensors and move to the correct device. """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)

        elif isinstance(data, torch.Tensor):
            data = data.to(self.device)

        elif isinstance(data, pd.DataFrame):
            data = torch.tensor(
                data.values, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(
                "Input data must be a NumPy array or a PyTorch tensor.")
        return data

    def _check_is_filiteraled(self):
        pass

    def get_params(self, deep=True):
        return {
            'model': self.model
        }

    def set_params(self, **parameters):
        return self
