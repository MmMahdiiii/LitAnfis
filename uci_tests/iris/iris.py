from sklearn import datasets
import numpy as np
import pandas as pd
# one hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

class Iris:
    def __init__(self) -> None:
        df = datasets.load_iris()
        y = df.target.reshape((-1, ))
        X = df.data
        
        # one hot encoding
        # enc = OneHotEncoder()
        # enc.fit(y.reshape(-1, 1))
        # y = enc.transform(y.reshape(-1, 1)).toarray()
        
        # standardization
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        
        self.X = X
        self.y = y
    
    def get_data(self):
        return self.X, self.y