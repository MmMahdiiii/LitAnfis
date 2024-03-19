import numpy as np 
import pandas as pd
# one hot encoding
from sklearn.preprocessing import OneHotEncoder

class Musk:
    def __init__(self) -> None:
        path = "./uci_tests/musk/musk.data"
        df = pd.read_csv(path, header=None)
        df = pd.read_csv('./clean2.data', header=None)
        df = df.drop([0, 1], axis=1, inplace=True)
        y = df[167].values
        X = df.drop([167], axis=1).values
        # one hot encoding
        enc = OneHotEncoder()
        enc.fit(y.reshape(-1, 1))
        y = enc.transform(y.reshape(-1, 1)).toarray()
        self.X = X
        self.y = y
    
    def get_data(self):
        return self.X, self.y