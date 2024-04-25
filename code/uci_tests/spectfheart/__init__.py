import numpy as np
import pandas as pd
# one hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

class Haberman:
    def __init__(self) -> None:
        path = "./uci_tests/spectfheart/spectfheart.csv"
        df = pd.read_csv(path, header=None)
        y = df[3].values
        X = df.drop([3], axis=1).values
        
        # one hot encoding
        enc = OneHotEncoder()
        enc.fit(y.reshape(-1, 1))
        y = enc.transform(y.reshape(-1, 1)).toarray()
        
        # standardization
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        
        self.X = X
        self.y = y
    
    def get_data(self):
        return self.X, self.y