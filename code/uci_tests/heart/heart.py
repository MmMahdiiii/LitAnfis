import numpy as np
import pandas as pd
# one hot encoding
from sklearn.preprocessing import StandardScaler

class Heart:
    def __init__(self) -> None:
        path = "./uci_tests/heart/processed.cleveland.data"
        df = pd.read_csv(path, header=None)
        df.replace('?', pd.NA, inplace=True)
        df = df.dropna(axis=0)

        y = df[13].values
        X = df.drop([13], axis=1).values
        
        y[y > 0] = 1
        
        # standardization
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        
        self.X = X
        self.y = y
    
    def get_data(self):
        return self.X, self.y