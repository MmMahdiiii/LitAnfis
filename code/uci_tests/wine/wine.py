import numpy as np
import pandas as pd
# one hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

class Wine:
    def __init__(self) -> None:
        path = "./uci_tests/wine/wine.data"
        df = pd.read_csv(path, header=None)
        y = df[0].values - 1
        X = df.drop([0], axis=1).values
        
        # standardization
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        
        self.X = X
        self.y = y
    
    def get_data(self):
        return self.X, self.y