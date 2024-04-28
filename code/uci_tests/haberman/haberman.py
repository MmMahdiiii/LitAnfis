import numpy as np
import pandas as pd
# one hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

class Haberman:
    def __init__(self) -> None:
        path = "./uci_tests/haberman/haberman.data"
        df = pd.read_csv(path, header=None)
        y = df[3].values
        X = df.drop([3], axis=1).values
        
        # standardization
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        
        self.X = X
        self.y = y - 1
    
    def get_data(self):
        return self.X, self.y