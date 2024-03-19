import numpy as np
import pandas as pd
# one hot encoding
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Abalone:
    def __init__(self) -> None:
        path = "./regression/abalone/abalone.csv"
        df = pd.read_csv(path)
        df = df.dropna(axis=0)
        
        y = df['Rings'].values.reshape((-1, 1))
        X = df.drop(['Rings'], axis=1)
        
        sex = X['Sex']
        X = X.drop(['Sex'], axis=1)

        # one hot encoding
        enc = OneHotEncoder()
        sex = enc.fit_transform(sex.values.reshape(-1, 1)).toarray()
        X = np.concatenate((X.values, sex), axis=1)
        
        # standardization
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        
        scaler = MinMaxScaler()
        scaler.fit(y)
        y = scaler.transform(y)
        
        self.X = X
        self.y = y
    
    def get_data(self):
        return self.X, self.y