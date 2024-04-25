import numpy as np
import pandas as pd
# one hot encoding
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class CalHousing:
    def __init__(self) -> None:
        path = "./regression/cal_house/housing.csv"
        df = pd.read_csv(path)
        df = df.dropna(axis=0)
        
        y = df['median_house_value'].values.reshape((-1, 1))
        X = df.drop(['median_house_value'], axis=1)
        
        ocean_p = X['ocean_proximity']
        X = X.drop(['ocean_proximity'], axis=1)
        # one hot encoding
        enc = OrdinalEncoder()
        ocean_p = enc.fit_transform(ocean_p.values.reshape(-1, 1))
        X = np.concatenate((X.values, ocean_p), axis=1)
        
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