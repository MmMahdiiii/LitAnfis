import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class Cryotheraphy:
    def __init__(self) -> None:
        
        file_path = './uci_tests/cryotherapy/Cryotherapy.xlsx'
        data = pd.read_excel(file_path)
        data = data.dropna(axis=0)

        y = data['Result_of_Treatment'].values
        X = data.drop('Result_of_Treatment', axis=1).values

        # Convert y to numpy array
        # encoder = OneHotEncoder()
        # y = encoder.fit_transform(y.values.reshape(-1, 1)).toarray()
        
        # standardization
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        self.X = X
        self.y = y

    def get_data(self):
        return self.X, self.y
