import numpy as np
import pandas as pd
# one hot encoding
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

class Seismic:
    def __init__(self) -> None:
        path = "./uci_tests/seismic/seismic-bumps.arff"

        column_names = [
            'seismic', 'seismoacoustic', 'shift', 'genergy', 'gpuls', 'gdenergy',
            'gdpuls', 'ghazard', 'nbumps', 'nbumps2', 'nbumps3', 'nbumps4', 'nbumps5',
            'nbumps6', 'nbumps7', 'nbumps89', 'energy', 'maxenergy', 'class'
        ]
        data = pd.read_csv(path, delimiter=',', skiprows=154, names=column_names)

        print(data.shape)

        # Encode categorical attributes
        le = LabelEncoder()
        for col in ['seismic', 'seismoacoustic', 'shift', 'ghazard']:
            data[col] = le.fit_transform(data[col])

        # Separate features (X) and target variable (y)
        X = data.drop('class', axis=1).values
        y = data['class'].values
        
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