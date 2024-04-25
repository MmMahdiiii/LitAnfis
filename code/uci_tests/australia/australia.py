import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class Australia:
    def __init__(self) -> None:
        
        file_path = './uci_tests/australia/australian.dat'  # Replace with your file path
        column_names = [
            'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
            'A11', 'A12', 'A13', 'A14', 'A15'
        ]
        data = pd.read_csv(file_path, header=None, delimiter=' ', names=column_names)
        data = data.dropna(axis=0)

        # Separate features (X) and target variable (y)
        X = data.drop('A15', axis=1)  # Features
        y = data['A15']

        # One-hot encode categorical columns in X
        categorical_cols = ['A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12']
        cat = X[categorical_cols].values
        num = X.drop(categorical_cols, axis=1).values
        encoder = OneHotEncoder()
        X = np.concatenate([encoder.fit_transform(cat).toarray(), num], axis=1)

        # Convert y to numpy array
        encoder = OneHotEncoder()
        y = encoder.fit_transform(y.values.reshape(-1, 1)).toarray()
        
        # standardization
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        self.X = X
        self.y = y

    def get_data(self):
        return self.X, self.y
