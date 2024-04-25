import numpy as np
import pandas as pd
# one hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

class Adult:
    def __init__(self) -> None:
        
        path = "./uci_tests/adult/adult.data"
        
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'class'
        ]
        
        data = pd.read_csv(path, names=column_names, delimiter=',\s', na_values='?', engine='python')

        # Drop rows with missing values
        data.dropna(axis=0)

        # Separate features (X) and target variable (y)
        X = data.drop('class', axis=1)
        y = data['class']

        # Encode categorical columns in X
        categorical_cols = [
            'workclass', 'education', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'native-country'
        ]
        
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
        