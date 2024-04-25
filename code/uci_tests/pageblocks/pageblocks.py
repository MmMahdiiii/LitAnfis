import numpy as np
import pandas as pd
# one hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

class PageBlocks:
    def __init__(self) -> None:
        path = "./uci_tests/pageblocks/page-blocks.data"
        
        column_names = [
            'height', 'length', 'area', 'eccen', 'p_black', 'p_and', 'mean_tr',
            'blackpix', 'blackand', 'wb_trans', 'class'
        ]
        data = pd.read_csv(path, delimiter='\s+', names=column_names)

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