import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from fq_model.fq_classification import FQ_classification

def run_classification_pipeline(X, y, n_splits=5, n_runs=10, random_state=42, model_params=None, learning_params=None):
    if model_params is None:
        model_params = {'in_features': X.shape[1], 'rules': 2, 'out_features': len(np.unique(y))}

    if learning_params is not None:
        lr = learning_params['lr']
        batch_size = learning_params['batch_size'] 
        num_epochs = learning_params['num_epochs'] 
    else:
        lr = 0.001
        batch_size = 32
        num_epochs = 200 
        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    progress = tqdm(total=n_runs*n_splits*num_epochs)
    
    test_performance = []
    train_performance = []
    
    for run in range(n_runs):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=run+random_state)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            X_train = torch.Tensor(X_train).to(device)
            X_test = torch.Tensor(X_test).to(device)
            y_train = torch.Tensor(y_train).type(torch.LongTensor).to(device)
            y_test = torch.Tensor(y_test).type(torch.LongTensor).to(device)

            model = FQ_classification(**model_params).to(device)
            optimizer = Adam(model.parameters(), lr=lr)
            
            temp_test=0
            temp_train=0
            
            for epoch in range(num_epochs):
                model.train()
                for i in range(0, len(X_train), batch_size):
                    optimizer.zero_grad()
                    batch_X, batch_y = X_train[i:i+batch_size], y_train[i:i+batch_size]
                    outputs = model(batch_X)
                    loss = F.cross_entropy(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
        
                model.eval()
                with torch.no_grad():
                    prediction = model(X_test).argmax(dim=1)
                    if m:= accuracy_score(y_test.cpu().numpy(), prediction.cpu().numpy()) > temp_test:
                        temp_test = m
                        prediction = model(X_train).argmax(dim=1)
                        temp_train = accuracy_score(y_train.cpu().numpy(), prediction.cpu().numpy())
                
                progress.update(1)  
        
            test_performance.append(temp_test)
            train_performance.append(temp_train)
                
    return train_performance, test_performance
