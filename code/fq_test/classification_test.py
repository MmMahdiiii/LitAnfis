from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fq_model.fq_classification import FQ_classification as FQ_mamdani
from fq_model.fq_tsk_classification import FQ_classification as FQ_tsk

def run_classification_pipeline(X, y,
                                test_size=0.2,
                                n_runs=10,
                                random_state=42,
                                min_acc=0.94,
                                model='mamdani',
                                model_params=None,
                                learning_params=None):
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
    
    progress = tqdm(total=n_runs)
    
    test_performance = []
    train_performance = []
    
    run = n_runs 
    # for run in range(n_runs):
    while run > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state+run)
    

        X_train = torch.Tensor(X_train).to(device)
        X_test = torch.Tensor(X_test).to(device)
        y_train = torch.Tensor(y_train).type(torch.LongTensor).to(device)
        y_test = torch.Tensor(y_test).type(torch.LongTensor).to(device)

        if model == 'mamdani':
            model = FQ_mamdani(**model_params).to(device)
        else:
            model = FQ_tsk(**model_params).to(device)
            
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
            temp_test = accuracy_score(y_test.cpu().numpy(), prediction.cpu().numpy())
            prediction = model(X_train).argmax(dim=1)
            temp_train = accuracy_score(y_train.cpu().numpy(), prediction.cpu().numpy())
        
        if temp_train > min_acc:
            test_performance.append(temp_test)
            train_performance.append(temp_train)
            run -= 1
            progress.update(1)  
                
    return train_performance, test_performance
