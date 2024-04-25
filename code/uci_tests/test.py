import torch.optim as optim
import torch
from uci_tests.iris.iris import Iris
from fq import FQ
import torch.nn.functional as F 
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split


def test():
    X, y = Iris().get_data()
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=5)
    model = FQ(in_features=X.shape[1], rules=5, out_featues=y.shape[1])
    
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)
    
    # Choose an optimizer, for example, SGD
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    
    num_epochs = 20
    
    print(X_train.shape)
    print(y_train.shape)
    
    # Training loop
    for epoch in range(num_epochs):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        
        # Compute loss
        loss = F.cross_entropy(outputs, y_train)
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        prediction = model(X_test)
        print(f'epoch {epoch}, test_acc: ', accuracy_score(y_test, prediction.argmax(dim=1)))