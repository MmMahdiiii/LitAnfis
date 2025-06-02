import seaborn as sns
import os
from sklearn.metrics import classification_report
from train.early_stop import EarlyStopping
from torch.utils.data import DataLoader
from tests.uci_test import Wine
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim
import torch
from model.litanfis import LitAnfis, SklearnLitAnfisWrapper
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


# config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_size = 0.7
batch_size = 20
random_state = 91992

lr = 0.0001
max_lr = 0.01
epochs = 1000

alpha = 5.0
min_alpha = 0.01


wine = Wine(train_size=train_size, session_id=random_state)
train_dataset = wine.train_dataset(device)
test_dataset = wine.test_dataset(device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


model_params = {
    'in_features': wine.train_numpy()[0].shape[1],
    'out_features': len(np.unique(wine.train_numpy()[1])),
    'rules': 3,
    'drop_out_p': 0.3
}

model = LitAnfis(**model_params, dtype=torch.float32)
sk_model = SklearnLitAnfisWrapper(model, device=device)


optimizer = optim.Adam(model.parameters(), lr=lr)

steps_per_epoch = len(train_loader)
scheduler = OneCycleLR(optimizer, max_lr=0.01,
                       steps_per_epoch=steps_per_epoch, epochs=epochs)

# Early stopping
early_stopping = EarlyStopping(patience=5, delta=0.01)

alpha_decaying = np.power(min_alpha / alpha, steps_per_epoch * epochs)

cross = nn.CrossEntropyLoss()
cos = nn.CosineSimilarity(dim=1)


def criterion(batch_X, batch_y, outputs, reconstructed, alpha):
    return cross(outputs, batch_y.long()) - cos(reconstructed, batch_X).mean() * alpha


for epoch in range(epochs):

    model.train()
    train_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        outputs, reconstructed = model(batch_X)

        loss = criterion(batch_X, batch_y, outputs, reconstructed, alpha)

        loss.backward()
        optimizer.step()
    
        alpha *= alpha_decaying

        train_loss += loss.item() * batch_X.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            output, _ = model(data)
            loss = cross(output, target.long())
            val_loss += loss.item() * data.size(0)

    val_loss /= len(test_loader.dataset)

    print(
        f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    plt.plot(train_loss)
    plt.plot(val_loss)


print('train')
print(classification_report(wine.train_numpy()[
      1], sk_model.predict(wine.train_numpy()[0])))
print('test')
print(classification_report(wine.test_numpy()[
      1], sk_model.predict(wine.test_numpy()[0])))
