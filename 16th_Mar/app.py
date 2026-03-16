import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from export import export_model # getting the function
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)

data = pd.read_csv("data.csv")

X = torch.tensor(data.drop("y", axis = 1).values, dtype = torch.float32)
Y = torch.tensor(data['y'].values, dtype = torch.float32).reshape(-1,1)

class MyDataset(Dataset):
    def __init__(self, X, Y ):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


Dataset = MyDataset(X,Y)

loader = DataLoader(
    Dataset, batch_size = 10 # we can now update the weights based on the batch snd when we go through eveything this is only one epoch
)

model = nn.Linear(2,1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100

for eppoch in range(epochs):
    for X,Y in loader:
        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y_hat, Y)
        loss.backward()
        optimizer.step()

    print(loss)

# conolution pooling, hidden layer and binary classification