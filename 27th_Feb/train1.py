import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from export import export_model

data = pd.read_csv("data.csv")

X = torch.tensor(data.drop("y", axis = 1).values, dtype = torch.float32)
Y = torch.tensor(data['y'].values, dtype = torch.float32).reshape(-1, 1)

model = nn.Linear(2,1)
crtiterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1)

epochs = 250

for epoch in range(epochs):
    y_hat = model(X)
    loss = crtiterion(y_hat, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss)

export_model(model, "model.json")