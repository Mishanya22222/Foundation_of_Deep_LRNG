import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np


# loading the data and defining the features
data = pd.read_csv("data.csv")
features = torch.tensor(data.drop("Salary ($1000s)", axis=1).to_numpy()).float() 
target = torch.tensor(data["Salary ($1000s)"].to_numpy()).float().reshape(-1,1) # because we want it to be as column

# standrdizing the data 
fm = features.mean(axis = 0, keepdim = True)
fs = features.std(axis = 0, keepdim = True)
tm = target.mean(axis = 0, keepdim = True)
ts = target.std(axis = 0, keepdim = True)

X = (features - fm)/fs
Y = (target - tm)/ts

# defying the model, criterai and the optimizer
model = nn.Linear(2,1)
criteria = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = .1)

# training the model
epochs = 150

for epoch in range(epochs):
    y_hat = model(X)
    loss = criteria(y_hat, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss)

# saving the weights
torch.save({
    "fm":fm,
    "fs":fs,
    "tm":tm,
    "ts":ts,
    "prarameters":model.state_dict()
}, "model_pth")