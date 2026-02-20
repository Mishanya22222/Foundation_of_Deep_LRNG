import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

torch.manual_seed(1)

data = pd.read_csv("data.csv")

features = torch.tensor(data.drop("MPG", axis = 1).to_numpy()).float()
target = torch.tensor(data['MPG'].to_numpy()).float().reshape(-1,1)

fm = features.mean(axis = 0, keepdim = True) # axis 0 means we average the columns and not rows
fs = features.std(axis = 0, keepdim = True) # keepdim keeps the dimension 
tm = target.mean(axis = 0, keepdim = True)
ts = target.std(axis = 0, keepdim = True)

X = (features - fm)/ fs
Y = (target - tm)/ ts

print(Y)

model = nn.Linear(2,1) # because we have two features and one output
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = .1)

epochs = 1000

for epoch in range(epochs):
    y_hat = model(X) # pass through the nn which is y_hat = X @ w + b
    loss = criterion(y_hat, Y) # compute the loss, which is r, then SSE, and then MSE
    loss.backward() # compute the derivative
    optimizer.step() # make the step, updating the parameters
    optimizer.zero_grad() # zero out the gradient

# saving optimal parameters
torch.save({
    "fm":fm,
    "fs":fs,
    "tm":tm,
    "ts":ts,
    "parameters":model.state_dict()
}, "model.pth")