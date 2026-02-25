import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

data = pd.read_csv("data.csv") 
data["Diagnosis"] = data["Diagnosis"].map({"Benign":0, "Malignant":1}) # changing the string target as an binry integers

features = torch.tensor(data.drop("Diagnosis", axis = 1).to_numpy()).float()
target = torch.tensor(data["Diagnosis"].to_numpy()).float().reshape(-1,1)

fm = features.mean(axis = 0, keepdim = True).reshape(-1,1) # look up the difference of axis = 0 and axis = 1
fs = features.std(axis = 0,keepdim = True).reshape(-1,1)
# we have to keep the Y at 0 and 1 and therefore we don't need to standardize it

X = (features - fm)/fs
Y = target

model = nn.Linear(1,1) # we don't have to put the sigmoid fucntion here
criterian = nn.BCEWithLogitsLoss() # because it's included here
optimizer = optim.SGD(model.parameters(), lr = .1)

epochs = 250

for epoch in range(epochs):
    y_hat = model(X)
    loss = criterian(y_hat, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# saving optimal parameters
torch.save({
    "fm":fm,
    "fs":fs,
    "parameters": model.state_dict()
}, "model.pth")