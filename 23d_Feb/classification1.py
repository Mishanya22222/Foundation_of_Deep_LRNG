import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 

torch.manual_seed(42)

data = pd.read_csv("dataset1.csv")
data["Result"] = data["Result"].map({"Fail":0, "Pass":1}) # changing the string target as an binry integers

features = torch.tensor(data.drop("Result", axis=1).values, dtype=torch.float32)
target = torch.tensor(data['Result'].values, dtype= torch.float32).reshape(-1,1)

fm = features.mean(axis = 0, keepdim = True)
fs = features.std(axis = 0, keepdim = True)
tm = target.mean(axis = 0, keepdim = True)
ts = target.std(axis = 0, keepdim = True)

X = (features - fm) / fs
Y = target

model = nn.Linear(2,1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

epochs = 250
for epoch in range(epochs):
    y_hat = model(X)
    loss = criterion(y_hat, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

torch.save({
    "fm": fm,
    "fs": fs,
    "tm": tm,
    "ts": ts,
    "parameters": model.state_dict()
}, "model1.pth")
