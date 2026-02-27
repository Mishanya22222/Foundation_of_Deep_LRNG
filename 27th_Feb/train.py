import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from export import export_model

df = pd.read_csv("data.csv")

X = torch.tensor(df.drop("y", axis = 1).values).float() # instead of to_numpy we can also use values, and the difference is that values are list
Y = torch.tensor(df["y"].values).float().reshape(-1,1)

model = nn.Linear(2,1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1)

epochs = 1000

for epoch in range(epochs):
    y_hat = model(X)
    loss = criterion(y_hat,Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss)

export_model(model, "model.json") # name of the model ane the name we want to give 