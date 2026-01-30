import torch
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')

X = torch.tensor(data.drop('Y', axis=1).to_numpy()).float()
Y = torch.tensor(data['Y'].to_numpy()).float().reshape(-1,1)

w = torch.tensor([
    [-2.4],
    [-0.7],
    [2.6],
    [-0.5],
    [1.1]
])

b = torch.tensor([
    [-1.8]
])

y_hat = X@w + b
r = y_hat - Y
SSE = r.T @ r
loss = SSE / len(Y)
print(f"Loss: {loss}")

# calculus (back propogation) for optimizing the weights 