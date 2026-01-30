import torch
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')

X = torch.tensor(data.drop('Y', axis=1).to_numpy()).float() # droping the target column, axis = 1 meaning we drop columns
Y = torch.tensor(data['Y'].to_numpy()).float().reshape(-1,1) # selecting only the target column from the dataframe and reshaping it to be a column vector 

W = torch.tensor([
    [3.0],
    [2.0]])

# What ever the Y looks like the bias should be like
b = torch.tensor([
    [1.0]])

y_hat = X@W + b
r = y_hat - Y
SSE = r.T @ r
loss = SSE / len(Y) 

print(f"Loss: {loss}")