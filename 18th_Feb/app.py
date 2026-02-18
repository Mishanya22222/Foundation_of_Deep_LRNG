import pandas as pd  
import numpy as np 
import torch  
import torch.nn as nn  
import torch.optim as optim  

data = pd.read_csv('data.csv')

features = torch.tensor(data.drop('Exam Score (%)',axis = 1).to_numpy()).float()
target = torch.tensor(data['Exam Score (%)'].to_numpy()).float().reshape(-1,1)

fm = features.mea().reshape(-1,1)
fs = features.std().reshape(-1,1)
tm = target.mean().reshape(-1,1)
ts = target.std().reshape(-1,1)