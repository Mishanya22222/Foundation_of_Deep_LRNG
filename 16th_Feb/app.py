import pandas as pd  
import numpy as np 
import torch  
import torch.nn as nn  
import torch.optim as optim  
# what are all this imports

data = pd.read_csv('data.csv')

# taking the first column out of the data and converting it to a tensor
features = torch.tensor(data.drop('Price',axis = 1).to_numpy()).float()
target = torch.tensor(data['Price'].to_numpy()).float().reshape(-1,1)

# finding the mean and standard deviation of the features and target, we will use this to normalize the data
fm = features.mean().reshape(-1,1) # reshape for 2 dimensional tensor
fs = features.std().reshape(-1,1)
tm = target.mean().reshape(-1,1)
ts = target.std().reshape(-1,1)

# standardize the data, we do this to make the training faster and more stable instead of using very small learning rate
X = (features - fm)/fs
Y = (target - tm)/ts

model = nn.Linear(1,1) # what does this do? it creates a linear model with 1 input and 1 output, it will learn the parameters of the model during training
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = .1)

# training the model
epochs = 100
for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat,Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# saving the model parameters and the mean and standard deviation of the features and target, we will use this to normalize the data during inference
torch.save({
    'fm':fm,
    'fs':fs,
    'tm':tm,
    'ts':ts,
    'parameters':model.state_dict()
},'model.pth')
