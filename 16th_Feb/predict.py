import torch  
import torch.nn as nn  

# loading the model parameters
model_data = torch.load('model.pth')
fm = model_data['fm']
fs = model_data['fs']
tm = model_data['tm']
ts = model_data['ts']
parameters = model_data['parameters']

features = torch.tensor([
    [1500.0]
])

# standardizing the data
X = (features - fm)/fs

model = nn.Linear(1,1) # one input and one output
model.load_state_dict(parameters) # assigning optimal weights

prediction = model(X)

# unstandardizing the prediction
# by multiplying the prediction with the st deviation and adding the mean
prediction = prediction*ts + tm
print(prediction)