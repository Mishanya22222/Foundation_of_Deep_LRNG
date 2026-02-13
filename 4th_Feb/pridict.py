import torch

X = torch.tensor([
    [7.0]
])

w = torch.tensor([
    [-.375]
])

b = torch.tensor([
    [6.875]
])

prediction = X@w + b
print(prediction)