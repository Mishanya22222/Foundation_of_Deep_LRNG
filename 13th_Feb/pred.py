import torch

Xm = torch.tensor([
    5.0
]) # mean of the original data

Xs = torch.tensor([
    4.0
])

Ym = torch.tensor([
    5.0
])

Ys = torch.tensor([
    3.0
])

# we have a raw input value of 7.0 not the normalized one
X_raw = torch.tensor([
    [7.0]
])

X = (X_raw - Xm) / Xs # normalize the input value

w = torch.tensor([
    [-.5]
])

b = torch.tensor([
    [0.0]
])

y_hat = X@w + b # make the prediction using the normalized input value
print(y_hat)

# to get the original value we need to unnormalize the prediction
# by multiplying by the standard deviation and adding the mean
# always standardize back before making the final prediction
print(y_hat * Ys + Ym)