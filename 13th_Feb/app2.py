import torch

X_raw = torch.tensor([
    [1.0],
    [5.0],
    [9.0]
])

Y_raw = torch.tensor([
    [5.0],
    [8.0],
    [2.0]
])

# normalize the data, we do this to make the training faster and more stable instead os usin very small lernig rate
Xm = X_raw.mean() # mean of the original data
Xs = X_raw.std() # standard deviation of the original data

Ym = Y_raw.mean() # mean of the original data
Ys = Y_raw.std() # standard deviation of the original data

X = (X_raw- Xm) / Xs 
Y = (Y_raw- Ym) / Ys

w = torch.tensor([
    [0.0]
], requires_grad=True)

b = torch.tensor([
    [0.0]
], requires_grad=True)

epochs = 1000
lr = 0.01

for epoch in range(epochs):
    y_hat = X@w + b
    r = y_hat - Y
    SSE = r.T @ r
    loss = SSE / 3

    loss.backward() # compute the derivative

    with torch.no_grad():
        w -= lr * w.grad # update w
        b -= lr * b.grad # update b
    
    w.grad.zero_() # reset the gradient to zero
    b.grad.zero_()

    print(loss.item(), w, b)
