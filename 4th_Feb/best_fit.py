import torch

X = torch.tensor([
    [1.0],
    [5.0],
    [9.0]
])

Y = torch.tensor([
    [5.0],
    [8.0],
    [2.0]
])

w = torch.tensor([
    [0.0]
], requires_grad=True)

b = torch.tensor([
    [0.0]
], requires_grad=True)

lr = 0.01 # learning rate
epochs = 5000

for epoch in range(epochs):
    # one pass through the network
    y_hat = X@w + b
    r = y_hat - Y
    SSE = r.T @ r
    loss = SSE / 3

    loss.backward() # compute the derivative

    with torch.no_grad():
        w -= lr * w.grad # update w
        b -= lr * b.grad # update b
    
    print(loss.item(), w,b)
    w.grad.zero_() # reset the gradient to zero
    b.grad.zero_() # reset the gradient to zero

print(7.0*w+b)