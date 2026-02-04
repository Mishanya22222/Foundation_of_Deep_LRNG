import torch

X = torch.tensor([
    [1.0],
    [5.0],
    [8.0]
])

Y = torch.tensor([
    [3.0],
    [6.0],
    [1.0]
])

w = torch.tensor([
    [0.0]
], requires_grad=True)

b = torch.tensor([
    [0.0]
], requires_grad=True)

lr = 0.05 # learning rate
epochs = 100


# one pass through the network
y_hat = X@w + b
r = y_hat - Y
SSE = r.T @ r
loss = SSE / 3

# find the gradients for w and b
loss.backward() # compute the derivative

with torch.no_grad():
    w -= lr * w.grad # update w
    b -= lr * b.grad # update b
    
w.grad.zero_() # reset the gradient to zero
b.grad.zero_() # reset the gradient to zero

print(w,b)
print(loss)