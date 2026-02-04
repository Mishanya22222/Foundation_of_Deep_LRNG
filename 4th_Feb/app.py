import torch

X = torch.tensor([
    [3.0]
])

Y = torch.tensor([
    [10.0]
])

w = torch.tensor([
    [6.0]
], requires_grad=True) # grad is the derivative and it changes

b = torch.tensor([
    [1.0]
], requires_grad=True) # grad is the derivative and it changes

lr = .2 # learning rate
# one pass through the network
y_hat = X@w + b
r = y_hat - Y
SSE = r.T @ r
loss = SSE / 1

# find the gradients for w and b
loss.backward() # compute the derivative

# updating weights and biases
with torch.no_grad():
    w -= lr * w.grad # update w
    b -= lr * b.grad # update b

w.grad.zero_()  # reset the gradient to zero
b.grad.zero_()  # reset the gradient to zero

print(w,b)