import torch

# make sure its a float and requires gradient computation
x = torch.tensor(3.0, requires_grad=True) # grad is the derivative
f = x**2 # function of y

f.backward() # compute the derivative
print(x.grad) # prints 6.0 which is the derivative of f at x=3
x.grad.zero_()  # reset the gradient to zero


f = x**2 # function of y

f.backward() # compute the derivative
print(x.grad) # prints 6.0 which is the derivative of f at x=3