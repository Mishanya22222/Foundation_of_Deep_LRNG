import torch
'''
x = torch.tensor(-1.0, requires_grad=True) # grad is the derivative
y = torch.tensor(-3.0, requires_grad=True)

f = -5*x**2*y**3 + 5*x*y + x**3

f.backward() # compute the derivative
print(x.grad, y.grad) # prints the derivatives of f at x=-1, y=-3
x.grad.zero_()  # reset the gradient to zero
'''

'''
x = torch.tensor(-2.0, requires_grad=True) # grad is the derivative
f = (-x - 4*x**3) / (4*x**2 + 3)
f.backward() # compute the derivative
print(x.grad) # prints the derivative of f at x=-2
print(f)
x.grad.zero_()  # reset the gradient to zero
'''

x = torch.tensor(1.0, requires_grad=True) # grad is the derivative

f = (5*x**3 - 3*x**2) / (1*x + 1)
f.backward() # compute the derivative
print(x.grad) # prints the derivative of f at x=-1
print(f)

# how to get to use the differentiation into the neural networks

# updating weights and biases
# w_new = w_old - lr * dw
# b_old = b_old - lr * db