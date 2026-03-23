from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from grid import save_image_grid
import torch.nn as nn
import torch
import torch.optim as optim 
import matplotlib
import math

torch.manual_seed(1)

dataset = datasets.MNIST(
    root = "./data", # makes another folder called data
    train = True, # we want the training data,
    download = True,
    transform = transforms.ToTensor() # comes in as tensors 
)

loader = DataLoader(
    dataset, 
    batch_size = 64,
    shuffle = True # send all of the images through for the first epoch, than shiffle and then send again for the second epoch, and so on.
)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),  # input
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64,10) # output
)

criterion = nn.CrossEntropyLoss() # loss function for classification problems, which has the softmax and will produce the probability
optimizer = optim.Adam(model.parameters(), lr= 0.001)

epochs = 10

# how to print only last loose withn batch
for epoch in range(epochs):
    for images, labels in loader:
        y_hat = model(images)
        loss = criterion(y_hat, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"epoch {epoch+1}/{epochs} last batch loss = {loss.item():.4f}")



