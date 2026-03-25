from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from grid import save_image_grid
import torch.nn as nn
import torch
import torch.optim as optim 

torch.manual_seed(1)

dataset = datasets.MNIST(
    root = "./data", # makes another folder called data
    train = True, # we want the training data,
    download = True,
    transform = transforms.ToTensor() # comes in as tensors 
)

test_dataset = datasets.MNIST(
    root = "./data",
    train = False,
    download= True,
    transform = transforms.ToTensor()
)

loader = DataLoader(
    dataset, 
    batch_size = 64,
    shuffle = True # send all of the images through for the first epoch, than shiffle and then send again for the second epoch, and so on.
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 10000,
    shuffle = False
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
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        optimizer.zero_grad()
        y_hat = model(images)
        loss = criterion(y_hat, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total += labels.size(0) # number of samples in the batch
        correct += (y_hat.argmax(1) == labels).sum().item() # number of correct predictions in the batch
        
    print(f"Epoch: {epoch}, Loss: {total_loss/len(loader)}, Accuracy: {correct/total}") # batches in an epoch, so we are getting the average loss per batch


