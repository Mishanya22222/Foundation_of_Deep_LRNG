from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from grid import save_image_grid
import matplotlib
import math

dataset = datasets.MNIST(
    root = "./data", # makes another folder called data
    train = True, # we want the training data,
    download = True,
    transform = transforms.ToTensor() # comes in as tensors 
)

# image, label = dataset[0]
# image.save("image.png")



loader = DataLoader(
    dataset, 
    batch_size = 10
)

for i,(images, labels) in enumerate(loader):
    save_image_grid(images)
    if i ==9:
        break