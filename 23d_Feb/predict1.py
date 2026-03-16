import torch
import torch.nn as nn

model_data = torch.load("model1.pth")

fm = model_data["fm"]
fs = model_data["fs"]
tm = model_data["tm"]
ts = model_data["ts"]
parameters = model_data["parameters"]

linear = nn.Linear(2,1)
linear.load_state_dict(parameters)

model = nn.Sequential(
    linear,
    nn.Sigmoid()
)

row_data = torch.tensor([
    [6,1]
]).float()

X = (row_data - fm) / fs

probability = model(X).item()
if probability > 0.5:
    print("Pass")
else:
    print("Fail")

print(probability)
