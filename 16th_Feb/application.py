import torch
import pandas as pd  
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim  
import gradio as gr

def f(square):
    model_data = torch.load('model.pth')
    fm = model_data['fm']
    fs = model_data['fs']
    tm = model_data['tm']
    ts = model_data['ts']
    parameters = model_data['parameters']

    features = torch.tensor([
        [square]
    ])

    # standardizing the data
    X = (features - fm)/fs

    model = nn.Linear(1,1) # one input and one output
    model.load_state_dict(parameters) # assigning optimal weights
    prediction = model(X)
    price = prediction*ts + tm
    return price.item()

with gr.Blocks() as iface:
    input_box = gr.Number(label="type in square feet")
    output_box = gr.Number(label = "this is the price of the house at that square feet")
    input_box.change(fn = f, inputs = [input_box], outputs = [output_box])

iface.launch()
