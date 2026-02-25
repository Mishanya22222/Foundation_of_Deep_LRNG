import torch
import torch.nn as nn
import gradio as gr

model_data = torch.load("model.pth")
fm = model_data["fm"]
fs = model_data["fs"]
parameters = model_data["parameters"]

linear = nn.Linear(1,1)
linear.load_state_dict(parameters) # loading the proper weights, bias and the sigmoid

# When we are making the prediction we have to specify the sigmoid
model = nn.Sequential(
    linear,
    nn.Sigmoid()
)

def pred(size):
    val = float(size)
    features = torch.tensor([[val]], dtype=torch.float32)

    # Normalize with small epsilon to avoid div-by-zero
    X = (features - fm) / fs
    prob = model(X).item()

    if prob > 0.5:
        classification = "Malignant"
    else:
        classification = "Benign"
    return classification

with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            size = gr.Number(label = "Please enter the size of the tumor Diagnozed")
        with gr.Column():
            output = gr.Textbox(label = "Prediction of the Tumor that you have")
        size.change(fn = pred, inputs = [size], outputs = [output])

iface.launch()


