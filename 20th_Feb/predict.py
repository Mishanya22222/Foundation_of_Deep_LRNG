import torch
import torch.nn as nn
import gradio as gr


model_data = torch.load("model.pth")
fm = model_data["fm"].detach().clone()
fs = model_data["fs"].detach().clone()
tm = model_data["tm"].detach().clone()
ts = model_data["ts"].detach().clone()
parameters = model_data["parameters"]

model = nn.Linear(2,1)
model.load_state_dict(parameters)
model.eval()

@torch.inference_mode()
def pred(weight, engine_size):
    features = torch.tensor([[weight, engine_size]], dtype=torch.float32)

    X = (features - fm) / fs
    y_hat = model(X)

    mpg = y_hat * ts + tm

    return float(mpg.item())

with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            weight = gr.Number(label="Car weight (1000 lbs)")
            engine_size = gr.Number(label="Engine size")
        with gr.Column():
            output = gr.Number(label="Estimated MPG")

    weight.change(fn = pred, inputs = [weight, engine_size], outputs = [output])
    engine_size.change(fn = pred, inputs = [weight, engine_size], outputs = [output])

iface.launch()


    
