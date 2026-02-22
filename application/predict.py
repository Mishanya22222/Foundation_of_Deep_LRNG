import torch
import torch.nn as nn
import gradio as gr

model_data = torch.load("model_pth")
fm = model_data["fm"]
fs = model_data["fs"]
tm = model_data["tm"]
ts = model_data["ts"]
parameters = model_data["prarameters"]

model = nn.Linear(2,1)
model.load_state_dict(parameters)
model.eval()

def f(years_exp, years_edu):
    features = torch.tensor([[years_exp, years_edu]], dtype=torch.float32)

    X = (features - fm)/fs
    y_hat = model(X)

    salary = y_hat * ts + tm
    return salary.item()

with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            years_exp = gr.Number(label = "Input years of Experience")
            years_edu =  gr.Number(label = "Input years of Education")
        with gr.Column():
            salary = gr.Number(label = "Expected salary")
    years_exp.change(fn = f, inputs = [years_exp,years_edu], outputs = [salary])
    years_edu.change(fn = f, inputs = [years_exp,years_edu], outputs = [salary])
    
iface.launch()