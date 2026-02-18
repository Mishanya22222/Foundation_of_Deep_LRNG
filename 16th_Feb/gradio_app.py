import gradio as gr
import torch

def f(x):
    return x **2

with gr.Blocks() as iface:
    x_box = gr.Number(label="type in a number")
    square_box = gr.Number(label = "this is a squre of the number")
    # we need inputs in the x_box and outputs in the square_box
    x_box.change(fn = f, inputs = [x_box], outputs = [square_box])

iface.launch()