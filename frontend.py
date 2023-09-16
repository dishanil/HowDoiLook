import numpy as np
import gradio as gr


def flip_text(x):
    return x[::-1]


def flip_image(x):
    return np.fliplr(x)


with gr.Blocks() as demo:
    gr.Markdown("Upload outfit to get started!")
    with gr.Tab("How do I look"):
        image_input = gr.Image()
        text_output = gr.Textbox()
        text_button = gr.Button("Get suggestions")
        with gr.Accordion("Appropriateness"):
            gr.Markdown("Look at me...")
        with gr.Accordion("Color scheme"):
            gr.Markdown("Look at me...")
        with gr.Accordion("XYZ"):
            gr.Markdown("Look at me...")
    with gr.Tab("What others wear"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("Flip")

    text_button.click(flip_text, inputs=image_input, outputs=text_output)
    image_button.click(flip_image, inputs=image_input, outputs=image_output)

demo.launch()