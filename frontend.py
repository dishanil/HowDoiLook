import re

import numpy as np
import gradio as gr
from blip2_inference import blip2_load_model, blip2_inference
from prompt_utils import promptEngineer, gpt_inference

TITLE = "Hey, how do I look?"


def inference(img, occasion_desc):
    gender, attire_desc = blip2_inference(img, blip2_model, blip2_processor)

    prompt_to_gpt = promptEngineer(gender, attire_desc, occasion_desc)

    print(f"{prompt_to_gpt = }")

    howDoILook = gpt_inference(prompt_to_gpt)

    return howDoILook

# Load the model once for faster inference
blip2_model, blip2_processor = blip2_load_model()

print("BLIP2 model loaded and ready to infer")

demo = gr.Interface(
    fn=inference,
    inputs=[
        gr.inputs.Image(type="pil"),
        gr.Textbox(
                    label="Occasion",
                    info="Interview - Tech or Consulting? \n Wedding - Mandarin or Indian? \n Season - Summer or Fall?",
                    lines=3,
                ),
       ],
    outputs=gr.outputs.Textbox(label="How do I look?"),
    title=TITLE,
).queue(concurrency_count=1)

# can put a description box about how your outfit is rated?

demo.launch(share=True)
