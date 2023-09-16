import numpy as np
import gradio as gr
from blip2_inference import blip2_load_model, blip2_inference
from prompt_utils import promptEngineer, gpt_inference, getSuggestions

SUGGESTIONS_TEXT = "Yes, please!"
TITLE = "Hey, how do I look?"
DESCRIPTION = ("Your attire will be judged based on \n 1. Appropriateness \n 2. Color scheme")


def inference(img, occasion_desc, suggestions_needed):
    suggestions_needed = True if suggestions_needed == SUGGESTIONS_TEXT else False

    attire_desc = blip2_inference(img, blip2_model, blip2_processor)

    prompt_to_gpt = promptEngineer(attire_desc, occasion_desc)

    print(f"{prompt_to_gpt = }")

    howDoILook = gpt_inference(prompt_to_gpt)

    suggestions = getSuggestions(occasion_desc)

    howDoILook = howDoILook + "\n" + suggestions

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
        gr.Checkbox(SUGGESTIONS_TEXT, label="Suggestions?", info="Do you need any suggestions about what others generally wear for this situation?"),
    ],
    outputs=gr.outputs.Textbox(label="How do I look?"),
    title=TITLE,
    description=DESCRIPTION,
).queue(concurrency_count=1)

demo.launch(share=True)
