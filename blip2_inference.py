import time

from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def blip2_load_model():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )

    model.to(DEVICE)

    return model, processor

def blip2_inference(image, model, processor):
    start = time.monotonic()
    prompt = "Question: Describe the outfit of the person in a descriptive way. Answer:"

    inputs = processor(image, text=prompt, return_tensors="pt").to(DEVICE, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    print(f"Time taken: {time.monotonic() - start}s")

    return generated_text
