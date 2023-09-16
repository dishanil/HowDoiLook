from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print("processor Start")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
print("processor End")

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
print("Model end")

model.to(device)

import time

start = time.monotonic()
image = Image.open("kurta.jpg")

# print("inputs Start")
# inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
print("inputs End")

prompt = "Question: Describe the outfit of the person in a descriptive way. Answer:"

inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=10)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

print(time.monotonic() - start)
