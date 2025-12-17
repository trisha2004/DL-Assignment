"""
Path 1: LLM + Tools
Speech -> Text -> LLM -> Image
"""

import whisper
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch

print("\n=== PATH 1: LLM + TOOLS ===")

# -------- Speech to Text --------
audio_file = "input.wav"  # provide a short audio file
whisper_model = whisper.load_model("base")
result = whisper_model.transcribe(audio_file)
text_prompt = result["text"]

print("\n[Input] Speech to Text Output:")
print(text_prompt)

# -------- LLM Reasoning --------
generator = pipeline("text-generation", model="distilgpt2")
llm_output = generator(text_prompt, max_length=50)[0]["generated_text"]

print("\n[Output] LLM Generated Text:")
print(llm_output)

# -------- Text to Image --------
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

image = pipe(llm_output).images[0]
image.save("path1_output.png")

print("\n[Final Output] Image saved as path1_output.png")

