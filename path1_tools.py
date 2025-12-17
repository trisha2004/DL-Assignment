

import whisper
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch


def speech_to_text(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]


def text_reasoning(text):
    generator = pipeline("text-generation", model="distilgpt2")
    output = generator(text, max_length=50)
    return output[0]["generated_text"]


def text_to_image(prompt):
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    image = pipe(prompt).images[0]
    image.save("result.png")


if __name__ == "__main__":
    print("Path 1 module loaded")
