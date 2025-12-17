"""
Path 2: LLM + Adapters
Frozen models with trainable adapters.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, GPT2LMHeadModel


class Adapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)


clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

for p in clip.parameters():
    p.requires_grad = False

for p in gpt2.parameters():
    p.requires_grad = False


adapter = Adapter(512, gpt2.config.n_embd)

dummy_image_embedding = torch.randn(1, 512)
aligned_output = adapter(dummy_image_embedding)

print("Adapter output shape:", aligned_output.shape)
