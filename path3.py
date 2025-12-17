"""
Path 3: Unified Multimodal Model
Single model with shared representations.
"""

import torch
import torch.nn as nn

print("\n=== PATH 3: UNIFIED MULTIMODAL MODEL ===")

# -------- Unified Model --------
class UnifiedMultimodalModel(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.text_embed = nn.Embedding(10000, embed_dim)
        self.image_embed = nn.Linear(2048, embed_dim)
        self.audio_embed = nn.Linear(128, embed_dim)

        self.transformer = nn.Transformer(
            d_model=embed_dim,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

    def forward(self, text, image, audio):
        t = self.text_embed(text).mean(dim=1)
        i = self.image_embed(image)
        a = self.audio_embed(audio)

        combined = (t + i + a) / 3
        combined = combined.unsqueeze(0)

        return self.transformer(combined, combined)

# -------- Inputs --------
text_input = torch.randint(0, 10000, (1, 10))
image_input = torch.randn(1, 2048)
audio_input = torch.randn(1, 128)

print("\n[Input] Text shape:", text_input.shape)
print("[Input] Image shape:", image_input.shape)
print("[Input] Audio shape:", audio_input.shape)

# -------- Model Output --------
model = UnifiedMultimodalModel()
output = model(text_input, image_input, audio_input)

print("\n[Output] Unified model output shape:", output.shape)
