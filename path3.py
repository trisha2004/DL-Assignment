"""
Path 3: Unified Multimodal Model
Shared representation for all modalities.
"""

import torch
import torch.nn as nn


class UnifiedMultimodalModel(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        self.text_encoder = nn.Embedding(10000, embed_dim)
        self.image_encoder = nn.Linear(2048, embed_dim)
        self.audio_encoder = nn.Linear(128, embed_dim)

        self.transformer = nn.Transformer(
            d_model=embed_dim,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

    def forward(self, text, image, audio):
        t = self.text_encoder(text).mean(dim=1)
        i = self.image_encoder(image)
        a = self.audio_encoder(audio)

        combined = (t + i + a) / 3
        combined = combined.unsqueeze(0)

        return self.transformer(combined, combined)


if __name__ == "__main__":
    model = UnifiedMultimodalModel()

    text = torch.randint(0, 10000, (1, 10))
    image = torch.randn(1, 2048)
    audio = torch.randn(1, 128)

    output = model(text, image, audio)
    print("Output shape:", output.shape)
