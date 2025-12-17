"""
Path 2: LLM + Adapters
Frozen models with a trainable adapter layer.
"""

import torch
import torch.nn as nn

print("\n=== PATH 2: LLM + ADAPTERS ===")

# -------- Adapter Definition --------
class Adapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# -------- Simulated Image Embedding --------
image_embedding = torch.randn(1, 512)
print("\n[Input] Image embedding shape:", image_embedding.shape)

# -------- Adapter Alignment --------
adapter = Adapter(512, 768)
aligned_features = adapter(image_embedding)

print("\n[Output] Aligned feature shape:", aligned_features.shape)
print("Sample values:", aligned_features[0][:5])
