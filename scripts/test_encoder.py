"""Smoke test for CNN Encoder."""
import sys
sys.path.insert(0, ".")

import torch
from src.encoder import CNNEncoder

encoder = CNNEncoder(embed_dim=768)

# Count parameters
total_params = sum(p.numel() for p in encoder.parameters())
trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print(f"Encoder parameters: {total_params:,} total, {trainable_params:,} trainable")
print(f"  ~{total_params / 1e6:.1f}M params")

# Forward pass
batch_size = 4
x = torch.randn(batch_size, 2, 256, 256)
z = encoder(x)

print(f"\nInput:  {x.shape}")
print(f"Output: {z.shape}")
print(f"Output range: [{z.min().item():.3f}, {z.max().item():.3f}]")
print(f"Output mean:  {z.mean().item():.3f}")
print(f"Output std:   {z.std().item():.3f}")

# Verify no NaN/Inf
assert not torch.isnan(z).any(), "NaN in output!"
assert not torch.isinf(z).any(), "Inf in output!"

# Verify BatchNorm is present
bn_count = sum(1 for m in encoder.modules() if isinstance(m, torch.nn.BatchNorm2d))
print(f"\nBatchNorm layers: {bn_count} (MUST be > 0 for LeJEPA)")

print("\n✅ Encoder test passed!")