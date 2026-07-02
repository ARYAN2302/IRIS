"""
CNN Encoder for LeJEPA on RF spectrograms.

Input:  (B, 2, 256, 256) — 2-channel spectrogram (log-power + phase)
Output: (B, 768) — embedding vector

Architecture: 5 conv blocks with BatchNorm + GELU + adaptive pooling + linear projection.
~3.4M parameters.

CRITICAL: BatchNorm is mandatory for LeJEPA. The paper shows 36% representation
collapse without it. SIGReg's Gaussian assumption depends on the normalization
that BatchNorm provides during training.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + GELU."""
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),  # MANDATORY — do not remove
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNEncoder(nn.Module):
    """
    CNN Encoder for RF spectrograms.
    
    Architecture:
      Conv(2→64, k=7, s=2, p=3)  + BN + GELU   → 64×128×128
      Conv(64→128, k=3, s=2, p=1) + BN + GELU   → 128×64×64
      Conv(128→256, k=3, s=2, p=1) + BN + GELU  → 256×32×32
      Conv(256→384, k=3, s=2, p=1) + BN + GELU  → 384×16×16
      Conv(384→512, k=3, s=2, p=1) + BN + GELU  → 512×8×8
      AdaptiveAvgPool2d(1)                       → 512×1×1
      Flatten                                     → 512
      Linear(512→768)                             → 768
    
    Total: ~3.4M parameters
    """
    
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        
        self.features = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),    # → 64×128×128
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),  # → 128×64×64
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1), # → 256×32×32
            ConvBlock(256, 384, kernel_size=3, stride=2, padding=1), # → 384×16×16
            ConvBlock(384, 512, kernel_size=3, stride=2, padding=1), # → 512×8×8
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, 256, 256) spectrogram tensor
        
        Returns:
            z: (B, 768) embedding vector
        """
        h = self.features(x)   # (B, 512, 8, 8)
        h = self.pool(h)       # (B, 512, 1, 1)
        h = h.flatten(1)       # (B, 512)
        z = self.proj(h)       # (B, 768)
        return z