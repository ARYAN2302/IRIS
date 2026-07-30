"""
Predictor network for LeJEPA.

Input:  context embedding z_ctx (B, 768)
Output: predicted target embedding ŷ (B, 768)

Simple MLP with one hidden layer. The predictor is lightweight —
the heavy lifting is in the encoder. This follows the paper's design.
"""

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """
    MLP predictor for LeJEPA.
    
    Architecture:
      Linear(768→768) → GELU → Linear(768→768)
    
    ~590K parameters. Deliberately small — the predictor should be
    a bottleneck so the encoder learns good representations, not
    the predictor learning to memorize.
    """
    
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
    
    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_ctx: (B, 768) context embedding from encoder
        
        Returns:
            y_pred: (B, 768) predicted target embedding
        """
        return self.net(z_ctx)