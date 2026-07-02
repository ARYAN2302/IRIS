"""LeJEPA model with projection head.

Architecture:
  Encoder (CNN) → z (768d)  ← used for evaluation / downstream
  Projector     → p (768d)  ← used for SIGReg + invariance loss
  Predictor     → y (768d)  ← predicts target projection from context projection
"""

import torch
import torch.nn as nn

from src.encoder import CNNEncoder
from src.predictor import Predictor
from src.sigreg import SIGReg, InvarianceLoss, LeJEPALoss


class ProjectionHead(nn.Module):
    """Projection head: encoder space → loss space decoupling.

    Lets the encoder learn general representations while the
    projection head handles SIGReg + invariance optimization.
    For evaluation, we use the pre-projection encoder output.
    """
    def __init__(self, in_dim=768, hidden_dim=768, out_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class LeJEPA(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, proj_dim=768, K=256, lam=1e-3):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.projector = ProjectionHead(in_dim=embed_dim, hidden_dim=embed_dim, out_dim=proj_dim)
        self.predictor = Predictor(in_dim=proj_dim, hidden_dim=proj_dim, out_dim=proj_dim)
        self.criterion = LeJEPALoss(embed_dim=proj_dim, K=K, lam=lam)

        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

    def forward(self, x1, x2):
        """Full forward pass for training.

        Returns:
            z1, z2: encoder outputs (for monitoring, not loss)
            p1, p2: projected outputs (for loss computation)
            y_pred: predictor output
            losses: dict with 'sig', 'inv', 'total'
        """
        # Encode both views
        z1 = self.encoder(x1)   # (B, embed_dim)
        z2 = self.encoder(x2)   # (B, embed_dim)

        # Project to loss space
        p1 = self.projector(z1)  # (B, proj_dim)
        p2 = self.projector(z2)  # (B, proj_dim)

        # Predict: predictor maps context projection → target projection
        y_pred = self.predictor(p1)  # (B, proj_dim)

        # Compute losses in projection space
        losses = self.criterion(p2, y_pred)  # p2 is target, y_pred is prediction

        return z1, z2, p1, p2, y_pred, losses

    def encode(self, x):
        """Get encoder representation (NO projection) for evaluation."""
        return self.encoder(x)

    def param_count(self):
        enc = sum(p.numel() for p in self.encoder.parameters())
        proj = sum(p.numel() for p in self.projector.parameters())
        pred = sum(p.numel() for p in self.predictor.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {'encoder': enc, 'projector': proj, 'predictor': pred, 'total': total}