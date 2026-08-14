"""
Late Fusion Layer with Modality Dropout.

Concatenates embeddings from RF, acoustic, and radar encoders (768-dim total),
then projects to 256-dim unified embedding. Modality dropout ensures the
fusion head never hard-depends on any single modality — it can work RF-silent.

Key design: encoders are frozen during fusion training. Only the fusion head
and downstream heads are trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ModalityDropout(nn.Module):
    """
    Randomly zero out one modality's embedding during training.
    This forces the fusion head to work with any modality missing.

    p=0.3 means each modality has 30% chance of being dropped per batch.
    Expected number of active modalities per forward pass: ~2.1 out of 3.
    """
    def __init__(self, n_modalities=3, p=0.3):
        super().__init__()
        self.n_modalities = n_modalities
        self.p = p

    def forward(self, embeddings_list):
        """
        embeddings_list: list of (B, D) tensors, one per modality
        Returns: (B, n_modalities * D) concatenated with dropout applied
        """
        if not self.training:
            return torch.cat(embeddings_list, dim=1)

        dropped = []
        for emb in embeddings_list:
            if torch.rand(1).item() < self.p:
                dropped.append(torch.zeros_like(emb))
            else:
                dropped.append(emb)

        return torch.cat(dropped, dim=1)


class FusionHead(nn.Module):
    """
    Late fusion: concat 3 modalities → project to unified embedding.

    Architecture: Linear(768, 256) → BatchNorm → GELU → Linear(256, 256) → BatchNorm

    Input: list of 3 embeddings, each (B, 256)
    Output: (B, 256) unified embedding
    """
    def __init__(self, embed_dim=256, n_modalities=3, use_modality_dropout=True, dropout_p=0.3):
        super().__init__()
        self.input_dim = embed_dim * n_modalities
        self.embed_dim = embed_dim
        self.n_modalities = n_modalities

        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        if use_modality_dropout:
            self.modality_dropout = ModalityDropout(n_modalities=n_modalities, p=dropout_p)
        else:
            self.modality_dropout = None

    def forward(self, embeddings_list):
        """
        embeddings_list: list of (B, D) tensors from each encoder
        """
        if self.modality_dropout is not None:
            x = self.modality_dropout(embeddings_list)
        else:
            x = torch.cat(embeddings_list, dim=1)
        return self.projection(x)

    def forward_silent(self, embeddings_list, silent_modality=None):
        """
        Inference mode with specific modality silenced.
        silent_modality: index of modality to zero out (0=RF, 1=acoustic, 2=radar)
        If None, uses all modalities.
        """
        if silent_modality is not None:
            embeddings_list = list(embeddings_list)
            embeddings_list[silent_modality] = torch.zeros_like(embeddings_list[silent_modality])
        x = torch.cat(embeddings_list, dim=1)
        return self.projection(x)


class FusedDetectionHead(nn.Module):
    """
    Full fused detection system: fusion head + drone/BG classifier.
    Encoders are frozen; only this module is trained.
    """
    def __init__(self, embed_dim=256, n_modalities=3, use_modality_dropout=True, dropout_p=0.3):
        super().__init__()
        self.fusion = FusionHead(embed_dim, n_modalities, use_modality_dropout, dropout_p)
        from .backbone import DroneBGHead, SIGRegLoss
        self.bg_head = DroneBGHead(d=embed_dim)
        self.sigreg = SIGRegLoss(embed_dim=embed_dim)

    def forward(self, embeddings_list):
        z = self.fusion(embeddings_list)
        return z

    def compute_loss(self, z, labels):
        sig_loss = self.sigreg(z)
        logits = self.bg_head(z)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        return sig_loss + bce_loss, sig_loss, bce_loss
