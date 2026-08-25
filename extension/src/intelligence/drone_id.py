"""
Drone Type Classifier — flips the receiver fingerprint bug into a feature.

The receiver fingerprint that's 100% linearly decodable is a BUG for detection.
But drone identity fingerprinting is a FEATURE nobody has solved well.

This module adds a parallel head on the encoder:
  - Detection head: receiver-invariant (SCF features)
  - ID head: drone-fingerprint-sensitive (recognizes specific drones)

Uses conditional contrastive learning:
  - Same drone, different receivers = positive pair
  - Different drone, same receiver = negative pair
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DroneIDHead(nn.Module):
    """
    Drone identification head — recognizes specific drone types
    from RF fingerprint features.

    Architecture: Linear(256, 128) → GELU → Linear(128, n_types)
    """
    def __init__(self, embed_dim=256, n_drone_types=30, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_drone_types)
        )

    def forward(self, z):
        return self.net(z)


class ConditionalContrastiveLoss(nn.Module):
    """
    Contrastive loss for drone ID disentanglement.

    Positive pair: same drone type, different receiver
    Negative pair: different drone type, same receiver

    This directly optimizes what we want: the embedding should encode
    drone identity (same across receivers) but not receiver identity
    (different across drones).
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, drone_labels, receiver_labels):
        """
        z: (B, D) embeddings
        drone_labels: (B,) drone type indices
        receiver_labels: (B,) receiver/source indices

        For each anchor i:
          positive = j where drone_labels[j] == drone_labels[i] AND receiver_labels[j] != receiver_labels[i]
          negative = j where drone_labels[j] != drone_labels[i] (regardless of receiver)
        """
        B = z.size(0)
        z_norm = F.normalize(z, dim=1)

        # Similarity matrix
        sim = torch.matmul(z_norm, z_norm.T) / self.temperature

        # Positive mask: same drone, different receiver
        drone_same = (drone_labels.unsqueeze(0) == drone_labels.unsqueeze(1))
        receiver_diff = (receiver_labels.unsqueeze(0) != receiver_labels.unsqueeze(1))
        positive_mask = drone_same & receiver_diff
        # Remove diagonal
        positive_mask.fill_diagonal_(False)

        # Negative mask: different drone
        drone_diff = (drone_labels.unsqueeze(0) != drone_labels.unsqueeze(1))
        negative_mask = drone_diff

        # For each anchor, compute contrastive loss
        loss = 0
        count = 0
        for i in range(B):
            pos_indices = positive_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = negative_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            # InfoNCE: -log(exp(sim_pos) / (exp(sim_pos) + sum(exp(sim_neg))))
            pos_sim = sim[i, pos_indices]
            neg_sim = sim[i, neg_indices]

            logits = torch.cat([pos_sim, neg_sim])
            labels = torch.cat([torch.ones(len(pos_indices)),
                               torch.zeros(len(neg_indices))]).to(z.device)

            loss += F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0).long())
            count += 1

        return loss / max(count, 1)


class DisentangledEncoder(nn.Module):
    """
    Full disentangled encoder with detection + ID heads.

    z_drone: used for drone type identification (fingerprint-sensitive)
    z_detect: used for drone detection (receiver-invariant, from SCF)

    The encoder produces a 256-dim embedding. The first 128 dims (z_detect)
    are used for detection (receiver-invariant). The last 128 dims (z_drone)
    are used for ID (drone-fingerprint-sensitive).
    """
    def __init__(self, in_ch=2, embed_dim=256, n_drone_types=30):
        super().__init__()
        from .backbone import CNNEncoder, SIGRegLoss, DroneBGHead
        self.encoder = CNNEncoder(in_ch=in_ch, embed_dim=embed_dim)
        self.sigreg = SIGRegLoss(embed_dim=embed_dim)
        self.bg_head = DroneBGHead(d=embed_dim // 2)
        self.id_head = DroneIDHead(embed_dim=embed_dim // 2, n_drone_types=n_drone_types)
        self.contrastive = ConditionalContrastiveLoss(temperature=0.07)

    def forward(self, x):
        z = self.encoder(x)
        z_detect = z[:, :128]  # first half: detection
        z_drone = z[:, 128:]   # second half: ID
        return z, z_detect, z_drone

    def compute_loss(self, z, z_detect, z_drone, bg_labels,
                     drone_labels=None, receiver_labels=None):
        """Full loss: SIGReg + BCE + ID CE + contrastive."""
        from .backbone import SIGRegLoss
        sig_loss = self.sigreg(z)

        # Detection loss (on z_detect)
        bg_logits = self.bg_head(z_detect)  # Note: bg_head needs to work on 128-dim
        bce_loss = F.binary_cross_entropy_with_logits(bg_logits, bg_labels)

        total = sig_loss + bce_loss

        # ID loss (on z_drone) — only for drone samples
        if drone_labels is not None:
            drone_mask = bg_labels == 1
            if drone_mask.sum() > 0:
                id_logits = self.id_head(z_drone[drone_mask])
                id_loss = F.cross_entropy(id_logits, drone_labels[drone_mask].long())
                total = total + id_loss

            # Contrastive loss
            if receiver_labels is not None:
                drone_mask = bg_labels == 1
                if drone_mask.sum() > 1:
                    contrast_loss = self.contrastive(
                        z_drone[drone_mask],
                        drone_labels[drone_mask].long(),
                        receiver_labels[drone_mask].long()
                    )
                    total = total + 0.1 * contrast_loss

        return total
