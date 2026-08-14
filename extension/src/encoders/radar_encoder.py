"""
Radar Encoder — drone detection from range-Doppler maps.

Input: range-Doppler map (1 channel, 256×256)
Data: RDRD (Kaggle, 17,485 samples, 3 classes — drones/vehicles/pedestrians)

Range-Doppler maps are already 2D images — resize to 256×256, log-scale, done.
No receiver fingerprint problem — radar hardware fingerprint is different from RF.
"""

import numpy as np
from .backbone import CNNEncoder, SIGRegLoss, DroneBGHead
import torch
import torch.nn as nn


class RadarEncoder(nn.Module):
    """Radar drone detection encoder using range-Doppler maps."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.encoder = CNNEncoder(in_ch=1, embed_dim=embed_dim)  # single channel
        self.sigreg = SIGRegLoss(embed_dim=embed_dim)
        self.bg_head = DroneBGHead(d=embed_dim)

    def forward(self, x):
        return self.encoder(x)

    def compute_loss(self, z, labels):
        sig_loss = self.sigreg(z)
        bg_logits = self.bg_head(z)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(bg_logits, labels)
        return sig_loss + bce_loss, sig_loss, bce_loss


def rd_map_to_image(rd_map, target_size=256):
    """
    Convert range-Doppler map to (1, 256, 256) for CNN input.

    Parameters:
        rd_map: 2D array (range_bins, doppler_bins)

    Returns: (1, 256, 256) float32
    """
    # Log-scale for dynamic range compression
    img = np.log1p(np.abs(rd_map).astype(np.float64))

    # Resize to 256×256
    h, w = img.shape
    if h != target_size or w != target_size:
        t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(
            t, size=(target_size, target_size), mode='bilinear', align_corners=False
        )
        img = t.squeeze().numpy()

    # Z-score
    std = img.std()
    img = (img - img.mean()) / (std + 1e-8)

    return img[np.newaxis, :, :].astype(np.float32)
