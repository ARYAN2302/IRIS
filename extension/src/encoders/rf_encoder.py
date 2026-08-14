"""
RF Encoder — trains on SCF (cyclostationary) features instead of spectrograms.
Receiver-invariant by construction (COH channel cancels receiver gain).

Supports:
  - SCF-only input (2 channels: log|SCF| + |COH|)
  - Hybrid input (4 channels: SCF + autocorrelation + HOM)
  - IQ augmentation before SCF computation (preserves COH invariance)
  - DRIFT-style disentanglement (GRL + domain head)
  - MixStyle for feature statistic mixing
"""

from .backbone import CNNEncoder, SIGRegLoss, DroneBGHead, MixStyle, GRL, DomainHead, dann_lambda
from ..scf_features import iq_to_scf_image, iq_to_hybrid_image
from ..iq_augment import augment_batch
import torch
import torch.nn as nn
import numpy as np


class RFEncoder(nn.Module):
    """
    Full RF encoder with optional MixStyle and DRIFT components.

    in_ch=2: SCF only (log|SCF| + |COH|)
    in_ch=4: Hybrid (SCF + autocorr + HOM)
    """
    def __init__(self, in_ch=2, embed_dim=256, use_mixstyle=True,
                 mixstyle_after_block=2, use_drift=False, n_domains=2):
        super().__init__()
        self.encoder = CNNEncoder(in_ch=in_ch, embed_dim=embed_dim)
        self.sigreg = SIGRegLoss(embed_dim=embed_dim)
        self.bg_head = DroneBGHead(d=embed_dim)
        self.use_drift = use_drift

        if use_mixstyle:
            self.mixstyle = MixStyle(p=0.5, alpha=0.1)
            self.mixstyle_after_block = mixstyle_after_block
        else:
            self.mixstyle = None

        if use_drift:
            self.domain_head = DomainHead(d=embed_dim, n_domains=n_domains)
        else:
            self.domain_head = None

    def forward(self, x):
        # Pass through conv blocks, insert MixStyle after specified block
        if self.mixstyle is not None:
            for i, layer in enumerate(self.encoder.conv):
                x = layer(x)
                if i == self.mixstyle_after_block * 2 - 1:  # after block N's MaxPool
                    x = self.mixstyle(x)
        else:
            x = self.encoder.conv(x)
        z = self.encoder.head(x)
        return z

    def compute_loss(self, z, labels, sources=None, lam=0.0):
        """Compute total loss: SIGReg + BCE + optional domain CE."""
        sig_loss = self.sigreg(z)
        bg_logits = self.bg_head(z)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(bg_logits, labels)

        total = sig_loss + bce_loss

        if self.domain_head is not None and sources is not None:
            drone_mask = sources != -1
            if drone_mask.sum() > 0:
                z_drones = z[drone_mask]
                sources_drones = sources[drone_mask]
                domain_logits = self.domain_head(z_drones)
                domain_loss = torch.nn.functional.cross_entropy(domain_logits, sources_drones)
                total = total + lam * domain_loss

        return total, sig_loss, bce_loss


def prepare_scf_training_data(zenodo_iq_list, dronerf_iq_list, bg_specs,
                               augment_factor=20, use_hybrid=False, seed=42):
    """
    Prepare SCF training data with IQ augmentation.

    Parameters:
        zenodo_iq_list: list of complex IQ arrays from Zenodo
        dronerf_iq_list: list of (iq, drone_type) tuples from DroneRF
        bg_specs: numpy array of BG spectrograms (2, 256, 256) — format mismatch acceptable
        augment_factor: number of augmented copies per real IQ sample
        use_hybrid: if True, produce 4-channel hybrid input

    Returns: (specs, labels, sources) numpy arrays
    """
    np.random.seed(seed)
    specs = []
    labels = []
    sources = []

    # Process Zenodo IQ → SCF
    for iq in zenodo_iq_list:
        if use_hybrid:
            scf_img = iq_to_hybrid_image(iq)
        else:
            scf_img = iq_to_scf_image(iq)
        specs.append(scf_img)
        labels.append(1)
        sources.append(0)  # Zenodo = source 0

        # Augmented copies
        augmented = augment_batch(iq, n_augments=augment_factor)
        for aug_iq in augmented:
            if use_hybrid:
                aug_img = iq_to_hybrid_image(aug_iq)
            else:
                aug_img = iq_to_scf_image(aug_iq)
            specs.append(aug_img)
            labels.append(1)
            sources.append(0)

    # Process DroneRF IQ → SCF
    for iq, dtype in dronerf_iq_list:
        if use_hybrid:
            scf_img = iq_to_hybrid_image(iq)
        else:
            scf_img = iq_to_scf_image(iq)
        specs.append(scf_img)
        labels.append(1)
        sources.append(1)  # DroneRF = source 1

        # Augmented copies
        augmented = augment_batch(iq, n_augments=augment_factor)
        for aug_iq in augmented:
            if use_hybrid:
                aug_img = iq_to_hybrid_image(aug_iq)
            else:
                aug_img = iq_to_scf_image(aug_iq)
            specs.append(aug_img)
            labels.append(1)
            sources.append(1)

    # Add BG spectrograms (already preprocessed, different format)
    for spec in bg_specs:
        specs.append(spec)
        labels.append(0)
        sources.append(-1)  # BG excluded from domain loss

    return np.stack(specs), np.array(labels, dtype=np.float32), np.array(sources, dtype=np.int64)
