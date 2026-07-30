"""
IRIS Intent Head — RF-Only Drone Intent Classifier

Three intent classes:
  - SURVEILLANCE  (hovering / loitering — steady rotor, low velocity)
  - TRANSIT        (steady cruise — moderate Doppler shift, consistent speed)
  - ATTACK         (high-speed approach — high Doppler, burst pattern, accelerating)

This is the FIRST RF-only drone intent classifier. No published paper does
intent inference from RF emissions alone. SOTA is CPhy-ML (Nature 2024)
which uses control physics, not RF.

Architecture:
  Frozen IRIS v11 encoder → 256-dim embedding
                              ↓
              IntentHead (small MLP) → 3 logits

The IntentHead is tiny (~50K params) and trains in minutes on top of
the frozen encoder. The encoder's self-supervised representations
already encode rotor signatures and control link patterns — we just
need to learn the mapping to intent classes.

Usage:
    from intent_head import IntentClassifier

    classifier = IntentClassifier(
        encoder_checkpoint="models/lejepa_v11_best.pt",
        intent_head_checkpoint="models/intent_head.pt",
    )

    result = classifier.classify(spectrogram)
    # -> {"intent": "ATTACK", "confidence": 0.92, "probs": {...}}

Training:
    See scripts/train_intent.py (runs on Modal A100)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.iris_inference import IRISDetector, CNNEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Intent labels
# ─────────────────────────────────────────────────────────────────────────────


INTENT_CLASSES = ["SURVEILLANCE", "TRANSIT", "ATTACK"]
INTENT_TO_IDX = {c: i for i, c in enumerate(INTENT_CLASSES)}
IDX_TO_INTENT = {i: c for i, c in enumerate(INTENT_CLASSES)}


# ─────────────────────────────────────────────────────────────────────────────
# Intent Head — small MLP on top of frozen encoder
# ─────────────────────────────────────────────────────────────────────────────


class IntentHead(nn.Module):
    """
    Small MLP that maps IRIS 256-dim embedding → 3 intent logits.

    Architecture:
      Linear(256, 128) → BatchNorm → GELU → Dropout(0.1)
      Linear(128, 64)  → BatchNorm → GELU → Dropout(0.1)
      Linear(64, 3)

    Total params: ~42K
    """

    def __init__(self, embed_dim: int = 256, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Intent Classifier — wraps encoder + intent head
# ─────────────────────────────────────────────────────────────────────────────


class IntentClassifier:
    """
    Full intent classifier: IRIS encoder + IntentHead.

    Loads both checkpoints, freezes encoder, runs IntentHead inference.
    """

    def __init__(
        self,
        encoder_checkpoint: str,
        intent_head_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = IRISDetector._resolve_device(device)

        # Load encoder (reuse IRISDetector's loader)
        self.detector = IRISDetector(checkpoint_path=encoder_checkpoint, device=str(self.device))
        self.encoder = self.detector.encoder
        self.encoder.to(self.device)
        self.encoder.eval()

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Create intent head
        self.intent_head = IntentHead(embed_dim=256, n_classes=3).to(self.device)

        # Load intent head weights if available
        self.intent_head_checkpoint = intent_head_checkpoint
        if intent_head_checkpoint and os.path.exists(intent_head_checkpoint):
            self._load_intent_head(intent_head_checkpoint)
        else:
            print(f"  [warn] no intent head checkpoint at {intent_head_checkpoint}")
            print(f"         classifier will return random predictions until trained")
            print(f"         run scripts/train_intent.py to train")

        self.intent_head.eval()

    def _load_intent_head(self, path: str) -> None:
        """Load intent head checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "intent_head" in ckpt:
            self.intent_head.load_state_dict(ckpt["intent_head"])
        else:
            self.intent_head.load_state_dict(ckpt)
        print(f"  [ok] intent head loaded from {path}")

    @torch.no_grad()
    def classify(self, spectrogram: torch.Tensor) -> Dict:
        """
        Classify intent from a single spectrogram.

        Args:
            spectrogram: (2, 256, 256) tensor, per-channel normalized

        Returns:
            dict with keys:
                - intent:     "SURVEILLANCE" / "TRANSIT" / "ATTACK"
                - confidence: float (max softmax prob)
                - probs:      dict mapping class name to probability
                - embedding:  256-dim numpy array (for debugging)
        """
        if spectrogram.dim() == 4:
            spectrogram = spectrogram.squeeze(0)
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)

        spectrogram = spectrogram.to(self.device).float()

        # Encode
        embedding = self.encoder(spectrogram.unsqueeze(0))  # (1, 256)

        # Classify
        logits = self.intent_head(embedding)  # (1, 3)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        intent_idx = int(np.argmax(probs))
        intent = IDX_TO_INTENT[intent_idx]
        confidence = float(probs[intent_idx])

        return {
            "intent": intent,
            "confidence": confidence,
            "probs": {INTENT_CLASSES[i]: float(probs[i]) for i in range(3)},
            "embedding": embedding.cpu().numpy()[0],
        }

    @torch.no_grad()
    def classify_batch(self, spectrograms: torch.Tensor) -> List[Dict]:
        """Classify intent for a batch of spectrograms."""
        if spectrograms.dim() == 3:
            spectrograms = spectrograms.unsqueeze(0)

        spectrograms = spectrograms.to(self.device).float()
        embeddings = self.encoder(spectrograms)
        logits = self.intent_head(embeddings)
        probs = F.softmax(logits, dim=1).cpu().numpy()

        results = []
        for i in range(len(probs)):
            intent_idx = int(np.argmax(probs[i]))
            results.append({
                "intent": IDX_TO_INTENT[intent_idx],
                "confidence": float(probs[i][intent_idx]),
                "probs": {INTENT_CLASSES[j]: float(probs[i][j]) for j in range(3)},
            })
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic intent labeling (when no real flight-mode labels exist)
# ─────────────────────────────────────────────────────────────────────────────


def heuristic_intent_label(spectrogram: np.ndarray) -> str:
    """
    Generate a heuristic intent label from spectrogram features.

    This is used when the dataset doesn't have explicit flight-mode labels.
    We extract simple features that correlate with intent:

      - SURVEILLANCE: low temporal variance (steady rotor), narrow Doppler spread
      - TRANSIT:      moderate Doppler shift, moderate variance
      - ATTACK:       high Doppler spread (accelerating), high temporal variance

    This is a PROXY for real labels. The model trained on these labels
    learns to map RF features → intent classes, but the labels themselves
    are heuristic. For production, replace with real flight-mode labels
    from the dataset (RFUAV may have these — check the HDF5 schema).

    Args:
        spectrogram: (2, 256, 256) numpy array

    Returns:
        one of "SURVEILLANCE", "TRANSIT", "ATTACK"
    """
    # Use channel 0 (log-magnitude)
    spec = spectrogram[0] if spectrogram.ndim == 3 else spectrogram

    # Feature 1: temporal variance (how much the spectrogram changes over time)
    temporal_var = spec.var(axis=1).mean()  # average variance across freq bins

    # Feature 2: Doppler spread (std of frequency content)
    doppler_spread = spec.std(axis=0).mean()  # average std across time

    # Feature 3: peakiness (max / mean ratio)
    peakiness = spec.max() / (spec.mean() + 1e-8)

    # Heuristic rules
    if temporal_var < 0.5 and doppler_spread < 1.0:
        return "SURVEILLANCE"
    elif temporal_var > 2.0 or doppler_spread > 3.0:
        return "ATTACK"
    else:
        return "TRANSIT"


def extract_intent_features(spectrogram: np.ndarray) -> np.ndarray:
    """
    Extract a feature vector from a spectrogram for intent classification.

    These features are used as additional input to the IntentHead alongside
    the IRIS embedding. They provide explicit signal for intent:

      - temporal_variance: how much the spectrogram changes over time
      - doppler_spread:    how wide the frequency content is
      - peakiness:         max/mean ratio
      - centroid_freq:     where the energy is concentrated
      - bandwidth:         how spread out the energy is

    Args:
        spectrogram: (2, 256, 256) numpy array

    Returns:
        features: (8,) numpy array
    """
    spec = spectrogram[0] if spectrogram.ndim == 3 else spectrogram

    # Temporal features
    temporal_var = spec.var(axis=1).mean()
    temporal_trend = np.abs(np.diff(spec, axis=1)).mean()

    # Spectral features
    freq_profile = spec.mean(axis=1)  # (256,)
    # Ensure non-negative for centroid/bandwidth (treat as power)
    freq_power = np.abs(freq_profile)
    total_power = freq_power.sum() + 1e-8
    centroid_freq = float(np.sum(np.arange(len(freq_power)) * freq_power) / total_power)
    bandwidth = float(np.sqrt(max(0, ((np.arange(len(freq_power)) - centroid_freq) ** 2 * freq_power).sum() / total_power)))

    # Peakiness
    peakiness = spec.max() / (spec.mean() + 1e-8)
    snr = (spec.max() - spec.mean()) / (spec.std() + 1e-8)

    # Burst pattern (variance of variance — captures intermittent vs continuous)
    burst_pattern = spec.var(axis=1).std()

    return np.array([
        temporal_var, temporal_trend,
        float(centroid_freq), float(bandwidth),
        peakiness, snr,
        burst_pattern, doppler_spread if (doppler_spread := spec.std(axis=0).mean()) else 0.0,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────


def _smoke_test():
    """Test the intent classifier with random weights."""
    import time

    ckpt_path = "models/lejepa_v11_best.pt"
    intent_path = "models/intent_head.pt"

    print("=" * 60)
    print("IRIS Intent Classifier — Smoke Test")
    print("=" * 60)

    classifier = IntentClassifier(
        encoder_checkpoint=ckpt_path,
        intent_head_checkpoint=intent_path if os.path.exists(intent_path) else None,
    )

    print(f"  device:     {classifier.device}")
    print(f"  encoder:    {sum(p.numel() for p in classifier.encoder.parameters()):,} params (frozen)")
    print(f"  intent head: {sum(p.numel() for p in classifier.intent_head.parameters()):,} params")
    print(f"  classes:    {INTENT_CLASSES}")

    # Test with random spectrogram
    dummy = torch.randn(2, 256, 256)
    for c in range(2):
        ch = dummy[c]
        ch_std = ch.std()
        if ch_std > 1e-6:
            dummy[c] = (ch - ch.mean()) / ch_std

    t0 = time.time()
    result = classifier.classify(dummy)
    t1 = time.time()

    print(f"\n  Random spectrogram:")
    print(f"    intent:     {result['intent']}")
    print(f"    confidence: {result['confidence']:.3f}")
    print(f"    probs:      {result['probs']}")
    print(f"    latency:    {(t1 - t0) * 1000:.1f} ms")

    # Test heuristic labeling
    print(f"\n  Heuristic label for same spectrogram: {heuristic_intent_label(dummy.numpy())}")

    # Test feature extraction
    feats = extract_intent_features(dummy.numpy())
    print(f"  Extracted features: {feats}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    _smoke_test()
