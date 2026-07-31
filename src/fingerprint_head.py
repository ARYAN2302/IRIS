"""
IRIS Fingerprint Head — Per-Transmitter Identification

A small contrastive head on top of the frozen IRIS v11 encoder that maps
256-dim embeddings → 128-dim RF fingerprints for per-transmitter IFF.

Used in the Cognitive EW / AVR-CL demo to show:
  - Enroll new drone types without forgetting old ones
  - Per-transmitter identification (not just type detection)

Architecture:
  Frozen IRIS v11 encoder → 256-dim embedding
                                ↓
              FingerprintHead (small MLP) → 128-dim fingerprint
                Linear(256→128) → BN → GELU → Dropout
                Linear(128→128) → BN → GELU
                L2-normalize

Training:
  Contrastive loss (SupCon) where positive pairs = same drone type,
  negative pairs = different drone types. The L2 normalization ensures
  cosine similarity = dot product, enabling fast nearest-neighbor matching.

Usage:
    from fingerprint_head import FingerprintClassifier

    classifier = FingerprintClassifier(
        encoder_checkpoint="models/lejepa_v11_best.pt",
        fingerprint_head_checkpoint="models/fingerprint_head.pt",
    )

    # Enroll a drone
    classifier.enroll("DJI_Mini_4_Pro_001", spectrogram)

    # Identify a drone
    result = classifier.identify(spectrogram)
    # -> {"matched": "DJI_Mini_4_Pro_001", "similarity": 0.92, "verdict": "FRIENDLY"}
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.iris_inference import IRISDetector, CNNEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint Head
# ─────────────────────────────────────────────────────────────────────────────


class FingerprintHead(nn.Module):
    """
    Small MLP that maps IRIS 256-dim embedding → 128-dim RF fingerprint.

    Architecture:
      Linear(256, 128) → BatchNorm → GELU → Dropout(0.1)
      Linear(128, 128)  → BatchNorm → GELU
      L2-normalize (during forward)

    Total params: ~50K
    """

    def __init__(self, embed_dim: int = 256, fp_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, fp_dim),
            nn.BatchNorm1d(fp_dim),
            nn.GELU(),
        )
        self.fp_dim = fp_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Output is L2-normalized for cosine similarity."""
        fp = self.net(x)
        return F.normalize(fp, p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint Classifier (encoder + head + registry)
# ─────────────────────────────────────────────────────────────────────────────


class FingerprintClassifier:
    """
    Full per-transmitter classifier: IRIS encoder + FingerprintHead + registry.

    The registry stores enrolled drone fingerprints as a dict:
      {drone_id: {"fingerprint": np.ndarray, "type": str, "enrolled_at": str}}

    At inference:
      1. Encode spectrogram → 256-dim embedding
      2. FingerprintHead → 128-dim L2-normalized fingerprint
      3. Cosine similarity (dot product) against all enrolled fingerprints
      4. If max similarity >= threshold → matched (FRIENDLY)
         else → unknown (HOSTILE)
    """

    def __init__(
        self,
        encoder_checkpoint: str,
        fingerprint_head_checkpoint: Optional[str] = None,
        registry_path: Optional[str] = None,
        device: Optional[str] = None,
        match_threshold: float = 0.85,
    ):
        self.device = IRISDetector._resolve_device(device)
        self.match_threshold = match_threshold

        # Load encoder (reuse IRISDetector's loader)
        self.detector = IRISDetector(checkpoint_path=encoder_checkpoint, device=str(self.device))
        self.encoder = self.detector.encoder
        self.encoder.to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Create fingerprint head
        self.fingerprint_head = FingerprintHead(embed_dim=256, fp_dim=128).to(self.device)

        # Load fingerprint head weights if available
        if fingerprint_head_checkpoint and os.path.exists(fingerprint_head_checkpoint):
            self._load_fingerprint_head(fingerprint_head_checkpoint)

        self.fingerprint_head.eval()

        # Registry of enrolled drones
        self.registry: Dict[str, dict] = {}
        self.registry_path = registry_path
        if registry_path and os.path.exists(registry_path):
            self._load_registry()

    def _load_fingerprint_head(self, path: str) -> None:
        """Load fingerprint head checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "fingerprint_head" in ckpt:
            self.fingerprint_head.load_state_dict(ckpt["fingerprint_head"])
        else:
            self.fingerprint_head.load_state_dict(ckpt)
        print(f"  [ok] fingerprint head loaded from {path}")

    def _load_registry(self) -> None:
        """Load enrolled drone registry from JSON."""
        with open(self.registry_path, "r") as f:
            data = json.load(f)
        self.registry = data
        print(f"  [ok] loaded {len(self.registry)} enrolled drones from {self.registry_path}")

    def _save_registry(self) -> None:
        """Save enrolled drone registry to JSON."""
        if self.registry_path:
            os.makedirs(os.path.dirname(self.registry_path) or ".", exist_ok=True)
            with open(self.registry_path, "w") as f:
                json.dump(self.registry, f, indent=2)

    @torch.no_grad()
    def extract_fingerprint(self, spectrogram: torch.Tensor) -> np.ndarray:
        """
        Extract 128-dim RF fingerprint from a spectrogram.

        Args:
            spectrogram: (2, 256, 256) tensor, per-channel normalized

        Returns:
            fingerprint: (128,) numpy array, L2-normalized
        """
        if spectrogram.dim() == 4:
            spectrogram = spectrogram.squeeze(0)
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)

        spectrogram = spectrogram.to(self.device).float()
        embedding = self.encoder(spectrogram.unsqueeze(0))  # (1, 256)
        fingerprint = self.fingerprint_head(embedding)  # (1, 128)
        return fingerprint.cpu().numpy()[0]

    @torch.no_grad()
    def extract_fingerprints_batch(self, spectrograms: torch.Tensor) -> np.ndarray:
        """Extract fingerprints for a batch of spectrograms."""
        if spectrograms.dim() == 3:
            spectrograms = spectrograms.unsqueeze(0)

        spectrograms = spectrograms.to(self.device).float()
        embeddings = self.encoder(spectrograms)
        fingerprints = self.fingerprint_head(embeddings)
        return fingerprints.cpu().numpy()

    def enroll(self, drone_id: str, drone_type: str, spectrogram: torch.Tensor) -> None:
        """
        Enroll a drone by its RF fingerprint.

        Args:
            drone_id: unique identifier (e.g., "DJI_Mini_4_Pro_001")
            drone_type: drone type name (e.g., "DJI Mini 4 Pro")
            spectrogram: (2, 256, 256) tensor
        """
        fingerprint = self.extract_fingerprint(spectrogram)
        self.registry[drone_id] = {
            "fingerprint": fingerprint.tolist(),
            "type": drone_type,
            "enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._save_registry()

    def enroll_from_embedding(self, drone_id: str, drone_type: str, embedding: np.ndarray) -> None:
        """Enroll using a pre-computed IRIS embedding (skips encoder)."""
        emb_tensor = torch.from_numpy(embedding).float().to(self.device)
        with torch.no_grad():
            fingerprint = self.fingerprint_head(emb_tensor.unsqueeze(0))
        self.registry[drone_id] = {
            "fingerprint": fingerprint.cpu().numpy()[0].tolist(),
            "type": drone_type,
            "enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._save_registry()

    def identify(self, spectrogram: torch.Tensor) -> Dict:
        """
        Identify a drone by matching its RF fingerprint against the registry.

        Returns:
            dict with:
              - matched_id: str or None
              - matched_type: str or None
              - similarity: float (max cosine sim)
              - verdict: "FRIENDLY" or "UNKNOWN"
              - all_similarities: dict of {drone_id: similarity}
        """
        if not self.registry:
            return {
                "matched_id": None,
                "matched_type": None,
                "similarity": 0.0,
                "verdict": "UNKNOWN",
                "all_similarities": {},
            }

        fingerprint = self.extract_fingerprint(spectrogram)

        # Compute cosine similarity (= dot product since L2-normalized)
        similarities = {}
        for drone_id, entry in self.registry.items():
            enrolled_fp = np.array(entry["fingerprint"], dtype=np.float32)
            sim = float(np.dot(fingerprint, enrolled_fp))
            similarities[drone_id] = sim

        best_id = max(similarities, key=similarities.get)
        best_sim = similarities[best_id]

        if best_sim >= self.match_threshold:
            return {
                "matched_id": best_id,
                "matched_type": self.registry[best_id]["type"],
                "similarity": best_sim,
                "verdict": "FRIENDLY",
                "all_similarities": similarities,
            }
        else:
            return {
                "matched_id": None,
                "matched_type": None,
                "similarity": best_sim,
                "verdict": "UNKNOWN",
                "all_similarities": similarities,
            }

    def identify_from_embedding(self, embedding: np.ndarray) -> Dict:
        """Identify using a pre-computed IRIS embedding."""
        if not self.registry:
            return {
                "matched_id": None,
                "matched_type": None,
                "similarity": 0.0,
                "verdict": "UNKNOWN",
                "all_similarities": {},
            }

        emb_tensor = torch.from_numpy(embedding).float().to(self.device)
        with torch.no_grad():
            fingerprint = self.fingerprint_head(emb_tensor.unsqueeze(0))
        fingerprint = fingerprint.cpu().numpy()[0]

        similarities = {}
        for drone_id, entry in self.registry.items():
            enrolled_fp = np.array(entry["fingerprint"], dtype=np.float32)
            sim = float(np.dot(fingerprint, enrolled_fp))
            similarities[drone_id] = sim

        best_id = max(similarities, key=similarities.get)
        best_sim = similarities[best_id]

        if best_sim >= self.match_threshold:
            return {
                "matched_id": best_id,
                "matched_type": self.registry[best_id]["type"],
                "similarity": best_sim,
                "verdict": "FRIENDLY",
                "all_similarities": similarities,
            }
        else:
            return {
                "matched_id": None,
                "matched_type": None,
                "similarity": best_sim,
                "verdict": "UNKNOWN",
                "all_similarities": similarities,
            }

    def list_enrolled(self) -> List[Dict]:
        """List all enrolled drones."""
        return [
            {"id": did, "type": d["type"], "enrolled_at": d["enrolled_at"]}
            for did, d in self.registry.items()
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Supervised Contrastive Loss for fingerprint training
# ─────────────────────────────────────────────────────────────────────────────


def supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised contrastive loss (Khosla et al., NeurIPS 2020).

    Same label = positive pair, different label = negative pair.
    L2-normalized embeddings → cosine similarity = dot product.

    Args:
        embeddings: (B, D) L2-normalized
        labels: (B,) class labels
        temperature: softmax temperature

    Returns:
        scalar loss
    """
    device = embeddings.device
    B = embeddings.shape[0]

    # Normalize (in case not already)
    embeddings = F.normalize(embeddings, dim=1)

    # Similarity matrix
    sim = torch.mm(embeddings, embeddings.t()) / temperature  # (B, B)
    sim = sim.clamp(-10.0, 10.0)

    # Mask: same label = positive
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()).float()  # (B, B)
    diag_mask = torch.eye(B, device=device)
    pos_mask = pos_mask - diag_mask  # exclude self
    pos_mask = pos_mask.clamp(min=0)

    # Numerator: exp(sim) for positives
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim_stable = sim - sim_max.detach()
    exp_sim = torch.exp(sim_stable)

    # Denominator: exp(sim) for all non-self
    denom_mask = 1.0 - diag_mask
    denom = (exp_sim * denom_mask).sum(dim=1, keepdim=True)

    # Log prob for positives
    numer = (exp_sim * pos_mask).sum(dim=1, keepdim=True)
    log_prob = torch.log(numer + 1e-8) - torch.log(denom + 1e-8)

    # Mean over positives per sample
    n_positives = pos_mask.sum(dim=1)
    valid = n_positives > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    mean_log_prob = (log_prob * pos_mask).sum(dim=1) / (n_positives + 1e-8)
    loss = -mean_log_prob[valid].mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────


def _smoke_test():
    """Test the fingerprint head."""
    import time

    ckpt_path = "models/lejepa_v11_best.pt"
    fp_path = "models/fingerprint_head.pt"

    print("=" * 60)
    print("IRIS Fingerprint Head — Smoke Test")
    print("=" * 60)

    if not os.path.exists(ckpt_path):
        print(f"  [skip] no encoder checkpoint at {ckpt_path}")
        return

    classifier = FingerprintClassifier(
        encoder_checkpoint=ckpt_path,
        fingerprint_head_checkpoint=fp_path if os.path.exists(fp_path) else None,
        registry_path="data/test_fingerprint_registry.json",
    )

    print(f"  device:          {classifier.device}")
    print(f"  encoder params:  {sum(p.numel() for p in classifier.encoder.parameters()):,} (frozen)")
    print(f"  fp head params:  {sum(p.numel() for p in classifier.fingerprint_head.parameters()):,}")
    print(f"  match threshold: {classifier.match_threshold}")

    # Test with random spectrogram
    dummy = torch.randn(2, 256, 256)
    for c in range(2):
        ch = dummy[c]
        ch_std = ch.std()
        if ch_std > 1e-6:
            dummy[c] = (ch - ch.mean()) / ch_std

    t0 = time.time()
    fp = classifier.extract_fingerprint(dummy)
    t1 = time.time()

    print(f"\n  Fingerprint shape: {fp.shape}")
    print(f"  Fingerprint norm:  {np.linalg.norm(fp):.4f} (should be ~1.0)")
    print(f"  Latency:           {(t1 - t0) * 1000:.1f} ms")

    # Test enrollment
    print(f"\n  Enrolling 3 test drones...")
    for i in range(3):
        spec = torch.randn(2, 256, 256)
        for c in range(2):
            ch = spec[c]
            ch_std = ch.std()
            if ch_std > 1e-6:
                spec[c] = (ch - ch.mean()) / ch_std
        classifier.enroll(f"test_drone_{i}", f"TestType{i}", spec)
    print(f"  Registry now has {len(classifier.registry)} drones")

    # Test identification
    result = classifier.identify(dummy)
    print(f"\n  Identification result:")
    print(f"    matched:   {result['matched_id']}")
    print(f"    type:      {result['matched_type']}")
    print(f"    similarity: {result['similarity']:.4f}")
    print(f"    verdict:   {result['verdict']}")

    # Cleanup
    if os.path.exists("data/test_fingerprint_registry.json"):
        os.remove("data/test_fingerprint_registry.json")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    _smoke_test()
