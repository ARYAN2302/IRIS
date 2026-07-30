"""
IRIS Inference Module — v11

Clean, dependency-light inference for the IRIS drone detector.
Loads the trained v11 encoder + Mahalanobis centroid/covariance.

Usage:
    from iris_inference import IRISDetector

    detector = IRISDetector(
        checkpoint_path="models/lejepa_v11_best.pt",
        centroid_path="models/drone_centroid.npz",
    )

    result = detector.detect(spectrogram_2x256x256)
    # -> {"verdict": "DRONE", "confidence": 0.97, "mahal_dist": 12.3, "threshold": 27.4}

The spectrogram must be:
    - shape: (2, 256, 256) — 2 channels (log-magnitude + gradient)
    - per-channel normalized (mean 0, std 1) — same as training
    - torch.float32

Architecture reproduced EXACTLY from scripts/train_modal_v11.py
DO NOT MODIFY — checkpoint loading will break.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# ENCODER — exact reproduction from train_modal_v11.py
# DO NOT CHANGE. Checkpoint state_dict keys depend on this.
# ─────────────────────────────────────────────────────────────────────────────


class ConvBlock(nn.Module):
    """Double conv block: (Conv→BN→GELU) × 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNEncoder(nn.Module):
    """
    6-layer CNN encoder for 2-channel 256×256 RF spectrograms.

    Architecture (depth=6, width=64, embed_dim=256):
      Block 0: in_ch=2  -> 64    + MaxPool  (128x128)
      Block 1: 64       -> 64    + MaxPool  (64x64)
      Block 2: 64       -> 128   + MaxPool  (32x32)
      Block 3: 128      -> 128   + MaxPool  (16x16)
      Block 4: 128      -> 256   + MaxPool  (8x8)
      Block 5: 256      -> 256   + MaxPool  (4x4)
      AdaptiveAvgPool -> 256 channels at 4x4 -> flatten 4096
      Linear(4096 -> 256) + BatchNorm1d(256)

    Total params: ~3.4M
    """

    def __init__(self, in_ch: int = 2, width: int = 64, depth: int = 6, embed_dim: int = 256):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(depth):
            out_ch = min(width * (2 ** (i // 2)), 512)
            layers.append(ConvBlock(ch, out_ch))
            layers.append(nn.MaxPool2d(2))
            ch = out_ch
        self.conv = nn.Sequential(*layers)

        # Compute flatten size with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 256, 256)
            out = self.conv(dummy)
            flat = out.numel() // out.shape[0]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


# ─────────────────────────────────────────────────────────────────────────────
# MAHALANOBIS DETECTOR
# ─────────────────────────────────────────────────────────────────────────────


def compute_mahalanobis(
    embeddings: np.ndarray,
    centroid: np.ndarray,
    cov_inv: np.ndarray,
    l2_normalize: bool = True,
) -> np.ndarray:
    """
    Compute Mahalanobis distance from each embedding to the centroid.

    Args:
        embeddings: (N, D) array
        centroid:   (D,) array
        cov_inv:    (D, D) array — pre-inverted covariance
        l2_normalize: if True, L2-normalize embeddings first (Mahalanobis++ 2025 finding)

    Returns:
        distances: (N,) array of Mahalanobis distances
    """
    if l2_normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings = embeddings / norms

    diff = embeddings - centroid  # (N, D)
    # Mahalanobis: sqrt(diff @ cov_inv @ diff^T) per row
    mahal_sq = np.sum(diff @ cov_inv * diff, axis=1)  # (N,)
    return np.sqrt(np.maximum(mahal_sq, 0.0))


def fit_mahalanobis(
    embeddings: np.ndarray,
    reg: float = 1e-3,
    l2_normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit Mahalanobis centroid + inverse covariance from training embeddings.

    Args:
        embeddings: (N, D) training drone embeddings
        reg: diagonal regularization added to covariance
        l2_normalize: if True, L2-normalize embeddings first

    Returns:
        centroid: (D,)
        cov_inv:  (D, D) — pre-inverted for fast inference
    """
    if l2_normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings = embeddings / norms

    centroid = embeddings.mean(axis=0)
    D = embeddings.shape[1]
    cov = np.cov(embeddings.T) + reg * np.eye(D)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    return centroid, cov_inv


# ─────────────────────────────────────────────────────────────────────────────
# IRIS DETECTOR
# ─────────────────────────────────────────────────────────────────────────────


class IRISDetector:
    """
    Full IRIS drone detector: encoder + Mahalanobis distance.

    Loads a trained v11 checkpoint, extracts the encoder, and runs
    Mahalanobis distance detection against a precomputed drone centroid.

    The Mahalanobis centroid + covariance can be:
      1. Pre-computed and saved as .npz (preferred for deployment)
      2. Computed on-the-fly from a sample of training drone spectrograms

    L2-normalization is applied BEFORE Mahalanobis (Mahalanobis++ 2025 finding —
    improves OOD detection significantly, especially for cross-dataset transfer).
    """

    def __init__(
        self,
        checkpoint_path: str,
        centroid_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
        l2_normalize: bool = True,
    ):
        """
        Args:
            checkpoint_path: path to lejepa_v11_best.pt (Modal download)
            centroid_path:   path to drone_centroid.npz (optional — can compute on the fly)
            device:          "mps" (M1), "cuda", or "cpu". Auto-detected if None.
            threshold:       Mahalanobis distance threshold for DRONE vs BG.
                             If None, uses 27.42 (v11 optimum from your bootstrap CI).
            l2_normalize:    apply L2 norm before Mahalanobis (default True — Mahalanobis++)
        """
        self.device = self._resolve_device(device)
        self.l2_normalize = l2_normalize

        # Load encoder
        self.encoder = self._load_encoder(checkpoint_path)
        self.encoder.to(self.device)
        self.encoder.eval()

        # Default threshold from v11 bootstrap CI analysis
        self.threshold = threshold if threshold is not None else 27.42
        self.threshold_source = "v11_default" if threshold is None else "user"

        # Load or initialize Mahalanobis params
        self.centroid: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.train_percentiles: Optional[np.ndarray] = None

        if centroid_path and os.path.exists(centroid_path):
            self._load_centroid(centroid_path)

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_encoder(self, checkpoint_path: str) -> CNNEncoder:
        """Load v11 checkpoint, extract encoder weights, return CNNEncoder."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Run scripts/pull_from_modal.py to download from Modal."
            )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)

        # Extract only encoder.* keys
        encoder_state = {}
        for key, val in state_dict.items():
            if key.startswith("encoder."):
                encoder_state[key[len("encoder."):]] = val

        if not encoder_state:
            raise ValueError(
                "No 'encoder.*' keys found in checkpoint. "
                "Is this a v11 LeJEPASupConV11 checkpoint?"
            )

        encoder = CNNEncoder(in_ch=2, width=64, depth=6, embed_dim=256)
        try:
            encoder.load_state_dict(encoder_state, strict=True)
        except RuntimeError as e:
            # Try non-strict as fallback (BatchNorm running stats may differ)
            print(f"  [warn] strict load failed, trying non-strict: {e}")
            encoder.load_state_dict(encoder_state, strict=False)

        # Restore training config if available
        if "cfg" in ckpt:
            self.cfg = ckpt["cfg"]
        else:
            self.cfg = {}

        return encoder

    def _load_centroid(self, path: str) -> None:
        """Load precomputed Mahalanobis centroid + covariance from .npz."""
        data = np.load(path)
        self.centroid = data["centroid"]
        self.cov_inv = data["cov_inv"]
        if "threshold" in data:
            self.threshold = float(data["threshold"])
            self.threshold_source = "computed"
        if "train_percentiles" in data:
            self.train_percentiles = data["train_percentiles"]

    def save_centroid(self, path: str) -> None:
        """Save current Mahalanobis centroid + covariance to .npz."""
        if self.centroid is None or self.cov_inv is None:
            raise RuntimeError("No centroid to save. Call fit_centroid() first.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            centroid=self.centroid,
            cov_inv=self.cov_inv,
            threshold=self.threshold,
            train_percentiles=self.train_percentiles if self.train_percentiles is not None else np.array([]),
        )
        print(f"  [ok] saved centroid to {path}")

    @torch.no_grad()
    def encode(self, spectrograms: torch.Tensor) -> np.ndarray:
        """
        Encode spectrograms → 256-dim embeddings.

        Args:
            spectrograms: (B, 2, 256, 256) tensor, already normalized per-channel

        Returns:
            embeddings: (B, 256) numpy array
        """
        if spectrograms.dim() == 3:
            spectrograms = spectrograms.unsqueeze(0)

        spectrograms = spectrograms.to(self.device).float()
        z = self.encoder(spectrograms)
        return z.cpu().numpy()

    def fit_centroid(
        self,
        spectrograms: torch.Tensor,
        batch_size: int = 64,
    ) -> None:
        """
        Compute Mahalanobis centroid + covariance from a sample of drone spectrograms.

        Call this once with ~500-5000 training drone spectrograms to fit the detector.

        Args:
            spectrograms: (N, 2, 256, 256) tensor of DRONE spectrograms (no backgrounds)
            batch_size: inference batch size
        """
        print(f"  [info] fitting Mahalanobis centroid from {len(spectrograms)} drone samples...")
        all_embs = []
        for i in range(0, len(spectrograms), batch_size):
            batch = spectrograms[i:i + batch_size]
            embs = self.encode(batch)
            all_embs.append(embs)
        all_embs = np.concatenate(all_embs, axis=0)

        self.centroid, self.cov_inv = fit_mahalanobis(
            all_embs, reg=1e-3, l2_normalize=self.l2_normalize
        )

        # Compute training percentiles for adaptive thresholding
        dists = compute_mahalanobis(
            all_embs, self.centroid, self.cov_inv, l2_normalize=self.l2_normalize
        )
        self.train_percentiles = np.percentile(dists, [50, 75, 90, 95, 99])

        # Set threshold at 99th percentile of training drones
        # (anything farther than 99% of training drones = background)
        self.threshold = float(self.train_percentiles[-1])
        self.threshold_source = "fit_99pct"
        print(f"  [ok] centroid fit. threshold={self.threshold:.2f} (99th pct of training drones)")

    def detect(self, spectrogram: torch.Tensor) -> Dict:
        """
        Detect drone in a single spectrogram.

        Args:
            spectrogram: (2, 256, 256) or (1, 2, 256, 256) tensor

        Returns:
            dict with keys:
                - verdict:    "DRONE" or "BACKGROUND"
                - confidence: 0.0-1.0 (1 - percentile)
                - mahal_dist: float
                - threshold:  float
                - percentile: float (% of training drones with HIGHER distance)
                              higher = more drone-like
        """
        if self.centroid is None or self.cov_inv is None:
            raise RuntimeError("Mahalanobis not fit. Call fit_centroid() or provide centroid_path.")

        emb = self.encode(spectrogram)  # (1, 256)
        dist = compute_mahalanobis(
            emb, self.centroid, self.cov_inv, l2_normalize=self.l2_normalize
        )[0]

        is_drone = dist <= self.threshold
        verdict = "DRONE" if is_drone else "BACKGROUND"

        # Percentile: what fraction of training drones have HIGHER distance?
        # higher percentile = more drone-like (closer to centroid)
        if self.train_percentiles is not None:
            # Interpolate percentile from training distribution
            pct = float(np.interp(dist, self.train_percentiles, [50, 75, 90, 95, 99]))
            if dist < self.train_percentiles[0]:
                pct = 50.0 - (self.train_percentiles[0] - dist) * 5  # extrapolate down
            confidence = max(0.0, min(1.0, pct / 100.0))
        else:
            # Simple confidence from distance ratio
            confidence = max(0.0, min(1.0, 1.0 - dist / (self.threshold * 2)))

        return {
            "verdict": verdict,
            "confidence": float(confidence),
            "mahal_dist": float(dist),
            "threshold": float(self.threshold),
            "percentile": float(pct) if self.train_percentiles is not None else 0.0,
        }

    def detect_batch(self, spectrograms: torch.Tensor) -> list:
        """Detect drones in a batch of spectrograms. Returns list of dicts."""
        if spectrograms.dim() == 3:
            spectrograms = spectrograms.unsqueeze(0)
        embs = self.encode(spectrograms)
        dists = compute_mahalanobis(
            embs, self.centroid, self.cov_inv, l2_normalize=self.l2_normalize
        )
        results = []
        for dist in dists:
            is_drone = dist <= self.threshold
            results.append({
                "verdict": "DRONE" if is_drone else "BACKGROUND",
                "mahal_dist": float(dist),
                "threshold": float(self.threshold),
            })
        return results


# ─────────────────────────────────────────────────────────────────────────────
# SPECTROGRAM PREPROCESSING — for live I/Q input
# ─────────────────────────────────────────────────────────────────────────────


def iq_to_spectrogram(
    iq: np.ndarray,
    n_fft: int = 1024,
    hop_length: Optional[int] = None,
    target_size: int = 256,
    normalize_per_channel: bool = True,
) -> torch.Tensor:
    """
    Convert complex I/Q samples to a 2-channel 256×256 spectrogram tensor
    matching IRIS v11 training format.

    Channel 0: log-magnitude STFT
    Channel 1: magnitude gradient (Sobel-like, captures temporal dynamics)

    Args:
        iq: complex128 or complex64 array of I/Q samples
        n_fft: FFT size (default 1024)
        hop_length: hop in samples (default n_fft // 4)
        target_size: output spatial dim (256 for v11)
        normalize_per_channel: per-channel mean/std normalization

    Returns:
        spectrogram: (2, 256, 256) torch.float32 tensor
    """
    import torch
    from scipy import signal as scipy_signal

    if hop_length is None:
        hop_length = n_fft // 4

    # Ensure complex
    if not np.iscomplexobj(iq):
        if iq.ndim == 2 and iq.shape[1] == 2:
            iq = iq[:, 0] + 1j * iq[:, 1]
        else:
            raise ValueError(f"iq must be complex or (N, 2) real, got shape {iq.shape}")

    # Compute STFT
    f, t, Zxx = scipy_signal.stft(
        iq,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        return_onesided=False,
        boundary=None,
    )

    # Shift DC to center
    Zxx = np.fft.fftshift(Zxx, axes=0)
    magnitude = np.abs(Zxx)

    # Log-magnitude
    log_mag = 20.0 * np.log10(magnitude + 1e-8)

    # Gradient channel (captures temporal dynamics — rotor modulation, bursts)
    grad = np.gradient(log_mag, axis=1)

    # Stack to 2-channel
    spec = np.stack([log_mag, grad], axis=0)  # (2, F, T)

    # Resize to 256x256
    spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)  # (1, 2, F, T)
    spec_tensor = F.interpolate(
        spec_tensor, size=(target_size, target_size),
        mode="bilinear", align_corners=False
    ).squeeze(0)  # (2, 256, 256)

    # Per-channel normalize
    if normalize_per_channel:
        for c in range(spec_tensor.shape[0]):
            ch = spec_tensor[c]
            ch_std = ch.std()
            if ch_std > 1e-6:
                spec_tensor[c] = (ch - ch.mean()) / ch_std
            else:
                spec_tensor[c] = ch - ch.mean()

    return spec_tensor


# ─────────────────────────────────────────────────────────────────────────────
# CLI — quick smoke test
# ─────────────────────────────────────────────────────────────────────────────


def _smoke_test():
    """Quick smoke test: load detector, encode random spectrogram."""
    import time

    ckpt_path = "models/lejepa_v11_best.pt"
    centroid_path = "models/drone_centroid.npz"

    print("=" * 60)
    print("IRIS Inference Smoke Test")
    print("=" * 60)

    detector = IRISDetector(
        checkpoint_path=ckpt_path,
        centroid_path=centroid_path if os.path.exists(centroid_path) else None,
    )
    print(f"  device:   {detector.device}")
    print(f"  encoder:  {sum(p.numel() for p in detector.encoder.parameters()):,} params")
    print(f"  embed_dim: 256")
    print(f"  threshold: {detector.threshold:.2f} ({detector.threshold_source})")

    # Random spectrogram (won't be a real drone — just tests pipeline)
    dummy = torch.randn(2, 256, 256)
    # Per-channel normalize
    for c in range(2):
        ch = dummy[c]
        ch_std = ch.std()
        if ch_std > 1e-6:
            dummy[c] = (ch - ch.mean()) / ch_std

    if detector.centroid is not None:
        t0 = time.time()
        result = detector.detect(dummy)
        t1 = time.time()
        print(f"\n  Random spectrogram (should be BACKGROUND):")
        print(f"    verdict:    {result['verdict']}")
        print(f"    confidence: {result['confidence']:.3f}")
        print(f"    mahal_dist: {result['mahal_dist']:.2f}")
        print(f"    threshold:  {result['threshold']:.2f}")
        print(f"    latency:    {(t1 - t0) * 1000:.1f} ms")
    else:
        print("\n  No centroid loaded — encoding only.")
        t0 = time.time()
        emb = detector.encode(dummy)
        t1 = time.time()
        print(f"    embedding shape: {emb.shape}")
        print(f"    embedding norm:  {np.linalg.norm(emb):.3f}")
        print(f"    latency:         {(t1 - t0) * 1000:.1f} ms")

    print("\n" + "=" * 60)
    print("Smoke test complete.")
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()
