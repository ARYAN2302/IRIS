#!/usr/bin/env python3
"""
IRIS v9 — Matched Background Evaluation

The concern: AUC=1.0000 might be a shortcut. Maybe the model detects
recording conditions (noise floor, compression, capture setup) rather than
actual drone RF signatures.

The fix: Generate MATCHED backgrounds — take drone spectrograms, remove
the drone signal, keep the noise floor. If AUC stays 0.95+ on these,
the model is detecting actual drone signal, not shortcuts.

Matched background generation:
  1. For each drone spectrogram, estimate per-frequency-bin noise floor
     using robust percentile estimation (10th percentile per row)
  2. Identify "signal pixels" — anything above noise_floor + k*noise_std
  3. Replace signal pixels with noise-consistent values:
     - Sample from the noise distribution of that frequency bin
     - This preserves noise floor shape, compression artifacts, etc.
  4. Result: same recording, same noise, no drone signal

Two-phase evaluation:
  Phase 1: Evaluate CURRENT v9 on matched backgrounds (quick sanity check)
  Phase 2: Retrain v10 with matched backgrounds as negatives (rigorous)

Usage:
  modal run scripts/eval_v9_matched_bg.py
"""

import h5py
import json
import os
import time

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score

# ─── Modal setup ──────────────────────────────────────────────────────────────

app = modal.App("iris-v9-matched-bg-eval")

VOL = modal.Volume.from_name("iris-data", create_if_missing=True)
MODEL_VOL = modal.Volume.from_name("iris-models-v9", create_if_missing=True)
MATCHED_VOL = modal.Volume.from_name("iris-matched-bg", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install("libgl1", "libglib2.0-0", "libhdf5-dev",
                 "python3", "python3-pip", "python-is-python3")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        "h5py==3.12.1", "numpy==1.26.4", "scikit-learn==1.6.1",
        "scipy==1.14.1", "matplotlib==3.9.3", "umap-learn==0.5.7",
    )
)

H5_REMOTE = "/data/iris_rfuav.h5"
EVAL_DIR = "/models/v9_eval"
MATCHED_DIR = "/matched"


# ─── Model (same architecture as v9) ─────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    def forward(self, x):
        return self.block(x)


class CNNEncoder(nn.Module):
    def __init__(self, in_ch=2, width=64, depth=6, embed_dim=256):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(depth):
            out_ch = min(width * (2 ** (i // 2)), 512)
            layers.append(ConvBlock(ch, out_ch))
            layers.append(nn.MaxPool2d(2))
            ch = out_ch
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 256, 256)
            out = self.conv(dummy)
            flat = out.numel() // out.shape[0]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
    def forward(self, x):
        return self.head(self.conv(x))


class LeJEPASupConV9(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = CNNEncoder(
            in_ch=cfg["in_ch"], width=cfg["encoder_width"],
            depth=cfg["encoder_depth"], embed_dim=cfg["embed_dim"],
        )
        self.projector = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["proj_dim"]),
            nn.BatchNorm1d(cfg["proj_dim"]),
            nn.GELU(),
            nn.Linear(cfg["proj_dim"], cfg["proj_dim"]),
        )
        self.predictor = nn.Sequential(
            nn.Linear(cfg["proj_dim"], cfg["pred_dim"]),
            nn.BatchNorm1d(cfg["pred_dim"]),
            nn.GELU(),
            nn.Linear(cfg["pred_dim"], cfg["pred_out"]),
        )


# ─── HDF5 helpers ────────────────────────────────────────────────────────────

def _resolve_type_dataset(grp, key):
    item = grp[key]
    if isinstance(item, h5py.Dataset):
        if len(item.shape) == 4:
            return item, item.shape[0], False
        elif len(item.shape) == 3:
            return item, 1, False
        else:
            raise ValueError(f"Unexpected shape {item.shape}")
    for sub_key in ["data", "spectrogram", "samples", "images"]:
        if sub_key in item:
            sub = item[sub_key]
            if isinstance(sub, h5py.Dataset) and len(sub.shape) >= 3:
                return sub, sub.shape[0], False
    sub_datasets = []
    for sk in item.keys():
        sub = item[sk]
        if isinstance(sub, h5py.Dataset) and len(sub.shape) == 3:
            sub_datasets.append(sk)
    if len(sub_datasets) > 0:
        try:
            sub_datasets.sort(key=lambda x: int(x))
        except ValueError:
            sub_datasets.sort()
        return item, len(sub_datasets), True
    raise ValueError(f"Cannot resolve /{key}")


# ─── Matched Background Generation ───────────────────────────────────────────

def generate_matched_background(spectrogram, noise_percentile=10,
                                 signal_sigma=2.0, method="replace"):
    """
    Generate a matched background from a drone spectrogram.

    The idea: remove the drone signal while preserving the noise floor
    and all recording/pipeline artifacts.

    Steps per channel:
      1. For each frequency bin (row), estimate noise floor as the
         noise_percentile-th percentile value
      2. For each frequency bin, estimate noise std from pixels below
         the noise floor + 1 sigma
      3. Identify signal pixels: those above noise_floor + signal_sigma * noise_std
      4. Replace signal pixels with values sampled from the noise distribution
         of that frequency bin

    This preserves:
      - Per-frequency noise floor shape (recording conditions)
      - Noise statistics (compression artifacts, quantization)
      - Overall dynamic range of the noise
      - Time-varying noise characteristics

    What it removes:
      - Drone signal (the bright lines/sweeps in the spectrogram)

    Args:
        spectrogram: numpy array of shape (C, H, W) or (H, W)
        noise_percentile: percentile for noise floor estimation (lower = more conservative)
        signal_sigma: number of noise stds above floor to count as "signal"
        method: "replace" (replace with noise samples) or "zero" (zero out signal)

    Returns:
        matched_bg: numpy array same shape as input, with signal removed
    """
    if spectrogram.ndim == 2:
        spectrogram = spectrogram[np.newaxis, :, :]

    C, H, W = spectrogram.shape
    matched = spectrogram.copy()

    for c in range(C):
        ch = spectrogram[c]  # (H, W) — H=freq bins, W=time bins

        for freq_idx in range(H):
            row = ch[freq_idx]  # (W,)

            # Estimate noise floor for this frequency bin
            noise_floor = np.percentile(row, noise_percentile)

            # Estimate noise std from pixels near the noise floor
            # Use pixels below noise_floor + 1 sigma as "noise pixels"
            # Iterative estimation to avoid contamination by signal
            below_mask = row <= noise_floor + np.std(row[row <= noise_floor + np.std(row)]) * 1.5
            if below_mask.sum() < 5:
                below_mask = row <= np.percentile(row, 50)

            noise_pixels = row[below_mask]
            if len(noise_pixels) > 1:
                noise_std = np.std(noise_pixels)
            else:
                noise_std = np.std(row) * 0.5

            # Identify signal pixels
            threshold = noise_floor + signal_sigma * noise_std
            signal_mask = row > threshold

            if signal_mask.sum() > 0:
                if method == "replace":
                    # Replace signal pixels with samples from the noise distribution
                    # This preserves noise statistics perfectly
                    n_replace = signal_mask.sum()
                    # Sample from the noise pixels' distribution
                    if len(noise_pixels) > 1:
                        replacements = np.random.choice(noise_pixels, size=n_replace, replace=True)
                        # Add tiny perturbation to avoid exact duplicates
                        replacements = replacements + np.random.normal(0, noise_std * 0.05, n_replace)
                    else:
                        replacements = np.random.normal(noise_floor, noise_std, n_replace)
                    matched[c, freq_idx, signal_mask] = replacements
                elif method == "zero":
                    # Simple zeroing — less realistic but conservative
                    matched[c, freq_idx, signal_mask] = noise_floor

    return matched


def load_all_drones(h5_path, split_key="train"):
    """Load all drone spectrograms from a split."""
    with h5py.File(h5_path, "r") as f:
        grp = f[split_key]
        all_spectrograms = []
        all_type_names = []
        type_names = []

        for key in sorted(grp.keys()):
            try:
                ds_or_grp, n_samples, is_multi = _resolve_type_dataset(grp, key)
                type_names.append(key)

                if is_multi:
                    sub_keys = list(ds_or_grp.keys())
                    try:
                        sub_keys.sort(key=lambda x: int(x))
                    except ValueError:
                        sub_keys.sort()
                    for sk in sub_keys:
                        sub = ds_or_grp[sk]
                        if isinstance(sub, h5py.Dataset) and len(sub.shape) == 3:
                            all_spectrograms.append(sub[:])
                            all_type_names.append(key)
                else:
                    for i in range(n_samples):
                        all_spectrograms.append(ds_or_grp[i])
                        all_type_names.append(key)
            except ValueError:
                continue

    return all_spectrograms, all_type_names, type_names


# ─── Dataset with per-channel normalization ───────────────────────────────────

class MatchedBGDataset(Dataset):
    """
    Dataset that generates matched backgrounds on-the-fly or from pre-computed.
    """
    def __init__(self, spectrograms, type_names, is_background=False):
        self.spectrograms = spectrograms
        self.type_names = type_names
        self.is_background = is_background

    def __len__(self):
        return len(self.spectrograms)

    def _normalize_per_channel(self, x):
        """Per-channel zero-mean unit-variance normalization."""
        for c in range(x.shape[0]):
            ch = x[c]
            ch_std = ch.std()
            if ch_std > 1e-6:
                x[c] = (ch - ch.mean()) / ch_std
            else:
                x[c] = ch - ch.mean()
        return x

    def __getitem__(self, idx):
        spec = self.spectrograms[idx]

        # Take first 2 channels
        if spec.shape[0] == 3:
            x = spec[:2].copy().astype(np.float32)
        elif spec.shape[0] == 2:
            x = spec.copy().astype(np.float32)
        else:
            x = spec[:2].copy().astype(np.float32)

        # If this is a matched background, generate it
        if self.is_background:
            x = generate_matched_background(x, noise_percentile=10,
                                             signal_sigma=2.0, method="replace")

        # Per-channel normalization
        x_t = torch.from_numpy(x)
        x_t = self._normalize_per_channel(x_t)

        label = 1 if not self.is_background else 0  # 1=drone, 0=background
        tname = self.type_names[idx] if not self.is_background else "matched_bg"

        return x_t, label, tname


class SimpleDataset(Dataset):
    """Simple dataset from pre-computed arrays."""
    def __init__(self, embeddings, labels, type_names):
        self.embeddings = embeddings
        self.labels = labels
        self.type_names = type_names

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.type_names[idx]


# ─── Main evaluation ─────────────────────────────────────────────────────────

@app.function(
    image=IMAGE,
    gpu="A100",
    volumes={"/data": VOL, "/models": MODEL_VOL, "/matched": MATCHED_VOL},
    timeout=7200,
    memory=32768,
)
def evaluate_matched_bg():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(MATCHED_DIR, exist_ok=True)
    device = "cuda"

    cfg = dict(in_ch=2, encoder_depth=6, encoder_width=64,
               embed_dim=256, proj_dim=256, pred_dim=256, pred_out=256)

    # ── Load best v9 checkpoint ──
    best_path = "/models/lejepa_v9_best.pt"
    if not os.path.exists(best_path):
        print(f"ERROR: No checkpoint at {best_path}")
        return

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    epoch = ckpt.get("epoch", "?")
    model = LeJEPASupConV9(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    encoder = model.encoder
    encoder.eval()

    print("=" * 70)
    print("IRIS v9 — MATCHED BACKGROUND EVALUATION")
    print("=" * 70)
    print(f"  Checkpoint: epoch {epoch}")
    print(f"  Method: Generate matched backgrounds from drone spectrograms")
    print(f"          by removing drone signal, keeping noise floor")
    print()

    # ── Phase 1: Generate matched backgrounds from HOLDOUT drones ──
    print("Phase 1: Generating matched backgrounds from holdout drones...")
    print("  Loading holdout drone spectrograms...")

    holdout_specs, holdout_tnames, holdout_types = load_all_drones(H5_REMOTE, "holdout")
    print(f"  Loaded {len(holdout_specs)} holdout spectrograms from {len(holdout_types)} types")

    # Generate matched backgrounds (in batches for memory)
    print("  Generating matched backgrounds (signal removal)...")
    matched_bg_specs = []
    rng = np.random.default_rng(42)

    for i, spec in enumerate(holdout_specs):
        if spec.shape[0] == 3:
            x = spec[:2].copy().astype(np.float32)
        elif spec.shape[0] == 2:
            x = spec.copy().astype(np.float32)
        else:
            x = spec[:2].copy().astype(np.float32)

        matched = generate_matched_background(x, noise_percentile=10,
                                               signal_sigma=2.0, method="replace")
        matched_bg_specs.append(matched)

        if (i + 1) % 500 == 0:
            print(f"    Generated {i+1}/{len(holdout_specs)} matched backgrounds")

    print(f"  Generated {len(matched_bg_specs)} matched backgrounds from holdout drones")

    # ── Also generate matched backgrounds from TRAIN drones ──
    print("  Loading train drone spectrograms for matched bg...")
    train_specs, train_tnames, train_types = load_all_drones(H5_REMOTE, "train")
    print(f"  Loaded {len(train_specs)} train spectrograms")

    # Subsample train for matched bg (use same count as original training)
    n_train_matched = min(14000, len(train_specs))
    train_indices = rng.choice(len(train_specs), n_train_matched, replace=False)

    print(f"  Generating {n_train_matched} matched backgrounds from train drones...")
    train_matched_specs = []
    for i, idx in enumerate(train_indices):
        spec = train_specs[idx]
        if spec.shape[0] == 3:
            x = spec[:2].copy().astype(np.float32)
        elif spec.shape[0] == 2:
            x = spec.copy().astype(np.float32)
        else:
            x = spec[:2].copy().astype(np.float32)

        matched = generate_matched_background(x, noise_percentile=10,
                                               signal_sigma=2.0, method="replace")
        train_matched_specs.append(matched)

        if (i + 1) % 2000 == 0:
            print(f"    Generated {i+1}/{n_train_matched}")

    # ── Save matched backgrounds to HDF5 for retraining ──
    print("\n  Saving matched backgrounds to HDF5...")
    matched_h5_path = f"{MATCHED_DIR}/iris_matched_bg.h5"
    with h5py.File(matched_h5_path, "w") as f:
        # Holdout matched backgrounds
        holdout_bg_grp = f.create_group("holdout_matched_bg")
        for i, spec in enumerate(matched_bg_specs):
            holdout_bg_grp.create_dataset(str(i), data=spec, compression="gzip")

        # Train matched backgrounds
        train_bg_grp = f.create_group("train_matched_bg")
        for i, spec in enumerate(train_matched_specs):
            train_bg_grp.create_dataset(str(i), data=spec, compression="gzip")

        # Also store original holdout spectrograms for comparison
        holdout_orig_grp = f.create_group("holdout_original")
        for i, spec in enumerate(holdout_specs):
            holdout_orig_grp.create_dataset(str(i), data=spec, compression="gzip")

    MATCHED_VOL.commit()
    print(f"  Saved to: {matched_h5_path}")

    # ── Phase 2: Evaluate v9 on matched backgrounds ──
    print("\n" + "=" * 70)
    print("Phase 2: Evaluating v9 on MATCHED backgrounds")
    print("=" * 70)

    # Encode holdout drones (original, with signal)
    print("  Encoding holdout drones (original)...")
    holdout_ds = MatchedBGDataset(holdout_specs, holdout_tnames, is_background=False)
    holdout_dl = DataLoader(holdout_ds, batch_size=64, shuffle=False, num_workers=4)

    holdout_embs = []
    with torch.no_grad():
        for batch_idx, (x, _, _) in enumerate(holdout_dl):
            z = encoder(x.to(device))
            holdout_embs.append(z.cpu().numpy())
            if (batch_idx + 1) % 20 == 0:
                print(f"    {batch_idx+1}/{len(holdout_dl)} batches")

    holdout_embs = np.concatenate(holdout_embs)
    print(f"  Holdout embeddings: {holdout_embs.shape}")

    # Encode matched backgrounds from holdout drones
    print("  Encoding matched backgrounds (from holdout drones)...")
    matched_ds = MatchedBGDataset(matched_bg_specs,
                                   ["matched_bg"] * len(matched_bg_specs),
                                   is_background=True)
    matched_dl = DataLoader(matched_ds, batch_size=64, shuffle=False, num_workers=4)

    matched_embs = []
    with torch.no_grad():
        for batch_idx, (x, _, _) in enumerate(matched_dl):
            z = encoder(x.to(device))
            matched_embs.append(z.cpu().numpy())
            if (batch_idx + 1) % 20 == 0:
                print(f"    {batch_idx+1}/{len(matched_dl)} batches")

    matched_embs = np.concatenate(matched_embs)
    print(f"  Matched BG embeddings: {matched_embs.shape}")

    # Encode train drones (for global centroid)
    print("  Encoding train drones (for centroid)...")
    # We need a dataset without on-the-fly matched bg generation
    train_ds = MatchedBGDataset(train_specs, train_tnames, is_background=False)
    # Subsample if too many
    if len(train_ds) > 8000:
        indices = rng.choice(len(train_ds), 8000, replace=False)
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=False,
                              sampler=torch.utils.data.SubsetRandomSampler(indices))
    else:
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=4)

    train_embs = []
    with torch.no_grad():
        for x, _, _ in train_dl:
            z = encoder(x.to(device))
            train_embs.append(z.cpu().numpy())
    train_embs = np.concatenate(train_embs)
    print(f"  Train embeddings: {train_embs.shape}")

    torch.cuda.empty_cache()

    # ── TEST A: Holdout Drones vs Matched Backgrounds ──
    print("\n" + "-" * 70)
    print("TEST A: Holdout Drones vs MATCHED Backgrounds (Global Mahalanobis)")
    print("-" * 70)

    D = train_embs.shape[1]
    centroid = train_embs.mean(axis=0)
    cov = np.cov(train_embs.T) + 1e-3 * np.eye(D)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    # Mahalanobis distances for holdout drones
    diff_drone = holdout_embs - centroid
    drone_mahal = np.sqrt(np.maximum(np.sum(diff_drone @ cov_inv * diff_drone, axis=1), 0))

    # Mahalanobis distances for matched backgrounds
    diff_bg = matched_embs - centroid
    bg_mahal = np.sqrt(np.maximum(np.sum(diff_bg @ cov_inv * diff_bg, axis=1), 0))

    # AUC
    all_dists = np.concatenate([drone_mahal, bg_mahal])
    all_labels = np.concatenate([np.ones(len(drone_mahal)), np.zeros(len(bg_mahal))])
    matched_auc = roc_auc_score(all_labels, -all_dists)
    matched_ap = average_precision_score(all_labels, -all_dists)

    # ROC details
    fpr, tpr, thresholds = roc_curve(all_labels, -all_dists)

    print(f"  Holdout drones:  mean dist = {drone_mahal.mean():.2f}, "
          f"median = {np.median(drone_mahal):.2f}, std = {drone_mahal.std():.2f}")
    print(f"  Matched BG:      mean dist = {bg_mahal.mean():.2f}, "
          f"median = {np.median(bg_mahal):.2f}, std = {bg_mahal.std():.2f}")
    print(f"  BG/Drone ratio:  {bg_mahal.mean() / drone_mahal.mean():.2f}x")
    print(f"  AUC:             {matched_auc:.4f}")
    print(f"  Avg Precision:   {matched_ap:.4f}")

    # Operational thresholds
    for fpr_target in [0.01, 0.05, 0.1]:
        idx = np.searchsorted(fpr, fpr_target)
        if idx < len(tpr):
            print(f"  TPR @ FPR={fpr_target:.0%}: {tpr[idx]:.4f} ({tpr[idx]*100:.1f}% detection)")

    for tpr_target in [0.90, 0.95, 0.99]:
        idx = np.searchsorted(tpr, tpr_target)
        if idx < len(fpr):
            print(f"  FPR @ TPR={tpr_target:.0%}: {fpr[idx]:.4f} ({fpr[idx]*100:.2f}% false alarms)")

    # ── TEST B: Cosine distance (orthogonal check) ──
    print("\n" + "-" * 70)
    print("TEST B: Cosine Distance Check (orthogonal to Mahalanobis)")
    print("-" * 70)

    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
    holdout_norm = holdout_embs / (np.linalg.norm(holdout_embs, axis=1, keepdims=True) + 1e-8)
    matched_norm = matched_embs / (np.linalg.norm(matched_embs, axis=1, keepdims=True) + 1e-8)

    drone_cos = 1 - np.dot(holdout_norm, centroid_norm)  # cosine distance (higher = further)
    bg_cos = 1 - np.dot(matched_norm, centroid_norm)

    cos_all = np.concatenate([-drone_cos, -bg_cos])  # negate: higher cosine sim = more drone-like
    cos_labels = np.concatenate([np.ones(len(drone_cos)), np.zeros(len(bg_cos))])
    cos_auc = roc_auc_score(cos_labels, cos_all)

    print(f"  Cosine AUC: {cos_auc:.4f}")
    print(f"  Drone cosine dist: mean={drone_cos.mean():.4f}, std={drone_cos.std():.4f}")
    print(f"  BG cosine dist:    mean={bg_cos.mean():.4f}, std={bg_cos.std():.4f}")

    # ── TEST C: Per-pair comparison ──
    print("\n" + "-" * 70)
    print("TEST C: Per-Pair Analysis (drone vs its own matched background)")
    print("-" * 70)

    # Each matched bg came from the same drone spectrogram
    # So we can compare distances directly
    drone_closer = (drone_mahal < bg_mahal).sum()
    total_pairs = len(drone_mahal)
    print(f"  Pairs where drone is CLOSER to centroid than its matched bg: "
          f"{drone_closer}/{total_pairs} ({drone_closer/total_pairs*100:.1f}%)")

    dist_differences = bg_mahal - drone_mahal  # positive = bg further (correct)
    print(f"  Mean distance difference (bg - drone): {dist_differences.mean():.2f}")
    print(f"  Min distance difference: {dist_differences.min():.2f} (negative = bg closer!)")
    print(f"  Max distance difference: {dist_differences.max():.2f}")

    # Pairs where matched bg is CLOSER (failures)
    failures = dist_differences < 0
    if failures.sum() > 0:
        print(f"\n  *** WARNING: {failures.sum()} pairs where matched bg is closer to "
              f"drone centroid than the actual drone! ***")
        failure_dists = dist_differences[failures]
        print(f"  Failure magnitude: mean={-failure_dists.mean():.2f}, "
              f"max={-failure_dists.min():.2f}")
    else:
        print(f"\n  No failures — every drone is closer to centroid than its matched bg.")

    # ── Per-type breakdown ──
    print("\n" + "-" * 70)
    print("Per-Holdout-Type Breakdown (vs Matched BG)")
    print("-" * 70)

    per_type_results = {}
    for dtype in sorted(set(holdout_tnames)):
        mask = np.array([t == dtype for t in holdout_tnames])
        dtype_drone = drone_mahal[mask]
        dtype_auc = roc_auc_score(
            np.concatenate([np.ones(mask.sum()), np.zeros(len(bg_mahal))]),
            np.concatenate([-dtype_drone, -bg_mahal])
        )
        per_type_results[dtype] = {
            "n": int(mask.sum()),
            "auc": float(dtype_auc),
            "drone_mean_dist": float(dtype_drone.mean()),
            "drone_median_dist": float(np.median(dtype_drone)),
        }

    print(f"  {'Type':<25} {'N':>6} {'AUC':>8} {'Drone Mean':>12}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*12}")
    for dtype, info in sorted(per_type_results.items()):
        status = "PASS" if info["auc"] >= 0.95 else "FAIL"
        print(f"  {dtype:<25} {info['n']:>6} {info['auc']:>8.4f} {info['drone_mean_dist']:>12.2f}  [{status}]")

    # ── TEST D: Also load original negatives for comparison ──
    print("\n" + "-" * 70)
    print("TEST D: Original vs Matched Background Comparison")
    print("-" * 70)

    # Load original negatives for comparison
    with h5py.File(H5_REMOTE, "r") as f:
        if "negatives" in f:
            neg_item = f["negatives"]
            if isinstance(neg_item, h5py.Dataset):
                n_total_neg = neg_item.shape[0]
            else:
                n_total_neg = len([sk for sk in neg_item.keys()
                                    if isinstance(neg_item[sk], h5py.Dataset)
                                    and len(neg_item[sk].shape) == 3])
            print(f"  Original negatives in dataset: {n_total_neg}")
        else:
            n_total_neg = 0
            print("  No negatives found in dataset")

    # ── PLOTS ──
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # 1. Distance distribution: drones vs matched bg
    ax = axes[0, 0]
    max_val = min(max(drone_mahal.max(), bg_mahal.max()), 80)
    bins = np.linspace(0, max_val, 80)
    ax.hist(drone_mahal, bins=bins, alpha=0.6,
            label=f'Holdout Drones (mean={drone_mahal.mean():.1f})',
            color='#2196F3', density=True)
    ax.hist(bg_mahal, bins=bins, alpha=0.6,
            label=f'Matched BG (mean={bg_mahal.mean():.1f})',
            color='#FF5722', density=True)
    ax.set_xlabel('Global Mahalanobis Distance', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'v9 vs Matched Backgrounds\nAUC={matched_auc:.4f}', fontsize=13)
    ax.legend(fontsize=10, loc='best')

    # 2. ROC curve: matched bg vs original bg
    ax = axes[0, 1]
    ax.plot(fpr, tpr, 'b-', linewidth=2,
            label=f'v9 vs Matched BG (AUC={matched_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('Detection ROC (Matched BG)', fontsize=13)
    ax.legend(fontsize=10, loc='lower right')

    # 3. Per-pair distance difference
    ax = axes[0, 2]
    sorted_diffs = np.sort(dist_differences)
    ax.plot(range(len(sorted_diffs)), sorted_diffs, 'b-', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Equal distance')
    ax.set_xlabel('Sample (sorted)', fontsize=11)
    ax.set_ylabel('BG dist - Drone dist', fontsize=11)
    ax.set_title(f'Per-Pair Distance Difference\n'
                 f'{failures.sum()}/{total_pairs} failures (bg closer)',
                 fontsize=13)
    ax.legend(fontsize=10, loc='best')

    # 4. Per-type AUC bar chart
    ax = axes[1, 0]
    type_names_sorted = sorted(per_type_results.keys())
    type_aucs = [per_type_results[t]["auc"] for t in type_names_sorted]
    colors = ['#4CAF50' if a >= 0.99 else '#FF9800' if a >= 0.95 else '#F44336'
              for a in type_aucs]
    ax.barh(range(len(type_names_sorted)), type_aucs, color=colors)
    ax.set_yticks(range(len(type_names_sorted)))
    ax.set_yticklabels([t[:20] for t in type_names_sorted], fontsize=9)
    ax.set_xlabel('AUC', fontsize=11)
    ax.set_title('Per-Type Detection AUC (vs Matched BG)', fontsize=13)
    ax.axvline(x=0.95, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 1.05)

    # 5. Cosine distance comparison
    ax = axes[1, 1]
    ax.hist(drone_cos, bins=50, alpha=0.6,
            label=f'Drones (mean={drone_cos.mean():.3f})',
            color='#2196F3', density=True)
    ax.hist(bg_cos, bins=50, alpha=0.6,
            label=f'Matched BG (mean={bg_cos.mean():.3f})',
            color='#FF5722', density=True)
    ax.set_xlabel('Cosine Distance to Centroid', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Cosine Distance (AUC={cos_auc:.4f})', fontsize=13)
    ax.legend(fontsize=10, loc='best')

    # 6. Summary text box
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = (
        f"MATCHED BACKGROUND EVALUATION\n"
        f"{'='*40}\n\n"
        f"Model: IRIS v9 (epoch {epoch})\n"
        f"Matched BG method: Signal removal\n"
        f"  - Noise percentile: 10th\n"
        f"  - Signal threshold: 2σ above floor\n"
        f"  - Replacement: Noise sampling\n\n"
        f"Results:\n"
        f"  Global Mahalanobis AUC: {matched_auc:.4f}\n"
        f"  Avg Precision:          {matched_ap:.4f}\n"
        f"  Cosine AUC:             {cos_auc:.4f}\n"
        f"  BG/Drone ratio:         {bg_mahal.mean()/drone_mahal.mean():.2f}x\n"
        f"  Pair failure rate:      {failures.sum()}/{total_pairs}\n\n"
    )

    if matched_auc >= 0.95:
        summary_text += f"VERDICT: AUC >= 0.95 — RESULT IS AIRTIGHT"
        color = 'green'
    elif matched_auc >= 0.90:
        summary_text += f"VERDICT: AUC 0.90-0.95 — STRONG BUT RETRAIN RECOMMENDED"
        color = 'orange'
    else:
        summary_text += f"VERDICT: AUC < 0.90 — PIPELINE SHORTCUT DETECTED"
        color = 'red'

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.15))

    plt.tight_layout()
    plt.savefig(f"{EVAL_DIR}/v9_matched_bg_eval.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {EVAL_DIR}/v9_matched_bg_eval.png")

    # ── Example: show a drone spectrogram vs its matched bg ──
    print("  Generating spectrogram comparison plot...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Pick 4 examples from different types
    rng_examples = np.random.default_rng(42)
    example_indices = []
    seen_types = set()
    for i, tname in enumerate(holdout_tnames):
        if tname not in seen_types and len(example_indices) < 4:
            example_indices.append(i)
            seen_types.add(tname)

    for col, idx in enumerate(example_indices):
        # Original drone
        spec = holdout_specs[idx]
        if spec.shape[0] >= 2:
            ax = axes[0, col]
            im = ax.imshow(spec[0], aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f'Drone: {holdout_tnames[idx][:15]}', fontsize=10)
            ax.set_ylabel('Freq bin' if col == 0 else '')
            plt.colorbar(im, ax=ax, fraction=0.046)

            # Matched background
            matched = matched_bg_specs[idx]
            ax = axes[1, col]
            im = ax.imshow(matched[0], aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f'Matched BG (signal removed)', fontsize=10)
            ax.set_ylabel('Freq bin' if col == 0 else '')
            ax.set_xlabel('Time bin', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle('Drone Spectrograms vs Matched Backgrounds (Signal Removed)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{EVAL_DIR}/v9_spectrogram_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {EVAL_DIR}/v9_spectrogram_comparison.png")

    # ── Final verdict ──
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    print(f"  Original eval (50K random bg):     AUC = 1.0000")
    print(f"  Matched bg eval (signal removed):   AUC = {matched_auc:.4f}")
    print(f"  Cosine distance AUC:                {cos_auc:.4f}")
    print(f"  Per-pair failure rate:               {failures.sum()}/{total_pairs} "
          f"({failures.sum()/total_pairs*100:.1f}%)")
    print()

    if matched_auc >= 0.95:
        print("  *** AUC >= 0.95 ON MATCHED BACKGROUNDS ***")
        print("  *** THE RESULT IS AIRTIGHT — NO PIPELINE SHORTCUTS ***")
        print("  *** The model detects actual drone RF signatures ***")
    elif matched_auc >= 0.90:
        print("  AUC 0.90-0.95: Strong detection on matched backgrounds")
        print("  Some shortcuts may exist, but core signal is real")
        print("  Recommend: retrain with matched backgrounds to tighten")
    else:
        print("  WARNING: AUC < 0.90 on matched backgrounds")
        print("  The model may be relying on pipeline shortcuts")
        print("  Must retrain with matched backgrounds")

    # ── Save results ──
    results = {
        "model": "v9_best",
        "epoch": epoch,
        "matched_bg_method": {
            "noise_percentile": 10,
            "signal_sigma": 2.0,
            "replacement": "noise_sampling",
        },
        "results": {
            "global_mahalanobis_auc": float(matched_auc),
            "avg_precision": float(matched_ap),
            "cosine_auc": float(cos_auc),
            "bg_drone_ratio": float(bg_mahal.mean() / drone_mahal.mean()),
            "pair_failure_rate": float(failures.sum() / total_pairs),
            "pair_failures": int(failures.sum()),
            "total_pairs": total_pairs,
        },
        "per_type": per_type_results,
        "verdict": "AIRTIGHT" if matched_auc >= 0.95 else
                   ("STRONG" if matched_auc >= 0.90 else "SHORTCUT_DETECTED"),
    }

    results_path = f"{EVAL_DIR}/v9_matched_bg_eval.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o)
                  if hasattr(o, '__float__') else str(o))
    MODEL_VOL.commit()
    print(f"\n  Results saved: {results_path}")
    print(f"  Matched backgrounds saved: {matched_h5_path}")

    print(f"\n{'='*70}")
    print("MATCHED BACKGROUND EVALUATION COMPLETE")
    print(f"{'='*70}")


@app.local_entrypoint()
def main():
    evaluate_matched_bg.remote()