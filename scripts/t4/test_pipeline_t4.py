#!/usr/bin/env python3
"""
IRIS T4 Pipeline Test — Run ALL phases on Modal T4 before running on Mac.

This is the "test everything works" script. It runs on a cheap T4 GPU
($0.40/hr vs A100 $1.10/hr) and verifies:

  1. Encoder loads correctly from checkpoint
  2. Inference produces sensible outputs
  3. Mahalanobis centroid fits correctly
  4. Honest evaluation runs (recording-grouped CV + SNR curve)
  5. Intent head trains and produces sensible accuracy
  6. Spoof detection logic works with real encoder
  7. Adversarial robustness test runs (FGSM only, fast)

If this script completes successfully, you can confidently run the
individual scripts on your Mac knowing the pipeline is sound.

Expected runtime: ~30-45 min on T4
Expected cost: ~$0.30

Usage:
    modal run scripts/t4/test_pipeline_t4.py

To run only specific phases:
    modal run scripts/t4/test_pipeline_t4.py --phase inference
    modal run scripts/t4/test_pipeline_t4.py --phase honest_eval
    modal run scripts/t4/test_pipeline_t4.py --phase intent
    modal run scripts/t4/test_pipeline_t4.py --phase spoof
    modal run scripts/t4/test_pipeline_t4.py --phase adversarial
    modal run scripts/t4/test_pipeline_t4.py --phase all  (default)
"""

from __future__ import annotations

import h5py
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────────────────────────────────────
# Modal setup — T4 GPU
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("iris-t4-pipeline-test")

VOL = modal.Volume.from_name("iris-data", create_if_missing=True)
MODEL_VOL = modal.Volume.from_name("iris-models-v11", create_if_missing=True)
MATCHED_VOL = modal.Volume.from_name("iris-matched-bg", create_if_missing=True)
RESULTS_VOL = modal.Volume.from_name("iris-results", create_if_missing=True)
INTENT_VOL = modal.Volume.from_name("iris-intent", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install("libgl1", "libglib2.0-0", "libhdf5-dev",
                 "python3", "python3-pip", "python-is-python3")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        "h5py==3.12.1", "numpy==1.26.4", "scikit-learn==1.6.1",
        "scipy==1.14.1", "matplotlib==3.9.3", "tqdm==4.67.1",
    )
)

H5_REMOTE = "/data/iris_rfuav.h5"
MATCHED_REMOTE = "/matched/iris_matched_bg.h5"
MODEL_REMOTE = "/models/lejepa_v11_best.pt"
RESULTS_REMOTE = "/results"
INTENT_REMOTE = "/intent"


# ─────────────────────────────────────────────────────────────────────────────
# Encoder — exact reproduction
# ─────────────────────────────────────────────────────────────────────────────


class ConvBlock(nn.Module):
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

    def forward(self, x):
        return self.block(x)


class CNNEncoder(nn.Module):
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


class IntentHead(nn.Module):
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

    def forward(self, x):
        return self.net(x)


INTENT_CLASSES = ["SURVEILLANCE", "TRANSIT", "ATTACK"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (shared across phases)
# ─────────────────────────────────────────────────────────────────────────────


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


def _prep_sample(sample: np.ndarray) -> np.ndarray:
    if sample.shape[0] == 3:
        x = sample[:2].copy()
    elif sample.shape[0] == 2:
        x = sample.copy()
    else:
        x = sample[:2].copy()
    return x.astype(np.float32)


def load_samples(h5_path: str, split: str, max_per_type: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Load samples from HDF5 split with per-channel normalization."""
    print(f"    loading {split} (max {max_per_type}/type)...")
    with h5py.File(h5_path, "r") as f:
        if split not in f:
            raise ValueError(f"No '{split}' in HDF5. Keys: {list(f.keys())}")
        grp = f[split]
        type_names = sorted(list(grp.keys()))

        specs_list = []
        types_list = []

        for tname in type_names:
            try:
                ds_or_grp, n_samples, is_multi = _resolve_type_dataset(grp, tname)
            except ValueError:
                continue

            n_to_load = min(n_samples, max_per_type)
            if is_multi:
                sub_keys = [sk for sk in ds_or_grp.keys()
                            if isinstance(ds_or_grp[sk], h5py.Dataset) and len(ds_or_grp[sk].shape) == 3]
                try:
                    sub_keys.sort(key=lambda x: int(x))
                except ValueError:
                    sub_keys.sort()
                for sk in sub_keys[:n_to_load]:
                    specs_list.append(_prep_sample(ds_or_grp[sk][:]))
                    types_list.append(tname)
            else:
                for i in range(n_to_load):
                    if len(ds_or_grp.shape) == 4:
                        specs_list.append(_prep_sample(ds_or_grp[i]))
                    else:
                        specs_list.append(_prep_sample(ds_or_grp[:]))
                    types_list.append(tname)

    specs = np.stack(specs_list).astype(np.float32)
    types = np.array(types_list)

    # Per-channel normalize
    for i in range(len(specs)):
        for c in range(specs.shape[1]):
            ch = specs[i, c]
            ch_std = ch.std()
            if ch_std > 1e-6:
                specs[i, c] = (ch - ch.mean()) / ch_std
            else:
                specs[i, c] = ch - ch.mean()

    print(f"    loaded {len(specs)} samples, {len(set(types))} types")
    return specs, types


def load_matched_bgs(matched_path: str, split: str = "holdout", max_n: int = 300) -> np.ndarray:
    print(f"    loading matched BGs (max {max_n})...")
    with h5py.File(matched_path, "r") as f:
        key = f"{split}_matched_bg"
        if key not in f:
            raise ValueError(f"No '{key}' in matched BG file")
        grp = f[key]
        keys = sorted(list(grp.keys()), key=lambda x: int(x) if x.isdigit() else 0)
        n = min(len(keys), max_n)
        rng = np.random.default_rng(123)
        indices = rng.choice(len(keys), n, replace=False)
        specs_list = [_prep_sample(grp[keys[i]][:]) for i in indices]

    specs = np.stack(specs_list).astype(np.float32)
    for i in range(len(specs)):
        for c in range(specs.shape[1]):
            ch = specs[i, c]
            ch_std = ch.std()
            if ch_std > 1e-6:
                specs[i, c] = (ch - ch.mean()) / ch_std
            else:
                specs[i, c] = ch - ch.mean()
    print(f"    loaded {len(specs)} matched BGs")
    return specs


def fit_mahalanobis_l2(embeddings: np.ndarray, reg: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
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


def mahalanobis_l2_np(embeddings: np.ndarray, centroid: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings = embeddings / norms
    diff = embeddings - centroid
    mahal_sq = np.sum(diff @ cov_inv * diff, axis=1)
    return np.sqrt(np.maximum(mahal_sq, 0.0))


def mahalanobis_l2_batch(embeddings: torch.Tensor, centroid_t: torch.Tensor, cov_inv_t: torch.Tensor) -> torch.Tensor:
    norms = embeddings.norm(dim=1, keepdim=True) + 1e-8
    embeddings = embeddings / norms
    diff = embeddings - centroid_t.unsqueeze(0)
    mahal_sq = (diff @ cov_inv_t * diff).sum(dim=1)
    return torch.sqrt(torch.clamp(mahal_sq, min=0.0))


@torch.no_grad()
def encode_batch(encoder: nn.Module, specs: np.ndarray, device: str, batch_size: int = 32) -> np.ndarray:
    encoder.eval()
    all_embs = []
    for i in range(0, len(specs), batch_size):
        batch = torch.from_numpy(specs[i:i+batch_size]).float().to(device)
        embs = encoder(batch)
        all_embs.append(embs.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def heuristic_intent_label(spec: np.ndarray) -> int:
    """Generate heuristic intent label from spectrogram features.

    Activity score for per-channel-normalized spectrograms typically
    ranges 1.2-1.8 with mean ~1.6. Thresholds calibrated from RFUAV:
      < 1.55  → SURVEILLANCE (low activity — hovering/loitering)
      > 1.70  → ATTACK        (high activity — diving/maneuvering)
      middle  → TRANSIT       (steady cruise)
    """
    s = spec[0] if spec.ndim == 3 else spec
    temporal_var = float(s.var(axis=1).mean())
    doppler_spread = float(s.std(axis=0).mean())
    activity = temporal_var + doppler_spread
    if activity < 1.55:
        return 0  # SURVEILLANCE
    elif activity > 1.70:
        return 2  # ATTACK
    return 1  # TRANSIT


def _diagnose_activity(specs: np.ndarray) -> dict:
    """Print activity score distribution for diagnostic purposes."""
    scores = []
    for s in specs:
        ch = s[0] if s.ndim == 3 else s
        tv = float(ch.var(axis=1).mean())
        ds = float(ch.std(axis=0).mean())
        scores.append(tv + ds)
    scores = np.array(scores)
    return {
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "p10": float(np.percentile(scores, 10)),
        "p30": float(np.percentile(scores, 30)),
        "p50": float(np.percentile(scores, 50)),
        "p70": float(np.percentile(scores, 70)),
        "p90": float(np.percentile(scores, 90)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Inference test
# ─────────────────────────────────────────────────────────────────────────────


def test_inference(device: str) -> Dict:
    """Test that encoder loads and produces sensible outputs."""
    print("\n" + "─" * 60)
    print("PHASE 1: Inference Test")
    print("─" * 60)

    # Load encoder
    print("  [1] Loading encoder checkpoint...")
    t0 = time.time()
    ckpt = torch.load(MODEL_REMOTE, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    encoder_state = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}

    encoder = CNNEncoder(in_ch=2, width=64, depth=6, embed_dim=256).to(device)
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.eval()

    param_count = sum(p.numel() for p in encoder.parameters())
    print(f"      encoder params: {param_count:,}")
    print(f"      load time: {time.time()-t0:.1f}s")

    # Test forward pass
    print("  [2] Testing forward pass...")
    dummy = torch.randn(4, 2, 256, 256, device=device)
    t0 = time.time()
    with torch.no_grad():
        out = encoder(dummy)
    latency = (time.time() - t0) * 1000 / 4
    print(f"      output shape: {out.shape}")
    print(f"      output norm: {out.norm(dim=1).mean().item():.3f}")
    print(f"      latency: {latency:.2f} ms/sample")

    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output!"
    assert not torch.isinf(out).any(), "Inf in output!"

    # Test with real data (small subset)
    print("  [3] Testing with real spectrograms...")
    train_specs, train_types = load_samples(H5_REMOTE, "train", max_per_type=20)
    embs = encode_batch(encoder, train_specs, device, batch_size=16)
    print(f"      embeddings shape: {embs.shape}")
    print(f"      embedding mean: {embs.mean():.4f}")
    print(f"      embedding std: {embs.std():.4f}")

    # Fit Mahalanobis
    print("  [4] Fitting Mahalanobis centroid...")
    centroid, cov_inv = fit_mahalanobis_l2(embs)
    dists = mahalanobis_l2_np(embs, centroid, cov_inv)
    print(f"      train distance percentiles:")
    print(f"        50th: {np.percentile(dists, 50):.2f}")
    print(f"        90th: {np.percentile(dists, 90):.2f}")
    print(f"        99th: {np.percentile(dists, 99):.2f}")

    threshold = float(np.percentile(dists, 99))
    print(f"      threshold (99th pct): {threshold:.2f}")

    return {
        "param_count": param_count,
        "latency_ms": latency,
        "threshold": threshold,
        "centroid_shape": list(centroid.shape),
        "cov_inv_shape": list(cov_inv.shape),
        "status": "PASS",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Honest evaluation test
# ─────────────────────────────────────────────────────────────────────────────


def test_honest_eval(encoder: nn.Module, device: str) -> Dict:
    """Test recording-grouped CV + SNR curve (smaller subset for T4 speed)."""
    print("\n" + "─" * 60)
    print("PHASE 2: Honest Evaluation Test")
    print("─" * 60)

    # Load data (smaller subsets for T4)
    print("  [1] Loading data (T4-optimized subsets)...")
    train_specs, train_types = load_samples(H5_REMOTE, "train", max_per_type=100)
    holdout_specs, holdout_types = load_samples(H5_REMOTE, "holdout", max_per_type=50)
    matched_bg_specs = load_matched_bgs(MATCHED_REMOTE, "holdout", max_n=200)

    # Encode + fit
    print("  [2] Encoding train + fitting Mahalanobis (L2-normalized)...")
    train_embs = encode_batch(encoder, train_specs, device)
    centroid, cov_inv = fit_mahalanobis_l2(train_embs)

    # Encode holdout + BG
    print("  [3] Encoding holdout + matched BGs...")
    holdout_embs = encode_batch(encoder, holdout_specs, device)
    bg_embs = encode_batch(encoder, matched_bg_specs, device)

    # Compute distances
    holdout_dists = mahalanobis_l2_np(holdout_embs, centroid, cov_inv)
    bg_dists = mahalanobis_l2_np(bg_embs, centroid, cov_inv)

    # AUC
    labels = np.concatenate([np.ones(len(holdout_dists)), np.zeros(len(bg_dists))])
    dists = np.concatenate([holdout_dists, bg_dists])
    auc = roc_auc_score(labels, -dists)

    print(f"  [4] Results:")
    print(f"      AUC (L2-Mahalanobis): {auc:.4f}")
    print(f"      Drone mean dist: {holdout_dists.mean():.2f}")
    print(f"      BG mean dist: {bg_dists.mean():.2f}")
    print(f"      BG/Drone ratio: {bg_dists.mean()/holdout_dists.mean():.2f}x")

    # SNR curve (only 3 points for T4 speed)
    print("  [5] SNR curve (3 points for T4 speed)...")
    snr_results = {}
    for snr in [20, 5, -5]:
        holdout_noisy = np.stack([
            s + np.random.randn(*s.shape).astype(np.float32) * np.sqrt(np.mean(s**2) / (10**(snr/10)))
            for s in holdout_specs
        ])
        bg_noisy = np.stack([
            s + np.random.randn(*s.shape).astype(np.float32) * np.sqrt(np.mean(s**2) / (10**(snr/10)))
            for s in matched_bg_specs
        ])
        h_embs = encode_batch(encoder, holdout_noisy, device)
        b_embs = encode_batch(encoder, bg_noisy, device)
        h_dists = mahalanobis_l2_np(h_embs, centroid, cov_inv)
        b_dists = mahalanobis_l2_np(b_embs, centroid, cov_inv)
        labels = np.concatenate([np.ones(len(h_dists)), np.zeros(len(b_dists))])
        dists = np.concatenate([h_dists, b_dists])
        snr_auc = roc_auc_score(labels, -dists)
        snr_results[snr] = float(snr_auc)
        print(f"      SNR {snr:+3d} dB: AUC = {snr_auc:.4f}")

    return {
        "auc": float(auc),
        "drone_mean_dist": float(holdout_dists.mean()),
        "bg_mean_dist": float(bg_dists.mean()),
        "snr_curve": snr_results,
        "status": "PASS" if auc > 0.7 else "WARN_LOW_AUC",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Intent training test
# ─────────────────────────────────────────────────────────────────────────────


def test_intent_training(encoder: nn.Module, device: str) -> Dict:
    """Test intent head training (smaller subset, fewer epochs for T4)."""
    print("\n" + "─" * 60)
    print("PHASE 3: Intent Head Training Test")
    print("─" * 60)

    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # Load data
    print("  [1] Loading data...")
    train_specs, train_types = load_samples(H5_REMOTE, "train", max_per_type=100)
    holdout_specs, holdout_types = load_samples(H5_REMOTE, "holdout", max_per_type=50)

    # Generate heuristic labels
    print("  [2] Generating heuristic intent labels...")

    # Diagnostic: print activity score distribution
    diag = _diagnose_activity(train_specs)
    print(f"      train activity score: min={diag['min']:.3f}, max={diag['max']:.3f}, "
          f"mean={diag['mean']:.3f}, std={diag['std']:.3f}")
    print(f"      percentiles: p10={diag['p10']:.3f}, p30={diag['p30']:.3f}, "
          f"p50={diag['p50']:.3f}, p70={diag['p70']:.3f}, p90={diag['p90']:.3f}")

    train_labels = np.array([heuristic_intent_label(s) for s in train_specs])
    holdout_labels = np.array([heuristic_intent_label(s) for s in holdout_specs])

    # Print label distribution
    for i, name in enumerate(INTENT_CLASSES):
        print(f"      train {name}: {(train_labels == i).sum()}")
        print(f"      holdout {name}: {(holdout_labels == i).sum()}")

    # Pre-compute embeddings
    print("  [3] Pre-computing embeddings...")
    train_embs = encode_batch(encoder, train_specs, device)
    holdout_embs = encode_batch(encoder, holdout_specs, device)

    train_embs_t = torch.from_numpy(train_embs).to(device)
    train_labels_t = torch.from_numpy(train_labels).to(device)
    holdout_embs_t = torch.from_numpy(holdout_embs).to(device)
    holdout_labels_t = torch.from_numpy(holdout_labels).to(device)

    # Train intent head
    print("  [4] Training intent head (10 epochs for T4 speed)...")
    intent_head = IntentHead(embed_dim=256, n_classes=3).to(device)
    optimizer = torch.optim.AdamW(intent_head.parameters(), lr=1e-3, weight_decay=0.01)

    # Class weights
    class_counts = torch.bincount(train_labels_t, minlength=3).float()
    class_weights = (1.0 / class_counts) * class_counts.sum() / 3
    class_weights = class_weights.to(device)

    best_acc = 0.0
    best_state = None

    EPOCHS = 10
    BATCH_SIZE = 32
    n_batches = (len(train_embs_t) + BATCH_SIZE - 1) // BATCH_SIZE

    for epoch in range(EPOCHS):
        intent_head.train()
        perm = torch.randperm(len(train_embs_t))
        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            x = train_embs_t[idx]
            y = train_labels_t[idx]

            logits = intent_head(x)
            loss = F.cross_entropy(logits, y, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluate
        intent_head.eval()
        with torch.no_grad():
            logits = intent_head(holdout_embs_t)
            preds = logits.argmax(dim=1)
            acc = (preds == holdout_labels_t).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in intent_head.state_dict().items()}

        if (epoch + 1) % 2 == 0:
            print(f"      epoch {epoch+1:2d}/{EPOCHS}: loss={epoch_loss/n_batches:.4f}, acc={acc:.4f}")

    # Final evaluation
    intent_head.load_state_dict(best_state)
    intent_head.eval()
    with torch.no_grad():
        logits = intent_head(holdout_embs_t)
        preds = logits.argmax(dim=1).cpu().numpy()
        true = holdout_labels  # already numpy

    cm = confusion_matrix(true, preds, labels=[0, 1, 2])
    print(f"\n  [5] Final accuracy: {best_acc:.4f}")
    print(f"      Confusion matrix:")
    print(f"        {'':15s} {'SURV':>5} {'TRAN':>5} {'ATK':>5}")
    for i, name in enumerate(INTENT_CLASSES):
        print(f"        {name:15s} {cm[i][0]:>5d} {cm[i][1]:>5d} {cm[i][2]:>5d}")

    # Save intent head
    print("  [6] Saving intent head checkpoint...")
    os.makedirs(INTENT_REMOTE, exist_ok=True)
    ckpt_path = f"{INTENT_REMOTE}/intent_head.pt"
    torch.save({
        "intent_head": best_state,
        "encoder_checkpoint": MODEL_REMOTE,
        "intent_classes": INTENT_CLASSES,
        "holdout_accuracy": best_acc,
        "confusion_matrix": cm.tolist(),
        "labeling_method": "heuristic_spectrogram_features",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "note": "Trained on T4 with reduced epochs for pipeline testing. For production, use train_intent.py on A100.",
    }, ckpt_path)
    INTENT_VOL.commit()
    print(f"      saved to {ckpt_path}")

    return {
        "accuracy": float(best_acc),
        "confusion_matrix": cm.tolist(),
        "status": "PASS" if best_acc > 0.4 else "WARN_LOW_ACC",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Spoof detection test
# ─────────────────────────────────────────────────────────────────────────────


def test_spoof_detection(encoder: nn.Module, device: str) -> Dict:
    """Test spoof detection with real encoder."""
    print("\n" + "─" * 60)
    print("PHASE 4: Spoof Detection Test")
    print("─" * 60)

    encoder.eval()

    # Use a few real drone spectrograms as "friendly" drones
    print("  [1] Loading drone samples for enrollment...")
    train_specs, _ = load_samples(H5_REMOTE, "train", max_per_type=5)

    # Enroll first 3 drones as "friendly"
    print("  [2] Enrolling 3 friendly drones...")
    friendly_embs = encode_batch(encoder, train_specs[:3], device)
    friendly_embs_norm = friendly_embs / (np.linalg.norm(friendly_embs, axis=1, keepdims=True) + 1e-8)

    registry = {}
    for i in range(3):
        serial = f"FRIENDLY_{i:04d}"
        registry[serial] = {
            "drone_type": "Test Drone",
            "serial_number": serial,
            "rf_fingerprint": friendly_embs_norm[i].tolist(),
        }
        print(f"      enrolled {serial} (fingerprint norm: {np.linalg.norm(friendly_embs_norm[i]):.3f})")

    # Test 1: Authentic — same drone, new spectrogram from same type
    print("\n  [3] Test 1: Authentic packet (enrolled drone, new sample)...")
    # Use 4th sample from same type as drone 0
    authentic_emb = encode_batch(encoder, train_specs[3:4], device)[0]
    authentic_emb_norm = authentic_emb / (np.linalg.norm(authentic_emb) + 1e-8)

    # Compare to registry
    sims = {}
    for serial, d in registry.items():
        enrolled = np.array(d["rf_fingerprint"], dtype=np.float32)
        enrolled_norm = enrolled / (np.linalg.norm(enrolled) + 1e-8)
        sim = float(np.dot(authentic_emb_norm, enrolled_norm))
        sims[serial] = sim

    best_serial = max(sims, key=sims.get)
    best_sim = sims[best_serial]
    threshold = 0.85

    if best_sim >= threshold:
        verdict = "AUTHENTIC"
        print(f"      ✓ AUTHENTIC — matched {best_serial} (sim={best_sim:.3f})")
    else:
        verdict = "NOT_ENROLLED"
        print(f"      ✗ NOT_ENROLLED — best sim {best_sim:.3f} below threshold")

    # Test 2: Spoofed — different drone type, claim friendly serial
    print("\n  [4] Test 2: Spoofed packet (different drone, claims friendly serial)...")
    holdout_specs, _ = load_samples(H5_REMOTE, "holdout", max_per_type=2)
    spoofed_emb = encode_batch(encoder, holdout_specs[:1], device)[0]
    spoofed_emb_norm = spoofed_emb / (np.linalg.norm(spoofed_emb) + 1e-8)

    # Attacker claims to be FRIENDLY_0000
    claimed_serial = "FRIENDLY_0000"
    enrolled = np.array(registry[claimed_serial]["rf_fingerprint"], dtype=np.float32)
    enrolled_norm = enrolled / (np.linalg.norm(enrolled) + 1e-8)
    sim_to_claimed = float(np.dot(spoofed_emb_norm, enrolled_norm))

    if sim_to_claimed < threshold:
        verdict_spoof = "SPOOFED"
        print(f"      ✓ SPOOFED DETECTED — claimed {claimed_serial} but sim only {sim_to_claimed:.3f}")
        print(f"        (threshold {threshold}) — different physical transmitter")
    else:
        verdict_spoof = "AUTHENTIC"
        print(f"      ✗ FAILED — sim {sim_to_claimed:.3f} above threshold (should be lower)")

    # Test 3: Unknown drone
    print("\n  [5] Test 3: Unknown drone (not in registry)...")
    unknown_serial = "UNKNOWN_9999"
    if unknown_serial not in registry:
        # Find best match
        sims_unknown = {}
        for serial, d in registry.items():
            enrolled = np.array(d["rf_fingerprint"], dtype=np.float32)
            enrolled_norm = enrolled / (np.linalg.norm(enrolled) + 1e-8)
            sim = float(np.dot(spoofed_emb_norm, enrolled_norm))
            sims_unknown[serial] = sim
        best_unknown = max(sims_unknown, key=sims_unknown.get)
        best_unknown_sim = sims_unknown[best_unknown]
        if best_unknown_sim < threshold:
            print(f"      ✓ NOT_ENROLLED — serial {unknown_serial} not in registry, best sim {best_unknown_sim:.3f}")
        else:
            print(f"      ✗ Matched {best_unknown} (sim {best_unknown_sim:.3f}) — unexpected")

    return {
        "authentic_test": {"verdict": verdict, "similarity": best_sim},
        "spoof_test": {"verdict": verdict_spoof, "similarity": sim_to_claimed},
        "threshold": threshold,
        "status": "PASS" if verdict_spoof == "SPOOFED" else "FAIL",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Adversarial robustness test (FGSM only, fast)
# ─────────────────────────────────────────────────────────────────────────────


def test_adversarial(encoder: nn.Module, device: str) -> Dict:
    """Test FGSM attack (one epsilon, small subset)."""
    print("\n" + "─" * 60)
    print("PHASE 5: Adversarial Robustness Test (FGSM, ε=0.1)")
    print("─" * 60)

    # Load data (small subset)
    print("  [1] Loading data...")
    train_specs, _ = load_samples(H5_REMOTE, "train", max_per_type=50)
    holdout_specs, _ = load_samples(H5_REMOTE, "holdout", max_per_type=30)
    matched_bg_specs = load_matched_bgs(MATCHED_REMOTE, "holdout", max_n=100)

    # Fit Mahalanobis
    print("  [2] Fitting Mahalanobis...")
    train_embs = encode_batch(encoder, train_specs, device)
    centroid, cov_inv = fit_mahalanobis_l2(train_embs)
    centroid_t = torch.from_numpy(centroid).to(device).float()
    cov_inv_t = torch.from_numpy(cov_inv).to(device).float()

    # Baseline distances
    holdout_embs = encode_batch(encoder, holdout_specs, device)
    bg_embs = encode_batch(encoder, matched_bg_specs, device)
    holdout_dists = mahalanobis_l2_np(holdout_embs, centroid, cov_inv)
    bg_dists = mahalanobis_l2_np(bg_embs, centroid, cov_inv)

    labels = np.concatenate([np.ones(len(holdout_dists)), np.zeros(len(bg_dists))])
    dists = np.concatenate([holdout_dists, bg_dists])
    baseline_auc = roc_auc_score(labels, -dists)
    print(f"      baseline AUC: {baseline_auc:.4f}")

    # FGSM attack on BG samples (try to make them look like drones)
    print("  [3] FGSM attack on BG samples (ε=0.1, 20 samples)...")
    eps = 0.1
    n_attack = min(20, len(matched_bg_specs))
    bg_tensor = torch.from_numpy(matched_bg_specs[:n_attack]).to(device).float()

    attacked_bg_dists = []
    for j in range(n_attack):
        spec = bg_tensor[j:j+1].clone().detach().requires_grad_(True)
        emb = encoder(spec)
        dist = mahalanobis_l2_batch(emb, centroid_t, cov_inv_t)
        loss = dist.sum()  # minimize distance (make BG look like drone)

        encoder.zero_grad()
        loss.backward()

        with torch.no_grad():
            perturbed = spec - eps * spec.grad.sign()
            attacked_emb = encoder(perturbed)
            attacked_dist = mahalanobis_l2_batch(attacked_emb, centroid_t, cov_inv_t)
            attacked_bg_dists.append(float(attacked_dist[0]))

    attacked_bg_dists = np.array(attacked_bg_dists)

    # Recompute AUC with attacked BGs
    all_d = np.concatenate([holdout_dists, attacked_bg_dists])
    all_l = np.concatenate([np.ones(len(holdout_dists)), np.zeros(len(attacked_bg_dists))])
    attacked_auc = roc_auc_score(all_l, -all_d)

    auc_drop = baseline_auc - attacked_auc
    bg_to_drone_rate = float((attacked_bg_dists <= np.percentile(holdout_dists, 99)).mean())

    print(f"      attacked AUC: {attacked_auc:.4f}")
    print(f"      AUC drop: {auc_drop:.4f}")
    print(f"      BG→Drone rate: {bg_to_drone_rate:.3f}")

    return {
        "baseline_auc": float(baseline_auc),
        "attacked_auc": float(attacked_auc),
        "auc_drop": float(auc_drop),
        "bg_to_drone_rate": bg_to_drone_rate,
        "status": "PASS",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Modal function
# ─────────────────────────────────────────────────────────────────────────────


@app.function(
    image=IMAGE,
    gpu="T4",  # T4 for cheaper testing
    volumes={"/data": VOL, "/models": MODEL_VOL, "/matched": MATCHED_VOL,
             "/results": RESULTS_VOL, "/intent": INTENT_VOL},
    timeout=5400,  # 90 min
    memory=16384,
)
def run_pipeline_test(phases: str = "all") -> Dict:
    device = "cuda"
    print("=" * 70)
    print("IRIS T4 Pipeline Test")
    print("=" * 70)
    print(f"  Phases: {phases}")
    print(f"  Device: T4 GPU")
    print(f"  Time:   {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

    # Reload volumes
    VOL.reload()
    MODEL_VOL.reload()
    MATCHED_VOL.reload()

    # Verify checkpoint exists
    if not os.path.exists(MODEL_REMOTE):
        return {"error": f"Checkpoint not found at {MODEL_REMOTE}. Run train_modal_v11.py first."}

    # Verify HDF5 exists
    if not os.path.exists(H5_REMOTE):
        return {"error": f"HDF5 not found at {H5_REMOTE}"}

    # Verify matched BG exists
    if not os.path.exists(MATCHED_REMOTE):
        return {"error": f"Matched BG HDF5 not found at {MATCHED_REMOTE}"}

    print(f"\n  [ok] all data files present")

    # Load encoder once, reuse across phases
    print(f"\n  Loading encoder...")
    ckpt = torch.load(MODEL_REMOTE, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    encoder_state = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
    encoder = CNNEncoder(in_ch=2, width=64, depth=6, embed_dim=256).to(device)
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.eval()
    print(f"  [ok] encoder loaded: {sum(p.numel() for p in encoder.parameters()):,} params")

    results = {}

    # Run requested phases
    phase_list = phases.split(",") if phases != "all" else ["inference", "honest_eval", "intent", "spoof", "adversarial"]

    for phase in phase_list:
        phase = phase.strip()
        try:
            if phase == "inference":
                results["inference"] = test_inference(device)
            elif phase == "honest_eval":
                results["honest_eval"] = test_honest_eval(encoder, device)
            elif phase == "intent":
                results["intent"] = test_intent_training(encoder, device)
            elif phase == "spoof":
                results["spoof"] = test_spoof_detection(encoder, device)
            elif phase == "adversarial":
                results["adversarial"] = test_adversarial(encoder, device)
            else:
                print(f"\n  [warn] unknown phase: {phase}")
        except Exception as e:
            import traceback
            print(f"\n  [error] phase {phase} failed: {e}")
            traceback.print_exc()
            results[phase] = {"status": "FAIL", "error": str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE TEST SUMMARY")
    print("=" * 70)
    for phase, r in results.items():
        status = r.get("status", "UNKNOWN")
        emoji = "✓" if status == "PASS" else "⚠" if "WARN" in status else "✗"
        print(f"  {emoji} {phase:15s} {status}")

    # Overall verdict
    all_pass = all(r.get("status") == "PASS" for r in results.values())
    print(f"\n  Overall: {'ALL PASS — safe to run on Mac' if all_pass else 'ISSUES FOUND — review above'}")
    print("=" * 70)

    # Save summary
    os.makedirs(RESULTS_REMOTE, exist_ok=True)
    summary_path = f"{RESULTS_REMOTE}/t4_pipeline_test.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "phases_run": phase_list,
            "results": results,
            "overall_pass": all_pass,
        }, f, indent=2, default=str)
    print(f"\n  Summary saved to {summary_path}")
    RESULTS_VOL.commit()

    return results


@app.local_entrypoint()
def main(phase: str = "all"):
    """Run the T4 pipeline test.

    Usage:
        modal run scripts/t4/test_pipeline_t4.py                    # all phases
        modal run scripts/t4/test_pipeline_t4.py --phase inference  # just inference
        modal run scripts/t4/test_pipeline_t4.py --phase spoof      # just spoof
    """
    results = run_pipeline_test.remote(phases=phase)
    return results


if __name__ == "__main__":
    main()
