#!/usr/bin/env python3
"""
IRIS v11 — Inference Script (detect.py)

Feed any spectrogram → get DRONE / BACKGROUND verdict.

Pipeline: 256-dim embedding → Mahalanobis distance → threshold.

That's it. One centroid. One covariance. One threshold.

v11 uses Hierarchical SupCon (Salesforce CVPR 2022), which forces all drone
types into a unified region in embedding space. This means a simple global
Mahalanobis detector actually works — no PCA, no kNN, no LOTO needed.

v10's complex pipeline (PCA→kNN→LOTO) was a band-aid for broken embeddings
where drone types formed 30 isolated clusters. With merged clusters, the
simplest possible detector is also the correct one.

Based on:
  - Lee et al. 2018 (NeurIPS): A Simple Unified Framework for Detecting
    Out-of-Distribution Samples and Adversarial Attacks — Mahalanobis
    distance to class-conditional Gaussian is the baseline for OOD detection.
  - Hierarchical SupCon makes the "one class = drone" assumption valid.

Usage:
  # Demo: 5 random holdout drones + 5 matched BGs + 5 random BGs
  modal run detect.py

  # Demo with specific types
  modal run detect.py --types DJI --types FUTABA

  # Batch: all holdout + matched BGs
  modal run detect.py --batch

  # Single sample by index
  modal run detect.py --sample 42
"""

import h5py
import json
import math
import os
import time

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve

# ─── Modal setup ──────────────────────────────────────────────────────────────

app = modal.App("iris-detect")

VOL = modal.Volume.from_name("iris-data", create_if_missing=True)
V11_VOL = modal.Volume.from_name("iris-models-v11", create_if_missing=True)
MATCHED_VOL = modal.Volume.from_name("iris-matched-bg", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install("libgl1", "libglib2.0-0", "libhdf5-dev",
                 "python3", "python3-pip", "python-is-python3")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        "h5py==3.12.1", "numpy==1.26.4", "scikit-learn==1.6.1",
        "scipy==1.14.1",
    )
)

H5_REMOTE = "/data/iris_rfuav.h5"
MATCHED_REMOTE = "/matched/iris_matched_bg.h5"
MODEL_REMOTE = "/models/lejepa_v11_best.pt"


# ─── Model ────────────────────────────────────────────────────────────────────

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


# ─── Resolve HDF5 ─────────────────────────────────────────────────────────────

def _resolve_type_dataset(grp, key):
    item = grp[key]
    if isinstance(item, h5py.Dataset):
        if len(item.shape) == 4:
            return item, item.shape[0], False
        elif len(item.shape) == 3:
            return item, 1, False
        else:
            raise ValueError(f"Unexpected shape {item.shape} for /{key}")
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


# ─── Datasets ──────────────────────────────────────────────────────────────────

class EvalDataset(Dataset):
    def __init__(self, h5_path, split_key="holdout"):
        self.f = h5py.File(h5_path, "r")
        grp = self.f[split_key]
        self.type_names = []
        self._resolved = {}
        self._sub_keys = {}
        self.type_to_label = {}
        label_idx = 0
        for key in sorted(grp.keys()):
            try:
                ds_or_grp, n_samples, is_multi = _resolve_type_dataset(grp, key)
                self.type_names.append(key)
                self._resolved[key] = (ds_or_grp, n_samples, is_multi)
                self.type_to_label[key] = label_idx
                if is_multi:
                    sub_keys = []
                    for sk in ds_or_grp.keys():
                        sub = ds_or_grp[sk]
                        if isinstance(sub, h5py.Dataset) and len(sub.shape) == 3:
                            sub_keys.append(sk)
                    try:
                        sub_keys.sort(key=lambda x: int(x))
                    except ValueError:
                        sub_keys.sort()
                    self._sub_keys[key] = sub_keys
                label_idx += 1
            except ValueError:
                continue
        self.index = []
        for tname in self.type_names:
            _, n_samples, _ = self._resolved[tname]
            for i in range(n_samples):
                self.index.append((tname, i))

    def __len__(self):
        return len(self.index)

    def _read_sample(self, tname, local_idx):
        ds_or_grp, n_samples, is_multi = self._resolved[tname]
        if is_multi:
            sub_key = self._sub_keys[tname][local_idx]
            return ds_or_grp[sub_key][:]
        else:
            return ds_or_grp[local_idx]

    @staticmethod
    def _normalize_per_channel(x):
        for c in range(x.shape[0]):
            ch = x[c]
            ch_std = ch.std()
            if ch_std > 1e-6:
                x[c] = (ch - ch.mean()) / ch_std
            else:
                x[c] = ch - ch.mean()
        return x

    def __getitem__(self, idx):
        tname, local_idx = self.index[idx]
        sample = self._read_sample(tname, local_idx)
        if sample.shape[0] == 3:
            x = torch.from_numpy(sample[:2].copy()).float()
        elif sample.shape[0] == 2:
            x = torch.from_numpy(sample.copy()).float()
        else:
            x = torch.from_numpy(sample[:2].copy()).float()
        x = self._normalize_per_channel(x)
        return x, self.type_to_label[tname], tname


class MatchedBGDataset(Dataset):
    def __init__(self, matched_path, split_key="holdout"):
        self.f = h5py.File(matched_path, "r")
        mbg_key = f"{split_key}_matched_bg"
        if mbg_key not in self.f:
            raise ValueError(f"No '{mbg_key}' in {matched_path}")
        self.grp = self.f[mbg_key]
        self.keys = sorted(list(self.grp.keys()),
                          key=lambda x: int(x) if x.isdigit() else 0)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        sample = self.grp[key][:]
        if sample.shape[0] == 3:
            x = torch.from_numpy(sample[:2].copy()).float()
        elif sample.shape[0] == 2:
            x = torch.from_numpy(sample.copy()).float()
        else:
            x = torch.from_numpy(sample[:2].copy()).float()
        for c in range(x.shape[0]):
            ch = x[c]
            ch_std = ch.std()
            if ch_std > 1e-6:
                x[c] = (ch - ch.mean()) / ch_std
            else:
                x[c] = ch - ch.mean()
        return x, idx


class RandomBGDataset(Dataset):
    def __init__(self, h5_path, max_negatives=5000, seed=42):
        self.f = h5py.File(h5_path, "r")
        neg_item = self.f["negatives"]
        if isinstance(neg_item, h5py.Dataset):
            n_total = neg_item.shape[0]
            self._is_multi = False
            self._ds = neg_item
        else:
            sub_keys = [sk for sk in neg_item.keys()
                        if isinstance(neg_item[sk], h5py.Dataset)
                        and len(neg_item[sk].shape) == 3]
            try:
                sub_keys.sort(key=lambda x: int(x))
            except ValueError:
                sub_keys.sort()
            self._is_multi = True
            self._grp = neg_item
            self._sub_keys = sub_keys
            n_total = len(sub_keys)
        n_load = min(n_total, max_negatives)
        rng = np.random.default_rng(seed)
        self._indices = rng.choice(n_total, n_load, replace=False).tolist()

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        local_idx = self._indices[idx]
        if self._is_multi:
            sub_key = self._sub_keys[local_idx]
            sample = self._grp[sub_key][:]
        else:
            sample = self._ds[local_idx]
        if sample.ndim == 2:
            sample = np.stack([sample, sample], axis=0)
        elif sample.shape[0] == 1:
            sample = np.concatenate([sample, sample], axis=0)
        if sample.shape[0] >= 3:
            x = torch.from_numpy(sample[:2].copy()).float()
        elif sample.shape[0] == 2:
            x = torch.from_numpy(sample.copy()).float()
        else:
            x = torch.from_numpy(sample[:2].copy()).float()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.shape[0] == 1:
            x = x.repeat(2, 1, 1)
        for c in range(x.shape[0]):
            ch = x[c]
            ch_std = ch.std()
            if ch_std > 1e-6:
                x[c] = (ch - ch.mean()) / ch_std
            else:
                x[c] = ch - ch.mean()
        return x, 0


# ─── Detection Engine ─────────────────────────────────────────────────────────

class IRISDetector:
    """
    Drone detection: One centroid. One Mahalanobis distance. One threshold.

    Pipeline:
      256-dim embedding → Mahalanobis distance to drone centroid → threshold

    v11's hierarchical SupCon merges all drone types into one unified region.
    This makes a single global Gaussian a valid model for the "drone" class.
    No PCA needed. No kNN needed. No LOTO needed.

    Lower Mahalanobis distance = more drone-like.
    """
    def __init__(self, encoder, train_embs, bg_embs=None, device="cuda"):
        self.encoder = encoder
        self.device = device
        self.encoder.eval()

        # ── Compute drone centroid + covariance in 256D ──
        print(f"  Computing drone centroid from {len(train_embs)} training embeddings...")
        self.centroid = train_embs.mean(axis=0)
        D = train_embs.shape[1]
        cov = np.cov(train_embs.T) + 1e-3 * np.eye(D)
        try:
            self.cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self.cov_inv = np.linalg.pinv(cov)

        # Training drone Mahalanobis distances for percentile ranking
        train_mahal = self._mahalanobis_batch(train_embs)
        self.train_mahal_sorted = np.sort(train_mahal)
        self.train_mahal_mean = train_mahal.mean()
        self.train_mahal_std = train_mahal.std()

        # ── Calibrate threshold on matched BGs ──
        if bg_embs is not None and len(bg_embs) > 0:
            bg_mahal = self._mahalanobis_batch(bg_embs)
            print(f"  Training drone Mahalanobis: {train_mahal.mean():.2f} ± {train_mahal.std():.2f}")
            print(f"  Matched BG Mahalanobis:     {bg_mahal.mean():.2f} ± {bg_mahal.std():.2f}")
            print(f"  BG/Drone ratio: {bg_mahal.mean() / train_mahal.mean():.3f}")

            # Find optimal threshold via Youden's J on train drones vs matched BGs
            all_dists = np.concatenate([train_mahal, bg_mahal])
            all_labels = np.concatenate([np.ones(len(train_mahal)), np.zeros(len(bg_mahal))])
            fpr, tpr, thresholds = roc_curve(all_labels, -all_dists)
            j = tpr - fpr
            best_idx = np.argmax(j)
            self.dist_threshold = float(-thresholds[best_idx])
            self.youden_j = float(j[best_idx])
            self.tpr_at_threshold = float(tpr[best_idx])
            self.fpr_at_threshold = float(fpr[best_idx])
            self.threshold_source = (
                f"Youden's J on train drones vs matched BGs "
                f"(J={self.youden_j:.4f}, TPR={self.tpr_at_threshold:.3f}, "
                f"FPR={self.fpr_at_threshold:.3f})"
            )
            print(f"  Threshold: {self.dist_threshold:.2f} ({self.threshold_source})")
        else:
            # Fallback: 99th percentile of training drone distances
            self.dist_threshold = float(np.percentile(train_mahal, 99))
            self.threshold_source = "99th percentile of training drone Mahalanobis distance"
            self.youden_j = None
            self.tpr_at_threshold = None
            self.fpr_at_threshold = None
            print(f"  Threshold: {self.dist_threshold:.2f} ({self.threshold_source})")

    def _mahalanobis_batch(self, embs):
        diff = embs - self.centroid
        return np.sqrt(np.maximum(np.sum(diff @ self.cov_inv * diff, axis=1), 0))

    @torch.no_grad()
    def encode(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        z = self.encoder(x.to(self.device))
        return z.cpu().numpy().flatten()

    def detect(self, x):
        """
        Detection: spectrogram → 256D embedding → Mahalanobis distance → verdict.

        Returns dict with:
          - embedding: 256-dim numpy array
          - mahal_dist: Mahalanobis distance to drone centroid
          - percentile: % of training drones with HIGHER Mahalanobis distance
                       (high = clearly a drone, low = clearly background)
          - verdict: "DRONE" or "BACKGROUND"
          - confidence: descriptive string
        """
        z = self.encode(x)
        z_2d = z.reshape(1, -1)

        # Mahalanobis distance to drone centroid
        mahal_dist = float(self._mahalanobis_batch(z_2d)[0])

        # Percentile: what % of training drones have HIGHER Mahalanobis distance?
        n_higher = len(self.train_mahal_sorted) - np.searchsorted(
            self.train_mahal_sorted, mahal_dist, side='right'
        )
        percentile = (n_higher / len(self.train_mahal_sorted)) * 100

        # Verdict: Mahalanobis distance below threshold → DRONE
        if mahal_dist <= self.dist_threshold:
            verdict = "DRONE"
        else:
            verdict = "BACKGROUND"

        # Confidence based on margin from threshold (in std units)
        margin = (self.dist_threshold - mahal_dist) / max(self.train_mahal_std, 1e-6)

        if verdict == "DRONE":
            if margin > 3:
                confidence = "Very high confidence drone"
            elif margin > 1.5:
                confidence = "High confidence drone"
            elif margin > 0:
                confidence = "Moderate confidence drone"
            else:
                confidence = "Low confidence drone"
        else:
            if margin < -3:
                confidence = "Very high confidence background"
            elif margin < -1.5:
                confidence = "High confidence background"
            elif margin < 0:
                confidence = "Moderate confidence background"
            else:
                confidence = "Low confidence background"

        return {
            "embedding": z,
            "mahal_dist": mahal_dist,
            "percentile": float(percentile),
            "verdict": verdict,
            "confidence": confidence,
        }


# ─── Main ─────────────────────────────────────────────────────────────────────

@app.function(
    image=IMAGE,
    gpu="A100",
    volumes={
        "/data": VOL,
        "/models": V11_VOL,
        "/matched": MATCHED_VOL,
    },
    timeout=1800,
    memory=32768,
)
def detect(mode="demo", sample_idx=None, type_filters=None):
    """
    mode:
      - "demo": Pick 5 holdout drones + 5 matched BGs + 5 random BGs, show verdicts
      - "sample": Detect a single sample by index
      - "batch": Full holdout evaluation with per-sample verdicts
    """
    device = "cuda"
    t0 = time.time()

    print("=" * 70)
    print("IRIS v11 — DRONE DETECTOR (Mahalanobis)")
    print("=" * 70)

    # ── Load model ──────────────────────────────────────────────────────────
    print("\nLoading model...")
    V11_VOL.reload()
    ckpt = torch.load(MODEL_REMOTE, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    best_epoch = ckpt.get("epoch", -1)
    print(f"  Checkpoint: epoch {best_epoch}")

    encoder = CNNEncoder(
        in_ch=cfg["in_ch"],
        width=cfg["encoder_width"],
        depth=cfg["encoder_depth"],
        embed_dim=cfg["embed_dim"],
    ).to(device)

    full_state = ckpt["model"]
    encoder_state = {k.replace("encoder.", "", 1): v
                     for k, v in full_state.items()
                     if k.startswith("encoder.")}
    if encoder_state:
        encoder.load_state_dict(encoder_state)
    else:
        encoder.load_state_dict(full_state)
    encoder.eval()
    print(f"  Encoder loaded: {cfg['embed_dim']}-dim embeddings")

    # ── Encode training data ────────────────────────────────────────────────
    print("\nEncoding training drones...")
    train_ds = EvalDataset(H5_REMOTE, "train")
    if len(train_ds) > 5000:
        indices = np.random.default_rng(42).choice(len(train_ds), 5000, replace=False)
        train_dl = DataLoader(train_ds, batch_size=128, shuffle=False,
                              sampler=torch.utils.data.SubsetRandomSampler(indices))
    else:
        train_dl = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)

    train_embs = []
    with torch.no_grad():
        for x, label, tname in train_dl:
            z = encoder(x.to(device))
            train_embs.append(z.cpu().numpy())
    train_embs = np.concatenate(train_embs)
    print(f"  Train embeddings: {train_embs.shape}")

    # ── Encode matched BGs (for threshold calibration) ──────────────────────
    print("\nEncoding matched backgrounds (for threshold calibration)...")
    MATCHED_VOL.reload()
    matched_ds = MatchedBGDataset(MATCHED_REMOTE, "holdout")
    if len(matched_ds) > 500:
        cal_indices = np.random.default_rng(42).choice(len(matched_ds), 500, replace=False)
        matched_dl = DataLoader(matched_ds, batch_size=128, shuffle=False,
                                sampler=torch.utils.data.SubsetRandomSampler(cal_indices))
    else:
        matched_dl = DataLoader(matched_ds, batch_size=128, shuffle=False, num_workers=4)

    bg_embs = []
    with torch.no_grad():
        for x, _ in matched_dl:
            z = encoder(x.to(device))
            bg_embs.append(z.cpu().numpy())
    bg_embs = np.concatenate(bg_embs)
    print(f"  BG embeddings: {bg_embs.shape}")

    # ── Build detector ─────────────────────────────────────────────────────
    detector = IRISDetector(
        encoder, train_embs,
        bg_embs=bg_embs, device=device,
    )
    print(f"\n  Pipeline: 256D embedding → Mahalanobis distance → threshold")
    print(f"  Threshold: {detector.dist_threshold:.2f} ({detector.threshold_source})")

    # ═══════════════════════════════════════════════════════════════════════
    # DEMO MODE
    # ═══════════════════════════════════════════════════════════════════════
    if mode == "demo":
        print("\n" + "=" * 70)
        print("DEMO MODE — Sample Detections")
        print("=" * 70)

        # ── Holdout drones ──
        print("\n--- HOLDOUT DRONES ---\n")
        holdout_ds = EvalDataset(H5_REMOTE, "holdout")
        n_holdout = len(holdout_ds)

        type_names = holdout_ds.type_names
        if type_filters:
            type_names = [t for t in type_names if any(f.lower() in t.lower() for f in type_filters)]

        rng = np.random.default_rng(42)
        demo_types = rng.choice(type_names, min(5, len(type_names)), replace=False)

        for tname in demo_types:
            type_indices = [i for i, (tn, _) in enumerate(holdout_ds.index) if tn == tname]
            if not type_indices:
                continue
            idx = rng.choice(type_indices)
            x, label, _ = holdout_ds[idx]
            result = detector.detect(x)

            pct_desc = f"closer than {result['percentile']:.1f}%" if result['percentile'] > 0 else "farther than all"
            print(f"  [{tname}]")
            print(f"    Verdict:    {result['verdict']}")
            print(f"    Confidence: {result['confidence']}")
            print(f"    Percentile: {pct_desc} of training drones")
            print(f"    Mahal dist: {result['mahal_dist']:.2f}  (threshold: {detector.dist_threshold:.2f})")
            print()

        # ── Matched backgrounds ──
        print("--- MATCHED BACKGROUNDS ---\n")
        n_matched = len(matched_ds)
        matched_demo_indices = rng.choice(n_matched, min(5, n_matched), replace=False)

        for idx in matched_demo_indices:
            x, _ = matched_ds[int(idx)]
            result = detector.detect(x)

            pct_desc = f"closer than {result['percentile']:.1f}%" if result['percentile'] > 0 else "farther than all"
            print(f"  [Matched BG #{int(idx)}]")
            print(f"    Verdict:    {result['verdict']}")
            print(f"    Confidence: {result['confidence']}")
            print(f"    Percentile: {pct_desc} of training drones")
            print(f"    Mahal dist: {result['mahal_dist']:.2f}  (threshold: {detector.dist_threshold:.2f})")
            print()

        # ── Random backgrounds ──
        print("--- RANDOM BACKGROUNDS ---\n")
        random_ds = RandomBGDataset(H5_REMOTE, max_negatives=1000)
        n_random = len(random_ds)
        random_demo_indices = rng.choice(n_random, min(5, n_random), replace=False)

        for idx in random_demo_indices:
            x, _ = random_ds[int(idx)]
            result = detector.detect(x)

            pct_desc = f"closer than {result['percentile']:.1f}%" if result['percentile'] > 0 else "farther than all"
            print(f"  [Random BG #{int(idx)}]")
            print(f"    Verdict:    {result['verdict']}")
            print(f"    Confidence: {result['confidence']}")
            print(f"    Percentile: {pct_desc} of training drones")
            print(f"    Mahal dist: {result['mahal_dist']:.2f}  (threshold: {detector.dist_threshold:.2f})")
            print()

        # ── Summary ──
        print("=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print(f"  Pipeline: 256D embedding → Mahalanobis distance → threshold")
        print(f"  Threshold: {detector.dist_threshold:.2f} ({detector.threshold_source})")
        print("  Verdict: Mahalanobis distance <= threshold -> DRONE, else BACKGROUND.")
        print("  Percentile = % of training drones with HIGHER Mahalanobis distance.")
        print("  Higher = more drone-like (closer to drone centroid).")

    # ═══════════════════════════════════════════════════════════════════════
    # SINGLE SAMPLE MODE
    # ═══════════════════════════════════════════════════════════════════════
    elif mode == "sample" and sample_idx is not None:
        print("\n" + "=" * 70)
        print(f"SINGLE SAMPLE DETECTION — Index {sample_idx}")
        print("=" * 70)

        holdout_ds = EvalDataset(H5_REMOTE, "holdout")
        if sample_idx >= len(holdout_ds):
            print(f"  ERROR: Index {sample_idx} out of range (max {len(holdout_ds)-1})")
            return

        x, label, tname = holdout_ds[sample_idx]
        result = detector.detect(x)

        print(f"\n  Type:       {tname}")
        print(f"  Verdict:    {result['verdict']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Mahal dist: {result['mahal_dist']:.2f}  (threshold: {detector.dist_threshold:.2f})")

    # ═══════════════════════════════════════════════════════════════════════
    # BATCH MODE
    # ═══════════════════════════════════════════════════════════════════════
    elif mode == "batch":
        print("\n" + "=" * 70)
        print("BATCH MODE — Full Holdout Evaluation")
        print("=" * 70)

        holdout_ds = EvalDataset(H5_REMOTE, "holdout")
        MATCHED_VOL.reload()
        matched_ds = MatchedBGDataset(MATCHED_REMOTE, "holdout")

        # Encode all holdout
        print("\n  Encoding holdout drones...")
        holdout_dl = DataLoader(holdout_ds, batch_size=128, shuffle=False, num_workers=4)
        holdout_embs = []
        holdout_types = []
        with torch.no_grad():
            for x, label, tname in holdout_dl:
                z = encoder(x.to(device))
                holdout_embs.append(z.cpu().numpy())
                holdout_types.extend(list(tname))
        holdout_embs = np.concatenate(holdout_embs)
        holdout_types = np.array(holdout_types)

        # Encode matched BGs
        print("  Encoding matched backgrounds...")
        matched_dl = DataLoader(matched_ds, batch_size=128, shuffle=False, num_workers=4)
        matched_embs = []
        with torch.no_grad():
            for x, _ in matched_dl:
                z = encoder(x.to(device))
                matched_embs.append(z.cpu().numpy())
        matched_embs = np.concatenate(matched_embs)

        # Compute Mahalanobis distances
        holdout_mahal = detector._mahalanobis_batch(holdout_embs)
        matched_mahal = detector._mahalanobis_batch(matched_embs)

        # AUC
        labels = np.concatenate([np.ones(len(holdout_mahal)), np.zeros(len(matched_mahal))])
        all_dists = np.concatenate([holdout_mahal, matched_mahal])
        mahal_auc = roc_auc_score(labels, -all_dists)

        # Per-sample verdicts
        print(f"\n  Mahalanobis AUC: {mahal_auc:.4f}")
        print(f"  Threshold: {detector.dist_threshold:.2f}")
        print(f"\n  {'Type':<25} {'N':>5} {'DRONE':>7} {'BG':>7} {'Accuracy':>10} {'Mean Mahal':>11}")
        print(f"  {'-'*25} {'-'*5} {'-'*7} {'-'*7} {'-'*10} {'-'*11}")

        total_correct = 0
        total_samples = 0

        for tname in sorted(np.unique(holdout_types)):
            mask = holdout_types == tname
            type_mahal = holdout_mahal[mask]
            n_drone = (type_mahal <= detector.dist_threshold).sum()
            n_bg = (type_mahal > detector.dist_threshold).sum()
            acc = n_drone / len(type_mahal)

            total_correct += n_drone
            total_samples += len(type_mahal)

            print(f"  {tname:<25} {len(type_mahal):>5} {n_drone:>7} {n_bg:>7} {acc:>10.1%} {type_mahal.mean():>11.2f}")

        # Matched BG verdicts
        mbg_correct = (matched_mahal > detector.dist_threshold).sum()
        print(f"\n  Matched BGs: {mbg_correct}/{len(matched_mahal)} correctly classified as BACKGROUND ({100*mbg_correct/len(matched_mahal):.1f}%)")
        print(f"  Total accuracy: {total_correct + mbg_correct}/{total_samples + len(matched_mahal)} ({100*(total_correct+mbg_correct)/(total_samples+len(matched_mahal)):.1f}%)")

    elapsed = time.time() - t0
    print(f"\n  Detection completed in {elapsed:.1f}s")


@app.local_entrypoint()
def main(mode: str = "demo", sample: int = -1, types: str = ""):
    """
    IRIS Drone Detector

    Args:
      mode:  "demo" (default), "sample", or "batch"
      sample: sample index (only for mode="sample")
      types: comma-separated type filters, e.g. "DJI,FUTABA"
    """
    type_filters = [t.strip() for t in types.split(",") if t.strip()] if types else None
    detect.remote(
        mode=mode,
        sample_idx=sample if sample >= 0 else None,
        type_filters=type_filters,
    )
