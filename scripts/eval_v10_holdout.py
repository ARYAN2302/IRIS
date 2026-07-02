#!/usr/bin/env python3
"""
IRIS v11 — Full Holdout Evaluation

v11 uses Hierarchical SupCon (Salesforce CVPR 2022). The key change:
standard SupCon treated each drone type as a separate class, creating 30
isolated clusters. Hierarchical SupCon adds a coarse-grained loss that
pulls all drone types together into a unified "drone" region while
preserving type sub-clusters via the fine-grained loss.

This script runs the full stress test:
  1. ALL holdout drones (3,659) vs ALL matched backgrounds (3,659) — per-type AUC
  2. ALL holdout drones vs 50,000 random backgrounds — sanity check
  3. Per-pair analysis: what % of matched BGs are closer to centroid than their
     source drone? (v9 was 78.1% — must be <<50% for real detection)
  4. Global Mahalanobis detection (the real-world use case: one centroid, unseen types)
  5. Per-type Mahalanobis detection (for reference, fails on unseen types)
  6. UMAP visualization saved as PNG

Usage:
  modal run scripts/eval_v10_holdout.py
"""

import h5py
import json
import math
import os
import time
from pathlib import Path

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score

# ─── Modal setup ──────────────────────────────────────────────────────────────

app = modal.App("iris-v11-eval")

VOL = modal.Volume.from_name("iris-data", create_if_missing=True)
MODEL_VOL = modal.Volume.from_name("iris-models-v11", create_if_missing=True)
MATCHED_VOL = modal.Volume.from_name("iris-matched-bg", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install("libgl1", "libglib2.0-0", "libhdf5-dev",
                 "python3", "python3-pip", "python-is-python3")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        "h5py==3.12.1", "numpy==1.26.4", "scikit-learn==1.6.1",
        "scipy==1.14.1", "umap-learn==0.5.7", "matplotlib==3.9.3",
    )
)

H5_REMOTE = "/data/iris_rfuav.h5"
MATCHED_REMOTE = "/matched/iris_matched_bg.h5"
MODEL_REMOTE = "/models/lejepa_v11_best.pt"
RESULTS_DIR = "/results"


# ─── Helper: resolve HDF5 type datasets ───────────────────────────────────────

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


# ─── Model (must match training) ──────────────────────────────────────────────

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


# ─── Dataset for eval (no augmentation, deterministic) ─────────────────────────

class EvalDataset(Dataset):
    """Loads spectrograms with per-channel normalization, no augmentation."""
    def __init__(self, h5_path, split_key="holdout"):
        self.f = h5py.File(h5_path, "r")
        self.split_key = split_key
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

        # Build flat index: (type_name, local_idx)
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
    """Loads matched backgrounds from pre-computed HDF5."""
    def __init__(self, matched_path, split_key="holdout"):
        self.f = h5py.File(matched_path, "r")
        mbg_key = f"{split_key}_matched_bg"
        if mbg_key not in self.f:
            raise ValueError(f"No '{mbg_key}' in {matched_path}")
        self.grp = self.f[mbg_key]
        self.keys = sorted(list(self.grp.keys()),
                          key=lambda x: int(x) if x.isdigit() else 0)

        # Try to figure out which drone type each matched bg came from
        # The matched bg was generated from holdout drones in order
        # We'll load the holdout dataset to map indices → types
        self.source_types = None  # will be filled by main eval function

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

        # Per-channel normalization (same as training)
        for c in range(x.shape[0]):
            ch = x[c]
            ch_std = ch.std()
            if ch_std > 1e-6:
                x[c] = (ch - ch.mean()) / ch_std
            else:
                x[c] = ch - ch.mean()

        return x, idx


class RandomBGDataset(Dataset):
    """Loads random background (negative) samples."""
    def __init__(self, h5_path, max_negatives=50000, seed=42):
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

        return x


# ─── Main Evaluation ──────────────────────────────────────────────────────────

@app.function(
    image=IMAGE,
    gpu="A100",
    volumes={
        "/data": VOL,
        "/models": MODEL_VOL,
        "/matched": MATCHED_VOL,
        "/results": modal.Volume.from_name("iris-results", create_if_missing=True),
    },
    timeout=3600,  # 60 min
    memory=32768,
)
def evaluate():
    device = "cuda"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    # ── Load model ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("IRIS v11 — FULL HOLDOUT EVALUATION")
    print("=" * 70)

    MODEL_VOL.reload()
    ckpt = torch.load(MODEL_REMOTE, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    best_epoch = ckpt["epoch"]
    print(f"Checkpoint: epoch {best_epoch}")
    print(f"Training eval results: {json.dumps(ckpt.get('eval_results', {}), indent=2, default=str)}")

    encoder = CNNEncoder(
        in_ch=cfg["in_ch"],
        width=cfg["encoder_width"],
        depth=cfg["encoder_depth"],
        embed_dim=cfg["embed_dim"],
    ).to(device)

    # The checkpoint saves the full LeJEPASupConV11 model state_dict
    # Keys are like: encoder.conv.0.block.0.weight, projector.net.0.weight, ...
    # We need only the encoder.* keys with the prefix stripped
    full_state = ckpt["model"]
    encoder_state = {k.replace("encoder.", "", 1): v
                    for k, v in full_state.items()
                    if k.startswith("encoder.")}
    if encoder_state:
        encoder.load_state_dict(encoder_state)
        print(f"  Loaded {len(encoder_state)} encoder params from full model checkpoint")
    else:
        # Fallback: maybe it was saved with just encoder state
        encoder.load_state_dict(full_state)
        print(f"  Loaded encoder state directly ({len(full_state)} params)")

    encoder.eval()
    print(f"Encoder loaded: {cfg['embed_dim']}-dim embeddings")

    # ── Encode ALL train drones (for Mahalanobis centroid) ─────────────────
    print("\n--- Encoding train drones ---")
    train_ds = EvalDataset(H5_REMOTE, "train")
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    train_embs = []
    train_labels = []
    train_types = []
    with torch.no_grad():
        for x, label, tname in train_dl:
            z = encoder(x.to(device))
            train_embs.append(z.cpu().numpy())
            train_labels.extend(label.tolist() if hasattr(label, 'tolist') else list(label))
            train_types.extend(list(tname))
    train_embs = np.concatenate(train_embs)
    train_labels = np.array(train_labels)
    train_types = np.array(train_types)
    print(f"  Train embeddings: {train_embs.shape}")

    # ── Encode ALL holdout drones ──────────────────────────────────────────
    print("\n--- Encoding holdout drones ---")
    holdout_ds = EvalDataset(H5_REMOTE, "holdout")
    holdout_dl = DataLoader(holdout_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    holdout_embs = []
    holdout_labels = []
    holdout_types = []
    with torch.no_grad():
        for x, label, tname in holdout_dl:
            z = encoder(x.to(device))
            holdout_embs.append(z.cpu().numpy())
            holdout_labels.extend(label.tolist() if hasattr(label, 'tolist') else list(label))
            holdout_types.extend(list(tname))
    holdout_embs = np.concatenate(holdout_embs)
    holdout_labels = np.array(holdout_labels)
    holdout_types = np.array(holdout_types)
    print(f"  Holdout embeddings: {holdout_embs.shape}")
    print(f"  Holdout types: {sorted(np.unique(holdout_types))}")

    # ── Encode ALL matched backgrounds ─────────────────────────────────────
    print("\n--- Encoding matched backgrounds ---")
    MATCHED_VOL.reload()
    matched_ds = MatchedBGDataset(MATCHED_REMOTE, "holdout")
    matched_dl = DataLoader(matched_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    matched_embs = []
    with torch.no_grad():
        for x, _ in matched_dl:
            z = encoder(x.to(device))
            matched_embs.append(z.cpu().numpy())
    matched_embs = np.concatenate(matched_embs)
    print(f"  Matched BG embeddings: {matched_embs.shape}")

    # ── Encode random backgrounds ──────────────────────────────────────────
    print("\n--- Encoding random backgrounds (50K) ---")
    random_ds = RandomBGDataset(H5_REMOTE, max_negatives=50000)
    random_dl = DataLoader(random_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    random_embs = []
    with torch.no_grad():
        for batch in random_dl:
            z = encoder(batch.to(device))
            random_embs.append(z.cpu().numpy())
    random_embs = np.concatenate(random_embs)
    print(f"  Random BG embeddings: {random_embs.shape}")

    # ── Compute Mahalanobis distances ──────────────────────────────────────
    print("\n--- Computing Mahalanobis distances ---")
    D = train_embs.shape[1]

    # Global Mahalanobis (THE real-world detector: one centroid from all train drones)
    centroid = train_embs.mean(axis=0)
    cov = np.cov(train_embs.T) + 1e-3 * np.eye(D)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    def mahalanobis_dist(embs, center, cov_inv):
        diff = embs - center
        return np.sqrt(np.maximum(np.sum(diff @ cov_inv * diff, axis=1), 0))

    holdout_mahal = mahalanobis_dist(holdout_embs, centroid, cov_inv)
    matched_mahal = mahalanobis_dist(matched_embs, centroid, cov_inv)
    random_mahal = mahalanobis_dist(random_embs, centroid, cov_inv)

    # ── TEST 1: Overall AUC — Holdout drones vs matched backgrounds ───────
    print("\n" + "=" * 70)
    print("TEST 1: OVERALL DETECTION — Holdout Drones vs Matched BG")
    print("=" * 70)

    labels_1 = np.concatenate([np.ones(len(holdout_mahal)), np.zeros(len(matched_mahal))])
    dists_1 = np.concatenate([holdout_mahal, matched_mahal])
    auc_matched = roc_auc_score(labels_1, -dists_1)
    print(f"  Drones: {len(holdout_mahal)}, Matched BG: {len(matched_mahal)}")
    print(f"  *** MATCHED BG AUC = {auc_matched:.4f} ***")
    print(f"  Drone mean dist:  {holdout_mahal.mean():.2f} ± {holdout_mahal.std():.2f}")
    print(f"  Matched BG mean dist: {matched_mahal.mean():.2f} ± {matched_mahal.std():.2f}")
    print(f"  BG/Drone ratio: {matched_mahal.mean() / holdout_mahal.mean():.3f}")

    # ── TEST 2: Overall AUC — Holdout drones vs random backgrounds ────────
    print("\n" + "=" * 70)
    print("TEST 2: SANITY CHECK — Holdout Drones vs Random BG (50K)")
    print("=" * 70)

    labels_2 = np.concatenate([np.ones(len(holdout_mahal)), np.zeros(len(random_mahal))])
    dists_2 = np.concatenate([holdout_mahal, random_mahal])
    auc_random = roc_auc_score(labels_2, -dists_2)
    print(f"  Drones: {len(holdout_mahal)}, Random BG: {len(random_mahal)}")
    print(f"  Random BG AUC = {auc_random:.4f}")
    print(f"  Random BG mean dist: {random_mahal.mean():.2f} ± {random_mahal.std():.2f}")

    # ── TEST 3: Per-type AUC breakdown ────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 3: PER-TYPE DETECTION — Each holdout type vs matched BG")
    print("=" * 70)

    holdout_type_names = sorted(np.unique(holdout_types))
    per_type_results = {}

    print(f"  {'Type':<25} {'N':>5} {'Matched AUC':>12} {'Random AUC':>12} {'BG/Drone':>9} {'Drone dist':>11} {'BG dist':>11}")
    print(f"  {'-'*25} {'-'*5} {'-'*12} {'-'*12} {'-'*9} {'-'*11} {'-'*11}")

    for tname in holdout_type_names:
        mask = holdout_types == tname
        n_type = mask.sum()
        type_mahal = holdout_mahal[mask]
        type_embs = holdout_embs[mask]

        # AUC vs matched bg
        lab = np.concatenate([np.ones(n_type), np.zeros(len(matched_mahal))])
        dist = np.concatenate([type_mahal, matched_mahal])
        type_auc_matched = roc_auc_score(lab, -dist)

        # AUC vs random bg
        lab2 = np.concatenate([np.ones(n_type), np.zeros(len(random_mahal))])
        dist2 = np.concatenate([type_mahal, random_mahal])
        type_auc_random = roc_auc_score(lab2, -dist2)

        ratio = matched_mahal.mean() / type_mahal.mean() if type_mahal.mean() > 0 else 0

        per_type_results[tname] = {
            "n_samples": int(n_type),
            "matched_auc": float(type_auc_matched),
            "random_auc": float(type_auc_random),
            "drone_mean_dist": float(type_mahal.mean()),
            "bg_drone_ratio": float(ratio),
        }

        print(f"  {tname:<25} {n_type:>5} {type_auc_matched:>12.4f} {type_auc_random:>12.4f} {ratio:>9.3f} {type_mahal.mean():>11.2f} {matched_mahal.mean():>11.2f}")

    # Summary
    matched_aucs = [v["matched_auc"] for v in per_type_results.values()]
    print(f"\n  SUMMARY:")
    print(f"    Mean matched AUC:  {np.mean(matched_aucs):.4f}")
    print(f"    Min matched AUC:   {np.min(matched_aucs):.4f}")
    print(f"    Max matched AUC:   {np.max(matched_aucs):.4f}")
    print(f"    Types >= 0.90:     {sum(1 for a in matched_aucs if a >= 0.90)}/{len(matched_aucs)}")
    print(f"    Types >= 0.95:     {sum(1 for a in matched_aucs if a >= 0.95)}/{len(matched_aucs)}")
    print(f"    Types >= 0.99:     {sum(1 for a in matched_aucs if a >= 0.99)}/{len(matched_aucs)}")

    # ── TEST 4: Per-pair analysis ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 4: PER-PAIR ANALYSIS — Matched BG vs its source drone")
    print("=" * 70)
    print("  (Each matched BG was generated from a specific holdout drone.")
    print("   If the model truly detects drone signal, the source drone should")
    print("   be CLOSER to the centroid than its matched background.)")

    # The matched BG was generated in order from holdout drones
    # So matched_bg[i] was generated from holdout_drone[i]
    n_pairs = min(len(holdout_mahal), len(matched_mahal))
    drone_closer = np.sum(holdout_mahal[:n_pairs] < matched_mahal[:n_pairs])
    bg_closer = np.sum(matched_mahal[:n_pairs] < holdout_mahal[:n_pairs])
    tied = n_pairs - drone_closer - bg_closer

    print(f"\n  Pairs analyzed: {n_pairs}")
    print(f"  Drone closer to centroid:  {drone_closer:>6} ({100*drone_closer/n_pairs:.1f}%)")
    print(f"  Matched BG closer:         {bg_closer:>6} ({100*bg_closer/n_pairs:.1f}%)")
    print(f"  Tied:                      {tied:>6} ({100*tied/n_pairs:.1f}%)")
    print(f"\n  v9 result: 78.1% matched BG was CLOSER → shortcut detected")
    print(f"  v10 result: {100*bg_closer/n_pairs:.1f}% matched BG is closer")

    # Per-type per-pair
    print(f"\n  {'Type':<25} {'N pairs':>8} {'Drone closer':>13} {'BG closer':>10} {'BG closer %':>12}")
    print(f"  {'-'*25} {'-'*8} {'-'*13} {'-'*10} {'-'*12}")
    for tname in holdout_type_names:
        mask = holdout_types == tname
        type_indices = np.where(mask)[0]
        # Find pairs within n_pairs limit
        valid_indices = type_indices[type_indices < n_pairs]
        if len(valid_indices) == 0:
            continue
        d_closer = np.sum(holdout_mahal[valid_indices] < matched_mahal[valid_indices])
        b_closer = np.sum(matched_mahal[valid_indices] < holdout_mahal[valid_indices])
        print(f"  {tname:<25} {len(valid_indices):>8} {d_closer:>13} {b_closer:>10} {100*b_closer/len(valid_indices):>12.1f}%")

    # ── TEST 5: Matched BG artifact check ─────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 5: MATCHED BG ARTIFACT CHECK")
    print("=" * 70)
    print("  If matched BGs cluster separately from random BGs in embedding space,")
    print("  the model is detecting synthesis artifacts (signal-removal footprints),")
    print("  not just drone signal. If they mix, the artifacts are NOT a shortcut.")

    # AUC: Can the model distinguish matched BG from random BG?
    # If AUC ≈ 0.5, matched BG and random BG are in the same region → no artifact shortcut
    # If AUC >> 0.5, matched BG is distinguishable from random BG → artifact shortcut exists
    artifact_labels = np.concatenate([np.ones(len(matched_mahal)), np.zeros(len(random_mahal))])
    artifact_dists = np.concatenate([matched_mahal, random_mahal])
    artifact_auc = roc_auc_score(artifact_labels, -artifact_dists)
    # Also try the other direction
    artifact_auc_rev = roc_auc_score(artifact_labels, artifact_dists)

    print(f"\n  Matched BG vs Random BG AUC: {artifact_auc:.4f}")
    print(f"  (0.5 = indistinguishable, >0.5 = matched BG closer to drone centroid, <0.5 = matched BG farther)")
    print(f"  Reverse AUC: {artifact_auc_rev:.4f}")

    # Cosine distance between matched BG centroid and random BG centroid
    matched_centroid = matched_embs.mean(axis=0)
    random_centroid = random_embs.mean(axis=0)
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim_bg = cosine_similarity(matched_centroid.reshape(1, -1), random_centroid.reshape(1, -1))[0, 0]
    print(f"  Cosine similarity (matched BG centroid, random BG centroid): {cos_sim_bg:.4f}")

    # Also check: can a linear probe distinguish matched BG from random BG?
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    bg_embs = np.concatenate([matched_embs, random_embs])
    bg_labels = np.concatenate([np.ones(len(matched_embs)), np.zeros(len(random_embs))])
    lr_bg = LogisticRegression(max_iter=500, solver="lbfgs")
    bg_probe = cross_val_score(lr_bg, bg_embs, bg_labels, cv=3, scoring="accuracy")
    print(f"  Linear probe (matched BG vs random BG): {bg_probe.mean():.4f} ± {bg_probe.std():.4f}")
    print(f"  (0.5 = can't distinguish, 1.0 = perfectly distinguishable)")

    if artifact_auc > 0.7 or bg_probe.mean() > 0.8:
        print("\n  ⚠️  WARNING: Matched BGs are distinguishable from random BGs!")
        print("     The model may be detecting synthesis artifacts, not drone signal.")
        print("     The 0.941 matched AUC could be inflated.")
    elif artifact_auc < 0.6 and bg_probe.mean() < 0.65:
        print("\n  ✅ Matched BGs are NOT distinguishable from random BGs.")
        print("     Synthesis artifacts are not a detectable shortcut.")
        print("     The matched AUC reflects genuine drone signal detection.")
    else:
        print(f"\n  ⚡ INCONCLUSIVE: Some distinguishability (AUC={artifact_auc:.3f}, probe={bg_probe.mean():.3f})")
        print("     Matched BGs might have mild artifacts. Interpret matched AUC")
        print("     as an upper bound — real performance may be slightly lower.")

    # ── TEST 6: Per-manufacturer Silhouette ───────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 6: PER-MANUFACTURER CLUSTERING")
    print("=" * 70)
    print("  If k-NN accuracy is driven by manufacturer-level separation,")
    print("  within-manufacturer silhouette should be low.")

    # Map type names to manufacturers
    # Known prefixes in the dataset: DJI, FUTABA, FLYSKY, FRSKY, SIYI, etc.
    manufacturer_map = {}
    for tname in train_ds.type_names:
        prefix = tname.split("_")[0].split("-")[0].upper()
        if prefix in ["DJI", "FUTABA", "FLYSKY", "FRSKY", "SIYI", "HOBBYWING", "RADIOMASTER", "SPEKTRUM", "TBS"]:
            manufacturer_map[tname] = prefix
        else:
            # Try first word
            manufacturer_map[tname] = tname.split("_")[0].split(" ")[0].upper()

    # Group by manufacturer
    manufacturers = {}
    for tname in train_ds.type_names:
        mfr = manufacturer_map.get(tname, "UNKNOWN")
        if mfr not in manufacturers:
            manufacturers[mfr] = []
        manufacturers[mfr].append(tname)

    print(f"\n  Manufacturer groups found: {len(manufacturers)}")
    print(f"  {'Manufacturer':<15} {'Types':>5} {'Samples':>8} {'Silhouette':>12} {'k-NN CV':>10}")
    print(f"  {'-'*15} {'-'*5} {'-'*8} {'-'*12} {'-'*10}")

    manufacturer_results = {}
    for mfr, type_list in sorted(manufacturers.items()):
        if len(type_list) < 2:
            # Only one type from this manufacturer — can't compute within-mfr silhouette
            mask = np.isin(train_types, type_list)
            n_samples = mask.sum()
            print(f"  {mfr:<15} {len(type_list):>5} {n_samples:>8} {'N/A (1 type)':>12} {'N/A':>10}")
            manufacturer_results[mfr] = {"n_types": len(type_list), "n_samples": int(n_samples), "silhouette": None}
            continue

        mask = np.isin(train_types, type_list)
        mfr_embs = train_embs[mask]
        mfr_labels = train_labels[mask]
        n_samples = mask.sum()

        if len(np.unique(mfr_labels)) < 2 or n_samples < 20:
            print(f"  {mfr:<15} {len(type_list):>5} {n_samples:>8} {'N/A':>12} {'N/A':>10}")
            continue

        # Within-manufacturer silhouette
        try:
            sil_mfr = silhouette_score(mfr_embs, mfr_labels, metric="cosine")
        except:
            sil_mfr = float("nan")

        # Within-manufacturer k-NN
        try:
            knn_mfr = KNeighborsClassifier(n_neighbors=5, metric="cosine")
            cv_mfr = cross_val_score(knn_mfr, mfr_embs, mfr_labels, cv=min(3, len(type_list)), scoring="accuracy")
            knn_mfr_acc = float(cv_mfr.mean())
        except:
            knn_mfr_acc = float("nan")

        manufacturer_results[mfr] = {
            "n_types": len(type_list),
            "n_samples": int(n_samples),
            "silhouette": float(sil_mfr) if not math.isnan(sil_mfr) else None,
            "knn_cv": float(knn_mfr_acc) if not math.isnan(knn_mfr_acc) else None,
        }

        sil_str = f"{sil_mfr:.4f}" if not math.isnan(sil_mfr) else "N/A"
        knn_str = f"{knn_mfr_acc:.4f}" if not math.isnan(knn_mfr_acc) else "N/A"
        print(f"  {mfr:<15} {len(type_list):>5} {n_samples:>8} {sil_str:>12} {knn_str:>10}")

    # Overall silhouette for reference
    sil_train = silhouette_score(train_embs, train_labels, metric="cosine")
    print(f"\n  Overall silhouette (30 types): {sil_train:.4f}")

    # Manufacturer-level silhouette (each type → its manufacturer)
    mfr_labels_train = np.array([manufacturer_map.get(t, "UNKNOWN") for t in train_types])
    unique_mfrs = np.unique(mfr_labels_train)
    if len(unique_mfrs) > 1:
        sil_mfr_overall = silhouette_score(train_embs, mfr_labels_train, metric="cosine")
        print(f"  Manufacturer-level silhouette: {sil_mfr_overall:.4f}")
        print(f"  (If mfr-level silhouette >> type-level, separation is by manufacturer, not type)")

    # Binary silhouette: drone vs matched_bg
    bin_embs = np.concatenate([holdout_embs, matched_embs])
    bin_labels = np.concatenate([np.ones(len(holdout_embs)), np.zeros(len(matched_embs))])
    sil_binary = silhouette_score(bin_embs, bin_labels, metric="cosine")
    print(f"  Binary silhouette (drone vs matched BG): {sil_binary:.4f}")

    # ── TEST 7: Optimal threshold & confusion matrix ──────────────────────
    print("\n" + "=" * 70)
    print("TEST 7: OPTIMAL DETECTION THRESHOLD")
    print("=" * 70)

    # Find optimal threshold on matched BG task
    all_dists = np.concatenate([holdout_mahal, matched_mahal])
    all_labels = np.concatenate([np.ones(len(holdout_mahal)), np.zeros(len(matched_mahal))])
    fpr, tpr, thresholds = roc_curve(all_labels, -all_dists)

    # Youden's J statistic
    j = tpr - fpr
    best_idx = np.argmax(j)
    best_threshold = -thresholds[best_idx]  # negative because we negated dists for ROC
    best_tpr = tpr[best_idx]
    best_fpr = fpr[best_idx]

    print(f"  Optimal Mahalanobis threshold: {best_threshold:.2f}")
    print(f"  True Positive Rate (drone detected as drone): {best_tpr:.4f}")
    print(f"  False Positive Rate (matched BG detected as drone): {best_fpr:.4f}")
    print(f"  Youden's J: {j[best_idx]:.4f}")

    # Also at fixed FPR targets
    for target_fpr in [0.01, 0.05, 0.10]:
        idx = np.searchsorted(fpr, target_fpr)
        if idx < len(tpr):
            print(f"  At FPR={target_fpr:.0%}: TPR={tpr[idx]:.4f}")

    # ── TEST 8: Comparison with v9 ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON: v9 vs v10 vs v11")
    print("=" * 70)
    print(f"  {'Metric':<35} {'v9':>10} {'v10':>10} {'v11':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Random BG AUC':<35} {'1.0000':>10} {'1.0000':>10} {auc_random:>10.4f}")
    print(f"  {'Matched BG AUC':<35} {'0.3002':>10} {'0.9371':>10} {auc_matched:>10.4f}")
    print(f"  {'BG closer than drone (per-pair)':<35} {'78.1%':>10} {'??%':>10} {f'{100*bg_closer/n_pairs:.1f}%':>10}")
    print(f"  {'Holdout types with AUC > 0.5':<35} {'0/7':>10} {'5/5':>10} {f'{sum(1 for a in matched_aucs if a > 0.5)}/{len(matched_aucs)}':>10}")

    # ── UMAP Visualization ────────────────────────────────────────────────
    print("\n--- Generating UMAP visualization ---")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from umap import UMAP

    # Subsample for UMAP (too many points gets messy)
    rng = np.random.default_rng(42)

    # Train: 500 drones (spread across types)
    n_train_sample = min(500, len(train_embs))
    train_idx = rng.choice(len(train_embs), n_train_sample, replace=False)

    # Holdout: all (3,659)
    # Matched BG: all (3,659)
    # Random BG: 1000
    n_random_sample = min(1000, len(random_embs))
    random_idx = rng.choice(len(random_embs), n_random_sample, replace=False)

    all_embs_umap = np.concatenate([
        train_embs[train_idx],
        holdout_embs,
        matched_embs,
        random_embs[random_idx],
    ])
    all_labels_umap = np.concatenate([
        np.zeros(n_train_sample),       # 0 = train drone
        np.ones(len(holdout_embs)),     # 1 = holdout drone
        2 * np.ones(len(matched_embs)), # 2 = matched BG
        3 * np.ones(n_random_sample),   # 3 = random BG
    ])

    reducer = UMAP(n_components=2, metric="cosine", n_neighbors=30,
                   min_dist=0.1, random_state=42, verbose=True)
    embedding_2d = reducer.fit_transform(all_embs_umap)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Plot 1: By category (drone / matched BG / random BG)
    colors = {0: "#2196F3", 1: "#4CAF50", 2: "#F44336", 3: "#9E9E9E"}
    names = {0: "Train Drones", 1: "Holdout Drones", 2: "Matched BG", 3: "Random BG"}
    alphas = {0: 0.3, 1: 0.5, 2: 0.5, 3: 0.2}

    for cat in [3, 0, 2, 1]:  # draw background first, drones last (on top)
        mask = all_labels_umap == cat
        axes[0].scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                       c=colors[cat], s=8 if cat < 2 else 6,
                       alpha=alphas[cat], label=names[cat], edgecolors="none")
    axes[0].set_title(f"IRIS v11 — Detection Space (Matched AUC={auc_matched:.3f})", fontsize=14)
    axes[0].legend(loc="best", fontsize=11, markerscale=3)

    # Plot 2: Holdout drones colored by type, matched BG in grey
    holdout_type_colors = plt.cm.Set1(np.linspace(0, 1, len(holdout_type_names)))
    type_color_map = {tname: holdout_type_colors[i] for i, tname in enumerate(holdout_type_names)}

    # Plot matched BG first (as red, behind drones)
    mbg_mask = all_labels_umap == 2
    axes[1].scatter(embedding_2d[mbg_mask, 0], embedding_2d[mbg_mask, 1],
                   c="#D32F2F", s=4, alpha=0.3, label="Matched BG", edgecolors="none")

    # Plot random BG
    rbg_mask = all_labels_umap == 3
    axes[1].scatter(embedding_2d[rbg_mask, 0], embedding_2d[rbg_mask, 1],
                   c="#9E9E9E", s=3, alpha=0.15, label="Random BG", edgecolors="none")

    # Plot holdout drones by type
    # Recalculate indices for holdout in UMAP embedding
    holdout_start = n_train_sample
    holdout_end = holdout_start + len(holdout_embs)
    holdout_umap = embedding_2d[holdout_start:holdout_end]

    for i, tname in enumerate(holdout_type_names):
        mask_t = holdout_types == tname
        axes[1].scatter(holdout_umap[mask_t, 0], holdout_umap[mask_t, 1],
                       c=[holdout_type_colors[i]], s=15, alpha=0.7,
                       label=f"{tname} (AUC={per_type_results[tname]['matched_auc']:.3f})",
                       edgecolors="none")

    axes[1].set_title("Holdout Drones by Type vs Backgrounds", fontsize=14)
    axes[1].legend(loc="best", fontsize=8, markerscale=2)

    plt.tight_layout()
    umap_path = f"{RESULTS_DIR}/iris_v11_umap.png"
    plt.savefig(umap_path, dpi=150, bbox_inches="tight")
    print(f"  UMAP saved to {umap_path}")

    # ── Save full results ─────────────────────────────────────────────────
    results = {
        "model": "v11_lejepa_hierarchical_supcon",
        "best_epoch": int(best_epoch),
        "overall": {
            "n_holdout_drones": int(len(holdout_mahal)),
            "n_matched_bg": int(len(matched_mahal)),
            "n_random_bg": int(len(random_mahal)),
            "matched_bg_auc": float(auc_matched),
            "random_bg_auc": float(auc_random),
            "drone_mean_dist": float(holdout_mahal.mean()),
            "matched_bg_mean_dist": float(matched_mahal.mean()),
            "random_bg_mean_dist": float(random_mahal.mean()),
            "bg_drone_ratio": float(matched_mahal.mean() / holdout_mahal.mean()),
        },
        "per_pair": {
            "n_pairs": int(n_pairs),
            "drone_closer_count": int(drone_closer),
            "bg_closer_count": int(bg_closer),
            "bg_closer_pct": float(100 * bg_closer / n_pairs),
        },
        "per_type": per_type_results,
        "artifact_check": {
            "matched_vs_random_auc": float(artifact_auc),
            "matched_vs_random_probe": float(bg_probe.mean()),
            "cosine_sim_centroids": float(cos_sim_bg),
            "verdict": "inflated" if artifact_auc > 0.7 or bg_probe.mean() > 0.8 else
                       ("clean" if artifact_auc < 0.6 and bg_probe.mean() < 0.65 else "inconclusive"),
        },
        "manufacturer_clustering": {
            "overall_type_silhouette": float(sil_train),
            "manufacturer_level_silhouette": float(sil_mfr_overall) if len(unique_mfrs) > 1 else None,
            "per_manufacturer": manufacturer_results,
        },
        "threshold": {
            "optimal_mahalanobis": float(best_threshold),
            "tpr_at_optimal": float(best_tpr),
            "fpr_at_optimal": float(best_fpr),
        },
        "embedding_quality": {
            "train_silhouette": float(sil_train),
            "binary_silhouette": float(sil_binary),
        },
        "comparison": {
            "v9_matched_auc": 0.3002,
            "v10_matched_auc": 0.9371,
            "v11_matched_auc": float(auc_matched),
            "v9_bg_closer_pct": 78.1,
            "v11_bg_closer_pct": float(100 * bg_closer / n_pairs),
        },
    }

    results_path = f"{RESULTS_DIR}/iris_v11_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # ── Final Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"\n  ★★★ MATCHED BG AUC = {auc_matched:.4f} ★★★")
    print(f"  Random BG AUC      = {auc_random:.4f}")
    print(f"  BG closer than drone = {100*bg_closer/n_pairs:.1f}% (v9 was 78.1%)")
    print(f"  Best detection threshold: Mahal < {best_threshold:.2f}")
    print(f"    → TPR = {best_tpr:.1%}, FPR = {best_fpr:.1%}")
    print()

    if auc_matched >= 0.97:
        print("  🔥 AIRtight result! Hierarchical SupCon merged drone clusters.")
        print("     The model detects universal drone-ness, not type fingerprints.")
    elif auc_matched >= 0.93:
        print("  ✅ Strong result! v11 matches or beats v10's 0.9371.")
        print("     Drone types are merging. Detection is learning drone-ness.")
    elif auc_matched >= 0.90:
        print("  ⚡ Decent result. Clusters are partially merging.")
        print("     Consider adjusting w_fine/w_coarse weights.")
    else:
        print("  ⚠️  Model still struggling. Need further investigation.")

    # Results are on the Modal volume, no need for subprocess
    print(f"\n  Results available on Modal volume 'iris-results' at /results/")


@app.local_entrypoint()
def main():
    evaluate.remote()