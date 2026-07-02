#!/usr/bin/env python3
"""
IRIS v7 — Fingerprint Shortcut Sanity Check

Quick test: normalize both drone and background spectrograms to the same
value distribution (zero mean, unit variance per channel) BEFORE encoding.

If the JPG vs STFT pixel distribution was the ONLY thing separating them,
normalization will kill the shortcut and AUC will drop to ~0.5.

If there's a real "drone-ness" signal surviving in the normalized spectrograms,
AUC will stay above 0.5 even after normalization.

Takes ~5 minutes on A100.

Usage:
  modal run scripts/detect_normalized.py
"""

import h5py
import json
import os

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve

# ─── Modal setup ──────────────────────────────────────────────────────────────

app = modal.App("iris-v7-norm-check")

VOL = modal.Volume.from_name("iris-data", create_if_missing=True)
MODEL_VOL = modal.Volume.from_name("iris-models-v7", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install("libgl1", "libglib2.0-0", "libhdf5-dev",
                 "python3", "python3-pip", "python-is-python3")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        "h5py==3.12.1", "numpy==1.26.4", "scikit-learn==1.6.1",
        "scipy==1.14.1", "matplotlib==3.9.3",
    )
)

H5_REMOTE = "/data/iris_rfuav.h5"

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


class LeJEPASupCon(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = CNNEncoder(
            in_ch=cfg["in_ch"], width=cfg["encoder_width"],
            depth=cfg["encoder_depth"], embed_dim=cfg["embed_dim"],
        )
    def forward(self, x):
        return self.encoder(x)


# ─── Dataset ──────────────────────────────────────────────────────────────────

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


class NormCheckDataset(Dataset):
    """
    Returns samples with per-channel normalization applied.
    Each sample: zero mean, unit variance per channel.
    This removes JPG vs STFT value distribution differences.
    """
    def __init__(self, h5_path, split_key, include_negatives=False, max_per_type=None):
        self.f = h5py.File(h5_path, "r")
        self.samples = []
        self.labels = []
        self.type_names_list = []
        self._resolved = {}
        self._sub_keys = {}

        grp = self.f[split_key]
        type_names = []
        for key in sorted(grp.keys()):
            try:
                ds_or_grp, n_samples, is_multi = _resolve_type_dataset(grp, key)
                type_names.append(key)
                self._resolved[(split_key, key)] = (ds_or_grp, n_samples, is_multi)
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
                    self._sub_keys[(split_key, key)] = sub_keys
            except ValueError:
                continue

        for label_idx, tname in enumerate(type_names):
            _, n_samples, _ = self._resolved[(split_key, tname)]
            n_use = min(n_samples, max_per_type) if max_per_type else n_samples
            for i in range(n_use):
                self.samples.append((split_key, tname, i))
                self.labels.append(label_idx)
                self.type_names_list.append(tname)

        neg_label = len(type_names)
        if include_negatives and "negatives" in self.f:
            neg_item = self.f["negatives"]
            if isinstance(neg_item, h5py.Dataset):
                n_neg = neg_item.shape[0]
                n_use = min(n_neg, max_per_type) if max_per_type else n_neg
                self._resolved[("negatives", None)] = (neg_item, n_use, False)
            else:
                sub_keys = [sk for sk in neg_item.keys()
                            if isinstance(neg_item[sk], h5py.Dataset) and len(neg_item[sk].shape) == 3]
                try:
                    sub_keys.sort(key=lambda x: int(x))
                except ValueError:
                    sub_keys.sort()
                n_neg = len(sub_keys)
                n_use = min(n_neg, max_per_type) if max_per_type else n_neg
                self._resolved[("negatives", None)] = (neg_item, n_use, True)
                self._sub_keys[("negatives", None)] = sub_keys

            for i in range(n_use):
                self.samples.append(("negatives", None, i))
                self.labels.append(neg_label)
                self.type_names_list.append("background")

        self.type_names = type_names
        self.n_drone_types = len(type_names)

    def _read_sample(self, split_key, tname, local_idx):
        ds_or_grp, _, is_multi = self._resolved[(split_key, tname)]
        if is_multi:
            sub_key = self._sub_keys[(split_key, tname)][local_idx]
            return ds_or_grp[sub_key][:]
        else:
            return ds_or_grp[local_idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        split_key, tname, local_idx = self.samples[idx]
        sample = self._read_sample(split_key, tname, local_idx)

        # Drop channel 2
        if sample.shape[0] == 3:
            x = sample[:2].copy().astype(np.float32)
        elif sample.shape[0] == 2:
            x = sample.copy().astype(np.float32)
        else:
            x = sample[:2].copy().astype(np.float32)

        # ── PER-CHANNEL NORMALIZATION ──
        # Each channel: zero mean, unit variance
        # This removes JPG compression artifacts vs raw STFT value differences
        for c in range(x.shape[0]):
            ch = x[c]
            ch_std = ch.std()
            if ch_std > 1e-6:
                x[c] = (ch - ch.mean()) / ch_std
            else:
                x[c] = ch - ch.mean()

        return torch.from_numpy(x), self.labels[idx], self.type_names_list[idx]


# ─── Main ─────────────────────────────────────────────────────────────────────

@app.function(
    image=IMAGE,
    gpu="A100",
    volumes={"/data": VOL, "/models": MODEL_VOL},
    timeout=1800,
    memory=32768,
)
def check():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda"

    # ── Load v7 best model ──
    cfg = dict(in_ch=2, encoder_depth=6, encoder_width=64,
               embed_dim=256, proj_dim=256, pred_dim=256, pred_out=256)
    model = LeJEPASupCon(cfg).to(device)

    ckpt_path = "/models/lejepa_v7_best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "/models/lejepa_v7_epoch5.pt"
    if not os.path.exists(ckpt_path):
        import glob
        ckpts = sorted(glob.glob("/models/lejepa_v7_epoch*.pt"))
        if ckpts:
            ckpt_path = ckpts[-1]
        else:
            print("ERROR: No checkpoints found!")
            return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded: {ckpt_path} (epoch {epoch})")

    encoder = model.encoder

    # ── Step 1: Encode TRAIN drones (with normalization) → centroids ──
    print("\n=== Step 1: Train drone centroids (normalized spectrograms) ===")
    train_ds = NormCheckDataset(H5_REMOTE, "train", include_negatives=False, max_per_type=500)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=2)

    train_embs, train_labels, train_types = [], [], []
    with torch.no_grad():
        for x, label, tname in train_dl:
            z = encoder(x.to(device))
            train_embs.append(z.cpu().numpy())
            train_labels.extend(label.tolist())
            train_types.extend(list(tname))
    train_embs = np.concatenate(train_embs)
    train_labels = np.array(train_labels)
    train_types = np.array(train_types)
    print(f"  {train_embs.shape[0]} train embeddings, {train_ds.n_drone_types} types")

    # Compute centroids
    n_types = train_ds.n_drone_types
    D = train_embs.shape[1]
    centroids = np.zeros((n_types, D))
    cov_inv_list = []
    reg = 1e-3

    for k in range(n_types):
        mask = train_labels == k
        cluster = train_embs[mask]
        centroids[k] = cluster.mean(axis=0)
        cov = np.cov(cluster.T) + reg * np.eye(D)
        try:
            cov_inv_list.append(np.linalg.inv(cov))
        except np.linalg.LinAlgError:
            cov_inv_list.append(np.linalg.pinv(cov))

    # ── Step 2: Encode HOLDOUT drones (normalized) ──
    print("\n=== Step 2: Holdout drones (normalized) ===")
    holdout_ds = NormCheckDataset(H5_REMOTE, "holdout", include_negatives=False, max_per_type=500)
    holdout_dl = DataLoader(holdout_ds, batch_size=64, shuffle=False, num_workers=2)

    holdout_embs, holdout_types = [], []
    with torch.no_grad():
        for x, label, tname in holdout_dl:
            z = encoder(x.to(device))
            holdout_embs.append(z.cpu().numpy())
            holdout_types.extend(list(tname))
    holdout_embs = np.concatenate(holdout_embs)
    holdout_types = np.array(holdout_types)
    print(f"  {holdout_embs.shape[0]} holdout embeddings")

    # Mahalanobis for holdout
    N_h = holdout_embs.shape[0]
    holdout_mahal = np.full(N_h, np.inf)
    for k in range(n_types):
        diff = holdout_embs - centroids[k]
        mahal_sq = np.sum(diff @ cov_inv_list[k] * diff, axis=1)
        mahal = np.sqrt(np.maximum(mahal_sq, 0))
        holdout_mahal = np.minimum(holdout_mahal, mahal)
    print(f"  Mean Mahalanobis: {holdout_mahal.mean():.2f}")

    # ── Step 3: Encode BACKGROUND negatives (normalized) ──
    print("\n=== Step 3: Background negatives (normalized) ===")
    neg_ds = NormCheckDataset(H5_REMOTE, "train", include_negatives=True, max_per_type=500)
    neg_indices = [i for i, t in enumerate(neg_ds.type_names_list) if t == "background"][:2000]

    neg_subset = torch.utils.data.Subset(neg_ds, neg_indices)
    neg_dl = DataLoader(neg_subset, batch_size=64, shuffle=False, num_workers=2)

    neg_embs = []
    with torch.no_grad():
        for x, label, tname in neg_dl:
            z = encoder(x.to(device))
            neg_embs.append(z.cpu().numpy())
    neg_embs = np.concatenate(neg_embs)
    print(f"  {neg_embs.shape[0]} negative embeddings")

    # Mahalanobis for negatives
    N_n = neg_embs.shape[0]
    neg_mahal = np.full(N_n, np.inf)
    for k in range(n_types):
        diff = neg_embs - centroids[k]
        mahal_sq = np.sum(diff @ cov_inv_list[k] * diff, axis=1)
        mahal = np.sqrt(np.maximum(mahal_sq, 0))
        neg_mahal = np.minimum(neg_mahal, mahal)
    print(f"  Mean Mahalanobis: {neg_mahal.mean():.2f}")

    # ── Step 4: Compare ──
    print("\n" + "="*70)
    print("NORMALIZED SPECTROGRAM CHECK")
    print("="*70)

    all_mahal = np.concatenate([holdout_mahal, neg_mahal])
    all_labels = np.concatenate([np.ones(N_h), np.zeros(N_n)])

    auc = roc_auc_score(all_labels, -all_mahal)

    print(f"\n  Holdout drones: mean={holdout_mahal.mean():.2f}, median={np.median(holdout_mahal):.2f}")
    print(f"  Background:     mean={neg_mahal.mean():.2f}, median={np.median(neg_mahal):.2f}")
    print(f"  Ratio (bg/drone): {neg_mahal.mean()/holdout_mahal.mean():.3f}")
    print(f"\n  AUC: {auc:.4f}")

    # ── Comparison with unnormalized results ──
    print(f"\n  --- Comparison ---")
    print(f"  Unnormalized: AUC=0.648, holdout=46.79, bg=49.40, ratio=1.056")
    print(f"  Normalized:   AUC={auc:.3f}, holdout={holdout_mahal.mean():.2f}, bg={neg_mahal.mean():.2f}, ratio={neg_mahal.mean()/holdout_mahal.mean():.3f}")

    print(f"\n{'='*70}")
    if auc > 0.85:
        print("  STRONG SIGNAL: Normalization didn't kill the separation!")
        print("  Real 'drone-ness' exists beyond value distribution differences.")
        print("  Fixing the data pipeline should give you a real detector.")
    elif auc > 0.70:
        print("  WEAK SIGNAL: Some separation survives normalization.")
        print("  There might be a small 'drone-ness' signal, but it's not strong.")
        print("  Better features or architecture might help.")
    elif auc > 0.55:
        print("  MARGINAL: Barely above random after normalization.")
        print("  The 100% binary accuracy was almost entirely the fingerprint shortcut.")
        print("  Drone detection from spectrograms is very hard with current approach.")
    else:
        print("  NO SIGNAL: Normalization killed all separation.")
        print("  The binary accuracy was 100% shortcut.")
    print(f"{'='*70}")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, max(holdout_mahal.max(), neg_mahal.max()), 80)
    ax.hist(holdout_mahal, bins=bins, alpha=0.6,
            label=f'Holdout Drones (mean={holdout_mahal.mean():.1f})',
            color='#2196F3', density=True)
    ax.hist(neg_mahal, bins=bins, alpha=0.6,
            label=f'Background (mean={neg_mahal.mean():.1f})',
            color='#F44336', density=True)
    ax.set_xlabel('Mahalanobis Distance (normalized spectrograms)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'After Per-Channel Normalization: AUC={auc:.3f}\n'
                 f'(Unnormalized AUC was 0.648)', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, min(bins[-1], 80))

    save_dir = "/models/detection_plots"
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/normalized_check.png", dpi=150, bbox_inches='tight')
    plt.close()
    MODEL_VOL.commit()
    print(f"\n  Plot saved to {save_dir}/normalized_check.png")


@app.local_entrypoint()
def main():
    check.remote()
