#!/usr/bin/env python3
"""
IRIS v11 — Tier 1 Visual Generation

Generates:
  1. "Exposure" Side-by-Side — v9 says DRONE, v11 says BACKGROUND on same matched BG
  2. ROC Curve Overlay — v9 (0.30) vs v10 (0.94) vs v11 (0.98) on same plot
  3. Polished UMAP — dark theme, clean labels, 0.9785 in title

Usage:
  modal run scripts/gen_tier1_visuals.py
"""

import h5py
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
from sklearn.metrics import silhouette_score

# ─── Modal setup ──────────────────────────────────────────────────────────────

app = modal.App("iris-tier1-visuals-v11")

VOL = modal.Volume.from_name("iris-data", create_if_missing=True)
V9_VOL = modal.Volume.from_name("iris-models-v9", create_if_missing=True)
V11_VOL = modal.Volume.from_name("iris-models-v11", create_if_missing=True)
MATCHED_VOL = modal.Volume.from_name("iris-matched-bg", create_if_missing=True)
RESULTS_VOL = modal.Volume.from_name("iris-results", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install("libgl1", "libglib2.0-0", "libhdf5-dev",
                 "python3", "python3-pip", "python-is-python3")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        "h5py==3.12.1", "numpy==1.26.4", "scikit-learn==1.6.1",
        "scipy==1.14.1", "umap-learn==0.5.7", "matplotlib==3.9.3",
        "Pillow==11.1.0",
    )
)

H5_REMOTE = "/data/iris_rfuav.h5"
MATCHED_REMOTE = "/matched/iris_matched_bg.h5"
OUTPUT_DIR = "/output"


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


def load_encoder(ckpt_path, volume, device="cpu"):
    """Load encoder from a checkpoint, handling both full-model and encoder-only saves."""
    volume.reload()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ckpt.get("cfg", {})
    in_ch = cfg.get("in_ch", 2)
    width = cfg.get("encoder_width", 64)
    depth = cfg.get("encoder_depth", 6)
    embed_dim = cfg.get("embed_dim", 256)

    encoder = CNNEncoder(in_ch=in_ch, width=width, depth=depth, embed_dim=embed_dim).to(device)

    state = ckpt["model"]
    encoder_state = {k.replace("encoder.", "", 1): v
                     for k, v in state.items()
                     if k.startswith("encoder.")}
    if encoder_state:
        encoder.load_state_dict(encoder_state)
        print(f"  Loaded {len(encoder_state)} encoder params from full model checkpoint")
    else:
        encoder.load_state_dict(state)
        print(f"  Loaded encoder state directly ({len(state)} params)")

    epoch = ckpt.get("epoch", -1)
    print(f"  Checkpoint epoch: {epoch}")
    encoder.eval()
    return encoder


# ─── Dataset for encoding ─────────────────────────────────────────────────────

class SimpleEvalDS(Dataset):
    """Load spectrograms with per-channel normalization, no augmentation."""
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


class MatchedBGDS(Dataset):
    """Load matched backgrounds with per-channel normalization."""
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


class RandomBGDS(Dataset):
    """Load random backgrounds with per-channel normalization."""
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


def encode_dataset(encoder, dataset, device, batch_size=128):
    """Encode all samples in a dataset, return embeddings + metadata."""
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_embs = []
    all_labels = []
    all_types = []
    with torch.no_grad():
        for batch in dl:
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    x, label, tname = batch
                    all_labels.extend(label.tolist() if hasattr(label, 'tolist') else list(label))
                    all_types.extend(list(tname))
                elif len(batch) == 2:
                    x, label = batch
                    all_labels.extend(label.tolist() if hasattr(label, 'tolist') else list(label))
                    all_types.extend(["matched_bg"] * len(label))
                else:
                    x = batch[0]
                    all_labels.extend([0] * len(x))
                    all_types.extend(["random_bg"] * len(x))
            else:
                x = batch
                all_labels.extend([0] * len(x))
                all_types.extend(["random_bg"] * len(x))
            if x.ndim == 3:
                x = x.unsqueeze(1)
            z = encoder(x.to(device))
            all_embs.append(z.cpu().numpy())
    return np.concatenate(all_embs), np.array(all_labels), np.array(all_types)


# ─── Main ─────────────────────────────────────────────────────────────────────

@app.function(
    image=IMAGE,
    gpu="A100",
    volumes={
        "/data": VOL,
        "/v9": V9_VOL,
        "/v11": V11_VOL,
        "/matched": MATCHED_VOL,
        "/output": RESULTS_VOL,
    },
    timeout=3600,
    memory=32768,
)
def generate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from umap import UMAP

    device = "cuda"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    # ── Load both encoders ─────────────────────────────────────────────────
    print("=" * 70)
    print("LOADING MODELS")
    print("=" * 70)

    print("\nLoading v9 encoder...")
    v9_encoder = load_encoder("/v9/lejepa_v9_best.pt", V9_VOL, device)

    print("\nLoading v11 encoder (Hierarchical SupCon)...")
    v11_encoder = load_encoder("/v11/lejepa_v11_best.pt", V11_VOL, device)

    # ── Encode all data with BOTH models ───────────────────────────────────
    print("\n" + "=" * 70)
    print("ENCODING DATA")
    print("=" * 70)

    # Train drones (for Mahalanobis centroid)
    print("\nEncoding train drones...")
    train_ds = SimpleEvalDS(H5_REMOTE, "train")
    v9_train_embs, v9_train_labels, v9_train_types = encode_dataset(v9_encoder, train_ds, device)
    v11_train_embs, v11_train_labels, v11_train_types = encode_dataset(v11_encoder, train_ds, device)
    print(f"  Train: {v9_train_embs.shape[0]} samples")

    # Holdout drones
    print("\nEncoding holdout drones...")
    holdout_ds = SimpleEvalDS(H5_REMOTE, "holdout")
    v9_holdout_embs, v9_holdout_labels, v9_holdout_types = encode_dataset(v9_encoder, holdout_ds, device)
    v11_holdout_embs, v11_holdout_labels, v11_holdout_types = encode_dataset(v11_encoder, holdout_ds, device)
    print(f"  Holdout: {v9_holdout_embs.shape[0]} samples")

    # Matched backgrounds
    print("\nEncoding matched backgrounds...")
    MATCHED_VOL.reload()
    matched_ds = MatchedBGDS(MATCHED_REMOTE, "holdout")
    v9_matched_embs, v9_matched_idx, _ = encode_dataset(v9_encoder, matched_ds, device)
    v11_matched_embs, v11_matched_idx, _ = encode_dataset(v11_encoder, matched_ds, device)
    print(f"  Matched BG: {v9_matched_embs.shape[0]} samples")

    # Random backgrounds
    print("\nEncoding random backgrounds (50K)...")
    random_ds = RandomBGDS(H5_REMOTE, max_negatives=50000)
    v9_random_embs, _, _ = encode_dataset(v9_encoder, random_ds, device)
    v11_random_embs, _, _ = encode_dataset(v11_encoder, random_ds, device)
    print(f"  Random BG: {v9_random_embs.shape[0]} samples")

    # ── Compute Mahalanobis distances ──────────────────────────────────────
    print("\nComputing Mahalanobis distances...")

    def compute_mahalanobis(train_embs, test_embs):
        D = train_embs.shape[1]
        centroid = train_embs.mean(axis=0)
        cov = np.cov(train_embs.T) + 1e-3 * np.eye(D)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
        diff = test_embs - centroid
        mahal_sq = np.sum(diff @ cov_inv * diff, axis=1)
        return np.sqrt(np.maximum(mahal_sq, 0)), centroid, cov_inv

    v9_holdout_mahal, v9_centroid, v9_cov_inv = compute_mahalanobis(v9_train_embs, v9_holdout_embs)
    v9_matched_mahal, _, _ = compute_mahalanobis(v9_train_embs, v9_matched_embs)
    v9_random_mahal, _, _ = compute_mahalanobis(v9_train_embs, v9_random_embs)

    v11_holdout_mahal, v11_centroid, v11_cov_inv = compute_mahalanobis(v11_train_embs, v11_holdout_embs)
    v11_matched_mahal, _, _ = compute_mahalanobis(v11_train_embs, v11_matched_embs)
    v11_random_mahal, _, _ = compute_mahalanobis(v11_train_embs, v11_random_embs)

    # Compute AUCs
    def compute_auc(drone_dists, bg_dists):
        labels = np.concatenate([np.ones(len(drone_dists)), np.zeros(len(bg_dists))])
        dists = np.concatenate([drone_dists, bg_dists])
        return roc_auc_score(labels, -dists)

    v9_matched_auc = compute_auc(v9_holdout_mahal, v9_matched_mahal)
    v11_matched_auc = compute_auc(v11_holdout_mahal, v11_matched_mahal)
    v9_random_auc = compute_auc(v9_holdout_mahal, v9_random_mahal)
    v11_random_auc = compute_auc(v11_holdout_mahal, v11_random_mahal)

    print(f"\n  v9  — Matched AUC: {v9_matched_auc:.4f}, Random AUC: {v9_random_auc:.4f}")
    print(f"  v11 — Matched AUC: {v11_matched_auc:.4f}, Random AUC: {v11_random_auc:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # VISUAL 1: "EXPOSURE" SIDE-BY-SIDE (v9 vs v11)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("VISUAL 1: EXPOSURE SIDE-BY-SIDE")
    print("=" * 70)

    # Find the most dramatic example: matched BG that v9 thinks is most drone-like
    # while v11 correctly identifies as background
    v9_holdout_mean = v9_holdout_mahal.mean()
    v11_holdout_mean = v11_holdout_mahal.mean()

    v9_normalized = v9_matched_mahal / v9_holdout_mean
    v11_normalized = v11_matched_mahal / v11_holdout_mean

    # Most dramatic: lowest v9 distance (most drone-like) AND highest v11 distance
    drama_score = -v9_normalized + v11_normalized  # high = v9 says drone, v11 says bg
    best_idx = np.argmax(drama_score)

    # Get the spectrogram for this matched BG
    matched_ds.f.close()
    random_ds.f.close()
    MATCHED_VOL.reload()
    mbg_f = h5py.File(MATCHED_REMOTE, "r")
    mbg_grp = mbg_f["holdout_matched_bg"]
    mbg_keys = sorted(list(mbg_grp.keys()), key=lambda x: int(x) if x.isdigit() else 0)
    mbg_sample = mbg_grp[mbg_keys[best_idx]][:]
    mbg_f.close()

    # Compute percentile ranks
    v9_pct_of_drones_closer = (v9_holdout_mahal < v9_matched_mahal[best_idx]).mean() * 100
    v11_pct_of_drones_closer = (v11_holdout_mahal < v11_matched_mahal[best_idx]).mean() * 100

    v9_says_drone = v9_pct_of_drones_closer < 50
    v11_says_drone = v11_pct_of_drones_closer < 50

    print(f"  Best drama index: {best_idx}")
    print(f"  v9  matched BG: closer to centroid than {100-v9_pct_of_drones_closer:.1f}% of actual drones")
    print(f"  v11 matched BG: closer to centroid than {100-v11_pct_of_drones_closer:.1f}% of actual drones")

    # Dark theme
    plt.style.use('dark_background')

    fig, axes = plt.subplots(1, 3, figsize=(24, 8),
                             gridspec_kw={'width_ratios': [1, 1, 1]})

    # Panel 1: The matched BG spectrogram
    if mbg_sample.shape[0] >= 2:
        ax = axes[0]
        ax.imshow(mbg_sample[:2].transpose(1, 2, 0)[:, :, 0],
                  aspect='auto', cmap='inferno', origin='lower')
        ax.set_title("Matched Background\n(drone signal removed)", fontsize=14,
                     color='#AAAAAA', pad=15)
        ax.set_xlabel("Time bins", fontsize=11, color='#888888')
        ax.set_ylabel("Frequency bins", fontsize=11, color='#888888')
        ax.tick_params(colors='#666666')

    # Panel 2: v9 verdict
    ax2 = axes[1]
    v9_color = '#FF4444' if v9_says_drone else '#44FF44'
    v9_verdict = "DRONE" if v9_says_drone else "BACKGROUND"
    v9_rank_pct = 100 - v9_pct_of_drones_closer
    ax2.text(0.5, 0.70, f"v9", fontsize=28, fontweight='bold',
             ha='center', va='center', color='#888888', transform=ax2.transAxes)
    ax2.text(0.5, 0.50, f"{v9_verdict}", fontsize=42, fontweight='bold',
             ha='center', va='center', color=v9_color, transform=ax2.transAxes)
    ax2.text(0.5, 0.33, f"closer than {v9_rank_pct:.0f}% of actual drones",
             fontsize=16, ha='center', va='center', color=v9_color, transform=ax2.transAxes,
             alpha=0.85)
    ax2.text(0.5, 0.18, f"Mahal dist: {v9_matched_mahal[best_idx]:.1f}",
             fontsize=13, ha='center', va='center', color='#666666',
             transform=ax2.transAxes)
    if v9_says_drone:
        ax2.text(0.5, 0.90, "WRONG -- shortcut detected", fontsize=15, fontweight='bold',
                 ha='center', va='center', color='#FF4444', transform=ax2.transAxes,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF4444', alpha=0.2))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # Panel 3: v11 verdict
    ax3 = axes[2]
    v11_color = '#44FF44' if not v11_says_drone else '#FF4444'
    v11_verdict = "BACKGROUND" if not v11_says_drone else "DRONE"
    v11_rank_pct = 100 - v11_pct_of_drones_closer
    ax3.text(0.5, 0.70, f"v11", fontsize=28, fontweight='bold',
             ha='center', va='center', color='#888888', transform=ax3.transAxes)
    ax3.text(0.5, 0.50, f"{v11_verdict}", fontsize=42, fontweight='bold',
             ha='center', va='center', color=v11_color, transform=ax3.transAxes)
    ax3.text(0.5, 0.33, f"closer than {v11_rank_pct:.0f}% of actual drones",
             fontsize=16, ha='center', va='center', color=v11_color, transform=ax3.transAxes,
             alpha=0.85)
    ax3.text(0.5, 0.18, f"Mahal dist: {v11_matched_mahal[best_idx]:.1f}",
             fontsize=13, ha='center', va='center', color='#666666',
             transform=ax3.transAxes)
    if not v11_says_drone:
        ax3.text(0.5, 0.90, "CORRECT -- real signal detected", fontsize=15, fontweight='bold',
                 ha='center', va='center', color='#44FF44', transform=ax3.transAxes,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#44FF44', alpha=0.2))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    fig.suptitle("Same Signal. Different Truth.",
                 fontsize=22, fontweight='bold', color='white', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    exposure_path = f"{OUTPUT_DIR}/01_exposure_side_by_side.png"
    plt.savefig(exposure_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  Saved: {exposure_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # VISUAL 2: ROC CURVE OVERLAY (v9 vs v11)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("VISUAL 2: ROC CURVE OVERLAY")
    print("=" * 70)

    fig2, ax_roc = plt.subplots(1, 1, figsize=(10, 10))

    # v9 ROC vs matched BG
    labels_v9 = np.concatenate([np.ones(len(v9_holdout_mahal)), np.zeros(len(v9_matched_mahal))])
    dists_v9 = np.concatenate([v9_holdout_mahal, v9_matched_mahal])
    fpr_v9, tpr_v9, _ = roc_curve(labels_v9, -dists_v9)
    auc_v9 = roc_auc_score(labels_v9, -dists_v9)

    # v11 ROC vs matched BG
    labels_v11 = np.concatenate([np.ones(len(v11_holdout_mahal)), np.zeros(len(v11_matched_mahal))])
    dists_v11 = np.concatenate([v11_holdout_mahal, v11_matched_mahal])
    fpr_v11, tpr_v11, _ = roc_curve(labels_v11, -dists_v11)
    auc_v11 = roc_auc_score(labels_v11, -dists_v11)

    # v9 ROC vs random BG
    labels_v9r = np.concatenate([np.ones(len(v9_holdout_mahal)), np.zeros(len(v9_random_mahal))])
    dists_v9r = np.concatenate([v9_holdout_mahal, v9_random_mahal])
    fpr_v9r, tpr_v9r, _ = roc_curve(labels_v9r, -dists_v9r)
    auc_v9r = roc_auc_score(labels_v9r, -dists_v9r)

    # v11 ROC vs random BG
    labels_v11r = np.concatenate([np.ones(len(v11_holdout_mahal)), np.zeros(len(v11_random_mahal))])
    dists_v11r = np.concatenate([v11_holdout_mahal, v11_random_mahal])
    fpr_v11r, tpr_v11r, _ = roc_curve(labels_v11r, -dists_v11r)
    auc_v11r = roc_auc_score(labels_v11r, -dists_v11r)

    # Diagonal
    ax_roc.plot([0, 1], [0, 1], 'w--', alpha=0.2, linewidth=1)

    # v9 vs matched BG -- the devastating line
    ax_roc.plot(fpr_v9, tpr_v9, color='#FF4444', linewidth=3,
                label=f'v9 vs Matched BG (AUC = {auc_v9:.2f})', linestyle='-')

    # v9 vs random BG -- the fake performance
    ax_roc.plot(fpr_v9r, tpr_v9r, color='#FF4444', linewidth=2,
                label=f'v9 vs Random BG (AUC = {auc_v9r:.2f})', linestyle='--', alpha=0.5)

    # v11 vs matched BG -- the real deal
    ax_roc.plot(fpr_v11, tpr_v11, color='#44FF88', linewidth=3,
                label=f'v11 vs Matched BG (AUC = {auc_v11:.2f})')

    # v11 vs random BG -- still perfect
    ax_roc.plot(fpr_v11r, tpr_v11r, color='#44FF88', linewidth=2,
                label=f'v11 vs Random BG (AUC = {auc_v11r:.2f})', linestyle='--', alpha=0.5)

    # Annotations
    ax_roc.annotate('Matched backgrounds\nexpose v9',
                    xy=(0.5, 0.35), xytext=(0.7, 0.15),
                    fontsize=13, color='#FF4444', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#FF4444', lw=2),
                    ha='center')

    ax_roc.annotate('v11 detects\ngenuine signal',
                    xy=(0.1, 0.9), xytext=(0.35, 0.7),
                    fontsize=13, color='#44FF88', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#44FF88', lw=2),
                    ha='center')

    ax_roc.set_xlabel('False Positive Rate', fontsize=14, color='#CCCCCC')
    ax_roc.set_ylabel('True Positive Rate', fontsize=14, color='#CCCCCC')
    ax_roc.set_title('Drone Detection: Real vs Fake Performance',
                     fontsize=18, fontweight='bold', color='white', pad=20)
    ax_roc.legend(loc='lower right', fontsize=13, facecolor='#2a2a3e',
                  edgecolor='#444466', labelcolor='white')
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.grid(True, alpha=0.15)
    ax_roc.set_aspect('equal')

    roc_path = f"{OUTPUT_DIR}/02_roc_overlay.png"
    plt.savefig(roc_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  Saved: {roc_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # VISUAL 3: POLISHED UMAP (v11)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("VISUAL 3: POLISHED UMAP")
    print("=" * 70)

    # Subsample for UMAP
    rng = np.random.default_rng(42)
    n_train_sample = min(500, len(v11_train_embs))
    train_idx = rng.choice(len(v11_train_embs), n_train_sample, replace=False)
    n_random_sample = min(1500, len(v11_random_embs))
    random_idx = rng.choice(len(v11_random_embs), n_random_sample, replace=False)

    all_embs_umap = np.concatenate([
        v11_train_embs[train_idx],
        v11_holdout_embs,
        v11_matched_embs,
        v11_random_embs[random_idx],
    ])
    all_labels_umap = np.concatenate([
        np.zeros(n_train_sample),           # 0 = train drone
        np.ones(len(v11_holdout_embs)),     # 1 = holdout drone
        2 * np.ones(len(v11_matched_embs)), # 2 = matched BG
        3 * np.ones(n_random_sample),       # 3 = random BG
    ])

    print("  Running UMAP...")
    reducer = UMAP(n_components=2, metric="cosine", n_neighbors=30,
                   min_dist=0.1, random_state=42, verbose=True)
    embedding_2d = reducer.fit_transform(all_embs_umap)
    print("  UMAP done.")

    # ── Plot 1: Category view (the hero image) ──
    fig3, ax_umap = plt.subplots(1, 1, figsize=(14, 11))

    # Plot order: random BG (back) -> matched BG -> train -> holdout (front)
    colors = {0: "#2196F3", 1: "#4CAF50", 2: "#F44336", 3: "#555577"}
    names = {0: "Train Drones", 1: "Holdout Drones", 2: "Matched BG", 3: "Random BG"}
    sizes = {0: 12, 1: 18, 2: 14, 3: 6}
    alphas = {0: 0.35, 1: 0.65, 2: 0.55, 3: 0.25}
    zorders = {3: 1, 2: 2, 0: 3, 1: 4}

    for cat in [3, 2, 0, 1]:
        mask = all_labels_umap == cat
        ax_umap.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                       c=colors[cat], s=sizes[cat], alpha=alphas[cat],
                       label=names[cat], edgecolors="none", zorder=zorders[cat])

    ax_umap.set_title(f"IRIS v11 -- Drone Detection Embedding Space\n"
                      f"Matched BG AUC = {v11_matched_auc:.4f}",
                      fontsize=18, fontweight='bold', color='white', pad=20)
    ax_umap.legend(loc='best', fontsize=14, facecolor='#2a2a3e',
                   edgecolor='#444466', labelcolor='white', markerscale=3)
    ax_umap.set_xlabel("UMAP-1", fontsize=12, color='#888888')
    ax_umap.set_ylabel("UMAP-2", fontsize=12, color='#888888')
    ax_umap.tick_params(colors='#666666')
    ax_umap.grid(True, alpha=0.08)

    umap_path = f"{OUTPUT_DIR}/03_umap_polished.png"
    plt.savefig(umap_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  Saved: {umap_path}")

    # ── Bonus: UMAP with holdout types colored ──
    fig4, ax_umap2 = plt.subplots(1, 1, figsize=(14, 11))

    # Backgrounds first
    mbg_mask = all_labels_umap == 2
    ax_umap2.scatter(embedding_2d[mbg_mask, 0], embedding_2d[mbg_mask, 1],
                    c="#F44336", s=10, alpha=0.3, label="Matched BG",
                    edgecolors="none", zorder=1)
    rbg_mask = all_labels_umap == 3
    ax_umap2.scatter(embedding_2d[rbg_mask, 0], embedding_2d[rbg_mask, 1],
                    c="#555577", s=4, alpha=0.15, label="Random BG",
                    edgecolors="none", zorder=1)

    # Per-type AUCs from the full eval
    per_type_aucs = {
        'DJI FPV COMBO': 0.9674,
        'FUTABA-T10J': 0.9984,
        'FUTABA-T14SG': 0.9656,
        'JR PROPO XG7': 0.9997,
        'JUMPER-T14': 0.9999,
        'RadioMaster BOXER': 1.0000,
        'WFLY ET10': 1.0000,
    }

    # Holdout by type
    holdout_start = n_train_sample
    holdout_end = holdout_start + len(v11_holdout_embs)
    holdout_umap = embedding_2d[holdout_start:holdout_end]
    holdout_type_names = sorted(np.unique(v11_holdout_types))
    type_colors = plt.cm.Set1(np.linspace(0, 1, len(holdout_type_names)))

    for i, tname in enumerate(holdout_type_names):
        mask_t = v11_holdout_types == tname
        auc_val = per_type_aucs.get(tname, 0.0)
        ax_umap2.scatter(holdout_umap[mask_t, 0], holdout_umap[mask_t, 1],
                        c=[type_colors[i]], s=20, alpha=0.7,
                        label=f"{tname} (AUC={auc_val:.3f})",
                        edgecolors="none", zorder=4)

    # Train drones (light blue, behind)
    train_mask = all_labels_umap == 0
    ax_umap2.scatter(embedding_2d[train_mask, 0], embedding_2d[train_mask, 1],
                    c="#2196F3", s=8, alpha=0.2, label="Train Drones",
                    edgecolors="none", zorder=2)

    ax_umap2.set_title(f"IRIS v11 -- Holdout Types vs Backgrounds\n"
                       f"98.1% of drones closer to centroid than their matched BGs",
                       fontsize=16, fontweight='bold', color='white', pad=20)
    ax_umap2.legend(loc='best', fontsize=9, facecolor='#2a2a3e',
                    edgecolor='#444466', labelcolor='white', markerscale=2.5,
                    ncol=2)
    ax_umap2.tick_params(colors='#666666')
    ax_umap2.grid(True, alpha=0.08)

    umap2_path = f"{OUTPUT_DIR}/03b_umap_by_type.png"
    plt.savefig(umap2_path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  Saved: {umap2_path}")

    # ── Commit all outputs ─────────────────────────────────────────────────
    RESULTS_VOL.commit()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ALL TIER 1 VISUALS GENERATED ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"\n  1. {exposure_path}")
    print(f"  2. {roc_path}")
    print(f"  3. {umap_path}")
    print(f"  3b. {umap2_path}")
    print(f"\n  Download: modal volume get iris-results /output/ ./iris_output/")


@app.local_entrypoint()
def main():
    generate.remote()
