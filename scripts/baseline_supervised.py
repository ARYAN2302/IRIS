#!/usr/bin/env python3
"""
IRIS Baseline — Supervised Binary Classifier (Drone vs Matched BG)

Same CNN encoder architecture as v11, but trained with BCE loss instead of
self-supervised objectives. This is the upper bound: a model that gets
flight-level labels for free (drone=1, matched_bg=0).

If IRIS v11 gets close to this without flight labels, the self-supervised
approach is justified.

Evaluation: Same as v11 — Mahalanobis distance to drone centroid, tested on
7 holdout drone types with matched backgrounds.

Usage:
  modal run scripts/baseline_supervised.py
"""

import h5py
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, roc_curve

# ─── Modal setup ──────────────────────────────────────────────────────────────

app = modal.App("iris-baseline-supervised")

VOL = modal.Volume.from_name("iris-data", create_if_missing=True)
MODEL_VOL = modal.Volume.from_name("iris-models-baseline", create_if_missing=True)
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

# ─── Hyperparameters ──────────────────────────────────────────────────────────

CFG = dict(
    # Architecture (SAME as v11 — only the loss changes)
    in_ch=2,
    encoder_depth=6,
    encoder_width=64,
    embed_dim=256,

    # Classification head
    hidden_dim=128,

    # Loss
    label_smoothing=0.0,  # no smoothing for baseline — give it every advantage

    # Optimizer
    lr=1e-3,
    weight_decay=1e-4,
    warmup_steps=5000,
    batch_size=128,
    grad_accum_steps=1,

    # Training
    epochs=50,
    eval_every=1,
    early_stop_patience=10,

    # Data
    img_size=256,
    num_workers=4,
)


# ─── Augmentation ─────────────────────────────────────────────────────────────

class SpectrogramAugment:
    def __init__(self, img_size=256, freq_mask_ratio=0.08, time_mask_ratio=0.08,
                 noise_std=0.03, crop_range=(0.85, 1.0)):
        self.img_size = img_size
        self.freq_mask_ratio = freq_mask_ratio
        self.time_mask_ratio = time_mask_ratio
        self.noise_std = noise_std
        self.crop_range = crop_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        C, H, W = x.shape

        # Random crop + resize
        scale = torch.empty(1).uniform_(self.crop_range[0], self.crop_range[1]).item()
        new_h, new_w = int(H * scale), int(W * scale)
        top = torch.randint(0, H - new_h + 1, (1,)).item()
        left = torch.randint(0, W - new_w + 1, (1,)).item()
        x_aug = x[:, top:top+new_h, left:left+new_w]
        x_aug = F.interpolate(x_aug.unsqueeze(0), size=(H, W), mode='bilinear',
                              align_corners=False).squeeze(0)

        # Frequency masking
        freq_mask_size = int(H * self.freq_mask_ratio)
        if freq_mask_size > 0:
            f_start = torch.randint(0, H - freq_mask_size + 1, (1,)).item()
            x_aug[:, f_start:f_start+freq_mask_size, :] = 0

        # Time masking
        time_mask_size = int(W * self.time_mask_ratio)
        if time_mask_size > 0:
            t_start = torch.randint(0, W - time_mask_size + 1, (1,)).item()
            x_aug[:, :, t_start:t_start+time_mask_size] = 0

        # Additive noise
        if self.noise_std > 0:
            x_aug = x_aug + torch.randn_like(x_aug) * self.noise_std

        return x_aug


# ─── Dataset ──────────────────────────────────────────────────────────────────

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


class SupervisedBinaryDataset(Dataset):
    """
    Binary drone (1) vs matched background (0) dataset.
    Same data as v11 training, but labels are just 0/1 instead of type IDs.
    """
    def __init__(self, h5_path: str, matched_path: str, split_key: str = "train",
                 augment=None):
        self.augment = augment

        # ── Load drone data ──
        self.drone_f = h5py.File(h5_path, "r")
        grp = self.drone_f[split_key]
        self.type_names = []
        self._resolved = {}
        self._sub_keys = {}

        for key in sorted(grp.keys()):
            try:
                ds_or_grp, n_samples, is_multi = _resolve_type_dataset(grp, key)
                self.type_names.append(key)
                self._resolved[key] = (ds_or_grp, n_samples, is_multi)
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
            except ValueError as e:
                print(f"  Skipping '{key}': {e}")
                continue

        # Build drone index
        self.drone_index = []
        for tname in self.type_names:
            _, n_samples, _ = self._resolved[tname]
            for i in range(n_samples):
                self.drone_index.append((tname, i))

        n_drones = len(self.drone_index)
        self.n_drone_types = len(self.type_names)

        # ── Load matched backgrounds ──
        self.matched_f = h5py.File(matched_path, "r")
        matched_key = f"{split_key}_matched_bg"

        if matched_key not in self.matched_f:
            raise ValueError(f"No matched backgrounds found at '{matched_key}' in {matched_path}.")

        self.matched_grp = self.matched_f[matched_key]
        self.matched_keys = sorted(list(self.matched_grp.keys()),
                                    key=lambda x: int(x) if x.isdigit() else 0)
        n_matched = len(self.matched_keys)

        print(f"  SupervisedBinaryDataset: {self.n_drone_types} drone types + matched bg")
        print(f"    Drones: {n_drones}")
        print(f"    Matched BG: {n_matched}")

        # Combined index: (source, local_idx)
        n_pairs = min(n_drones, n_matched)
        self.index = []
        for i in range(n_pairs):
            self.index.append(("drone", i))
        for i in range(n_pairs):
            self.index.append(("matched_bg", i))

        print(f"    Training pairs: {n_pairs}")
        print(f"    Total samples: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def _read_drone(self, tname, local_idx):
        ds_or_grp, n_samples, is_multi = self._resolved[tname]
        if is_multi:
            sub_key = self._sub_keys[tname][local_idx]
            return ds_or_grp[sub_key][:]
        else:
            return ds_or_grp[local_idx]

    def _normalize_per_channel(self, x):
        for c in range(x.shape[0]):
            ch = x[c]
            ch_std = ch.std()
            if ch_std > 1e-6:
                x[c] = (ch - ch.mean()) / ch_std
            else:
                x[c] = ch - ch.mean()
        return x

    def _prepare_tensor(self, sample):
        if sample.shape[0] == 3:
            x = torch.from_numpy(sample[:2].copy()).float()
        elif sample.shape[0] == 2:
            x = torch.from_numpy(sample.copy()).float()
        else:
            x = torch.from_numpy(sample[:2].copy()).float()
        return x

    def __getitem__(self, idx):
        source, local_idx = self.index[idx]

        if source == "drone":
            tname, drone_local = self.drone_index[local_idx]
            sample = self._read_drone(tname, drone_local)
            x = self._prepare_tensor(sample)
            binary_label = 1  # drone
        else:  # matched_bg
            key = self.matched_keys[local_idx]
            sample = self.matched_grp[key][:]
            x = self._prepare_tensor(sample)
            binary_label = 0  # background

        # Per-channel normalization
        x = self._normalize_per_channel(x)

        if self.augment:
            x = self.augment(x)

        return x, binary_label


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


class SupervisedBinaryClassifier(nn.Module):
    """
    Same encoder as v11, but with a binary classification head instead of
    LeJEPA predictor/projector. The encoder weights are the only thing
    that matters for fair comparison — same architecture, different objective.
    """
    def __init__(self, cfg):
        super().__init__()
        # Same encoder as v11
        layers = []
        ch = cfg["in_ch"]
        for i in range(cfg["encoder_depth"]):
            out_ch = min(cfg["encoder_width"] * (2 ** (i // 2)), 512)
            layers.append(ConvBlock(ch, out_ch))
            layers.append(nn.MaxPool2d(2))
            ch = out_ch
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, cfg["in_ch"], 256, 256)
            out = self.conv(dummy)
            flat = out.numel() // out.shape[0]

        self.encoder_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, cfg["embed_dim"]),
            nn.BatchNorm1d(cfg["embed_dim"]),
        )

        # Binary classification head on top of encoder
        self.classifier = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["hidden_dim"]),
            nn.BatchNorm1d(cfg["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(cfg["hidden_dim"], 1),  # single logit for BCE
        )

    def encode(self, x):
        """Return 256-dim embedding (same as v11 encoder output)."""
        return self.encoder_head(self.conv(x))

    def forward(self, x):
        z = self.encode(x)
        logit = self.classifier(z)
        return z, logit


# ─── LR Schedule ──────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Evaluation (same protocol as v11) ────────────────────────────────────────

def evaluate_baseline(model, h5_path, matched_path, device="cuda"):
    """
    Evaluate the supervised baseline using the SAME detection protocol as v11:
    Mahalanobis distance to drone centroid, tested on holdout + matched BG.
    This makes the comparison apples-to-apples.
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    model.eval()
    encoder = model.encode

    # ── Encode train drones ──
    train_f = h5py.File(h5_path, "r")
    train_grp = train_f["train"]
    type_names = []
    _resolved = {}
    _sub_keys = {}

    for key in sorted(train_grp.keys()):
        try:
            ds_or_grp, n_samples, is_multi = _resolve_type_dataset(train_grp, key)
            type_names.append(key)
            _resolved[key] = (ds_or_grp, n_samples, is_multi)
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
                _sub_keys[key] = sub_keys
        except ValueError:
            continue

    # Encode train samples
    train_embs = []
    train_labels = []
    for label_idx, tname in enumerate(type_names):
        ds_or_grp, n_samples, is_multi = _resolved[tname]
        n_use = min(n_samples, 500)
        for i in range(n_use):
            if is_multi:
                sub_key = _sub_keys[tname][i]
                sample = ds_or_grp[sub_key][:]
            else:
                sample = ds_or_grp[i]
            if sample.shape[0] == 3:
                x = torch.from_numpy(sample[:2].copy()).float()
            elif sample.shape[0] == 2:
                x = torch.from_numpy(sample.copy()).float()
            else:
                x = torch.from_numpy(sample[:2].copy()).float()
            # Normalize
            for c in range(x.shape[0]):
                ch = x[c]
                ch_std = ch.std()
                if ch_std > 1e-6:
                    x[c] = (ch - ch.mean()) / ch_std
                else:
                    x[c] = ch - ch.mean()
            with torch.no_grad():
                z = encoder(x.unsqueeze(0).to(device))
            train_embs.append(z.cpu().numpy().squeeze())
            train_labels.append(label_idx)

    train_embs = np.array(train_embs)
    train_labels = np.array(train_labels)
    n_types = len(type_names)
    print(f"  Train embeddings: {train_embs.shape}, {n_types} types")

    # ── Compute global drone centroid + covariance ──
    D = train_embs.shape[1]
    centroid = train_embs.mean(axis=0)
    cov = np.cov(train_embs.T) + 1e-3 * np.eye(D)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    # ── Encode holdout drones ──
    holdout_f = h5py.File(h5_path, "r")
    holdout_grp = holdout_f["holdout"]
    holdout_type_names = []
    holdout_resolved = {}
    holdout_sub_keys = {}

    for key in sorted(holdout_grp.keys()):
        try:
            ds_or_grp, n_samples, is_multi = _resolve_type_dataset(holdout_grp, key)
            holdout_type_names.append(key)
            holdout_resolved[key] = (ds_or_grp, n_samples, is_multi)
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
                holdout_sub_keys[key] = sub_keys
        except ValueError:
            continue

    holdout_embs = []
    holdout_type_list = []
    for tname in holdout_type_names:
        ds_or_grp, n_samples, is_multi = holdout_resolved[tname]
        n_use = min(n_samples, 500)
        for i in range(n_use):
            if is_multi:
                sub_key = holdout_sub_keys[tname][i]
                sample = ds_or_grp[sub_key][:]
            else:
                sample = ds_or_grp[i]
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
            with torch.no_grad():
                z = encoder(x.unsqueeze(0).to(device))
            holdout_embs.append(z.cpu().numpy().squeeze())
            holdout_type_list.append(tname)

    holdout_embs = np.array(holdout_embs)
    holdout_type_list = np.array(holdout_type_list)
    print(f"  Holdout embeddings: {holdout_embs.shape}, types: {holdout_type_names}")

    # ── Encode matched backgrounds for holdout ──
    matched_f = h5py.File(matched_path, "r")
    mbg_key = "holdout_matched_bg"
    if mbg_key not in matched_f:
        print(f"  WARNING: No {mbg_key} found, trying train_matched_bg")
        mbg_key = "train_matched_bg"
    mbg_grp = matched_f[mbg_key]
    mbg_keys = sorted(list(mbg_grp.keys()),
                       key=lambda x: int(x) if x.isdigit() else 0)
    n_mbg = min(len(mbg_keys), 2000)

    mbg_embs = []
    rng = np.random.default_rng(123)
    mbg_indices = rng.choice(len(mbg_keys), n_mbg, replace=False)
    for j in mbg_indices:
        key = mbg_keys[j]
        sample = mbg_grp[key][:]
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
        with torch.no_grad():
            z = encoder(x.unsqueeze(0).to(device))
        mbg_embs.append(z.cpu().numpy().squeeze())

    mbg_embs = np.array(mbg_embs)
    print(f"  Matched BG embeddings: {mbg_embs.shape}")

    # ── Encode random backgrounds ──
    if "negatives" in holdout_f:
        neg_item = holdout_f["negatives"]
        if isinstance(neg_item, h5py.Dataset):
            n_neg = min(neg_item.shape[0], 2000)
            neg_samples = neg_item[:n_neg]
        else:
            neg_keys = [sk for sk in neg_item.keys()
                        if isinstance(neg_item[sk], h5py.Dataset) and len(neg_item[sk].shape) == 3]
            n_neg = min(len(neg_keys), 2000)
            neg_samples_list = [neg_item[sk][:] for sk in neg_keys[:n_neg]]
            neg_samples = np.stack(neg_samples_list) if len(neg_samples_list) > 0 else np.array([])

        rand_bg_embs = []
        for sample in neg_samples:
            if sample.ndim == 2:
                sample = sample[np.newaxis, :, :]
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
            with torch.no_grad():
                z = encoder(x.unsqueeze(0).to(device))
            rand_bg_embs.append(z.cpu().numpy().squeeze())
        rand_bg_embs = np.array(rand_bg_embs)
        print(f"  Random BG embeddings: {rand_bg_embs.shape}")
    else:
        rand_bg_embs = None

    # ── Mahalanobis distances ──
    holdout_diff = holdout_embs - centroid
    holdout_mahal = np.sqrt(np.maximum(np.sum(holdout_diff @ cov_inv * holdout_diff, axis=1), 0))

    mbg_diff = mbg_embs - centroid
    mbg_mahal = np.sqrt(np.maximum(np.sum(mbg_diff @ cov_inv * mbg_diff, axis=1), 0))

    # ── AUC vs matched backgrounds ──
    all_d_matched = np.concatenate([holdout_mahal, mbg_mahal])
    all_l_matched = np.concatenate([np.ones(len(holdout_mahal)), np.zeros(len(mbg_mahal))])
    matched_auc = float(roc_auc_score(all_l_matched, -all_d_matched))

    # ── AUC vs random backgrounds ──
    random_auc = None
    if rand_bg_embs is not None and len(rand_bg_embs) > 0:
        rand_bg_diff = rand_bg_embs - centroid
        rand_bg_mahal = np.sqrt(np.maximum(np.sum(rand_bg_diff @ cov_inv * rand_bg_diff, axis=1), 0))
        all_d_random = np.concatenate([holdout_mahal, rand_bg_mahal])
        all_l_random = np.concatenate([np.ones(len(holdout_mahal)), np.zeros(len(rand_bg_mahal))])
        random_auc = float(roc_auc_score(all_l_random, -all_d_random))

    # ── Per-type AUC vs matched BG ──
    per_type_auc = {}
    for tname in holdout_type_names:
        mask = holdout_type_list == tname
        t_mahal = holdout_mahal[mask]
        all_d_t = np.concatenate([t_mahal, mbg_mahal])
        all_l_t = np.concatenate([np.ones(len(t_mahal)), np.zeros(len(mbg_mahal))])
        per_type_auc[tname] = float(roc_auc_score(all_l_t, -all_d_t))

    # ── Per-pair test ──
    n_pairs = min(len(holdout_mahal), len(mbg_mahal))
    drone_closer = np.sum(holdout_mahal[:n_pairs] < mbg_mahal[:n_pairs])
    per_pair_pct = float(drone_closer / n_pairs * 100)

    # ── Youden's J threshold ──
    fpr, tpr, thresholds = roc_curve(all_l_matched, -all_d_matched)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = -thresholds[best_idx]
    youden_j = float(j_scores[best_idx])

    results = {
        "matched_bg_auc": matched_auc,
        "random_bg_auc": random_auc,
        "per_type_auc": per_type_auc,
        "per_pair_drone_closer_pct": per_pair_pct,
        "youden_j": youden_j,
        "best_threshold": float(best_threshold),
        "n_holdout": len(holdout_embs),
        "n_matched_bg": len(mbg_embs),
    }

    print(f"\n{'='*70}")
    print(f"BASELINE SUPERVISED RESULTS")
    print(f"{'='*70}")
    print(f"  Matched BG AUC: {matched_auc:.4f}")
    if random_auc is not None:
        print(f"  Random BG AUC:  {random_auc:.4f}")
    print(f"  Per-pair drone closer: {per_pair_pct:.1f}%")
    print(f"  Youden's J: {youden_j:.4f} (threshold={best_threshold:.2f})")
    print(f"\n  Per-type AUC:")
    for tname, auc in sorted(per_type_auc.items()):
        print(f"    {tname:25s}: {auc:.4f}")
    print(f"{'='*70}")

    train_f.close()
    holdout_f.close()
    matched_f.close()
    model.train()

    return results


# ─── Training Loop ────────────────────────────────────────────────────────────

@app.function(
    image=IMAGE,
    gpu="A100",
    volumes={"/data": VOL, "/models": MODEL_VOL, "/matched": MATCHED_VOL},
    timeout=5400,
    memory=32768,
)
def train():
    cfg = CFG.copy()
    device = "cuda"

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ── Data ──
    augment = SpectrogramAugment(
        img_size=cfg["img_size"],
        freq_mask_ratio=0.08,
        time_mask_ratio=0.08,
        noise_std=0.03,
        crop_range=(0.85, 1.0),
    )

    train_ds = SupervisedBinaryDataset(
        H5_REMOTE, MATCHED_REMOTE, split_key="train",
        augment=augment,
    )

    model = SupervisedBinaryClassifier(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    steps_per_epoch = math.ceil(len(train_ds) / cfg["batch_size"])
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup_steps = min(cfg["warmup_steps"], total_steps // 4)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps
    )

    print(f"\n{'='*70}")
    print(f"IRIS BASELINE — Supervised Binary Classifier (Drone vs Matched BG)")
    print(f"{'='*70}")
    print(f"Dataset: {len(train_ds)} samples")
    print(f"Architecture: Same encoder as v11 + binary classifier head")
    print(f"Loss: BCEWithLogitsLoss (label_smoothing={cfg['label_smoothing']})")
    print(f"Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"LR: {cfg['lr']}, Weight decay: {cfg['weight_decay']}")
    print()

    effective_bs = min(cfg["batch_size"], len(train_ds))
    dl = DataLoader(
        train_ds,
        batch_size=effective_bs,
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    criterion = nn.BCEWithLogitsLoss()

    global_step = 0
    best_matched_auc = 0.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (x, label) in enumerate(dl):
            x = x.to(device, non_blocking=True)
            label = label.float().to(device, non_blocking=True)

            z, logit = model(x)
            loss = criterion(logit.squeeze(), label)

            loss.backward()

            if (batch_idx + 1) % cfg.get("grad_accum_steps", 1) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_losses.append(loss.item())

            # Accuracy
            pred = (logit.squeeze() > 0).float()
            epoch_correct += (pred == label).sum().item()
            epoch_total += label.shape[0]

            if (batch_idx + 1) % 50 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step {global_step} | loss={loss.item():.4f} "
                      f"acc={epoch_correct/max(epoch_total,1):.3f} lr={lr_now:.6f}")

        if len(epoch_losses) == 0:
            continue

        avg_loss = np.mean(epoch_losses)
        train_acc = epoch_correct / max(epoch_total, 1)

        # ── Evaluate ──
        if (epoch + 1) % cfg["eval_every"] == 0:
            eval_results = evaluate_baseline(model, H5_REMOTE, MATCHED_REMOTE, device)
            matched_auc = eval_results["matched_bg_auc"]

            print(f"\n{'='*70}")
            print(f"EPOCH {epoch} SUMMARY")
            print(f"  Train loss: {avg_loss:.4f}  Train acc: {train_acc:.3f}")
            print(f"  Matched BG AUC: {matched_auc:.4f}")
            print(f"  Random BG AUC: {eval_results.get('random_bg_auc', 'N/A')}")
            print(f"{'='*70}\n")

            # Save best
            if matched_auc > best_matched_auc:
                best_matched_auc = matched_auc
                best_epoch = epoch
                patience_counter = 0

                ckpt_path = "/models/baseline_supervised_best.pt"
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "matched_auc": matched_auc,
                    "results": eval_results,
                }, ckpt_path)
                MODEL_VOL.commit()
                print(f"  ** New best: {matched_auc:.4f} at epoch {epoch} **")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{cfg['early_stop_patience']})")

            if patience_counter >= cfg["early_stop_patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ── Final evaluation with best model ──
    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION (best epoch {best_epoch})")
    print(f"{'='*70}")

    ckpt = torch.load("/models/baseline_supervised_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    final_results = evaluate_baseline(model, H5_REMOTE, MATCHED_REMOTE, device)

    final_results["best_epoch"] = best_epoch
    final_results["training_epochs"] = epoch + 1
    final_results["model_type"] = "supervised_binary_classifier"

    with open("/models/baseline_supervised_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    MODEL_VOL.commit()
    print(f"\n  Results saved to /models/baseline_supervised_results.json")


@app.local_entrypoint()
def main():
    train.remote()
