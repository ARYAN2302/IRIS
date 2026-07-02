#!/usr/bin/env python3
"""
IRIS v8 — Global Mahalanobis Detection Fix

The per-type Mahalanobis approach failed because holdout drones are from
UNSEEN types — they're not close to any specific train type centroid.

This script uses GLOBAL Mahalanobis instead:
  - Compute ONE centroid from ALL train drone embeddings (not per-type)
  - Background should be far from this global drone centroid
  - Holdout drones should be close (they're in the "drone region")

Also compares:
  1. v8 encoder (epoch6 — best AUC in holdout eval)
  2. v7 encoder + norm inference (already proven AUC=0.921)

Usage:
  modal run scripts/detect_global_mahalanobis.py
"""

import h5py
import json
import os
import glob

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import roc_auc_score, roc_curve

# ─── Modal setup ──────────────────────────────────────────────────────────────

app = modal.App("iris-v8-global-mahal")

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


# ─── Model (same architecture as v7/v8) ──────────────────────────────────────

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

    def forward(self, x):
        z = self.encoder(x)
        p = self.projector(z)
        return z, p


# ─── Dataset ─────────────────────────────────────────────────────────────────

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


class IRISDataset(Dataset):
    """
    Flexible dataset: optionally applies per-channel normalization.
    If normalize=True, each channel is zero-mean unit-variance (v8 style).
    If normalize=False, raw spectrograms (v7 style).
    """
    def __init__(self, h5_path, split_key="train", include_negatives=False,
                 max_negatives=2000, normalize=True):
        self.f = h5py.File(h5_path, "r")
        self.normalize = normalize
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
            for i in range(n_samples):
                self.samples.append((split_key, tname, i))
                self.labels.append(label_idx)
                self.type_names_list.append(tname)

        self.n_drone_types = len(type_names)
        self.type_names = type_names

        neg_label = len(type_names)
        if include_negatives and "negatives" in self.f:
            neg_item = self.f["negatives"]
            if isinstance(neg_item, h5py.Dataset):
                n_neg = min(neg_item.shape[0], max_negatives)
                self._resolved[("negatives", None)] = (neg_item, n_neg, False)
            else:
                sub_keys = [sk for sk in neg_item.keys()
                            if isinstance(neg_item[sk], h5py.Dataset)
                            and len(neg_item[sk].shape) == 3]
                try:
                    sub_keys.sort(key=lambda x: int(x))
                except ValueError:
                    sub_keys.sort()
                n_neg = min(len(sub_keys), max_negatives)
                self._resolved[("negatives", None)] = (neg_item, n_neg, True)
                self._sub_keys[("negatives", None)] = sub_keys

            for i in range(n_neg):
                self.samples.append(("negatives", None, i))
                self.labels.append(neg_label)
                self.type_names_list.append("background")

    def __len__(self):
        return len(self.samples)

    def _read_sample(self, split_key, tname, local_idx):
        ds_or_grp, _, is_multi = self._resolved[(split_key, tname)]
        if is_multi:
            sub_key = self._sub_keys[(split_key, tname)][local_idx]
            return ds_or_grp[sub_key][:]
        else:
            return ds_or_grp[local_idx]

    def __getitem__(self, idx):
        split_key, tname, local_idx = self.samples[idx]
        sample = self._read_sample(split_key, tname, local_idx)

        if sample.shape[0] == 3:
            x = sample[:2].copy().astype(np.float32)
        elif sample.shape[0] == 2:
            x = sample.copy().astype(np.float32)
        else:
            x = sample[:2].copy().astype(np.float32)

        # ── Per-channel normalization (optional) ──
        if self.normalize:
            for c in range(x.shape[0]):
                ch = x[c]
                ch_std = ch.std()
                if ch_std > 1e-6:
                    x[c] = (ch - ch.mean()) / ch_std
                else:
                    x[c] = ch - ch.mean()

        return torch.from_numpy(x), self.labels[idx], self.type_names_list[idx]


# ─── Encoding helper ─────────────────────────────────────────────────────────

def encode_dataset(encoder, dataset, device, batch_size=64):
    """Encode entire dataset."""
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    all_embs, all_labels, all_types = [], [], []
    with torch.no_grad():
        for x, label, tname in dl:
            z = encoder(x.to(device))
            all_embs.append(z.cpu().numpy())
            all_labels.extend(label.tolist() if hasattr(label, 'tolist') else list(label))
            all_types.extend(list(tname))
    return np.concatenate(all_embs), np.array(all_labels), np.array(all_types)


# ─── Detection methods ───────────────────────────────────────────────────────

def global_mahalanobis(train_drone_embs, test_embs, test_types, reg=1e-3):
    """
    GLOBAL Mahalanobis: one centroid from ALL train drone embeddings.
    Tests: "is this sample in the drone region of embedding space?"
    """
    D = train_drone_embs.shape[1]
    centroid = train_drone_embs.mean(axis=0)
    cov = np.cov(train_drone_embs.T) + reg * np.eye(D)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    diff = test_embs - centroid
    mahal_sq = np.sum(diff @ cov_inv * diff, axis=1)
    mahal = np.sqrt(np.maximum(mahal_sq, 0))

    drone_mask = test_types != "background"
    bg_mask = test_types == "background"

    drone_dists = mahal[drone_mask]
    bg_dists = mahal[bg_mask]

    if len(drone_dists) > 0 and len(bg_dists) > 0:
        all_dists = np.concatenate([drone_dists, bg_dists])
        all_labels = np.concatenate([np.ones(len(drone_dists)),
                                      np.zeros(len(bg_dists))])
        auc = roc_auc_score(all_labels, -all_dists)
    else:
        auc = 0.5

    return {
        "drone_mean": float(drone_dists.mean()) if len(drone_dists) > 0 else 0,
        "drone_median": float(np.median(drone_dists)) if len(drone_dists) > 0 else 0,
        "bg_mean": float(bg_dists.mean()) if len(bg_dists) > 0 else 0,
        "bg_median": float(np.median(bg_dists)) if len(bg_dists) > 0 else 0,
        "ratio": float(bg_dists.mean() / drone_dists.mean())
                 if len(drone_dists) > 0 and drone_dists.mean() > 0 else 0,
        "auc": float(auc),
        "drone_dists": drone_dists,
        "bg_dists": bg_dists,
        "centroid": centroid,
        "cov_inv": cov_inv,
    }


def per_type_mahalanobis(train_embs, train_labels, test_embs, test_types, n_types, reg=1e-3):
    """
    PER-TYPE Mahalanobis: nearest centroid from train type centroids.
    Tests: "which specific drone type is this closest to?"
    This fails for unseen types — included for comparison.
    """
    D = train_embs.shape[1]
    centroids = np.zeros((n_types, D))
    cov_inv_list = []

    for k in range(n_types):
        mask = train_labels == k
        cluster = train_embs[mask]
        centroids[k] = cluster.mean(axis=0)
        cov = np.cov(cluster.T) + reg * np.eye(D)
        try:
            cov_inv_list.append(np.linalg.inv(cov))
        except np.linalg.LinAlgError:
            cov_inv_list.append(np.linalg.pinv(cov))

    N = test_embs.shape[0]
    min_mahal = np.full(N, np.inf)
    for k in range(n_types):
        diff = test_embs - centroids[k]
        mahal_sq = np.sum(diff @ cov_inv_list[k] * diff, axis=1)
        mahal = np.sqrt(np.maximum(mahal_sq, 0))
        min_mahal = np.minimum(min_mahal, mahal)

    drone_mask = test_types != "background"
    bg_mask = test_types == "background"

    drone_dists = min_mahal[drone_mask]
    bg_dists = min_mahal[bg_mask]

    if len(drone_dists) > 0 and len(bg_dists) > 0:
        all_dists = np.concatenate([drone_dists, bg_dists])
        all_labels = np.concatenate([np.ones(len(drone_dists)),
                                      np.zeros(len(bg_dists))])
        auc = roc_auc_score(all_labels, -all_dists)
    else:
        auc = 0.5

    return {
        "drone_mean": float(drone_dists.mean()) if len(drone_dists) > 0 else 0,
        "drone_median": float(np.median(drone_dists)) if len(drone_dists) > 0 else 0,
        "bg_mean": float(bg_dists.mean()) if len(bg_dists) > 0 else 0,
        "bg_median": float(np.median(bg_dists)) if len(bg_dists) > 0 else 0,
        "ratio": float(bg_dists.mean() / drone_dists.mean())
                 if len(drone_dists) > 0 and drone_dists.mean() > 0 else 0,
        "auc": float(auc),
        "drone_dists": drone_dists,
        "bg_dists": bg_dists,
    }


def cosine_detection(train_drone_embs, test_embs, test_types):
    """
    Cosine distance to global drone centroid.
    Simpler than Mahalanobis — no covariance, just angle.
    """
    centroid = train_drone_embs.mean(axis=0)
    centroid_norm = centroid / np.linalg.norm(centroid)

    test_norms = test_embs / np.linalg.norm(test_embs, axis=1, keepdims=True)
    cos_sim = test_norms @ centroid_norm  # higher = closer to drone centroid

    drone_mask = test_types != "background"
    bg_mask = test_types == "background"

    drone_cos = cos_sim[drone_mask]
    bg_cos = cos_sim[bg_mask]

    if len(drone_cos) > 0 and len(bg_cos) > 0:
        all_cos = np.concatenate([drone_cos, bg_cos])
        all_labels = np.concatenate([np.ones(len(drone_cos)),
                                      np.zeros(len(bg_cos))])
        auc = roc_auc_score(all_labels, all_cos)  # higher cos = more likely drone
    else:
        auc = 0.5

    return {
        "drone_mean_cos": float(drone_cos.mean()) if len(drone_cos) > 0 else 0,
        "bg_mean_cos": float(bg_cos.mean()) if len(bg_cos) > 0 else 0,
        "auc": float(auc),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

@app.function(
    image=IMAGE,
    gpu="A100",
    volumes={"/data": VOL, "/models": MODEL_VOL},
    timeout=3600,
    memory=32768,
)
def run_detection():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda"
    cfg = dict(in_ch=2, encoder_depth=6, encoder_width=64,
               embed_dim=256, proj_dim=256, pred_dim=256, pred_out=256)

    print("=" * 70)
    print("IRIS — GLOBAL MAHALANOBIS DETECTION")
    print("=" * 70)
    print()
    print("ISSUE: Per-type Mahalanobis failed for holdout (unseen) types.")
    print("FIX:   Global Mahalanobis — one centroid from ALL drone embeddings.")
    print("       Tests 'is this in the drone region?' not 'which type is this?'")
    print()

    # ── Configurations to test ──
    configs = []

    # v8 epoch6 (best AUC in holdout eval, trained with normalization)
    if os.path.exists("/models/lejepa_v7_epoch6.pt"):
        configs.append({
            "name": "v8-epoch6 (norm train)",
            "ckpt": "/models/lejepa_v7_epoch6.pt",
            "normalize": True,  # data normalization matches training
        })

    # v8 epoch4 (best Sil during training, trained with normalization)
    if os.path.exists("/models/lejepa_v7_epoch4.pt"):
        configs.append({
            "name": "v8-epoch4 (norm train)",
            "ckpt": "/models/lejepa_v7_epoch4.pt",
            "normalize": True,
        })

    # v7 best + norm inference (already proven AUC=0.921 with per-type Mahal)
    if os.path.exists("/models/lejepa_v7_best.pt"):
        configs.append({
            "name": "v7-best (norm inference)",
            "ckpt": "/models/lejepa_v7_best.pt",
            "normalize": True,  # normalize at inference only
        })

    if not configs:
        print("ERROR: No checkpoints found!")
        return

    # ── Run each config ──
    all_results = {}

    for config in configs:
        name = config["name"]
        ckpt_path = config["ckpt"]
        normalize = config["normalize"]

        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Normalize: {normalize}")
        print(f"{'='*70}")

        # Load model
        model = LeJEPASupCon(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        encoder = model.encoder
        encoder.eval()
        epoch = ckpt.get("epoch", "?")
        print(f"  Loaded from epoch {epoch}")

        # ── Prepare datasets ──
        train_drone_ds = IRISDataset(H5_REMOTE, "train", include_negatives=False,
                                      normalize=normalize)
        holdout_ds = IRISDataset(H5_REMOTE, "holdout", include_negatives=False,
                                  normalize=normalize)
        test_ds = IRISDataset(H5_REMOTE, "holdout", include_negatives=True,
                               max_negatives=2000, normalize=normalize)

        print(f"  Train drones: {len(train_drone_ds)} ({train_drone_ds.n_drone_types} types)")
        print(f"  Holdout drones: {len(holdout_ds)} ({holdout_ds.n_drone_types} types)")
        print(f"  Test set (holdout+bg): {len(test_ds)}")

        # ── Encode ──
        print("  Encoding train drones...")
        train_embs, train_labels, train_types = encode_dataset(
            encoder, train_drone_ds, device)

        print("  Encoding test set (holdout + background)...")
        test_embs, test_labels, test_types = encode_dataset(
            encoder, test_ds, device)

        # ── Method 1: Global Mahalanobis ──
        print("\n  --- Global Mahalanobis (ONE drone centroid) ---")
        global_mahal = global_mahalanobis(train_embs, test_embs, test_types)
        print(f"    Drone mean dist: {global_mahal['drone_mean']:.2f}")
        print(f"    Drone median dist: {global_mahal['drone_median']:.2f}")
        print(f"    BG mean dist: {global_mahal['bg_mean']:.2f}")
        print(f"    BG median dist: {global_mahal['bg_median']:.2f}")
        print(f"    BG/Drone ratio: {global_mahal['ratio']:.3f}")
        print(f"    AUC: {global_mahal['auc']:.4f}")

        # ── Method 2: Per-type Mahalanobis (for comparison) ──
        print("\n  --- Per-type Mahalanobis (30 type centroids) ---")
        per_type = per_type_mahalanobis(train_embs, train_labels, test_embs,
                                         test_types, train_drone_ds.n_drone_types)
        print(f"    Drone mean dist: {per_type['drone_mean']:.2f}")
        print(f"    BG mean dist: {per_type['bg_mean']:.2f}")
        print(f"    BG/Drone ratio: {per_type['ratio']:.3f}")
        print(f"    AUC: {per_type['auc']:.4f}")

        # ── Method 3: Cosine distance to drone centroid ──
        print("\n  --- Cosine to drone centroid ---")
        cos_det = cosine_detection(train_embs, test_embs, test_types)
        print(f"    Drone mean cos: {cos_det['drone_mean_cos']:.4f}")
        print(f"    BG mean cos: {cos_det['bg_mean_cos']:.4f}")
        print(f"    AUC: {cos_det['auc']:.4f}")

        # ── Store ──
        all_results[name] = {
            "epoch": epoch,
            "normalize": normalize,
            "global_mahal_auc": global_mahal["auc"],
            "global_mahal_drone_mean": global_mahal["drone_mean"],
            "global_mahal_bg_mean": global_mahal["bg_mean"],
            "global_mahal_ratio": global_mahal["ratio"],
            "per_type_mahal_auc": per_type["auc"],
            "per_type_mahal_drone_mean": per_type["drone_mean"],
            "per_type_mahal_bg_mean": per_type["bg_mean"],
            "cosine_auc": cos_det["auc"],
        }

        # ── Plot: Global Mahalanobis distribution ──
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        drone_d = global_mahal["drone_dists"]
        bg_d = global_mahal["bg_dists"]

        # Histogram
        ax = axes[0]
        max_val = max(drone_d.max(), bg_d.max(), 1)
        bins = np.linspace(0, min(max_val, 100), 80)
        ax.hist(drone_d, bins=bins, alpha=0.6,
                label=f'Holdout Drones (mean={drone_d.mean():.1f})',
                color='#2196F3', density=True)
        ax.hist(bg_d, bins=bins, alpha=0.6,
                label=f'Background (mean={bg_d.mean():.1f})',
                color='#F44336', density=True)
        ax.set_xlabel('Global Mahalanobis Distance', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Global Mahalanobis: AUC={global_mahal["auc"]:.3f}', fontsize=14)
        ax.legend(fontsize=10)

        # Per-type histogram for comparison
        ax = axes[1]
        pt_drone = per_type["drone_dists"]
        pt_bg = per_type["bg_dists"]
        max_val2 = max(pt_drone.max(), pt_bg.max(), 1)
        bins2 = np.linspace(0, min(max_val2, 100), 80)
        ax.hist(pt_drone, bins=bins2, alpha=0.6,
                label=f'Holdout Drones (mean={pt_drone.mean():.1f})',
                color='#2196F3', density=True)
        ax.hist(pt_bg, bins=bins2, alpha=0.6,
                label=f'Background (mean={pt_bg.mean():.1f})',
                color='#F44336', density=True)
        ax.set_xlabel('Per-type Mahalanobis Distance', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Per-type Mahalanobis: AUC={per_type["auc"]:.3f}', fontsize=14)
        ax.legend(fontsize=10)

        # ROC curves comparison
        ax = axes[2]
        # Global Mahalanobis ROC
        all_dists_g = np.concatenate([drone_d, bg_d])
        all_labels_g = np.concatenate([np.ones(len(drone_d)), np.zeros(len(bg_d))])
        fpr_g, tpr_g, _ = roc_curve(all_labels_g, -all_dists_g)
        ax.plot(fpr_g, tpr_g, 'b-', linewidth=2,
                label=f'Global Mahal (AUC={global_mahal["auc"]:.3f})')

        # Per-type ROC
        all_dists_p = np.concatenate([pt_drone, pt_bg])
        all_labels_p = np.concatenate([np.ones(len(pt_drone)), np.zeros(len(pt_bg))])
        fpr_p, tpr_p, _ = roc_curve(all_labels_p, -all_dists_p)
        ax.plot(fpr_p, tpr_p, 'r--', linewidth=2,
                label=f'Per-type Mahal (AUC={per_type["auc"]:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Detection ROC Curves', fontsize=14)
        ax.legend(fontsize=10)

        plt.suptitle(f'{name} — Detection Methods Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        save_dir = "/models/v8_eval_plots"
        os.makedirs(save_dir, exist_ok=True)
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(f"{save_dir}/global_mahal_{safe_name}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Plot saved: {save_dir}/global_mahal_{safe_name}.png")

    # ── Summary table ──
    print(f"\n{'='*70}")
    print("DETECTION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<30} {'Global AUC':<12} {'PerType AUC':<13} {'Cosine AUC':<12} {'BG/Dr':<8}")
    print("-" * 75)
    for name, res in all_results.items():
        print(f"{name:<30} {res['global_mahal_auc']:<12.4f} "
              f"{res['per_type_mahal_auc']:<13.4f} "
              f"{res['cosine_auc']:<12.4f} "
              f"{res['global_mahal_ratio']:<8.3f}")

    print(f"\n{'='*70}")
    print("KEY INSIGHT:")
    print("  Per-type Mahalanobis fails for UNSEEN drone types because")
    print("  holdout drones aren't close to any specific train type centroid.")
    print()
    print("  Global Mahalanobis should work better because it tests")
    print("  'is this in the drone region of embedding space?' — a question")
    print("  that applies to ALL drones, seen and unseen.")
    print(f"{'='*70}")

    # Save results
    results_path = "/models/global_mahal_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: float(o)
                  if hasattr(o, '__float__') else str(o))
    MODEL_VOL.commit()
    print(f"\n  Results saved: {results_path}")


@app.local_entrypoint()
def main():
    run_detection.remote()