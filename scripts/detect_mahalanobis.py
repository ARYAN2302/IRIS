#!/usr/bin/env python3
"""
IRIS v7 — Mahalanobis Distance Drone Detection Test

The question this answers:
  "If I hand you a spectrogram from RF noise — is there a drone in it?"

Method:
  1. Load trained v7 encoder (epoch 5 best)
  2. Encode all TRAIN drone samples → compute per-type cluster centroids + covariance
  3. Encode HOLDOUT drone samples + BACKGROUND negatives
  4. For each test sample, compute Mahalanobis distance to nearest drone cluster centroid
  5. If drones are close and background is far → drone detection is REAL
  6. If they overlap → the 100% binary accuracy was a shortcut, detection unproven

Why Mahalanobis (not just Euclidean):
  - Accounts for cluster shape (elongated clusters get proper distance)
  - SIGReg enforces near-spherical clusters, so Mahalanobis ≈ scaled Euclidean
  - But it's the principled metric for Gaussian clusters

Usage:
  modal run scripts/detect_mahalanobis.py
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
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler

# ─── Modal setup (same as v7) ─────────────────────────────────────────────────

app = modal.App("iris-v7-detection")

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

# ─── Model definition (must match v7 training) ────────────────────────────────

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
            in_ch=cfg["in_ch"],
            width=cfg["encoder_width"],
            depth=cfg["encoder_depth"],
            embed_dim=cfg["embed_dim"],
        )

    def forward(self, x):
        return self.encoder(x)


# ─── Dataset (simplified — just returns samples, no augmentation) ──────────────

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


class SimpleHDF5Dataset(Dataset):
    """Returns (sample_tensor, label, split_name, type_name)."""
    def __init__(self, h5_path, split_key, include_negatives=False, max_per_type=None):
        self.f = h5py.File(h5_path, "r")
        self.samples = []  # (split_key, type_name, local_idx)
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
            # Optionally subsample per type
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
        print(f"  SimpleHDF5Dataset: {len(self.samples)} samples "
              f"({self.n_drone_types} types + {'background' if include_negatives else 'no negs'})")

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
            x = torch.from_numpy(sample[:2].copy()).float()
        elif sample.shape[0] == 2:
            x = torch.from_numpy(sample.copy()).float()
        else:
            x = torch.from_numpy(sample[:2].copy()).float()

        return x, self.labels[idx], self.type_names_list[idx]


# ─── Encode dataset ────────────────────────────────────────────────────────────

def encode_dataset(encoder, dataset, device="cuda", batch_size=64):
    """Encode all samples in dataset, return embeddings, labels, type_names."""
    encoder.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_embeddings = []
    all_labels = []
    all_types = []

    with torch.no_grad():
        for x, label, tname in dl:
            x = x.to(device)
            z = encoder(x)
            all_embeddings.append(z.cpu().numpy())
            all_labels.extend(label if isinstance(label, list) else label.tolist())
            all_types.extend(tname if isinstance(tname, list) else list(tname))

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.array(all_labels)
    types = np.array(all_types)

    return embeddings, labels, types


# ─── Mahalanobis Distance ─────────────────────────────────────────────────────

def compute_mahalanobis_distances(embeddings, centroids, cov_inv_list):
    """
    Compute Mahalanobis distance from each embedding to nearest centroid.

    Args:
        embeddings: (N, D) test embeddings
        centroids: (K, D) cluster centroids
        cov_inv_list: list of (D, D) inverse covariance matrices, one per cluster

    Returns:
        min_distances: (N,) Mahalanobis distance to nearest centroid
        nearest_cluster: (N,) index of nearest centroid
    """
    N = embeddings.shape[0]
    K = centroids.shape[0]
    min_distances = np.full(N, np.inf)
    nearest_cluster = np.zeros(N, dtype=int)

    for k in range(K):
        diff = embeddings - centroids[k]  # (N, D)
        # Mahalanobis: sqrt(diff^T * Cov_inv * diff)
        mahal_sq = np.sum(diff @ cov_inv_list[k] * diff, axis=1)  # (N,)
        mahal = np.sqrt(np.maximum(mahal_sq, 0))

        mask = mahal < min_distances
        min_distances[mask] = mahal[mask]
        nearest_cluster[mask] = k

    return min_distances, nearest_cluster


def compute_centroids_and_cov(embeddings, labels, n_clusters, reg=1e-3):
    """
    Compute per-cluster centroids and regularized inverse covariance matrices.

    Args:
        embeddings: (N, D) all embeddings
        labels: (N,) cluster labels (0..n_clusters-1)
        n_clusters: number of clusters
        reg: regularization for covariance inversion

    Returns:
        centroids: (K, D)
        cov_inv_list: list of (D, D)
    """
    D = embeddings.shape[1]
    centroids = np.zeros((n_clusters, D))
    cov_inv_list = []

    for k in range(n_clusters):
        mask = labels == k
        cluster_embs = embeddings[mask]
        centroids[k] = cluster_embs.mean(axis=0)

        # Covariance with regularization
        cov = np.cov(cluster_embs.T) + reg * np.eye(D)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
        cov_inv_list.append(cov_inv)

    return centroids, cov_inv_list


# ─── Main Detection Test ──────────────────────────────────────────────────────

@app.function(
    image=IMAGE,
    gpu="A100",
    volumes={"/data": VOL, "/models": MODEL_VOL},
    timeout=1800,
    memory=32768,
)
def detect():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda"

    # ── Load v7 best model ──
    cfg = dict(
        in_ch=2, encoder_depth=6, encoder_width=64,
        embed_dim=256, proj_dim=256, pred_dim=256, pred_out=256,
    )

    model = LeJEPASupCon(cfg).to(device)

    # Try best checkpoint first, then epoch 5
    ckpt_path = "/models/lejepa_v7_best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "/models/lejepa_v7_epoch5.pt"
    if not os.path.exists(ckpt_path):
        print("No v7 checkpoint found! Looking for any epoch...")
        import glob
        ckpts = sorted(glob.glob("/models/lejepa_v7_epoch*.pt"))
        if ckpts:
            ckpt_path = ckpts[-1]
        else:
            print("ERROR: No checkpoints found at all!")
            return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint: {ckpt_path} (epoch {epoch})")

    encoder = model.encoder

    # ── Step 1: Encode TRAIN drone samples → compute centroids ──
    print("\n=== Step 1: Encoding TRAIN drone samples ===")
    train_ds = SimpleHDF5Dataset(H5_REMOTE, "train", include_negatives=False, max_per_type=500)
    train_embs, train_labels, train_types = encode_dataset(encoder, train_ds, device)
    print(f"  Train embeddings: {train_embs.shape}")
    print(f"  Train types: {len(train_ds.type_names)}")

    # Compute per-type centroids + covariance
    n_types = train_ds.n_drone_types
    centroids, cov_inv_list = compute_centroids_and_cov(train_embs, train_labels, n_types)
    print(f"  Computed {n_types} centroids")

    # ── Sanity: train Mahalanobis distances ──
    train_mahal, train_nearest = compute_mahalanobis_distances(train_embs, centroids, cov_inv_list)
    print(f"  Train Mahalanobis — mean: {train_mahal.mean():.2f}, "
          f"median: {np.median(train_mahal):.2f}, "
          f"95th pct: {np.percentile(train_mahal, 95):.2f}")

    # ── Step 2: Encode HOLDOUT drone samples ──
    print("\n=== Step 2: Encoding HOLDOUT drone samples ===")
    holdout_ds = SimpleHDF5Dataset(H5_REMOTE, "holdout", include_negatives=False, max_per_type=500)
    holdout_embs, holdout_labels, holdout_types = encode_dataset(encoder, holdout_ds, device)
    print(f"  Holdout embeddings: {holdout_embs.shape}")
    print(f"  Holdout types: {len(holdout_ds.type_names)}")

    # Compute holdout Mahalanobis distances (to nearest TRAIN centroid)
    holdout_mahal, holdout_nearest = compute_mahalanobis_distances(holdout_embs, centroids, cov_inv_list)
    print(f"  Holdout Mahalanobis — mean: {holdout_mahal.mean():.2f}, "
          f"median: {np.median(holdout_mahal):.2f}, "
          f"95th pct: {np.percentile(holdout_mahal, 95):.2f}")

    # ── Step 3: Encode BACKGROUND negatives ──
    print("\n=== Step 3: Encoding BACKGROUND negatives ===")
    # Use 2000 negatives (enough for statistics, not too slow)
    neg_ds = SimpleHDF5Dataset(H5_REMOTE, "train", include_negatives=True, max_per_type=500)
    # Filter to only negatives
    neg_mask = neg_ds.type_names_list == "background"
    neg_indices = [i for i, t in enumerate(neg_ds.type_names_list) if t == "background"][:2000]

    neg_subset = torch.utils.data.Subset(neg_ds, neg_indices)
    neg_dl = DataLoader(neg_subset, batch_size=64, shuffle=False, num_workers=2)

    all_neg_embs = []
    with torch.no_grad():
        for x, label, tname in neg_dl:
            x = x.to(device)
            z = encoder(x)
            all_neg_embs.append(z.cpu().numpy())
    neg_embs = np.concatenate(all_neg_embs, axis=0)
    print(f"  Negative embeddings: {neg_embs.shape}")

    # Compute negative Mahalanobis distances (to nearest TRAIN centroid)
    neg_mahal, neg_nearest = compute_mahalanobis_distances(neg_embs, centroids, cov_inv_list)
    print(f"  Neg Mahalanobis — mean: {neg_mahal.mean():.2f}, "
          f"median: {np.median(neg_mahal):.2f}, "
          f"95th pct: {np.percentile(neg_mahal, 95):.2f}")

    # ── Step 4: Detection Analysis ──
    print("\n" + "="*70)
    print("DETECTION ANALYSIS")
    print("="*70)

    # Drone = positive class (close to centroid), Background = negative class (far)
    # We want: drones have LOW Mahalanobis distance, background has HIGH

    # Combine holdout drones + negatives for binary detection
    all_test_mahal = np.concatenate([holdout_mahal, neg_mahal])
    all_test_labels = np.concatenate([
        np.ones(len(holdout_mahal)),     # 1 = drone
        np.zeros(len(neg_mahal)),        # 0 = background
    ])

    # ROC-AUC (drone vs background based on Mahalanobis distance)
    # Lower distance = more likely drone, so we negate for sklearn
    auc = roc_auc_score(all_test_labels, -all_test_mahal)
    print(f"\n  Binary Detection (Holdout Drones vs Background):")
    print(f"  ROC-AUC: {auc:.4f}")

    # Find optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(all_test_labels, -all_test_mahal)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = -thresholds[best_idx]  # convert back to Mahalanobis distance
    print(f"  Best threshold: Mahalanobis distance = {best_threshold:.2f}")
    print(f"  At threshold: TPR = {tpr[best_idx]:.3f}, FPR = {fpr[best_idx]:.3f}")

    # At 95% TPR (catch 95% of drones), what's the FPR?
    idx_95 = np.searchsorted(tpr, 0.95)
    if idx_95 < len(fpr):
        fpr_at_95 = fpr[idx_95]
        print(f"  At 95% TPR: FPR = {fpr_at_95:.3f} ({fpr_at_95*100:.1f}% false alarms)")

    # ── Also check: holdout drones vs train drones ──
    print(f"\n  Distance comparison:")
    print(f"  Train drones:  mean={train_mahal.mean():.2f}, median={np.median(train_mahal):.2f}")
    print(f"  Holdout drones: mean={holdout_mahal.mean():.2f}, median={np.median(holdout_mahal):.2f}")
    print(f"  Background:    mean={neg_mahal.mean():.2f}, median={np.median(neg_mahal):.2f}")

    # ── Per-holdout-type breakdown ──
    print(f"\n  Per holdout type (Mahalanobis distance to nearest train centroid):")
    for tname in sorted(set(holdout_types)):
        mask = holdout_types == tname
        t_mahal = holdout_mahal[mask]
        print(f"    {tname:25s}: mean={t_mahal.mean():.2f}, median={np.median(t_mahal):.2f}")

    # ── Verdict ──
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")
    if auc > 0.95:
        print("  REAL DETECTION: Drone vs background is clearly separable.")
        print("  The encoder learned 'drone-ness' — background falls far from all drone clusters.")
    elif auc > 0.85:
        print("  PROMISING: Some separation, but not clean enough for a detector.")
        print("  Background partially overlaps with drone clusters.")
    elif auc > 0.7:
        print("  WEAK: Marginal separation. Background often falls near drone clusters.")
        print("  The binary accuracy in training was likely a dataset fingerprint shortcut.")
    else:
        print("  NO DETECTION: Background and drones overlap in embedding space.")
        print("  The 100% binary accuracy was definitely a shortcut.")

    print(f"\n  AUC = {auc:.4f}")

    # ── Step 5: Save results ──
    results = {
        "auc": float(auc),
        "best_threshold": float(best_threshold),
        "tpr_at_best": float(tpr[best_idx]),
        "fpr_at_best": float(fpr[best_idx]),
        "train_mahal_mean": float(train_mahal.mean()),
        "holdout_mahal_mean": float(holdout_mahal.mean()),
        "neg_mahal_mean": float(neg_mahal.mean()),
        "holdout_mahal_median": float(np.median(holdout_mahal)),
        "neg_mahal_median": float(np.median(neg_mahal)),
        "n_holdout": int(len(holdout_mahal)),
        "n_negatives": int(len(neg_mahal)),
        "epoch": epoch,
    }

    # Add per-holdout-type stats
    per_type = {}
    for tname in sorted(set(holdout_types)):
        mask = holdout_types == tname
        t_mahal = holdout_mahal[mask]
        per_type[tname] = {"mean": float(t_mahal.mean()), "median": float(np.median(t_mahal))}
    results["per_holdout_type"] = per_type

    results_path = "/models/v7_detection_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    MODEL_VOL.commit()
    print(f"\n  Results saved to {results_path}")

    # ── Step 6: Plots ──
    save_dir = "/models/detection_plots"
    os.makedirs(save_dir, exist_ok=True)

    # Plot 1: Mahalanobis distance distributions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    ax = axes[0]
    bins = np.linspace(0, max(holdout_mahal.max(), neg_mahal.max()), 80)
    ax.hist(holdout_mahal, bins=bins, alpha=0.6, label=f'Holdout Drones (n={len(holdout_mahal)})',
            color='#2196F3', density=True)
    ax.hist(neg_mahal, bins=bins, alpha=0.6, label=f'Background (n={len(neg_mahal)})',
            color='#F44336', density=True)
    ax.axvline(best_threshold, color='black', linestyle='--', linewidth=2,
               label=f'Threshold={best_threshold:.1f}')
    ax.set_xlabel('Mahalanobis Distance to Nearest Drone Centroid', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Drone Detection via Mahalanobis Distance (AUC={auc:.3f})', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, min(bins[-1], 50))  # cap x-axis for readability

    # ROC curve
    ax = axes[1]
    ax.plot(fpr, tpr, color='#2196F3', linewidth=2, label=f'ROC (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.scatter([fpr[best_idx]], [tpr[best_idx]], color='red', s=100, zorder=5,
               label=f'Best threshold (TPR={tpr[best_idx]:.2f}, FPR={fpr[best_idx]:.2f})')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve: Holdout Drones vs Background', fontsize=14)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plot_path = f"{save_dir}/mahalanobis_detection.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {plot_path}")

    # Plot 2: Per-holdout-type distance vs background
    fig, ax = plt.subplots(figsize=(14, 6))
    all_type_names = sorted(set(holdout_types))
    data_for_box = []
    labels_for_box = []

    for tname in all_type_names:
        mask = holdout_types == tname
        data_for_box.append(holdout_mahal[mask])
        labels_for_box.append(tname)
    data_for_box.append(neg_mahal)
    labels_for_box.append("BACKGROUND")

    bp = ax.boxplot(data_for_box, patch_artist=True, showfliers=False)
    colors = ['#2196F3'] * len(all_type_names) + ['#F44336']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticklabels(labels_for_box, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mahalanobis Distance', fontsize=12)
    ax.set_title('Per-Type Distance to Nearest Train Centroid\n(Holdout Drones vs Background)', fontsize=14)
    ax.axhline(best_threshold, color='black', linestyle='--', linewidth=1.5,
               label=f'Threshold={best_threshold:.1f}')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plot_path2 = f"{save_dir}/per_type_detection.png"
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {plot_path2}")

    MODEL_VOL.commit()
    print("\nDone! Check the plots and AUC for the verdict.")


@app.local_entrypoint()
def main():
    detect.remote()