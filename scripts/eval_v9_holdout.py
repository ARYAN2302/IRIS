#!/usr/bin/env python3
"""
IRIS v9 — Full Holdout Evaluation

This is THE critical evaluation. v8 looked amazing during training but collapsed
on holdout (AUC 0.312). v9 claims AUC 1.000 during training — we need to verify
that holds with a FULL stress test:

  1. ALL train drones (13,441) for centroids — no subsampling
  2. ALL holdout drones (3,659 from 7 UNSEEN types)
  3. 50,000+ background negatives (not the 2,000 used during training eval)
  4. Per-holdout-type AUC breakdown — which types are hardest?
  5. Global vs per-type Mahalanobis comparison
  6. Detection at operational FPR thresholds (0.1%, 1%, 5%)
  7. UMAP visualization of embedding space
  8. ROC curve + distance distribution plots

Usage:
  modal run scripts/eval_v9_holdout.py
"""

import h5py
import json
import os
import glob
import time

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    silhouette_score, accuracy_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ─── Modal setup ──────────────────────────────────────────────────────────────

app = modal.App("iris-v9-holdout-eval")

VOL = modal.Volume.from_name("iris-data", create_if_missing=True)
MODEL_VOL = modal.Volume.from_name("iris-models-v9", create_if_missing=True)

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
N_BG_EVAL = 50000  # Full stress test with 50K negatives


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
    """Same architecture as v9 training — loads encoder from checkpoint."""
    def __init__(self, cfg):
        super().__init__()
        self.encoder = CNNEncoder(
            in_ch=cfg["in_ch"], width=cfg["encoder_width"],
            depth=cfg["encoder_depth"], embed_dim=cfg["embed_dim"],
        )
        # Projector + Predictor needed to load full model state_dict
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


# ─── Dataset with per-channel normalization ───────────────────────────────────

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


def _resolve_negatives(f):
    """Resolve the negatives group/dataset from HDF5."""
    if "negatives" not in f:
        return None, 0, False, []
    neg_item = f["negatives"]
    if isinstance(neg_item, h5py.Dataset):
        return neg_item, neg_item.shape[0], False, []
    else:
        sub_keys = [sk for sk in neg_item.keys()
                    if isinstance(neg_item[sk], h5py.Dataset)
                    and len(neg_item[sk].shape) == 3]
        try:
            sub_keys.sort(key=lambda x: int(x))
        except ValueError:
            sub_keys.sort()
        return neg_item, len(sub_keys), True, sub_keys


class IRISNormDataset(Dataset):
    """
    Evaluation dataset WITH per-channel normalization.
    Matches v9 training pipeline: each channel is zero-mean, unit-variance.
    """
    def __init__(self, h5_path, split_key="train", include_negatives=False,
                 max_negatives=50000):
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
            for i in range(n_samples):
                self.samples.append((split_key, tname, i))
                self.labels.append(label_idx)
                self.type_names_list.append(tname)

        self.n_drone_types = len(type_names)
        self.type_names = type_names

        if include_negatives and "negatives" in self.f:
            neg_data, n_total_neg, neg_is_multi, neg_sub_keys = _resolve_negatives(self.f)
            n_neg = min(n_total_neg, max_negatives)
            self._neg_is_multi = neg_is_multi
            self._neg_sub_keys = neg_sub_keys
            self._resolved[("negatives", None)] = (neg_data, n_total_neg, neg_is_multi)

            # Sample subset using seed for reproducibility
            rng = np.random.default_rng(123)  # Different seed from training!
            if n_neg < n_total_neg:
                self._bg_indices = rng.choice(n_total_neg, n_neg, replace=False).tolist()
            else:
                self._bg_indices = list(range(n_total_neg))

            for i in self._bg_indices:
                self.samples.append(("negatives", None, i))
                self.labels.append(self.n_drone_types)  # bg label
                self.type_names_list.append("background")

    def __len__(self):
        return len(self.samples)

    def _read_sample(self, split_key, tname, local_idx):
        if split_key == "negatives":
            neg_data, _, neg_is_multi = self._resolved[("negatives", None)]
            if neg_is_multi:
                sub_key = self._neg_sub_keys[local_idx]
                return neg_data[sub_key][:]
            else:
                return neg_data[local_idx]
        else:
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

        # ── PER-CHANNEL NORMALIZATION (same as v9 training) ──
        x_t = torch.from_numpy(x)
        for c in range(x_t.shape[0]):
            ch = x_t[c]
            ch_std = ch.std()
            if ch_std > 1e-6:
                x_t[c] = (ch - ch.mean()) / ch_std
            else:
                x_t[c] = ch - ch.mean()

        return x_t, self.labels[idx], self.type_names_list[idx]


# ─── Encoding helper ─────────────────────────────────────────────────────────

def encode_dataset(encoder, dataset, device, batch_size=64, max_samples=None,
                   desc="Encoding"):
    """Encode entire dataset, return embeddings, labels, type names."""
    if max_samples and len(dataset) > max_samples:
        indices = np.random.default_rng(42).choice(len(dataset), max_samples, replace=False)
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        dl = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    else:
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_embs, all_labels, all_types = [], [], []
    with torch.no_grad():
        for batch_idx, (x, label, tname) in enumerate(dl):
            z = encoder(x.to(device))
            all_embs.append(z.cpu().numpy())
            all_labels.extend(label.tolist() if hasattr(label, 'tolist') else list(label))
            all_types.extend(list(tname))
            if (batch_idx + 1) % 50 == 0:
                print(f"    {desc}: {batch_idx + 1}/{len(dl)} batches")

    return np.concatenate(all_embs), np.array(all_labels), np.array(all_types)


# ─── Detection metrics ───────────────────────────────────────────────────────

def compute_global_mahalanobis(train_embs, test_embs):
    """
    Global Mahalanobis: one centroid from ALL train drones.
    This is what v9 should excel at — "is this in the drone region?"
    """
    D = train_embs.shape[1]
    centroid = train_embs.mean(axis=0)
    cov = np.cov(train_embs.T) + 1e-3 * np.eye(D)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    diff = test_embs - centroid
    mahal_sq = np.sum(diff @ cov_inv * diff, axis=1)
    return np.sqrt(np.maximum(mahal_sq, 0))


def compute_per_type_mahalanobis(train_embs, train_labels, n_types, test_embs):
    """
    Per-type Mahalanobis: distance to nearest train type centroid.
    This failed for v8 but might work for v9 with background class.
    """
    D = train_embs.shape[1]
    centroids = np.zeros((n_types, D))
    cov_inv_list = []
    reg = 1e-3

    for k in range(n_types):
        mask = train_labels == k
        cluster = train_embs[mask]
        if len(cluster) < 2:
            centroids[k] = train_embs.mean(axis=0)  # fallback to global
            cov_inv_list.append(np.linalg.inv(np.cov(train_embs.T) + reg * np.eye(D)))
            continue
        centroids[k] = cluster.mean(axis=0)
        cov = np.cov(cluster.T) + reg * np.eye(D)
        try:
            cov_inv_list.append(np.linalg.inv(cov))
        except np.linalg.LinAlgError:
            cov_inv_list.append(np.linalg.pinv(cov))

    min_mahal = np.full(len(test_embs), np.inf)
    for k in range(n_types):
        diff = test_embs - centroids[k]
        mahal_sq = np.sum(diff @ cov_inv_list[k] * diff, axis=1)
        mahal = np.sqrt(np.maximum(mahal_sq, 0))
        min_mahal = np.minimum(min_mahal, mahal)

    return min_mahal


def compute_detection_metrics(drone_dists, bg_dists):
    """Compute full detection metrics from distance arrays."""
    all_dists = np.concatenate([drone_dists, bg_dists])
    all_labels = np.concatenate([np.ones(len(drone_dists)),
                                  np.zeros(len(bg_dists))])

    # AUC
    auc = roc_auc_score(all_labels, -all_dists)

    # Average precision (area under PR curve — better for imbalanced data)
    ap = average_precision_score(all_labels, -all_dists)

    # ROC curve for threshold analysis
    fpr, tpr, thresholds = roc_curve(all_labels, -all_dists)

    # Detection rate at various FPR thresholds
    tpr_at_fpr = {}
    for fpr_target in [0.001, 0.005, 0.01, 0.05, 0.1]:
        idx = np.searchsorted(fpr, fpr_target)
        if idx < len(tpr):
            tpr_at_fpr[f"fpr_{fpr_target:.3f}_tpr"] = float(tpr[idx])
        else:
            tpr_at_fpr[f"fpr_{fpr_target:.3f}_tpr"] = float(tpr[-1])

    # FPR at various TPR thresholds (operational: what false alarm rate at 95% detection?)
    fpr_at_tpr = {}
    for tpr_target in [0.90, 0.95, 0.99, 0.999]:
        idx = np.searchsorted(tpr, tpr_target)
        if idx < len(fpr):
            fpr_at_tpr[f"tpr_{tpr_target:.3f}_fpr"] = float(fpr[idx])
        else:
            fpr_at_tpr[f"tpr_{tpr_target:.3f}_fpr"] = 1.0

    return {
        "auc": float(auc),
        "ap": float(ap),
        "drone_mean": float(drone_dists.mean()),
        "drone_median": float(np.median(drone_dists)),
        "drone_std": float(drone_dists.std()),
        "drone_min": float(drone_dists.min()),
        "drone_max": float(drone_dists.max()),
        "drone_p5": float(np.percentile(drone_dists, 5)),
        "drone_p95": float(np.percentile(drone_dists, 95)),
        "bg_mean": float(bg_dists.mean()),
        "bg_median": float(np.median(bg_dists)),
        "bg_std": float(bg_dists.std()),
        "bg_min": float(bg_dists.min()),
        "bg_max": float(bg_dists.max()),
        "bg_p5": float(np.percentile(bg_dists, 5)),
        "bg_p95": float(np.percentile(bg_dists, 95)),
        "bg_drone_ratio": float(bg_dists.mean() / drone_dists.mean())
                          if drone_dists.mean() > 0 else 0,
        **tpr_at_fpr,
        **fpr_at_tpr,
    }


# ─── Per-type breakdown ──────────────────────────────────────────────────────

def compute_per_holdout_type_auc(holdout_types, distances, is_drone_mask):
    """Compute AUC for each holdout drone type separately vs background."""
    bg_dists = distances[~is_drone_mask]
    results = {}

    for dtype in sorted(set(holdout_types[is_drone_mask])):
        dtype_mask = holdout_types == dtype
        dtype_dists = distances[dtype_mask]

        if len(dtype_dists) > 0 and len(bg_dists) > 0:
            all_d = np.concatenate([dtype_dists, bg_dists])
            all_l = np.concatenate([np.ones(len(dtype_dists)),
                                     np.zeros(len(bg_dists))])
            auc = roc_auc_score(all_l, -all_d)
            results[dtype] = {
                "n_samples": int(dtype_mask.sum()),
                "auc": float(auc),
                "mean_dist": float(dtype_dists.mean()),
                "median_dist": float(np.median(dtype_dists)),
            }

    return results


# ─── Main evaluation ─────────────────────────────────────────────────────────

@app.function(
    image=IMAGE,
    gpu="A100",
    volumes={"/data": VOL, "/models": MODEL_VOL},
    timeout=5400,  # 90 min — encoding 50K+ samples takes time
    memory=32768,
)
def evaluate_v9():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    os.makedirs(EVAL_DIR, exist_ok=True)
    device = "cuda"

    # Default v9 config
    cfg = dict(in_ch=2, encoder_depth=6, encoder_width=64,
               embed_dim=256, proj_dim=256, pred_dim=256, pred_out=256)

    # ── Load best checkpoint ──
    best_path = "/models/lejepa_v9_best.pt"
    if not os.path.exists(best_path):
        print(f"ERROR: No best checkpoint found at {best_path}")
        print("Available checkpoints:")
        for f in sorted(glob.glob("/models/lejepa_v9_*.pt")):
            print(f"  {f}")
        return

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    epoch = ckpt.get("epoch", "?")
    saved_eval = ckpt.get("eval_results", {})

    print("=" * 70)
    print("IRIS v9 — FULL HOLDOUT EVALUATION (STRESS TEST)")
    print("=" * 70)
    print(f"  Checkpoint: {best_path}")
    print(f"  Epoch: {epoch}")
    print(f"  Training-time metrics: Sil={saved_eval.get('silhouette_drone', 0):.3f}, "
          f"Global AUC={saved_eval.get('mahalanobis_auc', 0):.3f}, "
          f"Per-type AUC={saved_eval.get('per_type_mahalanobis_auc', 0):.3f}")
    print(f"  Background eval samples: {N_BG_EVAL}")
    print()

    # Load model
    model = LeJEPASupConV9(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    encoder = model.encoder
    encoder.eval()
    print("  Model loaded successfully.")

    # ── Prepare datasets ──
    print("\nPreparing datasets with per-channel normalization...")

    # ALL train drones (no subsampling for centroids)
    train_ds = IRISNormDataset(H5_REMOTE, "train", include_negatives=False)
    print(f"  Train drones: {len(train_ds)} samples, {train_ds.n_drone_types} types")

    # ALL holdout drones
    holdout_ds = IRISNormDataset(H5_REMOTE, "holdout", include_negatives=False)
    print(f"  Holdout drones: {len(holdout_ds)} samples, {holdout_ds.n_drone_types} types")
    print(f"  Holdout types: {holdout_ds.type_names}")

    # Full test: holdout drones + LARGE background sample
    test_ds = IRISNormDataset(H5_REMOTE, "holdout", include_negatives=True,
                               max_negatives=N_BG_EVAL)
    print(f"  Test (holdout + {N_BG_EVAL} bg): {len(test_ds)} samples")

    # ── Encode everything ──
    print("\nEncoding datasets...")

    print("  [1/3] Encoding ALL train drones (for centroids)...")
    t0 = time.time()
    train_embs, train_labels, train_types = encode_dataset(
        encoder, train_ds, device, desc="Train drones")
    print(f"    Done: {train_embs.shape} in {time.time()-t0:.1f}s")

    print("  [2/3] Encoding ALL holdout drones...")
    t0 = time.time()
    holdout_embs, holdout_labels, holdout_types = encode_dataset(
        encoder, holdout_ds, device, desc="Holdout drones")
    print(f"    Done: {holdout_embs.shape} in {time.time()-t0:.1f}s")

    print("  [3/3] Encoding test set (holdout + background)...")
    t0 = time.time()
    test_embs, test_labels, test_types = encode_dataset(
        encoder, test_ds, device, desc="Test (holdout+bg)")
    print(f"    Done: {test_embs.shape} in {time.time()-t0:.1f}s")

    torch.cuda.empty_cache()

    # ── Classification metrics ──
    print("\n" + "=" * 70)
    print("CLASSIFICATION METRICS")
    print("=" * 70)

    # Train drone clustering
    if len(np.unique(train_labels)) > 1:
        knn = KNeighborsClassifier(n_neighbors=10, metric="cosine")
        # Subsample for k-NN CV (too slow on 13K with CV)
        rng = np.random.default_rng(42)
        keep = []
        for t in np.unique(train_types):
            mask = train_types == t
            idx = np.where(mask)[0]
            n_keep = min(len(idx), 200)
            keep.extend(rng.choice(idx, n_keep, replace=False))
        X_knn = train_embs[keep]
        y_knn = train_labels[keep]

        knn_scores = cross_val_score(knn, X_knn, y_knn, cv=3, scoring="accuracy")
        train_knn = float(knn_scores.mean())

        sil = silhouette_score(train_embs, train_labels,
                               metric="cosine", sample_size=5000)
        train_sil = float(sil)

        print(f"  Train drone types: {train_ds.n_drone_types}")
        print(f"  Train k-NN CV:     {train_knn:.4f} (+/- {knn_scores.std():.4f})")
        print(f"  Train Silhouette:  {train_sil:.4f}")
    else:
        train_knn, train_sil = 0, 0
        print("  Not enough type diversity for classification metrics")

    # Holdout drone clustering (how well do unseen types cluster?)
    if len(np.unique(holdout_labels)) > 1 and len(holdout_embs) > 10:
        ho_sil = float(silhouette_score(holdout_embs, holdout_labels, metric="cosine"))
        knn_ho = KNeighborsClassifier(n_neighbors=10, metric="cosine")
        knn_ho_scores = cross_val_score(knn_ho, holdout_embs, holdout_labels,
                                         cv=3, scoring="accuracy")
        ho_knn = float(knn_ho_scores.mean())
        print(f"  Holdout Silhouette: {ho_sil:.4f}")
        print(f"  Holdout k-NN CV:    {ho_knn:.4f} (+/- {knn_ho_scores.std():.4f})")
    else:
        ho_sil, ho_knn = 0, 0

    # Binary accuracy (drone vs background)
    drone_mask_test = test_types != "background"
    bg_mask_test = test_types == "background"
    y_binary = drone_mask_test.astype(int)

    if len(np.unique(y_binary)) > 1:
        lr = LogisticRegression(max_iter=500, solver="lbfgs")
        bin_scores = cross_val_score(lr, test_embs, y_binary, cv=3, scoring="accuracy")
        binary_acc = float(bin_scores.mean())
        print(f"  Binary acc (drone vs bg, 3-fold CV): {binary_acc:.4f}")
    else:
        binary_acc = 0

    # ── DETECTION METRICS (THE MAIN EVENT) ──
    print("\n" + "=" * 70)
    print("DETECTION METRICS — FULL STRESS TEST")
    print("=" * 70)
    print(f"  Test: {drone_mask_test.sum()} holdout drones vs "
          f"{bg_mask_test.sum()} background samples")
    print()

    # ── Global Mahalanobis ──
    print("  --- Global Mahalanobis (one drone centroid) ---")
    global_mahal = compute_global_mahalanobis(train_embs, test_embs)
    global_drone_dists = global_mahal[drone_mask_test]
    global_bg_dists = global_mahal[bg_mask_test]
    global_metrics = compute_detection_metrics(global_drone_dists, global_bg_dists)

    print(f"  AUC:              {global_metrics['auc']:.4f}")
    print(f"  Avg Precision:    {global_metrics['ap']:.4f}")
    print(f"  Drone dist:       mean={global_metrics['drone_mean']:.2f}, "
          f"median={global_metrics['drone_median']:.2f}, "
          f"std={global_metrics['drone_std']:.2f}")
    print(f"  Background dist:  mean={global_metrics['bg_mean']:.2f}, "
          f"median={global_metrics['bg_median']:.2f}, "
          f"std={global_metrics['bg_std']:.2f}")
    print(f"  BG/Drone ratio:   {global_metrics['bg_drone_ratio']:.2f}x")
    print(f"\n  Detection at operational FPR thresholds:")
    for fpr_t in [0.001, 0.005, 0.01, 0.05, 0.1]:
        key = f"fpr_{fpr_t:.3f}_tpr"
        if key in global_metrics:
            print(f"    FPR={fpr_t:.1%} → TPR={global_metrics[key]:.4f} "
                  f"({global_metrics[key]*100:.2f}% detection)")
    print(f"\n  False alarm at operational TPR thresholds:")
    for tpr_t in [0.90, 0.95, 0.99]:
        key = f"tpr_{tpr_t:.3f}_fpr"
        if key in global_metrics:
            print(f"    TPR={tpr_t:.0%} → FPR={global_metrics[key]:.4f} "
                  f"({global_metrics[key]*100:.2f}% false alarms)")

    # ── Per-type Mahalanobis ──
    print("\n  --- Per-type Mahalanobis (30 type centroids) ---")
    pt_mahal = compute_per_type_mahalanobis(
        train_embs, train_labels, train_ds.n_drone_types, test_embs)
    pt_drone_dists = pt_mahal[drone_mask_test]
    pt_bg_dists = pt_mahal[bg_mask_test]
    pt_metrics = compute_detection_metrics(pt_drone_dists, pt_bg_dists)

    print(f"  AUC:              {pt_metrics['auc']:.4f}")
    print(f"  Avg Precision:    {pt_metrics['ap']:.4f}")
    print(f"  BG/Drone ratio:   {pt_metrics['bg_drone_ratio']:.2f}x")

    # ── Per-holdout-type breakdown ──
    print("\n  --- Per-Holdout-Type Breakdown (Global Mahalanobis) ---")
    per_type_breakdown = compute_per_holdout_type_auc(
        test_types, global_mahal, drone_mask_test)

    print(f"  {'Type':<25} {'N':>6} {'AUC':>8} {'Mean Dist':>10} {'Median':>10}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*10} {'-'*10}")
    for dtype, info in sorted(per_type_breakdown.items()):
        marker = " ***" if info["auc"] < 0.95 else ""
        print(f"  {dtype:<25} {info['n_samples']:>6} {info['auc']:>8.4f} "
              f"{info['mean_dist']:>10.2f} {info['median_dist']:>10.2f}{marker}")

    worst_type = min(per_type_breakdown.items(), key=lambda x: x[1]["auc"])
    best_type = max(per_type_breakdown.items(), key=lambda x: x[1]["auc"])
    print(f"\n  Best type:  {best_type[0]} (AUC={best_type[1]['auc']:.4f})")
    print(f"  Worst type: {worst_type[0]} (AUC={worst_type[1]['auc']:.4f})")

    # ── PLOT 1: Distance distribution + ROC ──
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Global Mahalanobis histogram
    ax = axes[0, 0]
    max_val = min(max(global_drone_dists.max(), global_bg_dists.max()), 100)
    bins = np.linspace(0, max_val, 100)
    ax.hist(global_drone_dists, bins=bins, alpha=0.6,
            label=f'Holdout Drones (n={len(global_drone_dists)}, '
                  f'mean={global_drone_dists.mean():.1f})',
            color='#2196F3', density=True)
    ax.hist(global_bg_dists, bins=bins, alpha=0.6,
            label=f'Background (n={len(global_bg_dists)}, '
                  f'mean={global_bg_dists.mean():.1f})',
            color='#F44336', density=True)
    ax.set_xlabel('Global Mahalanobis Distance', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'v9 Detection Distribution (Global Mahalanobis)\n'
                 f'AUC={global_metrics["auc"]:.4f}, '
                 f'BG/Drone={global_metrics["bg_drone_ratio"]:.1f}x',
                 fontsize=13)
    ax.legend(fontsize=11, loc='best')

    # ROC curve
    ax = axes[0, 1]
    all_dists = np.concatenate([global_drone_dists, global_bg_dists])
    all_labels = np.concatenate([np.ones(len(global_drone_dists)),
                                  np.zeros(len(global_bg_dists))])
    fpr, tpr, _ = roc_curve(all_labels, -all_dists)
    ax.plot(fpr, tpr, 'b-', linewidth=2,
            label=f'v9 Global (AUC={global_metrics["auc"]:.4f})')

    # Also plot per-type Mahalanobis ROC
    all_dists_pt = np.concatenate([pt_drone_dists, pt_bg_dists])
    all_labels_pt = np.concatenate([np.ones(len(pt_drone_dists)),
                                     np.zeros(len(pt_bg_dists))])
    fpr_pt, tpr_pt, _ = roc_curve(all_labels_pt, -all_dists_pt)
    ax.plot(fpr_pt, tpr_pt, 'g--', linewidth=2,
            label=f'v9 Per-type (AUC={pt_metrics["auc"]:.4f})')

    # Reference: v8 best
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Drone Detection ROC Curve', fontsize=13)
    ax.legend(fontsize=11, loc='lower right')

    # Mark operational points
    for fpr_target, marker in [(0.01, 'o'), (0.05, 's'), (0.1, '^')]:
        key = f"fpr_{fpr_target:.3f}_tpr"
        if key in global_metrics:
            idx = np.searchsorted(fpr, fpr_target)
            if idx < len(tpr):
                ax.plot(fpr_target, tpr[idx], marker, color='red', markersize=10)
                ax.annotate(f'{tpr[idx]*100:.1f}% @ FPR={fpr_target:.0%}',
                           xy=(fpr_target, tpr[idx]),
                           xytext=(fpr_target + 0.05, tpr[idx] - 0.05),
                           fontsize=9, arrowprops=dict(arrowstyle='->', color='red'))

    # Per-holdout-type AUC bar chart
    ax = axes[1, 0]
    type_names_sorted = sorted(per_type_breakdown.keys())
    type_aucs = [per_type_breakdown[t]["auc"] for t in type_names_sorted]
    type_counts = [per_type_breakdown[t]["n_samples"] for t in type_names_sorted]
    colors = ['#4CAF50' if a >= 0.99 else '#FF9800' if a >= 0.95 else '#F44336'
              for a in type_aucs]
    bars = ax.barh(range(len(type_names_sorted)), type_aucs, color=colors)
    ax.set_yticks(range(len(type_names_sorted)))
    ax.set_yticklabels([f'{t} (n={n})' for t, n in
                         zip(type_names_sorted, type_counts)], fontsize=9)
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title('Per-Holdout-Type Detection AUC', fontsize=13)
    ax.axvline(x=0.95, color='orange', linestyle='--', alpha=0.5, label='AUC=0.95')
    ax.axvline(x=0.99, color='green', linestyle='--', alpha=0.5, label='AUC=0.99')
    ax.set_xlim(0, 1.05)
    ax.legend(loc='lower right', fontsize=10)

    # Add AUC values on bars
    for i, (auc, bar) in enumerate(zip(type_aucs, bars)):
        ax.text(min(auc + 0.01, 1.0), i, f'{auc:.3f}', va='center', fontsize=9)

    # Per-holdout-type distance distribution (box plot)
    ax = axes[1, 1]
    drone_type_dists = []
    drone_type_labels = []
    for dtype in sorted(set(test_types[drone_mask_test])):
        mask = test_types == dtype
        drone_type_dists.append(global_mahal[mask])
        drone_type_labels.append(dtype)

    # Add background
    drone_type_dists.append(global_bg_dists)
    drone_type_labels.append("BACKGROUND")

    bp = ax.boxplot(drone_type_dists, labels=[l[:15] for l in drone_type_labels],
                     patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp['boxes']):
        if i < len(drone_type_dists) - 1:
            patch.set_facecolor('#2196F3')
        else:
            patch.set_facecolor('#F44336')
    ax.set_ylabel('Global Mahalanobis Distance', fontsize=12)
    ax.set_title('Distance Distribution by Type', fontsize=13)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"{EVAL_DIR}/v9_detection_stress_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {EVAL_DIR}/v9_detection_stress_test.png")

    # ── UMAP visualization ──
    print("\n  Computing UMAP visualization...")
    import umap

    # Subsample for UMAP (too many points = slow + cluttered)
    rng = np.random.default_rng(42)
    n_drone_umap = min(len(train_embs), 2000)
    n_holdout_umap = min(len(holdout_embs), 1500)
    n_bg_umap = min(int(bg_mask_test.sum()), 2000)

    # Sample indices
    train_idx = rng.choice(len(train_embs), n_drone_umap, replace=False)
    holdout_idx = rng.choice(len(holdout_embs), n_holdout_umap, replace=False)
    bg_idx_all = np.where(bg_mask_test)[0]
    bg_idx = rng.choice(bg_idx_all, min(n_bg_umap, len(bg_idx_all)), replace=False)

    umap_embs = np.concatenate([
        train_embs[train_idx],
        holdout_embs[holdout_idx],
        test_embs[bg_idx],
    ])

    umap_labels = (
        ["train_" + str(t) for t in train_types[train_idx]] +
        ["holdout_" + str(t) for t in holdout_types[holdout_idx]] +
        ["background"] * len(bg_idx)
    )
    umap_source = (
        ["train_drone"] * len(train_idx) +
        ["holdout_drone"] * len(holdout_idx) +
        ["background"] * len(bg_idx)
    )

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                         random_state=42)
    embedding_2d = reducer.fit_transform(umap_embs)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Plot background first (grey, small)
    bg_umap_mask = np.array([s == "background" for s in umap_source])
    ax.scatter(embedding_2d[bg_umap_mask, 0], embedding_2d[bg_umap_mask, 1],
               c='#BDBDBD', s=5, alpha=0.3, label=f'Background (n={bg_umap_mask.sum()})',
               rasterized=True)

    # Plot train drones (blue shades)
    train_umap_mask = np.array([s == "train_drone" for s in umap_source])
    ax.scatter(embedding_2d[train_umap_mask, 0], embedding_2d[train_umap_mask, 1],
               c='#1976D2', s=10, alpha=0.4,
               label=f'Train Drones (n={train_umap_mask.sum()}, 30 types)',
               rasterized=True)

    # Plot holdout drones (red, larger)
    holdout_umap_mask = np.array([s == "holdout_drone" for s in umap_source])
    ax.scatter(embedding_2d[holdout_umap_mask, 0], embedding_2d[holdout_umap_mask, 1],
               c='#D32F2F', s=15, alpha=0.6,
               label=f'Holdout Drones (n={holdout_umap_mask.sum()}, 7 unseen types)',
               marker='^', rasterized=True)

    ax.set_xlabel('UMAP 1', fontsize=13)
    ax.set_ylabel('UMAP 2', fontsize=13)
    ax.set_title(f'IRIS v9 Embedding Space\n'
                 f'Global AUC={global_metrics["auc"]:.4f} | '
                 f'Sil(train)={train_sil:.3f} | '
                 f'BG/Drone={global_metrics["bg_drone_ratio"]:.1f}x',
                 fontsize=14)
    ax.legend(fontsize=12, loc='best')

    plt.tight_layout()
    plt.savefig(f"{EVAL_DIR}/v9_umap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {EVAL_DIR}/v9_umap.png")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Checkpoint: lejepa_v9_best.pt (epoch {epoch})")
    print(f"  Train drone types: {train_ds.n_drone_types}")
    print(f"  Holdout drone types: {holdout_ds.n_drone_types} (UNSEEN)")
    print(f"  Background samples tested: {int(bg_mask_test.sum())}")
    print()
    print(f"  TRAIN CLUSTERING")
    print(f"    k-NN CV:    {train_knn:.4f}")
    print(f"    Silhouette: {train_sil:.4f}")
    print()
    print(f"  HOLDOUT CLUSTERING")
    print(f"    Silhouette: {ho_sil:.4f}")
    print(f"    k-NN CV:    {ho_knn:.4f}")
    print()
    print(f"  DETECTION (Global Mahalanobis)")
    print(f"    AUC:            {global_metrics['auc']:.4f}")
    print(f"    Avg Precision:  {global_metrics['ap']:.4f}")
    print(f"    BG/Drone ratio: {global_metrics['bg_drone_ratio']:.2f}x")
    print(f"    95% TPR @ FPR:  {global_metrics.get('tpr_0.950_fpr', 'N/A')}")
    print()
    print(f"  DETECTION (Per-type Mahalanobis)")
    print(f"    AUC:            {pt_metrics['auc']:.4f}")
    print(f"    BG/Drone ratio: {pt_metrics['bg_drone_ratio']:.2f}x")
    print()
    print(f"  PER-HOLDOUT-TYPE AUC")
    for dtype, info in sorted(per_type_breakdown.items()):
        status = "PASS" if info["auc"] >= 0.95 else "FAIL"
        print(f"    {dtype:<25} AUC={info['auc']:.4f}  [{status}]")

    # ── v8 comparison ──
    print()
    print(f"  v8 vs v9 COMPARISON (same test, different models)")
    print(f"    v8 best holdout AUC: 0.886 (epoch6, 2000 bg)")
    print(f"    v8 best holdout AUC: 0.312 (best Sil, 2000 bg)")
    print(f"    v9 holdout AUC:      {global_metrics['auc']:.4f} ({int(bg_mask_test.sum())} bg)")

    if global_metrics["auc"] >= 0.99:
        print()
        print("  *** AUC >= 0.99 WITH 50K NEGATIVES ***")
        print("  *** V9 PASSES THE FULL STRESS TEST ***")

    # ── Save all results to JSON ──
    all_results = {
        "checkpoint": "lejepa_v9_best.pt",
        "epoch": epoch,
        "n_bg_eval": N_BG_EVAL,
        "n_train_drones": int(len(train_embs)),
        "n_holdout_drones": int(drone_mask_test.sum()),
        "n_bg_tested": int(bg_mask_test.sum()),
        "train_types": train_ds.type_names,
        "holdout_types": holdout_ds.type_names,
        "classification": {
            "train_knn_cv": train_knn,
            "train_silhouette": train_sil,
            "holdout_silhouette": ho_sil,
            "holdout_knn_cv": ho_knn,
            "binary_accuracy_cv": binary_acc,
        },
        "global_mahalanobis": global_metrics,
        "per_type_mahalanobis": pt_metrics,
        "per_holdout_type": {k: v for k, v in per_type_breakdown.items()},
    }

    results_path = f"{EVAL_DIR}/v9_holdout_eval.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: float(o)
                  if hasattr(o, '__float__') else str(o))
    MODEL_VOL.commit()
    print(f"\n  Results saved: {results_path}")

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")


@app.local_entrypoint()
def main():
    evaluate_v9.remote()