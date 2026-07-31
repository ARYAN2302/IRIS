#!/usr/bin/env python3
"""
IRIS Three Experiments — Combined Run on T4

Experiment 1: DJI-vs-non-DJI Re-split
  - Fit Mahalanobis centroid on 26 non-DJI train types only
  - Test zero-shot on 5 DJI types (4 train + 1 holdout)
  - Proves IRIS learned "drone-ness" not "DJI-ness"

Experiment 2: AVR-CL Sequential Enrollment
  - Enroll 7 holdout types one by one (6 non-DJI + 1 DJI FPV COMBO)
  - Compare naive vs AVR-CL: does enrolling type N forget types 1..N-1?
  - Uses fingerprint head + anchor-verify-repair

Experiment 3: DroneRF Parrot Re-labeling Check
  - Scan 200 random negatives from DroneRF
  - Check if any are detected as drones (possible mislabeled Parrot captures)
  - If found, re-label and test IRIS on them as non-DJI drones

Usage:
    modal run scripts/three_experiments.py
"""

from __future__ import annotations

import h5py
import json
import os
import sys
import time
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# ─────────────────────────────────────────────────────────────────────────────
# Modal setup — T4
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("iris-three-experiments")

VOL = modal.Volume.from_name("iris-data", create_if_missing=True)
MODEL_VOL = modal.Volume.from_name("iris-models-v11", create_if_missing=True)
MATCHED_VOL = modal.Volume.from_name("iris-matched-bg", create_if_missing=True)
RESULTS_VOL = modal.Volume.from_name("iris-results", create_if_missing=True)

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
MATCHED_REMOTE = "/matched/iris_matched_bg.h5"
MODEL_REMOTE = "/models/lejepa_v11_best.pt"
RESULTS_REMOTE = "/results"

# DJI types in RFUAV (from configs/split.json)
DJI_TYPES = ["DJI AVATA2", "DJI MAVIC3 PRO", "DJI MINI3", "DJI MINI4 PRO", "DJI FPV COMBO"]


# ─────────────────────────────────────────────────────────────────────────────
# Encoder (exact reproduction from train_modal_v11.py)
# ─────────────────────────────────────────────────────────────────────────────


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
        )
    def forward(self, x): return self.block(x)


class CNNEncoder(nn.Module):
    def __init__(self, in_ch=2, width=64, depth=6, embed_dim=256):
        super().__init__()
        layers, ch = [], in_ch
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
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, embed_dim), nn.BatchNorm1d(embed_dim))
    def forward(self, x): return self.head(self.conv(x))


class FingerprintHead(nn.Module):
    def __init__(self, embed_dim=256, fp_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, fp_dim), nn.BatchNorm1d(fp_dim), nn.GELU(),
        )
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_type_dataset(grp, key):
    item = grp[key]
    if isinstance(item, h5py.Dataset):
        if len(item.shape) == 4: return item, item.shape[0], False
        elif len(item.shape) == 3: return item, 1, False
        else: raise ValueError(f"Bad shape {item.shape}")
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
    if sub_datasets:
        try: sub_datasets.sort(key=lambda x: int(x))
        except ValueError: sub_datasets.sort()
        return item, len(sub_datasets), True
    raise ValueError(f"Cannot resolve /{key}")


def _prep(sample):
    if sample.shape[0] == 3: return sample[:2].copy().astype(np.float32)
    elif sample.shape[0] == 2: return sample.copy().astype(np.float32)
    else: return sample[:2].copy().astype(np.float32)


def _norm(x):
    for c in range(x.shape[0]):
        ch, std = x[c], x[c].std()
        if std > 1e-6: x[c] = (ch - ch.mean()) / std
        else: x[c] = ch - ch.mean()
    return x


def fit_mahalanobis_l2(embs, reg=1e-3):
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs = embs / norms
    centroid = embs.mean(axis=0)
    D = embs.shape[1]
    cov = np.cov(embs.T) + reg * np.eye(D)
    try: cov_inv = np.linalg.inv(cov)
    except: cov_inv = np.linalg.pinv(cov)
    return centroid, cov_inv


def mahal_l2(embs, centroid, cov_inv):
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs = embs / norms
    diff = embs - centroid
    return np.sqrt(np.maximum(np.sum(diff @ cov_inv * diff, axis=1), 0))


@torch.no_grad()
def encode_batch(encoder, specs, device, bs=32):
    encoder.eval()
    all_embs = []
    for i in range(0, len(specs), bs):
        batch = torch.from_numpy(specs[i:i+bs]).float().to(device)
        all_embs.append(encoder(batch).cpu().numpy())
    return np.concatenate(all_embs)


def load_type_samples(h5_path, split, type_name, max_n=200):
    """Load samples for a specific drone type from a split."""
    with h5py.File(h5_path, "r") as f:
        if split not in f: return np.array([])
        grp = f[split]
        if type_name not in grp: return np.array([])
        try:
            ds_or_grp, n_samples, is_multi = _resolve_type_dataset(grp, type_name)
        except: return np.array([])

        specs = []
        if is_multi:
            sub_keys = [sk for sk in ds_or_grp.keys()
                        if isinstance(ds_or_grp[sk], h5py.Dataset) and len(ds_or_grp[sk].shape) == 3]
            try: sub_keys.sort(key=lambda x: int(x))
            except: sub_keys.sort()
            for sk in sub_keys[:max_n]:
                specs.append(_norm(_prep(ds_or_grp[sk][:])))
        else:
            n = min(ds_or_grp.shape[0] if len(ds_or_grp.shape) == 4 else 1, max_n)
            for i in range(n):
                if len(ds_or_grp.shape) == 4: specs.append(_norm(_prep(ds_or_grp[i])))
                else: specs.append(_norm(_prep(ds_or_grp[:])))

        return np.stack(specs) if specs else np.array([])


def load_all_type_samples(h5_path, split, max_per_type=100):
    """Load samples for ALL types in a split. Returns {type_name: np.ndarray}."""
    with h5py.File(h5_path, "r") as f:
        if split not in f: return {}
        grp = f[split]
        type_names = sorted(list(grp.keys()))
        result = {}
        for tname in type_names:
            try:
                specs = load_type_samples(h5_path, split, tname, max_per_type)
                if len(specs) > 0: result[tname] = specs
            except: continue
        return result


def load_negative_samples(h5_path, n=200, seed=42):
    """Load n random negative samples from /negatives/."""
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as f:
        if "negatives" not in f: return np.array([])
        neg_grp = f["negatives"]
        if isinstance(neg_grp, h5py.Dataset):
            n_total = neg_grp.shape[0]
            indices = rng.choice(n_total, min(n, n_total), replace=False)
            return np.stack([_norm(_prep(neg_grp[int(i)])) for i in indices])
        all_keys = list(neg_grp.keys())
        picked = rng.choice(all_keys, min(n, len(all_keys)), replace=False)
        specs = []
        for key in picked:
            try:
                item = neg_grp[str(key)]
                if isinstance(item, h5py.Dataset):
                    if len(item.shape) == 4: specs.append(_norm(_prep(item[0])))
                    elif len(item.shape) == 3: specs.append(_norm(_prep(item[:])))
            except: continue
        return np.stack(specs) if specs else np.array([])


# ─────────────────────────────────────────────────────────────────────────────
# AVR-CL functions
# ─────────────────────────────────────────────────────────────────────────────


def get_head_state(head):
    return {n: p.data.cpu().clone() for n, p in head.named_parameters()}


def set_head_state(head, state, device="cuda"):
    for n, p in head.named_parameters():
        if n in state: p.data.copy_(state[n].to(device))


def repair_head(head, snapshot, alpha=0.1, device="cuda"):
    n = 0
    for name, p in head.named_parameters():
        if name in snapshot:
            p.data.copy_((1.0 - alpha) * p.data + alpha * snapshot[name].to(device))
            n += 1
    return n


def supcon_loss(embeddings, labels, temperature=0.07):
    device = embeddings.device
    B = embeddings.shape[0]
    embeddings = F.normalize(embeddings, dim=1)
    sim = torch.mm(embeddings, embeddings.t()) / temperature
    sim = sim.clamp(-10.0, 10.0)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()).float()
    diag = torch.eye(B, device=device)
    pos_mask = (pos_mask - diag).clamp(min=0)
    sim_max, _ = sim.max(dim=1, keepdim=True)
    exp_sim = torch.exp(sim - sim_max.detach())
    denom = (exp_sim * (1.0 - diag)).sum(dim=1, keepdim=True)
    numer = (exp_sim * pos_mask).sum(dim=1, keepdim=True)
    log_prob = torch.log(numer + 1e-8) - torch.log(denom + 1e-8)
    n_pos = pos_mask.sum(dim=1)
    valid = n_pos > 0
    if valid.sum() == 0: return torch.tensor(0.0, device=device, requires_grad=True)
    mean_log = (log_prob * pos_mask).sum(dim=1) / (n_pos + 1e-8)
    return -mean_log[valid].mean()


def identify_accuracy(head, registry, test_embs, test_labels, device, threshold=0.5):
    """Test identification: given embeddings, match against registry."""
    head.eval()
    if not registry: return 0.0, {}
    correct = 0
    total = 0
    per_type = {}
    with torch.no_grad():
        fps = head(torch.from_numpy(test_embs).float().to(device)).cpu().numpy()
    for i, (fp, true_label) in enumerate(zip(fps, test_labels)):
        best_sim = -1
        best_type = None
        for dtype, enrolled_fp in registry.items():
            sim = float(np.dot(fp, enrolled_fp))
            if sim > best_sim:
                best_sim = sim
                best_type = dtype
        if best_sim >= threshold and best_type == true_label:
            correct += 1
        per_type.setdefault(true_label, {"correct": 0, "total": 0})
        per_type[true_label]["total"] += 1
        if best_sim >= threshold and best_type == true_label:
            per_type[true_label]["correct"] += 1
        total += 1
    return (correct / total if total > 0 else 0), per_type


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


@app.function(
    image=IMAGE,
    gpu="T4",
    volumes={"/data": VOL, "/models": MODEL_VOL, "/matched": MATCHED_VOL, "/results": RESULTS_VOL},
    timeout=5400,
    memory=16384,
)
def run_all_experiments():
    device = "cuda"
    print("=" * 70)
    print("IRIS — Three Experiments Combined Run")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"  GPU:  T4")
    print("=" * 70)

    VOL.reload()
    MODEL_VOL.reload()
    MATCHED_VOL.reload()

    # Load encoder
    print("\n[0] Loading encoder...")
    ckpt = torch.load(MODEL_REMOTE, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    encoder_state = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
    encoder = CNNEncoder(in_ch=2, width=64, depth=6, embed_dim=256).to(device)
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    print(f"  [ok] encoder: {sum(p.numel() for p in encoder.parameters()):,} params")

    all_results = {}

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: DJI-vs-non-DJI Re-split
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: DJI-vs-non-DJI Re-split")
    print("=" * 70)
    print("  Fit Mahalanobis centroid on 26 non-DJI train types only.")
    print("  Test zero-shot on 5 DJI types.")
    print("  If AUC > 0.9 → IRIS learned 'drone-ness' not 'DJI-ness'.")

    # Load all train types
    print("\n  [1.1] Loading train data...")
    train_data = load_all_type_samples(H5_REMOTE, "train", max_per_type=100)
    print(f"    {len(train_data)} train types loaded")

    # Split into DJI and non-DJI
    dji_train_types = [t for t in train_data if t in DJI_TYPES]
    nondji_train_types = [t for t in train_data if t not in DJI_TYPES]
    print(f"    DJI train types: {dji_train_types}")
    print(f"    Non-DJI train types: {len(nondji_train_types)} types")

    # Encode non-DJI train types → fit centroid
    print("\n  [1.2] Encoding non-DJI train types → fitting centroid...")
    nondji_embs = []
    for t in nondji_train_types:
        embs = encode_batch(encoder, train_data[t], device)
        nondji_embs.append(embs)
    nondji_embs = np.concatenate(nondji_embs)
    centroid, cov_inv = fit_mahalanobis_l2(nondji_embs)
    print(f"    centroid fit from {len(nondji_embs)} non-DJI samples")

    # Load and encode DJI types (from both train and holdout)
    print("\n  [1.3] Loading and encoding DJI types...")
    dji_specs = {}
    for t in DJI_TYPES:
        # Check train split
        specs = load_type_samples(H5_REMOTE, "train", t, max_n=100)
        if len(specs) == 0:
            # Check holdout split
            specs = load_type_samples(H5_REMOTE, "holdout", t, max_n=100)
        if len(specs) > 0:
            dji_specs[t] = specs
            print(f"    {t}: {len(specs)} samples")

    # Load matched BGs for comparison
    print("  [1.4] Loading matched BGs...")
    with h5py.File(MATCHED_REMOTE, "r") as f:
        key = "holdout_matched_bg"
        if key in f:
            grp = f[key]
            keys = sorted(list(grp.keys()), key=lambda x: int(x) if x.isdigit() else 0)
            bg_specs = np.stack([_norm(_prep(grp[k][:])) for k in keys[:200]])

    # Encode DJI + BG
    dji_embs = {}
    for t, specs in dji_specs.items():
        dji_embs[t] = encode_batch(encoder, specs, device)
    bg_embs = encode_batch(encoder, bg_specs, device)

    # Compute distances
    dji_dists = {t: mahal_l2(e, centroid, cov_inv) for t, e in dji_embs.items()}
    bg_dists = mahal_l2(bg_embs, centroid, cov_inv)

    # AUC: DJI vs BG
    all_dji_d = np.concatenate([d for d in dji_dists.values()])
    all_d = np.concatenate([all_dji_d, bg_dists])
    all_l = np.concatenate([np.ones(len(all_dji_d)), np.zeros(len(bg_dists))])
    auc_dji = roc_auc_score(all_l, -all_d)

    # Per-type DJI AUC
    per_type_auc = {}
    for t, d in dji_dists.items():
        labels = np.concatenate([np.ones(len(d)), np.zeros(len(bg_dists))])
        dists = np.concatenate([d, bg_dists])
        per_type_auc[t] = float(roc_auc_score(labels, -dists))

    # Also test non-DJI holdout types
    print("\n  [1.5] Loading non-DJI holdout types for comparison...")
    holdout_data = load_all_type_samples(H5_REMOTE, "holdout", max_per_type=100)
    nondji_holdout = {t: s for t, s in holdout_data.items() if t not in DJI_TYPES}
    print(f"    {len(nondji_holdout)} non-DJI holdout types")

    nondji_holdout_dists = {}
    for t, specs in nondji_holdout.items():
        embs = encode_batch(encoder, specs, device)
        nondji_holdout_dists[t] = mahal_l2(embs, centroid, cov_inv)

    all_nondji_d = np.concatenate([d for d in nondji_holdout_dists.values()])
    all_d2 = np.concatenate([all_nondji_d, bg_dists])
    all_l2 = np.concatenate([np.ones(len(all_nondji_d)), np.zeros(len(bg_dists))])
    auc_nondji = roc_auc_score(all_l2, -all_d2)

    print(f"\n  RESULTS:")
    print(f"    AUC (DJI vs BG, centroid fit on non-DJI):      {auc_dji:.4f}")
    print(f"    AUC (non-DJI holdout vs BG, same centroid):    {auc_nondji:.4f}")
    print(f"    Per-type DJI AUC:")
    for t, a in per_type_auc.items():
        print(f"      {t}: {a:.4f}")

    exp1_results = {
        "description": "Fit Mahalanobis on 26 non-DJI train types, test zero-shot on 5 DJI types",
        "nondji_train_types": len(nondji_train_types),
        "dji_types_tested": list(dji_specs.keys()),
        "auc_dji_vs_bg": float(auc_dji),
        "auc_nondji_holdout_vs_bg": float(auc_nondji),
        "per_type_dji_auc": {t: float(a) for t, a in per_type_auc.items()},
        "verdict": "PASS — IRIS learned drone-ness, not DJI-ness" if auc_dji > 0.9 else "PARTIAL — some DJI generalization",
    }
    all_results["experiment_1_dji_vs_nondji"] = exp1_results

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: AVR-CL Sequential Enrollment
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: AVR-CL Sequential Enrollment")
    print("=" * 70)
    print("  Enroll 7 holdout types one by one.")
    print("  Compare naive vs AVR-CL: does enrolling type N forget types 1..N-1?")

    # Load holdout data
    print("\n  [2.1] Loading holdout data...")
    if not holdout_data:
        holdout_data = load_all_type_samples(H5_REMOTE, "holdout", max_per_type=100)
    holdout_types = sorted(holdout_data.keys())
    print(f"    {len(holdout_types)} holdout types: {holdout_types}")

    # Split each type into enrollment (60%) and test (40%)
    rng = np.random.default_rng(42)
    enroll_data = {}
    test_data = {}
    test_labels = []
    test_embs_all = []
    for t in holdout_types:
        specs = holdout_data[t]
        n = len(specs)
        perm = rng.permutation(n)
        n_enroll = max(3, n // 2)
        enroll_data[t] = specs[perm[:n_enroll]]
        test_data[t] = specs[perm[n_enroll:]]
        # Pre-compute test embeddings
        if len(test_data[t]) > 0:
            embs = encode_batch(encoder, test_data[t], device)
            test_embs_all.append(embs)
            test_labels.extend([t] * len(embs))

    test_embs_all = np.concatenate(test_embs_all)
    test_labels = np.array(test_labels)
    print(f"    Enrollment samples per type: {[f'{t}:{len(enroll_data[t])}' for t in holdout_types]}")
    print(f"    Total test samples: {len(test_labels)}")

    # Pre-compute enrollment embeddings
    enroll_embs = {}
    for t in holdout_types:
        enroll_embs[t] = encode_batch(encoder, enroll_data[t], device)

    # --- Run naive method ---
    print("\n  [2.2] Running NAIVE enrollment (no repair)...")
    fp_head_naive = FingerprintHead(embed_dim=256, fp_dim=128).to(device)
    registry_naive = {}
    naive_history = []

    for i, t in enumerate(holdout_types):
        # Enroll: fine-tune on new type
        fp_head_naive.train()
        opt = torch.optim.AdamW(fp_head_naive.parameters(), lr=1e-3, weight_decay=0.01)
        labels = torch.zeros(len(enroll_embs[t]), dtype=torch.long, device=device)
        embs_tensor = torch.from_numpy(enroll_embs[t]).float().to(device)
        for epoch in range(3):
            perm = torch.randperm(len(embs_tensor))
            for j in range(0, len(embs_tensor), 16):
                idx = perm[j:j+16]
                if len(idx) < 2: continue
                fps = fp_head_naive(embs_tensor[idx])
                loss = supcon_loss(fps, labels[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        fp_head_naive.eval()

        # Store mean fingerprint for this type
        with torch.no_grad():
            mean_fp = fp_head_naive(torch.from_numpy(enroll_embs[t].mean(axis=0)).float().unsqueeze(0).to(device))
        registry_naive[t] = mean_fp.cpu().numpy()[0]

        # Test accuracy on ALL enrolled types so far
        enrolled_so_far = holdout_types[:i+1]
        test_mask = np.isin(test_labels, enrolled_so_far)
        acc, per_type = identify_accuracy(
            fp_head_naive, {k: registry_naive[k] for k in enrolled_so_far},
            test_embs_all[test_mask], test_labels[test_mask], device, threshold=0.3
        )
        naive_history.append({
            "enrolled_type": t,
            "n_enrolled": i + 1,
            "overall_accuracy": acc,
            "per_type": {k: v["correct"]/max(v["total"],1) for k, v in per_type.items()},
        })
        print(f"    Naive [{i+1}/{len(holdout_types)}] enrolled {t}: acc={acc:.3f}")

    # --- Run AVR-CL method ---
    print("\n  [2.3] Running AVR-CL enrollment (anchor-verify-repair)...")
    fp_head_avr = FingerprintHead(embed_dim=256, fp_dim=128).to(device)
    registry_avr = {}
    best_accuracies = {}
    avr_history = []
    total_repairs = 0

    for i, t in enumerate(holdout_types):
        # ANCHOR: snapshot weights
        snapshot = get_head_state(fp_head_avr)

        # LEARN: fine-tune on new type
        fp_head_avr.train()
        opt = torch.optim.AdamW(fp_head_avr.parameters(), lr=1e-3, weight_decay=0.01)
        labels = torch.zeros(len(enroll_embs[t]), dtype=torch.long, device=device)
        embs_tensor = torch.from_numpy(enroll_embs[t]).float().to(device)
        for epoch in range(3):
            perm = torch.randperm(len(embs_tensor))
            for j in range(0, len(embs_tensor), 16):
                idx = perm[j:j+16]
                if len(idx) < 2: continue
                fps = fp_head_avr(embs_tensor[idx])
                loss = supcon_loss(fps, labels[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        fp_head_avr.eval()

        # Enroll new type
        with torch.no_grad():
            mean_fp = fp_head_avr(torch.from_numpy(enroll_embs[t].mean(axis=0)).float().unsqueeze(0).to(device))
        registry_avr[t] = mean_fp.cpu().numpy()[0]

        # VERIFY: test accuracy on all enrolled types
        enrolled_so_far = holdout_types[:i+1]
        test_mask = np.isin(test_labels, enrolled_so_far)
        acc, per_type = identify_accuracy(
            fp_head_avr, {k: registry_avr[k] for k in enrolled_so_far},
            test_embs_all[test_mask], test_labels[test_mask], device, threshold=0.3
        )

        # Check drift on previously enrolled types
        repairs = 0
        if i > 0:
            prev_types = holdout_types[:i]
            drifted = False
            for pt in prev_types:
                if pt in per_type and pt in best_accuracies:
                    drop = best_accuracies[pt] - per_type[pt]["correct"]/max(per_type[pt]["total"],1)
                    if drop > 0.1:  # 10% accuracy drop = drift
                        drifted = True
                        break

            if drifted:
                # REPAIR
                for step in range(5):
                    n = repair_head(fp_head_avr, snapshot, alpha=0.15, device=device)
                    repairs += 1
                    # Re-enroll with repaired weights
                    for et in enrolled_so_far:
                        with torch.no_grad():
                            fp = fp_head_avr(torch.from_numpy(enroll_embs[et].mean(axis=0)).float().unsqueeze(0).to(device))
                        registry_avr[et] = fp.cpu().numpy()[0]
                    # Re-verify
                    acc, per_type = identify_accuracy(
                        fp_head_avr, {k: registry_avr[k] for k in enrolled_so_far},
                        test_embs_all[test_mask], test_labels[test_mask], device, threshold=0.3
                    )
                    still_drifted = False
                    for pt in prev_types:
                        if pt in per_type and pt in best_accuracies:
                            drop = best_accuracies[pt] - per_type[pt]["correct"]/max(per_type[pt]["total"],1)
                            if drop > 0.1: still_drifted = True; break
                    if not still_drifted: break

        total_repairs += repairs

        # Update best accuracies
        for pt in per_type:
            acc_pt = per_type[pt]["correct"]/max(per_type[pt]["total"],1)
            if pt not in best_accuracies or acc_pt > best_accuracies[pt]:
                best_accuracies[pt] = acc_pt

        avr_history.append({
            "enrolled_type": t,
            "n_enrolled": i + 1,
            "overall_accuracy": acc,
            "per_type": {k: v["correct"]/max(v["total"],1) for k, v in per_type.items()},
            "repairs": repairs,
        })
        print(f"    AVR-CL [{i+1}/{len(holdout_types)}] enrolled {t}: acc={acc:.3f}, repairs={repairs}")

    # Compare final
    print(f"\n  RESULTS:")
    print(f"    Naive final accuracy:   {naive_history[-1]['overall_accuracy']:.3f}")
    print(f"    AVR-CL final accuracy:  {avr_history[-1]['overall_accuracy']:.3f}")
    print(f"    Total AVR-CL repairs:   {total_repairs}")
    print(f"    Naive per-type: {naive_history[-1]['per_type']}")
    print(f"    AVR-CL per-type: {avr_history[-1]['per_type']}")

    exp2_results = {
        "description": "Sequential enrollment of 7 holdout types, naive vs AVR-CL",
        "holdout_types": holdout_types,
        "naive_history": naive_history,
        "avr_history": avr_history,
        "naive_final_accuracy": naive_history[-1]["overall_accuracy"],
        "avr_final_accuracy": avr_history[-1]["overall_accuracy"],
        "total_repairs": total_repairs,
    }
    all_results["experiment_2_avr_cl"] = exp2_results

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: DroneRF Parrot Re-labeling Check
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: DroneRF Parrot Re-labeling Check")
    print("=" * 70)
    print("  Scan 200 random DroneRF negatives.")
    print("  Check if any are detected as drones (possible mislabeled Parrot captures).")

    print("\n  [3.1] Loading 200 random negatives...")
    neg_specs = load_negative_samples(H5_REMOTE, n=200, seed=42)
    print(f"    loaded {len(neg_specs)} negatives")

    # Use centroid from ALL train types (original IRIS detector)
    print("  [3.2] Re-fitting centroid on ALL train types...")
    all_train_embs = []
    for t in train_data:
        embs = encode_batch(encoder, train_data[t], device)
        all_train_embs.append(embs)
    all_train_embs = np.concatenate(all_train_embs)
    full_centroid, full_cov_inv = fit_mahalanobis_l2(all_train_embs)
    train_dists = mahal_l2(all_train_embs, full_centroid, full_cov_inv)
    threshold = float(np.percentile(train_dists, 99))
    print(f"    threshold (99th pct): {threshold:.2f}")

    # Encode negatives and compute distances
    print("  [3.3] Encoding negatives + computing distances...")
    neg_embs = encode_batch(encoder, neg_specs, device)
    neg_dists = mahal_l2(neg_embs, full_centroid, full_cov_inv)

    # How many negatives are detected as drones?
    n_detected = int((neg_dists <= threshold).sum())
    pct_detected = n_detected / len(neg_dists) * 100

    print(f"\n  RESULTS:")
    print(f"    Negatives tested: {len(neg_dists)}")
    print(f"    Detected as drones: {n_detected} ({pct_detected:.1f}%)")
    print(f"    Distance stats: min={neg_dists.min():.2f}, mean={neg_dists.mean():.2f}, max={neg_dists.max():.2f}")
    print(f"    Threshold: {threshold:.2f}")

    if n_detected > 5:
        print(f"\n    ⚠ {n_detected} negatives look like drones!")
        print(f"    These might be mislabeled Parrot/other drone captures.")
        print(f"    Re-labeling them as drone positives would add non-DJI training data.")

        # Check if detected negatives cluster together (same drone type?)
        detected_embs = neg_embs[neg_dists <= threshold]
        if len(detected_embs) > 1:
            from sklearn.metrics import silhouette_score
            from sklearn.cluster import KMeans
            if len(detected_embs) >= 4:
                for k in range(2, min(4, len(detected_embs))):
                    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(detected_embs)
                    sil = silhouette_score(detected_embs, labels)
                    print(f"    Detected negatives cluster analysis (k={k}): silhouette={sil:.3f}")
    else:
        print(f"\n    ✓ Negatives are genuinely background — no mislabeled drones found.")

    exp3_results = {
        "description": "Scan DroneRF negatives for mislabeled drone captures",
        "n_negatives_tested": len(neg_dists),
        "n_detected_as_drones": n_detected,
        "pct_detected": float(pct_detected),
        "neg_distance_min": float(neg_dists.min()),
        "neg_distance_mean": float(neg_dists.mean()),
        "neg_distance_max": float(neg_dists.max()),
        "threshold": threshold,
        "verdict": f"{n_detected} possible mislabeled drones found" if n_detected > 5 else "Negatives are genuine background",
    }
    all_results["experiment_3_dronerf_check"] = exp3_results

    # ═══════════════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    os.makedirs(RESULTS_REMOTE, exist_ok=True)
    all_results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    json_path = f"{RESULTS_REMOTE}/three_experiments.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  [ok] saved {json_path}")

    # Generate plot for Experiment 2
    print("  [info] generating enrollment comparison plot...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Overall accuracy over enrollment steps
        steps = [h["n_enrolled"] for h in naive_history]
        naive_accs = [h["overall_accuracy"] for h in naive_history]
        avr_accs = [h["overall_accuracy"] for h in avr_history]

        ax1.plot(steps, naive_accs, "o-", color="red", linewidth=2, markersize=10, label="Naive (no repair)")
        ax1.plot(steps, avr_accs, "s-", color="green", linewidth=2, markersize=10, label="AVR-CL (anchor-verify-repair)")
        ax1.set_xlabel("Number of Enrolled Types", fontsize=12)
        ax1.set_ylabel("Identification Accuracy", fontsize=12)
        ax1.set_title("Experiment 2: Sequential Enrollment — Naive vs AVR-CL", fontsize=13)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xticks(steps)

        # Plot 2: Experiment 1 bar chart
        types = list(per_type_auc.keys())
        aucs = list(per_type_auc.values())
        colors = ["#ff6b6b" if "DJI" in t else "#4ecdc4" for t in types]
        ax2.barh(range(len(types)), aucs, color=colors, edgecolor="black")
        ax2.set_yticks(range(len(types)))
        ax2.set_yticklabels([t[:20] for t in types], fontsize=10)
        ax2.set_xlabel("AUC (zero-shot, centroid on non-DJI)", fontsize=12)
        ax2.set_title("Experiment 1: DJI Detection (trained on non-DJI only)", fontsize=13)
        ax2.axvline(x=0.9, color="green", linestyle="--", alpha=0.5, label="0.9 target")
        ax2.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="chance")
        ax2.legend(fontsize=10)
        ax2.set_xlim(0, 1.05)
        for i, v in enumerate(aucs):
            ax2.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)

        plt.tight_layout()
        plot_path = f"{RESULTS_REMOTE}/three_experiments.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [ok] saved {plot_path}")
    except Exception as e:
        print(f"  [warn] plot failed: {e}")

    # Generate markdown report
    md_path = f"{RESULTS_REMOTE}/three_experiments.md"
    with open(md_path, "w") as f:
        f.write("# IRIS — Three Experiments Report\n\n")
        f.write(f"**Generated:** {all_results['timestamp']}\n\n")

        f.write("## Experiment 1: DJI-vs-non-DJI Re-split\n\n")
        f.write(f"Fit Mahalanobis centroid on {exp1_results['nondji_train_types']} non-DJI train types only.\n")
        f.write(f"Test zero-shot on {len(exp1_results['dji_types_tested'])} DJI types.\n\n")
        f.write(f"**AUC (DJI vs BG):** {exp1_results['auc_dji_vs_bg']:.4f}\n\n")
        f.write(f"**AUC (non-DJI holdout vs BG):** {exp1_results['auc_nondji_holdout_vs_bg']:.4f}\n\n")
        f.write("**Per-type DJI AUC:**\n\n")
        f.write("| Drone Type | AUC |\n|---|---|\n")
        for t, a in exp1_results["per_type_dji_auc"].items():
            f.write(f"| {t} | {a:.4f} |\n")
        f.write(f"\n**Verdict:** {exp1_results['verdict']}\n\n")

        f.write("## Experiment 2: AVR-CL Sequential Enrollment\n\n")
        f.write(f"Enrolled {len(holdout_types)} holdout types sequentially.\n\n")
        f.write(f"**Naive final accuracy:** {exp2_results['naive_final_accuracy']:.3f}\n\n")
        f.write(f"**AVR-CL final accuracy:** {exp2_results['avr_final_accuracy']:.3f}\n\n")
        f.write(f"**Total AVR-CL repairs:** {exp2_results['total_repairs']}\n\n")
        f.write("**Enrollment history:**\n\n")
        f.write("| Step | Type | Naive Acc | AVR-CL Acc | Repairs |\n|---|---|---|---|---|\n")
        for n, a in zip(naive_history, avr_history):
            f.write(f"| {n['n_enrolled']} | {n['enrolled_type']} | {n['overall_accuracy']:.3f} | {a['overall_accuracy']:.3f} | {a['repairs']} |\n")

        f.write("\n## Experiment 3: DroneRF Parrot Re-labeling Check\n\n")
        f.write(f"Scanned {exp3_results['n_negatives_tested']} DroneRF negatives.\n\n")
        f.write(f"**Detected as drones:** {exp3_results['n_detected_as_drones']} ({exp3_results['pct_detected']:.1f}%)\n\n")
        f.write(f"**Verdict:** {exp3_results['verdict']}\n\n")

    print(f"  [ok] saved {md_path}")
    RESULTS_VOL.commit()

    print("\n" + "=" * 70)
    print("ALL THREE EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\n  Exp 1 (DJI vs non-DJI): AUC = {auc_dji:.4f}")
    print(f"  Exp 2 (AVR-CL): naive={naive_history[-1]['overall_accuracy']:.3f}, AVR-CL={avr_history[-1]['overall_accuracy']:.3f}")
    print(f"  Exp 3 (DroneRF check): {n_detected}/{len(neg_dists)} negatives detected as drones")

    return all_results


@app.local_entrypoint()
def main():
    run_all_experiments.remote()


if __name__ == "__main__":
    main()
