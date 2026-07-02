#!/usr/bin/env python3
"""
Artifact check for supervised baseline:
Can the model distinguish matched BG from random BG?

If AUC ≈ 1.0 → model learned recording artifacts, not pure drone-ness
If AUC ≈ 0.5 → model learned pure drone signal (ideal)

IRIS v11 artifact AUC = 0.9246

Usage:
  modal run scripts/artifact_check_supervised.py
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import modal
from sklearn.metrics import roc_auc_score

app = modal.App("iris-artifact-check")

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
    def __init__(self, cfg):
        super().__init__()
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

        self.classifier = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["hidden_dim"]),
            nn.BatchNorm1d(cfg["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(cfg["hidden_dim"], 1),
        )

    def encode(self, x):
        return self.encoder_head(self.conv(x))

    def forward(self, x):
        z = self.encode(x)
        logit = self.classifier(z)
        return z, logit


CFG = dict(in_ch=2, encoder_depth=6, encoder_width=64, embed_dim=256, hidden_dim=128)


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


def normalize_per_channel(x):
    for c in range(x.shape[0]):
        ch = x[c]
        ch_std = ch.std()
        if ch_std > 1e-6:
            x[c] = (ch - ch.mean()) / ch_std
        else:
            x[c] = ch - ch.mean()
    return x


def prepare_tensor(sample):
    if sample.shape[0] == 3:
        x = torch.from_numpy(sample[:2].copy()).float()
    elif sample.shape[0] == 2:
        x = torch.from_numpy(sample.copy()).float()
    else:
        x = torch.from_numpy(sample[:2].copy()).float()
    return x


@app.function(
    image=IMAGE,
    gpu="A100",
    volumes={"/data": VOL, "/models": MODEL_VOL, "/matched": MATCHED_VOL},
    timeout=1800,
    memory=32768,
)
def artifact_check():
    device = "cuda"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load best supervised model
    model = SupervisedBinaryClassifier(CFG).to(device)
    ckpt = torch.load("/models/baseline_supervised_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded best model from epoch {ckpt['epoch']}, matched AUC={ckpt['matched_auc']:.4f}")

    # ── Encode train drones for centroid ──
    train_f = h5py.File(H5_REMOTE, "r")
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

    train_embs = []
    for label_idx, tname in enumerate(type_names):
        ds_or_grp, n_samples, is_multi = _resolved[tname]
        n_use = min(n_samples, 500)
        for i in range(n_use):
            if is_multi:
                sub_key = _sub_keys[tname][i]
                sample = ds_or_grp[sub_key][:]
            else:
                sample = ds_or_grp[i]
            x = prepare_tensor(sample)
            x = normalize_per_channel(x)
            with torch.no_grad():
                z = model.encode(x.unsqueeze(0).to(device))
            train_embs.append(z.cpu().numpy().squeeze())

    train_embs = np.array(train_embs)
    print(f"Train embeddings: {train_embs.shape}")

    # ── Centroid + covariance ──
    D = train_embs.shape[1]
    centroid = train_embs.mean(axis=0)
    cov = np.cov(train_embs.T) + 1e-3 * np.eye(D)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    # ── Encode matched BG (holdout) ──
    matched_f = h5py.File(MATCHED_REMOTE, "r")
    mbg_key = "holdout_matched_bg"
    if mbg_key not in matched_f:
        mbg_key = "train_matched_bg"
    mbg_grp = matched_f[mbg_key]
    mbg_keys = sorted(list(mbg_grp.keys()), key=lambda x: int(x) if x.isdigit() else 0)
    n_mbg = min(len(mbg_keys), 2000)

    mbg_embs = []
    rng = np.random.default_rng(123)
    mbg_indices = rng.choice(len(mbg_keys), n_mbg, replace=False)
    for j in mbg_indices:
        key = mbg_keys[j]
        sample = mbg_grp[key][:]
        x = prepare_tensor(sample)
        x = normalize_per_channel(x)
        with torch.no_grad():
            z = model.encode(x.unsqueeze(0).to(device))
        mbg_embs.append(z.cpu().numpy().squeeze())

    mbg_embs = np.array(mbg_embs)
    print(f"Matched BG embeddings: {mbg_embs.shape}")

    # ── Encode random BG ──
    holdout_f = h5py.File(H5_REMOTE, "r")
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
            x = prepare_tensor(sample)
            x = normalize_per_channel(x)
            with torch.no_grad():
                z = model.encode(x.unsqueeze(0).to(device))
            rand_bg_embs.append(z.cpu().numpy().squeeze())
        rand_bg_embs = np.array(rand_bg_embs)
        print(f"Random BG embeddings: {rand_bg_embs.shape}")
    else:
        print("ERROR: No negatives found in holdout data!")
        return

    # ── Compute Mahalanobis distances ──
    mbg_diff = mbg_embs - centroid
    mbg_mahal = np.sqrt(np.maximum(np.sum(mbg_diff @ cov_inv * mbg_diff, axis=1), 0))

    rand_bg_diff = rand_bg_embs - centroid
    rand_bg_mahal = np.sqrt(np.maximum(np.sum(rand_bg_diff @ cov_inv * rand_bg_diff, axis=1), 0))

    # ── ARTIFACT AUC: matched BG vs random BG ──
    # If model can tell matched BG from random BG, it learned recording artifacts
    # Label: matched BG = 1, random BG = 0
    # Score: distance to centroid (matched BG should be closer if artifacts leak)
    artifact_labels = np.concatenate([np.ones(len(mbg_embs)), np.zeros(len(rand_bg_embs))])
    # Use negative distance (closer to centroid = more "drone-like")
    artifact_scores = np.concatenate([-mbg_mahal, -rand_bg_mahal])
    artifact_auc = roc_auc_score(artifact_labels, artifact_scores)

    # Also try the other direction (use raw Mahalanobis distance as score)
    artifact_auc_rev = roc_auc_score(artifact_labels, np.concatenate([mbg_mahal, rand_bg_mahal]))

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"ARTIFACT CHECK — Supervised Baseline")
    print(f"{'='*70}")
    print(f"  Matched BG mean dist to centroid: {mbg_mahal.mean():.2f}")
    print(f"  Random BG mean dist to centroid:  {rand_bg_mahal.mean():.2f}")
    print(f"  Matched BG std dist:  {mbg_mahal.std():.2f}")
    print(f"  Random BG std dist:   {rand_bg_mahal.std():.2f}")
    print(f"")
    print(f"  Artifact AUC (matched BG closer = 1): {artifact_auc:.4f}")
    print(f"  Artifact AUC (reversed):              {artifact_auc_rev:.4f}")
    print(f"")
    print(f"  Interpretation:")
    print(f"    AUC ≈ 0.5 → model learned pure drone signal (no artifact leakage)")
    print(f"    AUC ≈ 1.0 → model learned recording artifacts (matched BG closer)")
    print(f"")
    print(f"  IRIS v11 artifact AUC: 0.9246")
    print(f"  Supervised artifact AUC: {artifact_auc:.4f}")
    print(f"{'='*70}")

    train_f.close()
    holdout_f.close()
    matched_f.close()


@app.local_entrypoint()
def main():
    artifact_check.remote()


import os
