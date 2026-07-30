#!/usr/bin/env python3
"""
IRIS v11 — Bootstrap Confidence Intervals for AUC

Computes 95% CI for:
  - Overall matched BG AUC
  - Per-type matched AUC
  - Per-pair drone-closer rate

Usage:
  modal run bootstrap_ci.py
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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

app = modal.App("iris-bootstrap-ci")

VOL = modal.Volume.from_name("iris-data", create_if_missing=True)
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
        "scipy==1.14.1",
    )
)

H5_REMOTE = "/data/iris_rfuav.h5"
MATCHED_REMOTE = "/matched/iris_matched_bg.h5"
MODEL_REMOTE = "/models/lejepa_v11_best.pt"
N_BOOTSTRAP = 10000


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


@app.function(
    image=IMAGE,
    gpu="A100",
    volumes={
        "/data": VOL,
        "/models": V11_VOL,
        "/matched": MATCHED_VOL,
        "/output": RESULTS_VOL,
    },
    timeout=1800,
    memory=32768,
)
def bootstrap():
    device = "cuda"
    t0 = time.time()

    print("=" * 70)
    print("IRIS v11 — BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)

    # Load model
    V11_VOL.reload()
    ckpt = torch.load(MODEL_REMOTE, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    encoder = CNNEncoder(
        in_ch=cfg["in_ch"], width=cfg["encoder_width"],
        depth=cfg["encoder_depth"], embed_dim=cfg["embed_dim"],
    ).to(device)
    full_state = ckpt["model"]
    encoder_state = {k.replace("encoder.", "", 1): v for k, v in full_state.items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state if encoder_state else full_state)
    encoder.eval()
    print(f"  Loaded epoch {ckpt.get('epoch', -1)} checkpoint")

    # Encode training drones
    print("\n  Encoding training drones...")
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
    print(f"  Train: {train_embs.shape}")

    # Compute centroid + cov_inv
    D = train_embs.shape[1]
    centroid = train_embs.mean(axis=0)
    cov = np.cov(train_embs.T) + 1e-3 * np.eye(D)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    def mahal(embs):
        diff = embs - centroid
        return np.sqrt(np.maximum(np.sum(diff @ cov_inv * diff, axis=1), 0))

    # Encode holdout + matched BG
    print("  Encoding holdout drones...")
    holdout_ds = EvalDataset(H5_REMOTE, "holdout")
    holdout_dl = DataLoader(holdout_ds, batch_size=128, shuffle=False, num_workers=4)
    holdout_embs, holdout_types = [], []
    with torch.no_grad():
        for x, label, tname in holdout_dl:
            z = encoder(x.to(device))
            holdout_embs.append(z.cpu().numpy())
            holdout_types.extend(list(tname))
    holdout_embs = np.concatenate(holdout_embs)
    holdout_types = np.array(holdout_types)
    holdout_dists = mahal(holdout_embs)

    print("  Encoding matched backgrounds...")
    MATCHED_VOL.reload()
    matched_ds = MatchedBGDataset(MATCHED_REMOTE, "holdout")
    matched_dl = DataLoader(matched_ds, batch_size=128, shuffle=False, num_workers=4)
    matched_embs = []
    with torch.no_grad():
        for x, _ in matched_dl:
            z = encoder(x.to(device))
            matched_embs.append(z.cpu().numpy())
    matched_embs = np.concatenate(matched_embs)
    matched_dists = mahal(matched_embs)

    print(f"  Holdout: {len(holdout_dists)}, Matched BG: {len(matched_dists)}")

    # ── Bootstrap ──
    rng = np.random.default_rng(42)
    print(f"\n  Running {N_BOOTSTRAP} bootstrap iterations...")

    # Overall AUC
    overall_aucs = np.zeros(N_BOOTSTRAP)
    # Per-pair: drone closer than matched BG
    n_pairs = min(len(holdout_dists), len(matched_dists))
    pair_rates = np.zeros(N_BOOTSTRAP)
    # Per-type AUCs
    holdout_type_names = sorted(np.unique(holdout_types))
    per_type_boot = {t: np.zeros(N_BOOTSTRAP) for t in holdout_type_names}

    for i in range(N_BOOTSTRAP):
        # Resample with replacement
        d_idx = rng.choice(len(holdout_dists), len(holdout_dists), replace=True)
        m_idx = rng.choice(len(matched_dists), len(matched_dists), replace=True)

        d_boot = holdout_dists[d_idx]
        m_boot = matched_dists[m_idx]

        # Overall AUC
        labels = np.concatenate([np.ones(len(d_boot)), np.zeros(len(m_boot))])
        dists = np.concatenate([-d_boot, -m_boot])
        overall_aucs[i] = roc_auc_score(labels, dists)

        # Per-pair rate
        pair_d = holdout_dists[rng.choice(n_pairs, n_pairs, replace=True)]
        pair_m = matched_dists[rng.choice(n_pairs, n_pairs, replace=True)]
        pair_rates[i] = np.mean(pair_d < pair_m)

        # Per-type AUC
        for tname in holdout_type_names:
            t_mask = holdout_types == tname
            t_dists_all = holdout_dists[t_mask]
            n_t = len(t_dists_all)
            t_idx = rng.choice(n_t, n_t, replace=True)
            t_d_boot = t_dists_all[t_idx]
            m_idx_t = rng.choice(len(matched_dists), len(matched_dists), replace=True)
            t_m_boot = matched_dists[m_idx_t]
            t_labels = np.concatenate([np.ones(len(t_d_boot)), np.zeros(len(t_m_boot))])
            t_dists = np.concatenate([-t_d_boot, -t_m_boot])
            per_type_boot[tname][i] = roc_auc_score(t_labels, t_dists)

        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{N_BOOTSTRAP} done")

    # ── Compute CIs ──
    def ci(arr, alpha=0.05):
        return np.percentile(arr, 100 * alpha / 2), np.percentile(arr, 100 * (1 - alpha / 2))

    overall_ci = ci(overall_aucs)
    pair_ci = ci(pair_rates)

    print(f"\n{'='*70}")
    print(f"RESULTS — {N_BOOTSTRAP} bootstrap samples, 95% CI")
    print(f"{'='*70}")
    print(f"\n  Overall Matched BG AUC: {np.mean(overall_aucs):.4f} "
          f"[{overall_ci[0]:.4f}, {overall_ci[1]:.4f}]")
    print(f"  Per-pair drone-closer rate: {np.mean(pair_rates):.4f} "
          f"[{pair_ci[0]:.4f}, {pair_ci[1]:.4f}]")

    print(f"\n  {'Type':<25} {'AUC':>6} {'95% CI':>20}")
    print(f"  {'-'*25} {'-'*6} {'-'*20}")
    for tname in holdout_type_names:
        t_ci = ci(per_type_boot[tname])
        t_mean = np.mean(per_type_boot[tname])
        print(f"  {tname:<25} {t_mean:.4f} [{t_ci[0]:.4f}, {t_ci[1]:.4f}]")

    # Save
    results = {
        "n_bootstrap": N_BOOTSTRAP,
        "overall_auc": {
            "point_estimate": float(roc_auc_score(
                np.concatenate([np.ones(len(holdout_dists)), np.zeros(len(matched_dists))]),
                np.concatenate([-holdout_dists, -matched_dists])
            )),
            "mean": float(np.mean(overall_aucs)),
            "ci_lower": float(overall_ci[0]),
            "ci_upper": float(overall_ci[1]),
        },
        "per_pair_drone_closer_rate": {
            "point_estimate": float(np.mean(holdout_dists[:n_pairs] < matched_dists[:n_pairs])),
            "mean": float(np.mean(pair_rates)),
            "ci_lower": float(pair_ci[0]),
            "ci_upper": float(pair_ci[1]),
        },
        "per_type": {},
    }
    for tname in holdout_type_names:
        t_ci = ci(per_type_boot[tname])
        results["per_type"][tname] = {
            "mean": float(np.mean(per_type_boot[tname])),
            "ci_lower": float(t_ci[0]),
            "ci_upper": float(t_ci[1]),
        }

    results_path = "/output/bootstrap_ci_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    RESULTS_VOL.commit()

    elapsed = time.time() - t0
    print(f"\n  Saved: {results_path}")
    print(f"  Completed in {elapsed:.0f}s")


@app.local_entrypoint()
def main():
    bootstrap.remote()
