#!/usr/bin/env python3
"""
IRIS v10 — LeJEPA + SIGReg + SupCon with MATCHED Backgrounds

v9 just got exposed:
  - Original eval (50K random bg):    AUC = 1.0000
  - Matched bg eval (signal removed): AUC = 0.3002

The model was detecting recording conditions (noise floor, compression,
capture setup), NOT actual drone RF signatures.

v10 fix: Train with MATCHED backgrounds as the negative class.
  - Pre-computed matched backgrounds from eval script (iris_matched_bg.h5)
  - Same recording, same noise floor, drone signal removed
  - Forces model to learn ACTUAL drone RF signatures
  - If v10 achieves AUC 0.95+ on matched backgrounds, the result is airtight

SPEED: Uses pre-computed matched backgrounds → same speed as v9 (~10 min/epoch)
       NOT on-the-fly generation (which would be 40-60 min/epoch)

Usage:
  modal run scripts/train_modal_v10.py
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
from sklearn.metrics import roc_auc_score

# ─── Modal setup ──────────────────────────────────────────────────────────────

app = modal.App("iris-v10-lejepa-matchedbg")

VOL = modal.Volume.from_name("iris-data", create_if_missing=True)
MODEL_VOL = modal.Volume.from_name("iris-models-v10", create_if_missing=True)
MATCHED_VOL = modal.Volume.from_name("iris-matched-bg", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install("libgl1", "libglib2.0-0", "libhdf5-dev",
                 "python3", "python3-pip", "python-is-python3")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        "h5py==3.12.1", "numpy==1.26.4", "scikit-learn==1.6.1",
        "scipy==1.14.1", "umap-learn==0.5.7",
    )
)

H5_REMOTE = "/data/iris_rfuav.h5"
MATCHED_REMOTE = "/matched/iris_matched_bg.h5"

# ─── Hyperparameters ──────────────────────────────────────────────────────────

CFG = dict(
    # Architecture (same as v7/v8/v9)
    in_ch=2,
    encoder_depth=6,
    encoder_width=64,
    embed_dim=256,
    proj_dim=256,
    pred_dim=256,
    pred_out=256,

    # Loss weights — increase SupCon slightly for harder negatives
    lam_sig=1e-3,
    alpha_supcon=0.05,      # v9 was 0.03, bump up for harder negatives
    supcon_temperature=0.07,

    # SIGReg
    sigreg_k=256,

    # Optimizer
    lr=1e-3,
    weight_decay=0.0,
    warmup_steps=10000,
    batch_size=128,
    grad_accum_steps=2,

    # Training
    epochs=50,
    eval_every=1,
    early_stop_patience=7,  # slightly longer — harder task

    # Data
    img_size=256,
    num_workers=4,
)


# ─── SIGReg Loss ──────────────────────────────────────────────────────────────

class SIGRegLoss(nn.Module):
    def __init__(self, embed_dim: int, k: int = 256, seed: int = 42):
        super().__init__()
        self.embed_dim = embed_dim
        self.k = k
        generator = torch.Generator().manual_seed(seed)
        W = torch.randn(k, embed_dim, generator=generator)
        W = W / W.norm(dim=1, keepdim=True)
        self.register_buffer("W", W)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        projections = F.linear(z, self.W)
        var_per_proj = projections.var(dim=0)
        loss = ((var_per_proj - 1.0) ** 2).mean()
        return loss


# ─── Supervised Contrastive Loss ──────────────────────────────────────────────

def supcon_loss(embeddings, labels, temperature=0.07):
    device = embeddings.device
    B = embeddings.shape[0]

    norms = F.normalize(embeddings, dim=1)
    sim = torch.mm(norms, norms.t()) / temperature
    sim = sim.clamp(-10.0, 10.0)

    label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)

    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim_stable = sim - sim_max.detach()
    exp_sim = torch.exp(sim_stable)

    diag_mask = ~torch.eye(B, dtype=torch.bool, device=device)
    denom = (exp_sim * diag_mask.float()).sum(dim=1, keepdim=True)
    pos_mask = label_mask & diag_mask
    num_positives = pos_mask.sum(dim=1)
    valid = num_positives > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    numer = (exp_sim * pos_mask.float()).sum(dim=1, keepdim=True)
    log_prob = torch.log(numer + 1e-8) - torch.log(denom + 1e-8)
    mean_log_prob = (log_prob * pos_mask.float()).sum(dim=1) / (num_positives.float() + 1e-8)
    loss = -mean_log_prob[valid].mean()
    return loss


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

        scale = torch.empty(1).uniform_(self.crop_range[0], self.crop_range[1]).item()
        new_h, new_w = int(H * scale), int(W * scale)
        top = torch.randint(0, H - new_h + 1, (1,)).item()
        left = torch.randint(0, W - new_w + 1, (1,)).item()
        x_aug = x[:, top:top+new_h, left:left+new_w]
        x_aug = F.interpolate(x_aug.unsqueeze(0), size=(H, W), mode='bilinear',
                              align_corners=False).squeeze(0)

        freq_mask_size = int(H * self.freq_mask_ratio)
        if freq_mask_size > 0:
            f_start = torch.randint(0, H - freq_mask_size + 1, (1,)).item()
            x_aug[:, f_start:f_start+freq_mask_size, :] = 0

        time_mask_size = int(W * self.time_mask_ratio)
        if time_mask_size > 0:
            t_start = torch.randint(0, W - time_mask_size + 1, (1,)).item()
            x_aug[:, :, t_start:t_start+time_mask_size] = 0

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


class IRISDroneMatchedBGDataset(Dataset):
    """
    v10 Training dataset: Drones + MATCHED backgrounds.

    Loads PRE-COMPUTED matched backgrounds from iris_matched_bg.h5.
    This makes training just as fast as v9 (~10 min/epoch).

    - Drone samples: per-channel normalized, augmented, type labels 0..29
    - Matched BG samples: loaded from pre-computed HDF5, per-channel normalized,
      augmented, all share label 30
    - Balanced: 1:1 drone to matched-bg ratio
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
        self.bg_label = self.n_drone_types  # label 30
        self.type_to_label = {t: i for i, t in enumerate(self.type_names)}

        # ── Load matched backgrounds ──
        self.matched_f = h5py.File(matched_path, "r")
        matched_key = f"{split_key}_matched_bg"

        if matched_key not in self.matched_f:
            raise ValueError(f"No matched backgrounds found at '{matched_key}' in {matched_path}. "
                           f"Run eval_v9_matched_bg.py first to generate them.")

        self.matched_grp = self.matched_f[matched_key]
        self.matched_keys = sorted(list(self.matched_grp.keys()),
                                    key=lambda x: int(x) if x.isdigit() else 0)
        n_matched = len(self.matched_keys)

        print(f"  IRISDroneMatchedBGDataset: {self.n_drone_types} drone types + matched bg")
        print(f"    Drones: {n_drones}")
        print(f"    Matched BG (pre-computed): {n_matched}")

        # Combined index: alternate drone and matched bg
        # Use minimum of both to keep balanced
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
            label = self.type_to_label[tname]
        else:  # matched_bg
            key = self.matched_keys[local_idx]
            sample = self.matched_grp[key][:]
            x = self._prepare_tensor(sample)
            label = self.bg_label

        # Per-channel normalization
        x = self._normalize_per_channel(x)

        if self.augment:
            view1 = self.augment(x)
            view2 = self.augment(x)
        else:
            view1 = view2 = x

        return view1, view2, label


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


class Projector(nn.Module):
    def __init__(self, embed_dim=256, proj_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, proj_dim=256, pred_dim=256, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.GELU(),
            nn.Linear(pred_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class LeJEPASupConV10(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = CNNEncoder(
            in_ch=cfg["in_ch"],
            width=cfg["encoder_width"],
            depth=cfg["encoder_depth"],
            embed_dim=cfg["embed_dim"],
        )
        self.projector = Projector(cfg["embed_dim"], cfg["proj_dim"])
        self.predictor = Predictor(cfg["proj_dim"], cfg["pred_dim"], cfg["pred_out"])
        self.sigreg = SIGRegLoss(cfg["embed_dim"], k=cfg["sigreg_k"])

    def encode_project(self, x):
        z = self.encoder(x)
        p = self.projector(z)
        return z, p

    def forward(self, x1, x2):
        z1, p1 = self.encode_project(x1)
        z2, p2 = self.encode_project(x2)
        q1 = self.predictor(p1)
        q2 = self.predictor(p2)
        return z1, z2, p1, p2, q1, q2

    def compute_loss(self, z1, z2, p1, p2, q1, q2, labels,
                     lam_sig=1e-3, alpha_supcon=0.05, supcon_temperature=0.07):
        align_loss = (F.mse_loss(q1, p2.detach()) + F.mse_loss(q2, p1.detach())) / 2.0
        sig_loss = (self.sigreg(z1) + self.sigreg(z2)) / 2.0

        z_all = torch.cat([z1, z2], dim=0)
        labels_all = torch.cat([labels, labels], dim=0)
        sc_loss = supcon_loss(z_all, labels_all, temperature=supcon_temperature)

        lejepa_loss = lam_sig * sig_loss + (1 - lam_sig) * align_loss
        total_loss = lejepa_loss + alpha_supcon * sc_loss

        return total_loss, {
            "align_loss": align_loss.item(),
            "sig_loss": sig_loss.item(),
            "supcon_loss": sc_loss.item(),
            "lejepa_loss": lejepa_loss.item(),
            "total_loss": total_loss.item(),
        }


# ─── LR Schedule ──────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Collapse Detection ───────────────────────────────────────────────────────

def check_collapse(z: torch.Tensor, threshold_std=1e-3, threshold_eff_dim=2):
    with torch.no_grad():
        std_per_dim = z.std(dim=0)
        mean_std = std_per_dim.mean().item()
        min_std = std_per_dim.min().item()
        if std_per_dim.max() > 0:
            eff_dim = (std_per_dim > 0.01 * std_per_dim.max()).sum().item()
        else:
            eff_dim = 0
        z_norm = F.normalize(z, dim=1)
        n = min(100, z.shape[0])
        idx = torch.randperm(z.shape[0])[:n]
        cos_sim = torch.mm(z_norm[idx], z_norm[idx].T)
        mean_cos = (cos_sim.sum() - n) / (n * n - n)
        collapsed = mean_std < threshold_std or eff_dim < threshold_eff_dim
    return {"mean_std": mean_std, "min_std": min_std, "eff_dim": eff_dim,
            "mean_cos_sim": mean_cos.item(), "collapsed": collapsed}


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_with_detection(encoder, h5_path, matched_path, device="cuda"):
    """
    v10 evaluation: Test on BOTH random backgrounds AND matched backgrounds.

    The matched background AUC is the REAL metric. Random bg AUC is a
    sanity check — it should be high but we don't care much.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import silhouette_score, roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    encoder.eval()

    class NormEvalDS(Dataset):
        def __init__(self, h5_path, split_key, include_negatives=False,
                     max_negatives=2000, include_matched_bg=False,
                     max_matched_bg=2000, matched_path=None):
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
                    self.samples.append((split_key, tname, i, "drone"))
                    self.labels.append(label_idx)
                    self.type_names_list.append(tname)

            self.n_drone_types = len(type_names)
            self.type_names = type_names

            # Random negatives
            if include_negatives and "negatives" in self.f:
                neg_item = self.f["negatives"]
                if isinstance(neg_item, h5py.Dataset):
                    n_total_neg = neg_item.shape[0]
                    n_neg = min(n_total_neg, max_negatives)
                    self._resolved[("negatives", None)] = (neg_item, n_total_neg, False)
                    self._neg_is_multi = False
                else:
                    sub_keys = [sk for sk in neg_item.keys()
                                if isinstance(neg_item[sk], h5py.Dataset)
                                and len(neg_item[sk].shape) == 3]
                    try:
                        sub_keys.sort(key=lambda x: int(x))
                    except ValueError:
                        sub_keys.sort()
                    n_total_neg = len(sub_keys)
                    n_neg = min(n_total_neg, max_negatives)
                    self._resolved[("negatives", None)] = (neg_item, n_total_neg, True)
                    self._sub_keys[("negatives", None)] = sub_keys
                    self._neg_is_multi = True

                rng = np.random.default_rng(42)
                self._bg_indices = rng.choice(n_total_neg, n_neg, replace=False).tolist()

                for i in self._bg_indices:
                    self.samples.append(("negatives", None, i, "random_bg"))
                    self.labels.append(self.n_drone_types)
                    self.type_names_list.append("background")

            # Matched backgrounds
            self._matched_data = None
            self._matched_f = None
            if include_matched_bg and matched_path and os.path.exists(matched_path):
                self._matched_f = h5py.File(matched_path, "r")
                mbg_key = f"{split_key}_matched_bg"
                if mbg_key in self._matched_f:
                    mbg_grp = self._matched_f[mbg_key]
                    mbg_keys = sorted(list(mbg_grp.keys()),
                                      key=lambda x: int(x) if x.isdigit() else 0)
                    n_mbg = min(len(mbg_keys), max_matched_bg)
                    rng2 = np.random.default_rng(123)
                    mbg_indices = rng2.choice(len(mbg_keys), n_mbg, replace=False)
                    self._matched_grp = mbg_grp
                    self._matched_keys = mbg_keys
                    self._matched_indices = mbg_indices

                    for j in mbg_indices:
                        self.samples.append(("matched_bg", None, j, "matched_bg"))
                        self.labels.append(self.n_drone_types)
                        self.type_names_list.append("matched_bg")
                    print(f"    Loaded {n_mbg} matched backgrounds")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            split_key, tname, local_idx, source = self.samples[idx]

            if source == "matched_bg":
                key = self._matched_keys[local_idx]
                sample = self._matched_grp[key][:]
            elif source == "random_bg":
                if self._neg_is_multi:
                    sub_key = self._sub_keys[("negatives", None)][local_idx]
                    sample = self._resolved[("negatives", None)][0][sub_key][:]
                else:
                    sample = self._resolved[("negatives", None)][0][local_idx]
            else:
                ds_or_grp, _, is_multi = self._resolved[(split_key, tname)]
                if is_multi:
                    sub_key = self._sub_keys[(split_key, tname)][local_idx]
                    sample = ds_or_grp[sub_key][:]
                else:
                    sample = ds_or_grp[local_idx]

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

            return x, self.labels[idx], self.type_names_list[idx]

    results = {}

    # Encode train drones
    train_ds = NormEvalDS(h5_path, "train", include_negatives=False)
    if len(train_ds) > 5000:
        indices = np.random.default_rng(42).choice(len(train_ds), 5000, replace=False)
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=False,
                              sampler=torch.utils.data.SubsetRandomSampler(indices))
    else:
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=2)

    train_embs, train_labels, train_types = [], [], []
    with torch.no_grad():
        for x, label, tname in train_dl:
            z = encoder(x.to(device))
            train_embs.append(z.cpu().numpy())
            train_labels.extend(label.tolist() if hasattr(label, 'tolist') else list(label))
            train_types.extend(list(tname))
    train_embs = np.concatenate(train_embs)
    train_labels = np.array(train_labels)
    train_types = np.array(train_types)
    n_types = train_ds.n_drone_types

    if len(np.unique(train_labels)) > 1 and len(train_embs) > 10:
        knn = KNeighborsClassifier(n_neighbors=10, metric="cosine")
        cv_scores = cross_val_score(knn, train_embs, train_labels, cv=3, scoring="accuracy")
        results["per_type_knn_cv"] = float(cv_scores.mean())
        sil = silhouette_score(train_embs, train_labels, metric="cosine")
        results["silhouette_drone"] = float(sil)
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        probe_scores = cross_val_score(lr, train_embs, train_labels, cv=5, scoring="accuracy")
        results["linear_probe_cv"] = float(probe_scores.mean())

    # Encode holdout + backgrounds
    test_ds = NormEvalDS(h5_path, "holdout", include_negatives=True,
                          max_negatives=2000, include_matched_bg=True,
                          max_matched_bg=2000, matched_path=matched_path)
    if len(test_ds) > 5000:
        indices = np.random.default_rng(42).choice(len(test_ds), 5000, replace=False)
        test_dl = DataLoader(test_ds, batch_size=64, shuffle=False,
                             sampler=torch.utils.data.SubsetRandomSampler(indices))
    else:
        test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    test_embs, test_labels, test_types = [], [], []
    with torch.no_grad():
        for x, label, tname in test_dl:
            z = encoder(x.to(device))
            test_embs.append(z.cpu().numpy())
            test_labels.extend(label.tolist() if hasattr(label, 'tolist') else list(label))
            test_types.extend(list(tname))
    test_embs = np.concatenate(test_embs)
    test_labels = np.array(test_labels)
    test_types = np.array(test_types)

    # Global Mahalanobis
    D = train_embs.shape[1]
    centroid = train_embs.mean(axis=0)
    cov = np.cov(train_embs.T) + 1e-3 * np.eye(D)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    diff = test_embs - centroid
    mahal_sq = np.sum(diff @ cov_inv * diff, axis=1)
    mahal = np.sqrt(np.maximum(mahal_sq, 0))

    drone_mask = ~np.isin(test_types, ["background", "matched_bg"])
    random_bg_mask = test_types == "background"
    matched_bg_mask = test_types == "matched_bg"

    # AUC vs random backgrounds
    if drone_mask.sum() > 0 and random_bg_mask.sum() > 0:
        drone_dists = mahal[drone_mask]
        bg_dists = mahal[random_bg_mask]
        all_d = np.concatenate([drone_dists, bg_dists])
        all_l = np.concatenate([np.ones(len(drone_dists)), np.zeros(len(bg_dists))])
        results["random_bg_auc"] = float(roc_auc_score(all_l, -all_d))
        results["random_bg_ratio"] = float(bg_dists.mean() / drone_dists.mean())

    # *** THE KEY METRIC: AUC vs MATCHED backgrounds ***
    if drone_mask.sum() > 0 and matched_bg_mask.sum() > 0:
        drone_dists = mahal[drone_mask]
        mbg_dists = mahal[matched_bg_mask]
        all_d = np.concatenate([drone_dists, mbg_dists])
        all_l = np.concatenate([np.ones(len(drone_dists)), np.zeros(len(mbg_dists))])
        results["matched_bg_auc"] = float(roc_auc_score(all_l, -all_d))
        results["matched_bg_ratio"] = float(mbg_dists.mean() / drone_dists.mean())
        results["matched_drone_mean"] = float(drone_dists.mean())
        results["matched_bg_mean"] = float(mbg_dists.mean())

    # Per-type Mahalanobis for reference
    if drone_mask.sum() > 0 and random_bg_mask.sum() > 0:
        centroids = np.zeros((n_types, D))
        cov_inv_list = []
        for k in range(n_types):
            mask = train_labels == k
            cluster = train_embs[mask]
            centroids[k] = cluster.mean(axis=0)
            cov_k = np.cov(cluster.T) + 1e-3 * np.eye(D)
            try:
                cov_inv_list.append(np.linalg.inv(cov_k))
            except np.linalg.LinAlgError:
                cov_inv_list.append(np.linalg.pinv(cov_k))

        min_mahal = np.full(len(test_embs), np.inf)
        for k in range(n_types):
            diff_k = test_embs - centroids[k]
            mahal_sq_k = np.sum(diff_k @ cov_inv_list[k] * diff_k, axis=1)
            min_mahal = np.minimum(min_mahal, np.sqrt(np.maximum(mahal_sq_k, 0)))

        pt_drone = min_mahal[drone_mask]
        pt_bg = min_mahal[random_bg_mask]
        pt_all = np.concatenate([pt_drone, pt_bg])
        pt_labels = np.concatenate([np.ones(len(pt_drone)), np.zeros(len(pt_bg))])
        results["per_type_mahalanobis_auc"] = float(roc_auc_score(pt_labels, -pt_all))

    # Binary accuracy
    if "background" in test_types or "matched_bg" in test_types:
        y_binary = (~np.isin(test_types, ["background", "matched_bg"])).astype(int)
        if len(np.unique(y_binary)) > 1:
            lr = LogisticRegression(max_iter=500, solver="lbfgs")
            bin_scores = cross_val_score(lr, test_embs, y_binary, cv=3, scoring="accuracy")
            results["binary_acc_cv"] = float(bin_scores.mean())

    encoder.train()
    return results


# ─── Training Loop ────────────────────────────────────────────────────────────

@app.function(
    image=IMAGE,
    gpu="A100",
    volumes={"/data": VOL, "/models": MODEL_VOL, "/matched": MATCHED_VOL},
    timeout=5400,  # 90 min — same speed as v9 now
    memory=32768,
)
def train():
    cfg = CFG.copy()
    device = "cuda"

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    model = LeJEPASupConV10(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    augment = SpectrogramAugment(
        img_size=cfg["img_size"],
        freq_mask_ratio=0.08,
        time_mask_ratio=0.08,
        noise_std=0.03,
        crop_range=(0.85, 1.0),
    )

    train_ds = IRISDroneMatchedBGDataset(
        H5_REMOTE, MATCHED_REMOTE, split_key="train",
        augment=augment,
    )

    steps_per_epoch = math.ceil(len(train_ds) / cfg["batch_size"])
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup_steps = min(cfg["warmup_steps"], total_steps // 4)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps
    )

    print(f"\n{'='*70}")
    print(f"IRIS v10 — MATCHED BACKGROUND TRAINING (PRE-COMPUTED)")
    print(f"{'='*70}")
    print(f"Dataset: {len(train_ds)} samples "
          f"({train_ds.n_drone_types} drone types + matched bg)")
    print(f"Normalization: True (per-channel)")
    print(f"Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"λ_sig={cfg['lam_sig']}, α_supcon={cfg['alpha_supcon']}, T={cfg['supcon_temperature']}")
    print(f"BG label: {train_ds.bg_label} (MATCHED backgrounds)")
    print(f"Matched BG source: {MATCHED_REMOTE}")
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

    global_step = 0
    best_combined = -1.0
    best_epoch = -1
    best_matched_auc = 0.0
    grad_accum = cfg.get("grad_accum_steps", 1)
    patience_counter = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        optimizer.zero_grad()
        epoch_losses = []
        epoch_metrics = {"align_loss": [], "sig_loss": [], "supcon_loss": []}

        for batch_idx, (view1, view2, label) in enumerate(dl):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            z1, z2, p1, p2, q1, q2 = model(view1, view2)

            loss, metrics = model.compute_loss(
                z1, z2, p1, p2, q1, q2, label,
                lam_sig=cfg["lam_sig"],
                alpha_supcon=cfg["alpha_supcon"],
                supcon_temperature=cfg["supcon_temperature"],
            )
            loss = loss / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_losses.append(loss.item() * grad_accum)
            for k in ["align_loss", "sig_loss", "supcon_loss"]:
                epoch_metrics[k].append(metrics[k])

            if (batch_idx + 1) % (50 * grad_accum) == 0:
                collapse_info = check_collapse(z1)
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  step {global_step} | total={metrics['total_loss']:.4f} "
                    f"align={metrics['align_loss']:.4f} sig={metrics['sig_loss']:.4f} "
                    f"supcon={metrics['supcon_loss']:.4f} "
                    f"eff_dim={collapse_info['eff_dim']} std={collapse_info['mean_std']:.4f} "
                    f"cos={collapse_info['mean_cos_sim']:.3f} lr={lr_now:.6f}"
                )

        if len(epoch_losses) == 0:
            continue

        avg_loss = np.mean(epoch_losses)
        avg_align = np.mean(epoch_metrics["align_loss"])
        avg_sig = np.mean(epoch_metrics["sig_loss"])
        avg_supcon = np.mean(epoch_metrics["supcon_loss"])

        with torch.no_grad():
            z_check = model.encoder(view1)
            collapse_info = check_collapse(z_check)

        print(f"\n{'='*70}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"  Total: {avg_loss:.4f}  Align: {avg_align:.4f}  "
              f"SIGReg: {avg_sig:.4f}  SupCon: {avg_supcon:.4f}")
        print(f"  Eff dim: {collapse_info['eff_dim']}/{cfg['embed_dim']}  "
              f"Mean std: {collapse_info['mean_std']:.4f}  "
              f"Mean cos: {collapse_info['mean_cos_sim']:.3f}")
        print(f"{'='*70}\n")

        torch.cuda.empty_cache()

        # Save checkpoint every epoch
        ckpt_path = f"/models/lejepa_v10_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "cfg": cfg,
            "collapse_info": collapse_info,
        }, ckpt_path)
        MODEL_VOL.commit()

        if collapse_info["collapsed"]:
            print(f"Training stopped — representation collapsed at epoch {epoch}")
            break

        # Evaluate every epoch
        if (epoch + 1) % cfg["eval_every"] == 0:
            print("Running evaluation (random bg + matched bg)...")
            eval_results = evaluate_with_detection(
                model.encoder, H5_REMOTE, MATCHED_REMOTE, device=device)

            sil = eval_results.get("silhouette_drone", 0)
            random_auc = eval_results.get("random_bg_auc", 0)
            matched_auc = eval_results.get("matched_bg_auc", 0)
            pt_auc = eval_results.get("per_type_mahalanobis_auc", 0)
            knn = eval_results.get("per_type_knn_cv", 0)

            print(f"  Per-type k-NN CV: {knn:.3f}")
            print(f"  Silhouette (drone): {sil:.3f}")
            print(f"  Random BG AUC: {random_auc:.3f}")
            print(f"  *** MATCHED BG AUC: {matched_auc:.3f} ***")
            if matched_auc > 0:
                print(f"  *** Matched BG/Drone ratio: {eval_results.get('matched_bg_ratio', 0):.3f} ***")
                print(f"  *** Drone mean dist: {eval_results.get('matched_drone_mean', 0):.2f}, "
                      f"Matched BG mean dist: {eval_results.get('matched_bg_mean', 0):.2f} ***")

            # Combined metric: weight matched_auc 2x (the primary metric)
            combined = sil + 2.0 * matched_auc + random_auc * 0.5

            if matched_auc > best_matched_auc:
                best_matched_auc = matched_auc

            if combined > best_combined:
                best_combined = combined
                best_epoch = epoch
                patience_counter = 0
                best_path = "/models/lejepa_v10_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "eval_results": eval_results,
                }, best_path)
                MODEL_VOL.commit()
                print(f"  ★ New best! Combined={combined:.4f} "
                      f"(Sil={sil:.3f}, Random={random_auc:.3f}, "
                      f"Matched={matched_auc:.3f})")
            else:
                patience_counter += 1
                print(f"  Combined: {combined:.4f} (best: {best_combined:.4f}), "
                      f"patience: {patience_counter}/{cfg['early_stop_patience']}")

            eval_path = f"/models/lejepa_v10_epoch{epoch}_eval.json"
            with open(eval_path, "w") as f:
                json.dump(eval_results, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
            MODEL_VOL.commit()

            # Early stopping
            if patience_counter >= cfg["early_stop_patience"]:
                print(f"\n⛔ Early stopping at epoch {epoch}")
                print(f"  Best epoch: {best_epoch}")
                print(f"  Best matched BG AUC: {best_matched_auc:.4f}")
                break

    print(f"\nTraining complete. Best epoch: {best_epoch}")
    print(f"  Best matched BG AUC: {best_matched_auc:.4f}")
    print(f"  Best combined: {best_combined:.4f}")


@app.local_entrypoint()
def main():
    train.remote()