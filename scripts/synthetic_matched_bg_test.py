#!/usr/bin/env python3
"""
IRIS — Synthetic Matched BG Test (Surgery Artifact Check)

The question: Does the model detect the drone signal, or does it detect
the "surgery" performed to create matched backgrounds?

How it works:
  1. Take random BG spectrograms (no drone was ever present)
  2. Apply the EXACT same surgical operation used to create matched BGs:
     - Estimate per-freq-bin noise floor (10th percentile)
     - Identify "signal pixels" (above floor + 2σ)
     - Replace those pixels with noise samples
  3. Even though there's no real signal, the surgery will find and replace
     random noise spikes
  4. Feed both original random BG and "surgically modified" random BG to the model

Interpretation:
  - If model says "random BG" for both → it detects actual drone signal (CLEAN)
  - If model says "matched BG / drone-like" for the surgically modified ones →
    it's detecting the surgery artifact (CONTAMINATED)

Run on IRIS v11 AND supervised baseline to compare.

Usage:
  modal run scripts/synthetic_matched_bg_test.py
"""

import h5py
import json
import os
import numpy as np
import torch
import torch.nn as nn
import modal
from sklearn.metrics import roc_auc_score

app = modal.App("iris-synthetic-matched-bg")

VOL = modal.Volume.from_name("iris-data", create_if_missing=True)
MODEL_VOL_V11 = modal.Volume.from_name("iris-models-v11", create_if_missing=True)
MODEL_VOL_BASELINE = modal.Volume.from_name("iris-models-baseline", create_if_missing=True)
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


# ─── Model architectures ─────────────────────────────────────────────────────

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


class LeJEPASupConV11(nn.Module):
    """v11 encoder (only need encoder for embeddings)."""
    def __init__(self, cfg):
        super().__init__()
        self.encoder = CNNEncoder(
            in_ch=cfg["in_ch"], width=cfg["encoder_width"],
            depth=cfg["encoder_depth"], embed_dim=cfg["embed_dim"],
        )


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


# ─── HDF5 helpers ────────────────────────────────────────────────────────────

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


# ─── Matched BG generation (EXACT same as eval_v9_matched_bg.py) ─────────────

def generate_matched_background(spectrogram, noise_percentile=10,
                                 signal_sigma=2.0, method="replace"):
    """
    Generate a matched background from a spectrogram.
    EXACT copy from eval_v9_matched_bg.py lines 164-248.
    """
    if spectrogram.ndim == 2:
        spectrogram = spectrogram[np.newaxis, :, :]

    C, H, W = spectrogram.shape
    matched = spectrogram.copy()

    for c in range(C):
        ch = spectrogram[c]

        for freq_idx in range(H):
            row = ch[freq_idx]

            # Estimate noise floor for this frequency bin
            noise_floor = np.percentile(row, noise_percentile)

            # Estimate noise std from pixels near the noise floor
            below_mask = row <= noise_floor + np.std(row[row <= noise_floor + np.std(row)]) * 1.5
            if below_mask.sum() < 5:
                below_mask = row <= np.percentile(row, 50)

            noise_pixels = row[below_mask]
            if len(noise_pixels) > 1:
                noise_std = np.std(noise_pixels)
            else:
                noise_std = np.std(row) * 0.5

            # Identify signal pixels
            threshold = noise_floor + signal_sigma * noise_std
            signal_mask = row > threshold

            if signal_mask.sum() > 0:
                if method == "replace":
                    n_replace = signal_mask.sum()
                    if len(noise_pixels) > 1:
                        replacements = np.random.choice(noise_pixels, size=n_replace, replace=True)
                        replacements = replacements + np.random.normal(0, noise_std * 0.05, n_replace)
                    else:
                        replacements = np.random.normal(noise_floor, noise_std, n_replace)
                    matched[c, freq_idx, signal_mask] = replacements
                elif method == "zero":
                    matched[c, freq_idx, signal_mask] = noise_floor

    return matched


# ─── Helper functions ─────────────────────────────────────────────────────────

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


def encode_samples(samples, encoder_fn, device, batch_size=64):
    """Encode a list of numpy arrays into embeddings."""
    embs = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        batch_t = torch.stack(batch).to(device)
        with torch.no_grad():
            z = encoder_fn(batch_t)
        embs.append(z.cpu().numpy())
    return np.concatenate(embs)


@app.function(
    image=IMAGE,
    gpu="A100",
    volumes={"/data": VOL, "/models_v11": MODEL_VOL_V11,
             "/models_baseline": MODEL_VOL_BASELINE, "/matched": MATCHED_VOL},
    timeout=3600,
    memory=32768,
)
def run_synthetic_test():
    device = "cuda"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("=" * 70)
    print("SYNTHETIC MATCHED BG TEST — Surgery Artifact Check")
    print("=" * 70)
    print()
    print("Taking random BG spectrograms (no drone ever present)")
    print("and applying the SAME surgical operation used to create matched BGs.")
    print("If the model detects the surgery → it's contaminated.")
    print("If the model doesn't care → it's clean.")
    print()

    # ── Load random BG (negatives) ──
    print("Loading random backgrounds (negatives)...")
    h5_f = h5py.File(H5_REMOTE, "r")

    if "negatives" not in h5_f:
        print("ERROR: No negatives found in dataset!")
        h5_f.close()
        return

    neg_item = h5_f["negatives"]
    if isinstance(neg_item, h5py.Dataset):
        n_total = neg_item.shape[0]
        n_use = min(n_total, 2000)
        neg_indices = np.random.default_rng(42).choice(n_total, n_use, replace=False)
        neg_indices.sort()
        neg_samples = neg_item[list(neg_indices)]
    else:
        neg_keys = [sk for sk in neg_item.keys()
                    if isinstance(neg_item[sk], h5py.Dataset) and len(neg_item[sk].shape) == 3]
        n_use = min(len(neg_keys), 2000)
        rng = np.random.default_rng(42)
        chosen = rng.choice(len(neg_keys), n_use, replace=False)
        neg_samples = np.stack([neg_item[neg_keys[i]][:] for i in chosen])

    print(f"  Loaded {n_use} random BG spectrograms")

    # ── Apply surgery to random BGs ──
    print("Applying surgical operation to random BGs...")
    synthetic_matched = []
    n_surgerized = 0
    for i, sample in enumerate(neg_samples):
        if sample.ndim == 2:
            sample = sample[np.newaxis, :, :]

        # Take first 2 channels
        if sample.shape[0] >= 2:
            x = sample[:2].copy().astype(np.float32)
        else:
            x = np.stack([sample[0]] * 2).astype(np.float32)

        # Apply the EXACT same surgery
        surgerized = generate_matched_background(x, noise_percentile=10,
                                                  signal_sigma=2.0, method="replace")
        synthetic_matched.append(surgerized)

        # Count how many pixels were modified
        diff = np.abs(x - surgerized)
        if diff.sum() > 0:
            n_surgerized += 1

        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{n_use}")

    print(f"  {n_surgerized}/{n_use} samples had pixels modified by surgery")

    # ── Prepare tensors ──
    print("Preparing tensors...")

    original_tensors = []
    synthetic_tensors = []

    for i in range(n_use):
        # Original random BG
        sample = neg_samples[i]
        if sample.ndim == 2:
            sample = sample[np.newaxis, :, :]
        if sample.shape[0] >= 2:
            x_orig = sample[:2].copy().astype(np.float32)
        else:
            x_orig = np.stack([sample[0]] * 2).astype(np.float32)

        x_orig_t = torch.from_numpy(x_orig)
        x_orig_t = normalize_per_channel(x_orig_t)
        original_tensors.append(x_orig_t)

        # Synthetic matched BG (surgically modified)
        x_syn = synthetic_matched[i]
        x_syn_t = torch.from_numpy(x_syn)
        x_syn_t = normalize_per_channel(x_syn_t)
        synthetic_tensors.append(x_syn_t)

    # ── Also load real matched BGs and real drones for reference ──
    print("Loading real matched BGs for reference...")
    matched_f = h5py.File(MATCHED_REMOTE, "r")
    mbg_key = "holdout_matched_bg"
    if mbg_key not in matched_f:
        mbg_key = "train_matched_bg"
    mbg_grp = matched_f[mbg_key]
    mbg_keys = sorted(list(mbg_grp.keys()), key=lambda x: int(x) if x.isdigit() else 0)
    n_real_mbg = min(len(mbg_keys), 2000)

    real_mbg_tensors = []
    rng_mbg = np.random.default_rng(123)
    mbg_indices = rng_mbg.choice(len(mbg_keys), n_real_mbg, replace=False)
    for j in mbg_indices:
        key = mbg_keys[j]
        sample = mbg_grp[key][:]
        x = prepare_tensor(sample)
        x = normalize_per_channel(x)
        real_mbg_tensors.append(x)

    print(f"  Loaded {n_real_mbg} real matched BGs")

    # ── Load train drone embeddings for centroid ──
    print("Loading train drone samples for centroid...")
    train_grp = h5_f["train"]
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

    train_drone_tensors = []
    for tname in type_names:
        ds_or_grp, n_samples, is_multi = _resolved[tname]
        n_use_t = min(n_samples, 200)
        for i in range(n_use_t):
            if is_multi:
                sub_key = _sub_keys[tname][i]
                sample = ds_or_grp[sub_key][:]
            else:
                sample = ds_or_grp[i]
            x = prepare_tensor(sample)
            x = normalize_per_channel(x)
            train_drone_tensors.append(x)

    print(f"  Loaded {len(train_drone_tensors)} train drone samples")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST BOTH MODELS
    # ═══════════════════════════════════════════════════════════════════════

    results = {}

    for model_name, vol_path, model_class, cfg in [
        ("IRIS v11", "/models_v11/lejepa_v11_best.pt", LeJEPASupConV11,
         dict(in_ch=2, encoder_depth=6, encoder_width=64, embed_dim=256,
              proj_dim=256, pred_dim=256, pred_out=256)),
        ("Supervised Baseline", "/models_baseline/baseline_supervised_best.pt",
         SupervisedBinaryClassifier,
         dict(in_ch=2, encoder_depth=6, encoder_width=64, embed_dim=256, hidden_dim=128)),
    ]:
        print(f"\n{'='*70}")
        print(f"Testing: {model_name}")
        print(f"{'='*70}")

        # Load model
        if not os.path.exists(vol_path):
            print(f"  WARNING: No checkpoint at {vol_path}, skipping")
            continue

        ckpt = torch.load(vol_path, map_location=device, weights_only=False)
        model = model_class(cfg).to(device)

        if model_name == "IRIS v11":
            model.load_state_dict(ckpt["model"], strict=False)
            encoder_fn = model.encoder
        else:
            model.load_state_dict(ckpt["model"])
            encoder_fn = model.encode

        model.eval()

        # Encode train drones for centroid
        print("  Encoding train drones for centroid...")
        train_embs = encode_samples(train_drone_tensors, encoder_fn, device)
        centroid = train_embs.mean(axis=0)
        D = train_embs.shape[1]
        cov = np.cov(train_embs.T) + 1e-3 * np.eye(D)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        # Encode all test sets
        print("  Encoding original random BGs...")
        orig_embs = encode_samples(original_tensors, encoder_fn, device)

        print("  Encoding synthetic matched BGs (surgery on random BG)...")
        syn_embs = encode_samples(synthetic_tensors, encoder_fn, device)

        print("  Encoding real matched BGs...")
        real_mbg_embs = encode_samples(real_mbg_tensors, encoder_fn, device)

        # ── Compute Mahalanobis distances ──
        def mahalanobis(embs, centroid, cov_inv):
            diff = embs - centroid
            return np.sqrt(np.maximum(np.sum(diff @ cov_inv * diff, axis=1), 0))

        orig_mahal = mahalanobis(orig_embs, centroid, cov_inv)
        syn_mahal = mahalanobis(syn_embs, centroid, cov_inv)
        real_mbg_mahal = mahalanobis(real_mbg_embs, centroid, cov_inv)
        train_mahal = mahalanobis(train_embs, centroid, cov_inv)

        # ── THE KEY TEST: Original random BG vs Synthetic matched BG ──
        # If the surgery matters, synthetic matched BGs will be closer to
        # the drone centroid than original random BGs
        # Label: original BG = 0, synthetic BG = 1
        # Score: negative distance (closer to centroid = more "drone-like")
        surgery_labels = np.concatenate([np.zeros(len(orig_mahal)), np.ones(len(syn_mahal))])
        surgery_scores = np.concatenate([-orig_mahal, -syn_mahal])
        surgery_auc = roc_auc_score(surgery_labels, surgery_scores)

        # Also: are synthetic matched BGs closer than real matched BGs?
        syn_vs_real_labels = np.concatenate([np.zeros(len(real_mbg_mahal)), np.ones(len(syn_mahal))])
        syn_vs_real_scores = np.concatenate([-real_mbg_mahal, -syn_mahal])
        syn_vs_real_auc = roc_auc_score(syn_vs_real_labels, syn_vs_real_scores)

        # ── Print results ──
        print(f"\n  {'─'*60}")
        print(f"  DISTANCE TO DRONE CENTROID (Mahalanobis)")
        print(f"  {'─'*60}")
        print(f"  Train drones:      mean={train_mahal.mean():.2f}  std={train_mahal.std():.2f}")
        print(f"  Original random BG: mean={orig_mahal.mean():.2f}  std={orig_mahal.std():.2f}")
        print(f"  Synthetic matched:  mean={syn_mahal.mean():.2f}  std={syn_mahal.std():.2f}")
        print(f"  Real matched BG:    mean={real_mbg_mahal.mean():.2f}  std={real_mbg_mahal.std():.2f}")
        print(f"")
        print(f"  {'─'*60}")
        print(f"  KEY TEST: Does the surgery change the model's classification?")
        print(f"  {'─'*60}")
        print(f"  Original BG vs Synthetic matched BG AUC: {surgery_auc:.4f}")
        print(f"    (0.5 = surgery has no effect → MODEL IS CLEAN)")
        print(f"    (1.0 = surgery makes it look like matched BG → CONTAMINATED)")
        print(f"")
        print(f"  Synthetic matched vs Real matched BG AUC: {syn_vs_real_auc:.4f}")
        print(f"    (How similar are synthetic and real matched BGs to the model)")

        results[model_name] = {
            "surgery_auc": float(surgery_auc),
            "syn_vs_real_auc": float(syn_vs_real_auc),
            "orig_mean_dist": float(orig_mahal.mean()),
            "syn_mean_dist": float(syn_mahal.mean()),
            "real_mbg_mean_dist": float(real_mbg_mahal.mean()),
            "train_mean_dist": float(train_mahal.mean()),
        }

    # ── Final comparison ──
    print(f"\n\n{'='*70}")
    print(f"FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"")
    print(f"  {'Model':<25} {'Surgery AUC':>12} {'Interpretation':>30}")
    print(f"  {'─'*25} {'─'*12} {'─'*30}")

    for mname, mres in results.items():
        auc = mres["surgery_auc"]
        if auc < 0.55:
            interp = "CLEAN - no surgery artifact"
        elif auc < 0.65:
            interp = "MOSTLY CLEAN - minimal artifact"
        elif auc < 0.80:
            interp = "PARTIALLY CONTAMINATED"
        else:
            interp = "HEAVILY CONTAMINATED"
        print(f"  {mname:<25} {auc:>12.4f} {interp:>30}")

    print(f"")
    print(f"  IRIS v11 artifact AUC (matched vs random BG): 0.9246")
    if "Supervised Baseline" in results:
        print(f"  Supervised artifact AUC (matched vs random BG): 0.9618")
    print(f"")
    print(f"  The surgery AUC tells us how much of that artifact AUC")
    print(f"  is due to the SURGICAL OPERATION vs actual drone signal leakage.")
    print(f"{'='*70}")

    h5_f.close()
    matched_f.close()


@app.local_entrypoint()
def main():
    run_synthetic_test.remote()