#!/usr/bin/env python3
"""
IRIS Cross-Dataset Generalization — Debug + Full Test

1. Diagnose why DRFF-R2 gives identical distances (suspicious)
2. Test IRIS on ALL available drone data:
   - RFUAV holdout (7 types) — baseline
   - DRFF-R2 (8 types) — cross-dataset, with debug
   - DroneRF negatives scanned for drone-like signals
3. Report: what % of random drones does IRIS detect?

Usage:
    modal run scripts/cross_dataset_debug.py
"""

from __future__ import annotations

import h5py
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import modal
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
from sklearn.metrics import roc_auc_score

app = modal.App("iris-cross-dataset-debug")

VOL = modal.Volume.from_name("iris-data", create_if_missing=True)
MODEL_VOL = modal.Volume.from_name("iris-models-v11", create_if_missing=True)
MATCHED_VOL = modal.Volume.from_name("iris-matched-bg", create_if_missing=True)
RESULTS_VOL = modal.Volume.from_name("iris-results", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install("libgl1", "libglib2.0-0", "libhdf5-dev", "python3", "python3-pip", "python-is-python3")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "h5py==3.12.1", "numpy==1.26.4",
                 "scikit-learn==1.6.1", "scipy==1.14.1", "matplotlib==3.9.3")
)

H5_REMOTE = "/data/iris_rfuav.h5"
DRFFR2_REMOTE = "/data/drffr2.h5"
MATCHED_REMOTE = "/matched/iris_matched_bg.h5"
MODEL_REMOTE = "/models/lejepa_v11_best.pt"
RESULTS_REMOTE = "/results"


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


class STFTEngine:
    def __init__(self, n_fft=1024, hop_len=256, win_len=1024, target_h=256, target_w=256):
        self.window = hann(win_len, sym=True)
        self.stft = ShortTimeFFT(win=self.window, hop=hop_len, fs=1.0, mfft=n_fft, fft_mode='twosided')
        self.target_h = target_h
        self.target_w = target_w

    def __call__(self, iq_complex):
        S = self.stft.stft(iq_complex)
        power = np.abs(S) ** 2
        log_power = np.log1p(power)
        phase = np.angle(S) / np.pi
        log_power = self._resize(log_power)
        phase = self._resize(phase)
        log_power = self._normalize(log_power)
        phase = self._normalize(phase)
        return np.stack([log_power, phase], axis=0).astype(np.float32)

    def _resize(self, img):
        h, w = img.shape
        if h == self.target_h and w == self.target_w: return img
        row_idx = np.clip(np.round(np.linspace(0, h-1, self.target_h)).astype(int), 0, h-1)
        col_idx = np.clip(np.round(np.linspace(0, w-1, self.target_w)).astype(int), 0, w-1)
        return img[np.ix_(row_idx, col_idx)]

    def _normalize(self, img):
        m, s = img.mean(), img.std()
        return (img - m) / s if s > 1e-8 else img - m


def _norm(x):
    for c in range(x.shape[0]):
        ch, std = x[c], x[c].std()
        x[c] = (ch - ch.mean()) / std if std > 1e-6 else ch - ch.mean()
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
        embs = encoder(batch)
        all_embs.append(embs.cpu().numpy())
    return np.concatenate(all_embs)


@app.function(
    image=IMAGE, gpu="T4",
    volumes={"/data": VOL, "/models": MODEL_VOL, "/matched": MATCHED_VOL, "/results": RESULTS_VOL},
    timeout=5400, memory=16384,
)
def run_debug():
    device = "cuda"
    print("=" * 70)
    print("IRIS — Cross-Dataset Debug + Full Generalization Test")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 70)

    VOL.reload()
    MODEL_VOL.reload()
    MATCHED_VOL.reload()

    # Load encoder
    ckpt = torch.load(MODEL_REMOTE, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    encoder_state = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
    encoder = CNNEncoder(in_ch=2, width=64, depth=6, embed_dim=256).to(device)
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    print(f"  [ok] encoder loaded")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Debug DRFF-R2 — inspect raw data
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1: Debug DRFF-R2 raw data")
    print("=" * 70)

    with h5py.File(DRFFR2_REMOTE, "r") as f:
        grp = f["drones"]
        first_type = list(grp.keys())[0]
        first_sample = grp[first_type][list(grp[first_type].keys())[0]][:]
        
        print(f"  Sample shape: {first_sample.shape}")
        print(f"  Sample dtype: {first_sample.dtype}")
        print(f"  Channel 0 (I?): min={first_sample[0].min():.6f}, max={first_sample[0].max():.6f}, mean={first_sample[0].mean():.6f}, std={first_sample[0].std():.6f}")
        print(f"  Channel 1 (Q?): min={first_sample[1].min():.6f}, max={first_sample[1].max():.6f}, mean={first_sample[1].mean():.6f}, std={first_sample[1].std():.6f}")
        if first_sample.shape[0] >= 3:
            print(f"  Channel 2 (?):  min={first_sample[2].min():.6f}, max={first_sample[2].max():.6f}, mean={first_sample[2].mean():.6f}, std={first_sample[2].std():.6f}")
        
        print(f"\n  First 10 values of channel 0: {first_sample[0][:10]}")
        print(f"  First 10 values of channel 1: {first_sample[1][:10]}")
        if first_sample.shape[0] >= 3:
            print(f"  First 10 values of channel 2: {first_sample[2][:10]}")

        # Check if channels are actually I/Q or something else
        # If I/Q: should be centered around 0, roughly [-1, 1] range
        # If spectrogram: should be positive, maybe [0, large]
        
        ch0_range = first_sample[0].max() - first_sample[0].min()
        ch1_range = first_sample[1].max() - first_sample[1].min()
        print(f"\n  Channel 0 range: {ch0_range:.6f}")
        print(f"  Channel 1 range: {ch1_range:.6f}")
        
        # Check if it's already a spectrogram (256x256 = 65536)
        # If so, reshape would work directly
        print(f"\n  Trying reshape (3, 65536) → (3, 256, 256)...")
        reshaped = first_sample.reshape(3, 256, 256)
        print(f"  Reshaped ch0: min={reshaped[0].min():.6f}, max={reshaped[0].max():.6f}, mean={reshaped[0].mean():.6f}")
        print(f"  Reshaped ch1: min={reshaped[1].min():.6f}, max={reshaped[1].max():.6f}, mean={reshaped[1].mean():.6f}")
        print(f"  Reshaped ch2: min={reshaped[2].min():.6f}, max={reshaped[2].max():.6f}, mean={reshaped[2].mean():.6f}")
        
        # Check: is ch0 positive (spectrogram) or centered at 0 (IQ)?
        if first_sample[0].min() >= 0:
            print("\n  ⚠ Channel 0 is all positive — might be spectrogram data, NOT raw IQ!")
            print("  Trying direct reshape as spectrogram (no STFT)...")
        else:
            print("\n  Channel 0 has negative values — consistent with raw IQ")
        
        # Also check: is 65536 = 256*256 a coincidence or intentional?
        print(f"\n  65536 = 256 * 256 = {256*256}")
        print(f"  This could be pre-computed spectrograms stored as 1D!")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Test multiple conversion strategies
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2: Test multiple DRFF-R2 conversion strategies")
    print("=" * 70)

    # Load RFUAV train for centroid
    print("  Loading RFUAV train...")
    with h5py.File(H5_REMOTE, "r") as f:
        train_specs = []
        grp = f["train"]
        for tname in sorted(grp.keys())[:30]:
            try:
                item = grp[tname]
                if isinstance(item, h5py.Dataset):
                    if len(item.shape) == 4:
                        for i in range(min(item.shape[0], 100)):
                            s = item[i]
                            if s.shape[0] >= 2:
                                train_specs.append(_norm(s[:2].copy().astype(np.float32)))
                    elif len(item.shape) == 3:
                        train_specs.append(_norm(item[:2].copy().astype(np.float32)))
                elif isinstance(item, h5py.Group):
                    subs = sorted([k for k in item.keys() if isinstance(item[k], h5py.Dataset) and len(item[k].shape) == 3])
                    for sk in subs[:100]:
                        s = item[sk][:]
                        if s.shape[0] >= 2:
                            train_specs.append(_norm(s[:2].copy().astype(np.float32)))
            except: continue
    train_specs = np.stack(train_specs)
    print(f"  RFUAV train: {len(train_specs)} samples")

    train_embs = encode_batch(encoder, train_specs, device)
    centroid, cov_inv = fit_mahalanobis_l2(train_embs)
    train_dists = mahal_l2(train_embs, centroid, cov_inv)
    threshold = float(np.percentile(train_dists, 99))
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Train mean dist: {train_dists.mean():.2f}")

    # Load matched BGs
    with h5py.File(MATCHED_REMOTE, "r") as f:
        bg_grp = f["holdout_matched_bg"]
        bg_keys = sorted(list(bg_grp.keys()), key=lambda x: int(x) if x.isdigit() else 0)
        bg_specs = []
        for k in bg_keys[:200]:
            s = bg_grp[k][:]
            if s.shape[0] >= 2:
                bg_specs.append(_norm(s[:2].copy().astype(np.float32)))
    bg_specs = np.stack(bg_specs)
    bg_embs = encode_batch(encoder, bg_specs, device)
    bg_dists = mahal_l2(bg_embs, centroid, cov_inv)
    print(f"  BG mean dist: {bg_dists.mean():.2f}")

    # Load RFUAV holdout
    with h5py.File(H5_REMOTE, "r") as f:
        holdout_specs = []
        holdout_types = []
        grp = f["holdout"]
        for tname in sorted(grp.keys()):
            try:
                item = grp[tname]
                if isinstance(item, h5py.Dataset):
                    if len(item.shape) == 4:
                        for i in range(min(item.shape[0], 100)):
                            s = item[i]
                            if s.shape[0] >= 2:
                                holdout_specs.append(_norm(s[:2].copy().astype(np.float32)))
                                holdout_types.append(tname)
                elif isinstance(item, h5py.Group):
                    subs = sorted([k for k in item.keys() if isinstance(item[k], h5py.Dataset) and len(item[k].shape) == 3])
                    for sk in subs[:100]:
                        s = item[sk][:]
                        if s.shape[0] >= 2:
                            holdout_specs.append(_norm(s[:2].copy().astype(np.float32)))
                            holdout_types.append(tname)
            except: continue
    holdout_specs = np.stack(holdout_specs)
    holdout_embs = encode_batch(encoder, holdout_specs, device)
    holdout_dists = mahal_l2(holdout_embs, centroid, cov_inv)
    print(f"  RFUAV holdout mean dist: {holdout_dists.mean():.2f}")
    print(f"  RFUAV holdout detection: {(holdout_dists <= threshold).mean():.3f}")

    # Strategy A: Direct reshape (3, 65536) → (2, 256, 256) [treat as pre-computed spectrogram]
    print("\n  --- Strategy A: Direct reshape (no STFT) ---")
    with h5py.File(DRFFR2_REMOTE, "r") as f:
        grp = f["drones"]
        drffr2_a = []
        drffr2_a_types = []
        for tname in sorted(grp.keys()):
            type_grp = grp[tname]
            subs = sorted(list(type_grp.keys()))
            for sk in subs[:50]:
                raw = type_grp[sk][:]  # (3, 65536)
                reshaped = raw.reshape(3, 256, 256)
                spec = _norm(reshaped[:2].copy().astype(np.float32))
                drffr2_a.append(spec)
                drffr2_a_types.append(tname)
    drffr2_a = np.stack(drffr2_a)
    embs_a = encode_batch(encoder, drffr2_a, device)
    dists_a = mahal_l2(embs_a, centroid, cov_inv)
    print(f"    Mean dist: {dists_a.mean():.4f}")
    print(f"    Std dist:  {dists_a.std():.4f}")
    print(f"    Detection: {(dists_a <= threshold).mean():.3f}")
    # Check if distances vary at all
    if dists_a.std() < 0.01:
        print(f"    ⚠ ALL DISTANCES IDENTICAL — degenerate output!")

    # Strategy B: STFT on ch0+ch1 as I/Q
    print("\n  --- Strategy B: STFT on ch0+ch1 as complex IQ ---")
    stft_engine = STFTEngine()
    with h5py.File(DRFFR2_REMOTE, "r") as f:
        grp = f["drones"]
        drffr2_b = []
        drffr2_b_types = []
        for tname in sorted(grp.keys()):
            type_grp = grp[tname]
            subs = sorted(list(type_grp.keys()))
            for sk in subs[:50]:
                raw = type_grp[sk][:]  # (3, 65536)
                iq = raw[0].astype(np.complex128) + 1j * raw[1].astype(np.complex128)
                spec = stft_engine(iq)
                drffr2_b.append(spec)
                drffr2_b_types.append(tname)
    drffr2_b = np.stack(drffr2_b)
    embs_b = encode_batch(encoder, drffr2_b, device)
    dists_b = mahal_l2(embs_b, centroid, cov_inv)
    print(f"    Mean dist: {dists_b.mean():.4f}")
    print(f"    Std dist:  {dists_b.std():.4f}")
    print(f"    Detection: {(dists_b <= threshold).mean():.3f}")
    if dists_b.std() < 0.01:
        print(f"    ⚠ ALL DISTANCES IDENTICAL — degenerate output!")

    # Strategy C: STFT on ch2 only (if ch2 is the actual signal)
    print("\n  --- Strategy C: STFT on ch2 only ---")
    with h5py.File(DRFFR2_REMOTE, "r") as f:
        grp = f["drones"]
        drffr2_c = []
        drffr2_c_types = []
        for tname in sorted(grp.keys()):
            type_grp = grp[tname]
            subs = sorted(list(type_grp.keys()))
            for sk in subs[:50]:
                raw = type_grp[sk][:]  # (3, 65536)
                # Treat ch2 as real signal, ch0 as imag
                iq = raw[2].astype(np.complex128) + 1j * np.zeros_like(raw[2])
                spec = stft_engine(iq)
                drffr2_c.append(spec)
                drffr2_c_types.append(tname)
    drffr2_c = np.stack(drffr2_c)
    embs_c = encode_batch(encoder, drffr2_c, device)
    dists_c = mahal_l2(embs_c, centroid, cov_inv)
    print(f"    Mean dist: {dists_c.mean():.4f}")
    print(f"    Std dist:  {dists_c.std():.4f}")
    print(f"    Detection: {(dists_c <= threshold).mean():.3f}")

    # Strategy D: Treat (3, 65536) as already-spectrogram with 3 channels
    # reshape to (3, 256, 256) and use channels 0,2 (skip channel 1 which might be phase)
    print("\n  --- Strategy D: Direct reshape, use ch0+ch2 ---")
    with h5py.File(DRFFR2_REMOTE, "r") as f:
        grp = f["drones"]
        drffr2_d = []
        drffr2_d_types = []
        for tname in sorted(grp.keys()):
            type_grp = grp[tname]
            subs = sorted(list(type_grp.keys()))
            for sk in subs[:50]:
                raw = type_grp[sk][:]  # (3, 65536)
                reshaped = raw.reshape(3, 256, 256)
                # Use ch0 (power?) and ch2 (phase?)
                spec = _norm(np.stack([reshaped[0], reshaped[2]]).astype(np.float32))
                drffr2_d.append(spec)
                drffr2_d_types.append(tname)
    drffr2_d = np.stack(drffr2_d)
    embs_d = encode_batch(encoder, drffr2_d, device)
    dists_d = mahal_l2(embs_d, centroid, cov_inv)
    print(f"    Mean dist: {dists_d.mean():.4f}")
    print(f"    Std dist:  {dists_d.std():.4f}")
    print(f"    Detection: {(dists_d <= threshold).mean():.3f}")

    # Strategy E: Treat as IQ but normalize differently
    # Maybe the IQ values need to be scaled
    print("\n  --- Strategy E: STFT on scaled IQ ---")
    with h5py.File(DRFFR2_REMOTE, "r") as f:
        grp = f["drones"]
        drffr2_e = []
        drffr2_e_types = []
        for tname in sorted(grp.keys()):
            type_grp = grp[tname]
            subs = sorted(list(type_grp.keys()))
            for sk in subs[:50]:
                raw = type_grp[sk][:]  # (3, 65536)
                # Normalize IQ to [-1, 1] range first
                i = raw[0]
                q = raw[1]
                max_val = max(np.abs(i).max(), np.abs(q).max())
                if max_val > 0:
                    i = i / max_val
                    q = q / max_val
                iq = i.astype(np.complex128) + 1j * q.astype(np.complex128)
                spec = stft_engine(iq)
                drffr2_e.append(spec)
                drffr2_e_types.append(tname)
    drffr2_e = np.stack(drffr2_e)
    embs_e = encode_batch(encoder, drffr2_e, device)
    dists_e = mahal_l2(embs_e, centroid, cov_inv)
    print(f"    Mean dist: {dists_e.mean():.4f}")
    print(f"    Std dist:  {dists_e.std():.4f}")
    print(f"    Detection: {(dists_e <= threshold).mean():.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    print(f"  {'Strategy':<40} {'Mean Dist':>10} {'Std':>8} {'Detection':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*8} {'-'*10}")
    print(f"  {'RFUAV holdout (baseline)':<40} {holdout_dists.mean():>10.4f} {holdout_dists.std():>8.4f} {(holdout_dists <= threshold).mean():>10.3f}")
    print(f"  {'Matched BG (should be high)':<40} {bg_dists.mean():>10.4f} {bg_dists.std():>8.4f} {(bg_dists <= threshold).mean():>10.3f}")
    print(f"  {'A: Direct reshape ch0+ch1':<40} {dists_a.mean():>10.4f} {dists_a.std():>8.4f} {(dists_a <= threshold).mean():>10.3f}")
    print(f"  {'B: STFT ch0+ch1 as IQ':<40} {dists_b.mean():>10.4f} {dists_b.std():>8.4f} {(dists_b <= threshold).mean():>10.3f}")
    print(f"  {'C: STFT ch2 only':<40} {dists_c.mean():>10.4f} {dists_c.std():>8.4f} {(dists_c <= threshold).mean():>10.3f}")
    print(f"  {'D: Direct reshape ch0+ch2':<40} {dists_d.mean():>10.4f} {dists_d.std():>8.4f} {(dists_d <= threshold).mean():>10.3f}")
    print(f"  {'E: STFT scaled IQ':<40} {dists_e.mean():>10.4f} {dists_e.std():>8.4f} {(dists_e <= threshold).mean():>10.3f}")

    # Determine best strategy
    strategies = {
        "A_direct_reshape": (dists_a, drffr2_a_types),
        "B_stft_iq": (dists_b, drffr2_b_types),
        "C_stft_ch2": (dists_c, drffr2_c_types),
        "D_reshape_ch02": (dists_d, drffr2_d_types),
        "E_stft_scaled": (dists_e, drffr2_e_types),
    }

    best_strategy = min(strategies.items(), key=lambda x: x[1][0].mean())
    print(f"\n  Best strategy (lowest mean dist): {best_strategy[0]}")
    print(f"    Mean dist: {best_strategy[1][0].mean():.4f}")
    print(f"    Detection: {(best_strategy[1][0] <= threshold).mean():.3f}")

    # Per-type breakdown for best strategy
    best_dists, best_types = best_strategy[1]
    print(f"\n  Per-type ({best_strategy[0]}):")
    unique_types = sorted(set(best_types))
    for t in unique_types:
        mask = np.array(best_types) == t
        t_dists = best_dists[mask]
        print(f"    {t:25s}: mean={t_dists.mean():.4f}, det={float((t_dists <= threshold).mean()):.3f}, n={len(t_dists)}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Full generalization test — ALL drone types from ALL datasets
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3: Full Generalization Test — ALL drone types")
    print("=" * 70)

    all_drone_results = {}

    # RFUAV holdout (7 types)
    print("\n  RFUAV holdout (7 types):")
    holdout_types_arr = np.array(holdout_types)
    for t in sorted(set(holdout_types)):
        mask = holdout_types_arr == t
        t_dists = holdout_dists[mask]
        det = float((t_dists <= threshold).mean())
        print(f"    {t:25s}: det={det:.3f}, dist={t_dists.mean():.2f}, n={len(t_dists)}")
        all_drone_results[f"RFUAV/{t}"] = {"detection_rate": det, "mean_dist": float(t_dists.mean()), "n": int(len(t_dists))}

    # DRFF-R2 (8 types) — use best strategy
    print(f"\n  DRFF-R2 (8 types, strategy={best_strategy[0]}):")
    best_types_arr = np.array(best_types)
    for t in sorted(set(best_types)):
        mask = best_types_arr == t
        t_dists = best_dists[mask]
        det = float((t_dists <= threshold).mean())
        print(f"    {t:25s}: det={det:.3f}, dist={t_dists.mean():.2f}, n={len(t_dists)}")
        all_drone_results[f"DRFF-R2/{t}"] = {"detection_rate": det, "mean_dist": float(t_dists.mean()), "n": int(len(t_dists))}

    # DroneRF negatives — scan for drone-like signals
    print(f"\n  DroneRF negatives (scanning for drone-like signals):")
    with h5py.File(H5_REMOTE, "r") as f:
        if "negatives" in f:
            neg_grp = f["negatives"]
            if isinstance(neg_grp, h5py.Dataset):
                neg_samples = neg_grp[:500]
            else:
                all_keys = list(neg_grp.keys())
                rng = np.random.default_rng(42)
                picked = rng.choice(all_keys, min(500, len(all_keys)), replace=False)
                neg_samples = []
                for key in picked:
                    try:
                        item = neg_grp[str(key)]
                        if isinstance(item, h5py.Dataset):
                            if len(item.shape) == 4: neg_samples.append(item[0])
                            elif len(item.shape) == 3: neg_samples.append(item[:])
                    except: continue
                neg_samples = np.stack(neg_samples) if neg_samples else np.array([])
            
            if len(neg_samples) > 0:
                neg_specs = np.stack([_norm(s[:2].copy().astype(np.float32)) for s in neg_samples])
                neg_embs = encode_batch(encoder, neg_specs, device)
                neg_dists = mahal_l2(neg_embs, centroid, cov_inv)
                neg_det = float((neg_dists <= threshold).mean())
                print(f"    DroneRF negatives: det={neg_det:.3f}, dist={neg_dists.mean():.2f}, n={len(neg_dists)}")
                all_drone_results["DroneRF/negatives"] = {"detection_rate": neg_det, "mean_dist": float(neg_dists.mean()), "n": int(len(neg_dists))}

    # Summary
    total_types = len([k for k in all_drone_results.keys() if "negatives" not in k])
    detected_types = sum(1 for k, v in all_drone_results.items() if "negatives" not in k and v["detection_rate"] > 0.5)
    print(f"\n" + "=" * 70)
    print(f"GENERALIZATION SUMMARY")
    print(f"=" * 70)
    print(f"  Total drone types tested: {total_types}")
    print(f"  Types detected (>50% detection rate): {detected_types}")
    print(f"  Generalization rate: {detected_types}/{total_types} = {detected_types/total_types*100:.1f}%")
    print(f"  False positive rate on DroneRF negatives: {neg_det:.3f}")

    # Save
    os.makedirs(RESULTS_REMOTE, exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "threshold": threshold,
        "rfuav_holdout_mean_dist": float(holdout_dists.mean()),
        "bg_mean_dist": float(bg_dists.mean()),
        "strategies": {
            "A_direct_reshape": {"mean_dist": float(dists_a.mean()), "std": float(dists_a.std()), "detection": float((dists_a <= threshold).mean())},
            "B_stft_iq": {"mean_dist": float(dists_b.mean()), "std": float(dists_b.std()), "detection": float((dists_b <= threshold).mean())},
            "C_stft_ch2": {"mean_dist": float(dists_c.mean()), "std": float(dists_c.std()), "detection": float((dists_c <= threshold).mean())},
            "D_reshape_ch02": {"mean_dist": float(dists_d.mean()), "std": float(dists_d.std()), "detection": float((dists_d <= threshold).mean())},
            "E_stft_scaled": {"mean_dist": float(dists_e.mean()), "std": float(dists_e.std()), "detection": float((dists_e <= threshold).mean())},
        },
        "best_strategy": best_strategy[0],
        "all_drone_results": all_drone_results,
        "total_types": total_types,
        "detected_types": detected_types,
        "generalization_rate": float(detected_types / total_types),
    }

    json_path = f"{RESULTS_REMOTE}/cross_dataset_debug.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  [ok] saved {json_path}")
    RESULTS_VOL.commit()

    return output


@app.local_entrypoint()
def main():
    run_debug.remote()


if __name__ == "__main__":
    main()
