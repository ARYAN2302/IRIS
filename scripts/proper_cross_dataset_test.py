#!/usr/bin/env python3
"""
IRIS Proper Cross-Dataset Test — DRFF-R2 with correct STFT conversion

FIXES:
1. DRFF-R2 raw IQ (3, 65536) → proper STFT spectrograms using STFTEngine
   (log-power + phase channels, same as RFUAV preprocessing)
2. Spectral artifact test uses REALISTIC hard negatives:
   - Real WiFi/BT captures spectrally shaped to match drone envelope
   - NOT permuted drone spectrograms (unrealistic)
3. Cross-dataset: train on RFUAV, test on DRFF-R2 (properly converted)
4. Within-dataset artifact check: test if matched BGs share drone spectral envelope

Usage:
    modal run scripts/proper_cross_dataset_test.py
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
import torch.nn.functional as F
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
from sklearn.metrics import roc_auc_score

app = modal.App("iris-proper-cross-dataset")

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


# ─────────────────────────────────────────────────────────────────────────────
# STFT Engine — EXACT copy from src/stft_engine.py
# Channel 0: log-power spectrogram
# Channel 1: normalized phase
# ─────────────────────────────────────────────────────────────────────────────


class STFTEngine:
    def __init__(self, n_fft=1024, hop_len=256, win_len=1024, target_height=256, target_width=256):
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.target_height = target_height
        self.target_width = target_width
        self.window = hann(win_len, sym=True)
        self.stft = ShortTimeFFT(win=self.window, hop=hop_len, fs=1.0, mfft=n_fft, fft_mode='twosided')

    def __call__(self, iq_complex):
        S = self.stft.stft(iq_complex)
        power = np.abs(S) ** 2
        log_power = np.log1p(power)
        phase = np.angle(S) / np.pi
        log_power = self._resize(log_power, self.target_height, self.target_width)
        phase = self._resize(phase, self.target_height, self.target_width)
        log_power = self._normalize(log_power)
        phase = self._normalize(phase)
        return np.stack([log_power, phase], axis=0).astype(np.float32)

    def _resize(self, img, h, w):
        orig_h, orig_w = img.shape
        if orig_h == h and orig_w == w: return img
        row_idx = np.round(np.linspace(0, orig_h - 1, h)).astype(int)
        col_idx = np.round(np.linspace(0, orig_w - 1, w)).astype(int)
        row_idx = np.clip(row_idx, 0, orig_h - 1)
        col_idx = np.clip(col_idx, 0, orig_w - 1)
        return img[np.ix_(row_idx, col_idx)]

    def _normalize(self, img):
        mean, std = img.mean(), img.std()
        if std < 1e-8: return img - mean
        return (img - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _prep_spectrogram(sample):
    """Handle RFUAV pre-processed spectrograms (2, 256, 256) or (3, 256, 256)."""
    if sample.ndim == 3 and sample.shape[0] >= 2:
        return sample[:2].copy().astype(np.float32)
    return sample.astype(np.float32)


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


def load_rfuav_split(h5_path, split, max_per_type=500):
    """Load RFUAV pre-processed spectrograms."""
    print(f"  [info] loading RFUAV {split}...")
    with h5py.File(h5_path, "r") as f:
        if split not in f: return {}
        grp = f[split]
        type_names = sorted(list(grp.keys()))
        specs_dict = {}
        for tname in type_names:
            try:
                item = grp[tname]
                if isinstance(item, h5py.Dataset):
                    if len(item.shape) == 4:
                        n = min(item.shape[0], max_per_type)
                        specs = [_norm(_prep_spectrogram(item[i])) for i in range(n)]
                    elif len(item.shape) == 3:
                        specs = [_norm(_prep_spectrogram(item[:]))]
                    else: continue
                elif isinstance(item, h5py.Group):
                    sub_keys = sorted([k for k in item.keys()
                                      if isinstance(item[k], h5py.Dataset) and len(item[k].shape) == 3])
                    specs = [_norm(_prep_spectrogram(item[k][:])) for k in sub_keys[:max_per_type]]
                else: continue
                if specs: specs_dict[tname] = np.stack(specs)
            except: continue
        print(f"  [ok] {sum(len(v) for v in specs_dict.values())} samples, {len(specs_dict)} types")
        return specs_dict


def load_drffr2_with_stft(h5_path, max_per_type=200):
    """Load DRFF-R2 raw IQ and convert through STFTEngine (same as RFUAV preprocessing)."""
    print(f"  [info] loading DRFF-R2 with proper STFT conversion...")
    stft_engine = STFTEngine(n_fft=1024, hop_len=256, win_len=1024)
    specs_dict = {}

    with h5py.File(h5_path, "r") as f:
        if "drones" not in f:
            print("  [error] no 'drones' group in DRFF-R2")
            return {}

        grp = f["drones"]
        type_names = sorted(list(grp.keys()))
        print(f"    types: {type_names}")

        for tname in type_names:
            try:
                type_grp = grp[tname]
                if not isinstance(type_grp, h5py.Group): continue
                sub_keys = sorted(list(type_grp.keys()))
                n_to_load = min(len(sub_keys), max_per_type)
                specs = []
                for sk in sub_keys[:n_to_load]:
                    raw = type_grp[sk][:]  # shape (3, 65536)
                    # Convert to complex IQ using channels 0,1
                    if raw.ndim == 2 and raw.shape[0] >= 2:
                        iq = raw[0] + 1j * raw[1]
                        # Apply STFT
                        spec = stft_engine(iq)
                        specs.append(spec)
                if specs:
                    specs_dict[tname] = np.stack(specs)
                    print(f"      {tname}: {len(specs)} samples (STFT converted)")
            except Exception as e:
                print(f"      {tname}: skipped ({e})")

    print(f"  [ok] {sum(len(v) for v in specs_dict.values())} DRFF-R2 samples, {len(specs_dict)} types")
    return specs_dict


def load_matched_bgs(matched_path, n=500):
    """Load matched backgrounds."""
    print(f"  [info] loading {n} matched BGs...")
    with h5py.File(matched_path, "r") as f:
        key = "holdout_matched_bg"
        if key not in f: return np.array([])
        grp = f[key]
        keys = sorted(list(grp.keys()), key=lambda x: int(x) if x.isdigit() else 0)
        n = min(len(keys), n)
        specs = [_norm(_prep_spectrogram(grp[keys[i]][:])) for i in range(n)]
        return np.stack(specs) if specs else np.array([])


def load_rfuav_negatives(h5_path, n=500, seed=42):
    """Load RFUAV /negatives/ (real WiFi/BT/environmental)."""
    print(f"  [info] loading {n} RFUAV negatives...")
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as f:
        if "negatives" not in f: return np.array([])
        neg_grp = f["negatives"]
        if isinstance(neg_grp, h5py.Dataset):
            n_total = neg_grp.shape[0]
            indices = rng.choice(n_total, min(n, n_total), replace=False)
            return np.stack([_norm(_prep_spectrogram(neg_grp[int(i)])) for i in indices])
        all_keys = list(neg_grp.keys())
        picked = rng.choice(all_keys, min(n, len(all_keys)), replace=False)
        specs = []
        for key in picked:
            try:
                item = neg_grp[str(key)]
                if isinstance(item, h5py.Dataset):
                    if len(item.shape) == 4: specs.append(_norm(_prep_spectrogram(item[0])))
                    elif len(item.shape) == 3: specs.append(_norm(_prep_spectrogram(item[:])))
            except: continue
        return np.stack(specs) if specs else np.array([])


def generate_realistic_hard_negatives(real_neg_specs, drone_specs, n=500, seed=42):
    """
    Generate REALISTIC hard negatives by spectrally shaping real RF noise
    to match drone spectral envelopes.

    Method: Take real WiFi/BT captures (which IRIS correctly rejects at 0% FPR),
    apply a spectral filter that matches the average drone spectral envelope.
    This creates realistic adversarial backgrounds that share drone spectral
    shape but have real RF noise temporal structure.

    This is a FAIR test — unlike permuted drone spectrograms, these are
    real signals with realistic temporal structure, just spectrally shaped.
    """
    print(f"  [info] generating {n} realistic spectrally-shaped hard negatives...")
    rng = np.random.default_rng(seed)

    # Compute average drone spectral envelope (channel 0 = log-power)
    all_drones = np.concatenate(list(drone_specs.values()))
    drone_envelope = all_drones[:, 0, :].mean(axis=0)  # (256, 256) avg power
    drone_envelope_1d = drone_envelope.mean(axis=1)  # (256,) freq profile
    drone_envelope_1d = np.maximum(drone_envelope_1d, 0)  # ensure non-negative

    hard_negs = []
    for i in range(n):
        # Pick a random real negative
        idx = rng.integers(0, len(real_neg_specs))
        neg = real_neg_specs[idx].copy()

        # Apply spectral shaping: scale each frequency bin by drone envelope
        for c in range(neg.shape[0]):
            # Get frequency profile of this negative
            neg_profile = neg[c].mean(axis=1)
            neg_profile = np.maximum(neg_profile, 0)

            # Scale to match drone envelope
            scale = drone_envelope_1d / (neg_profile + 1e-8)
            scale = np.clip(scale, 0.5, 2.0)  # limit shaping to avoid artifacts
            neg[c] = neg[c] * scale[:, np.newaxis]

        # Re-normalize
        neg = _norm(neg)
        hard_negs.append(neg)

    print(f"  [ok] generated {len(hard_negs)} realistic hard negatives")
    return np.stack(hard_negs)


@app.function(
    image=IMAGE, gpu="T4",
    volumes={"/data": VOL, "/models": MODEL_VOL, "/matched": MATCHED_VOL, "/results": RESULTS_VOL},
    timeout=5400, memory=16384,
)
def run_proper_tests():
    device = "cuda"
    print("=" * 70)
    print("IRIS — Proper Cross-Dataset + Artifact Tests (Fixed)")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
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

    # Load all data
    print("\n[1] Loading data...")
    rfuav_train = load_rfuav_split(H5_REMOTE, "train", max_per_type=500)
    rfuav_holdout = load_rfuav_split(H5_REMOTE, "holdout", max_per_type=500)
    drffr2 = load_drffr2_with_stft(DRFFR2_REMOTE, max_per_type=200)
    matched_bg = load_matched_bgs(MATCHED_REMOTE, n=500)
    real_negs = load_rfuav_negatives(H5_REMOTE, n=500)

    all_train = np.concatenate(list(rfuav_train.values()))
    all_holdout = np.concatenate(list(rfuav_holdout.values()))

    # Fit centroid on RFUAV train
    print("\n[2] Fitting Mahalanobis centroid on RFUAV train...")
    train_embs = encode_batch(encoder, all_train, device)
    centroid, cov_inv = fit_mahalanobis_l2(train_embs)
    train_dists = mahal_l2(train_embs, centroid, cov_inv)
    threshold = float(np.percentile(train_dists, 99))
    print(f"  [ok] threshold: {threshold:.2f}")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 1: Cross-Dataset Transfer (RFUAV → DRFF-R2, proper STFT)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 1: Cross-Dataset Transfer (RFUAV → DRFF-R2, PROPER STFT)")
    print("=" * 70)

    if drffr2:
        all_drffr2 = np.concatenate(list(drffr2.values()))
        print(f"  DRFF-R2: {len(all_drffr2)} samples, {len(drffr2)} types")

        # Encode
        print("  [info] encoding DRFF-R2...")
        drffr2_embs = encode_batch(encoder, all_drffr2, device)
        bg_embs = encode_batch(encoder, matched_bg, device)
        holdout_embs = encode_batch(encoder, all_holdout, device)

        # Distances
        drffr2_dists = mahal_l2(drffr2_embs, centroid, cov_inv)
        bg_dists = mahal_l2(bg_embs, centroid, cov_inv)
        holdout_dists = mahal_l2(holdout_embs, centroid, cov_inv)

        # AUCs
        labels_r = np.concatenate([np.ones(len(holdout_dists)), np.zeros(len(bg_dists))])
        dists_r = np.concatenate([holdout_dists, bg_dists])
        auc_rfuav = roc_auc_score(labels_r, -dists_r)

        labels_d = np.concatenate([np.ones(len(drffr2_dists)), np.zeros(len(bg_dists))])
        dists_d = np.concatenate([drffr2_dists, bg_dists])
        auc_drffr2 = roc_auc_score(labels_d, -dists_d)

        # Per-type DRFF-R2
        per_type = {}
        for tname, specs in drffr2.items():
            embs = encode_batch(encoder, specs, device)
            dists = mahal_l2(embs, centroid, cov_inv)
            labels = np.concatenate([np.ones(len(dists)), np.zeros(len(bg_dists))])
            d = np.concatenate([dists, bg_dists])
            per_type[tname] = {
                "n": len(specs),
                "auc": float(roc_auc_score(labels, -d)),
                "mean_dist": float(dists.mean()),
                "detection_rate": float((dists <= threshold).mean()),
            }

        print(f"\n  RESULTS:")
        print(f"    AUC (RFUAV holdout vs BG):  {auc_rfuav:.4f}")
        print(f"    AUC (DRFF-R2 vs BG):        {auc_drffr2:.4f}")
        print(f"    RFUAV holdout mean dist:    {holdout_dists.mean():.2f}")
        print(f"    DRFF-R2 drone mean dist:    {drffr2_dists.mean():.2f}")
        print(f"    BG mean dist:               {bg_dists.mean():.2f}")
        print(f"    DRFF-R2 detection rate:     {(drffr2_dists <= threshold).mean():.3f}")
        print(f"    RFUAV holdout detection:    {(holdout_dists <= threshold).mean():.3f}")
        print(f"\n    Per-type DRFF-R2:")
        for t, r in per_type.items():
            print(f"      {t:25s}: AUC={r['auc']:.4f}, det={r['detection_rate']:.3f}, dist={r['mean_dist']:.2f}")

        all_results["test_1_cross_dataset"] = {
            "auc_rfuav_holdout": float(auc_rfuav),
            "auc_drffr2": float(auc_drffr2),
            "rfuav_mean_dist": float(holdout_dists.mean()),
            "drffr2_mean_dist": float(drffr2_dists.mean()),
            "bg_mean_dist": float(bg_dists.mean()),
            "drffr2_detection_rate": float((drffr2_dists <= threshold).mean()),
            "rfuav_detection_rate": float((holdout_dists <= threshold).mean()),
            "per_type": per_type,
            "status": "PASS" if auc_drffr2 > 0.9 else ("PARTIAL" if auc_drffr2 > 0.7 else "FAIL"),
        }
    else:
        print("  [error] DRFF-R2 not loaded")
        all_results["test_1_cross_dataset"] = {"status": "ERROR", "reason": "DRFF-R2 not loaded"}

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 2: Realistic Spectral Artifact Check
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 2: Realistic Spectral Artifact Check")
    print("=" * 70)
    print("  Take REAL WiFi/BT captures, spectrally shape them to match drone envelope.")
    print("  If IRIS false-alarms → it's using spectral shape (artifact).")
    print("  If IRIS rejects → it learned temporal patterns.")

    if len(real_negs) > 0:
        # Generate realistic hard negatives
        hard_negs = generate_realistic_hard_negatives(real_negs, rfuav_holdout, n=500, seed=42)

        # Encode
        print("  [info] encoding hard negatives...")
        hard_neg_embs = encode_batch(encoder, hard_negs, device)
        real_neg_embs = encode_batch(encoder, real_negs, device)

        # Distances
        hard_neg_dists = mahal_l2(hard_neg_embs, centroid, cov_inv)
        real_neg_dists = mahal_l2(real_neg_embs, centroid, cov_inv)
        holdout_dists = mahal_l2(holdout_embs, centroid, cov_inv)

        # AUCs
        labels_h = np.concatenate([np.ones(len(holdout_dists)), np.zeros(len(hard_neg_dists))])
        dists_h = np.concatenate([holdout_dists, hard_neg_dists])
        auc_hard = roc_auc_score(labels_h, -dists_h)

        labels_r = np.concatenate([np.ones(len(holdout_dists)), np.zeros(len(real_neg_dists))])
        dists_r = np.concatenate([holdout_dists, real_neg_dists])
        auc_real = roc_auc_score(labels_r, -dists_r)

        # Detection rates
        hard_fp = (hard_neg_dists <= threshold).mean()
        real_fp = (real_neg_dists <= threshold).mean()
        drone_det = (holdout_dists <= threshold).mean()

        print(f"\n  RESULTS:")
        print(f"    AUC (drones vs spectrally-shaped real RF): {auc_hard:.4f}")
        print(f"    AUC (drones vs real RF):                   {auc_real:.4f}")
        print(f"    Drone detection rate:    {drone_det:.3f}")
        print(f"    Hard neg FP rate:        {hard_fp:.3f}")
        print(f"    Real RF FP rate:         {real_fp:.3f}")
        print(f"    Hard neg mean dist:      {hard_neg_dists.mean():.2f}")
        print(f"    Drone mean dist:         {holdout_dists.mean():.2f}")
        print(f"    Real neg mean dist:      {real_neg_dists.mean():.2f}")

        print(f"\n  INTERPRETATION:")
        if auc_hard > 0.95 and hard_fp < 0.1:
            print(f"    ✅ IRIS learned REAL temporal patterns — spectral shaping doesn't fool it")
        elif auc_hard > 0.85:
            print(f"    ⚠ Partial — IRIS uses both temporal and spectral features")
            print(f"    Spectral shaping increases FP from {real_fp:.1%} to {hard_fp:.1%}")
        else:
            print(f"    ❌ IRIS relies on spectral shape — shaped RF defeats it")

        all_results["test_2_spectral_artifact"] = {
            "auc_hard_negatives": float(auc_hard),
            "auc_real_negatives": float(auc_real),
            "hard_neg_fp_rate": float(hard_fp),
            "real_neg_fp_rate": float(real_fp),
            "drone_detection_rate": float(drone_det),
            "hard_neg_mean_dist": float(hard_neg_dists.mean()),
            "drone_mean_dist": float(holdout_dists.mean()),
            "real_neg_mean_dist": float(real_neg_dists.mean()),
            "verdict": "PASS — learned temporal patterns" if auc_hard > 0.95 and hard_fp < 0.1 else
                       "PARTIAL — uses both" if auc_hard > 0.85 else
                       "FAIL — spectral artifact",
        }
    else:
        all_results["test_2_spectral_artifact"] = {"status": "SKIPPED", "reason": "no real negatives"}

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 3: Cross-Dataset Per-Transmitter IFF (if DRFF-R2 has individual units)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 3: DRFF-R2 Individual Unit Analysis")
    print("=" * 70)
    print("  DRFF-R2 has individual units (e.g., mini4PRO_u1, mini5PRO_u1)")
    print("  Check if IRIS can distinguish individual units of the same model.")

    if drffr2:
        # Group by model (strip _uN suffix)
        model_groups = {}
        for tname in drffr2.keys():
            # e.g., "mini4PRO_u1" → model="mini4PRO", unit="u1"
            parts = tname.rsplit("_", 1)
            if len(parts) == 2:
                model, unit = parts
                if model not in model_groups:
                    model_groups[model] = []
                model_groups[model].append((tname, unit))

        print(f"  Models with multiple units:")
        for model, units in model_groups.items():
            if len(units) > 1:
                print(f"    {model}: {len(units)} units — {units}")

        # For models with multiple units, check if embeddings cluster by unit
        unit_separation = {}
        for model, units in model_groups.items():
            if len(units) < 2: continue
            unit_embs = {}
            for tname, unit in units:
                specs = drffr2[tname]
                embs = encode_batch(encoder, specs, device)
                unit_embs[unit] = embs

            # Compute mean embedding per unit
            unit_means = {u: e.mean(axis=0) for u, e in unit_embs.items()}
            # Compute pairwise cosine similarity between unit means
            unit_list = list(unit_means.keys())
            sims = []
            for i in range(len(unit_list)):
                for j in range(i+1, len(unit_list)):
                    u1, u2 = unit_list[i], unit_list[j]
                    e1, e2 = unit_means[u1], unit_means[u2]
                    sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
                    sims.append(sim)
                    print(f"    {model} {u1} vs {u2}: cosine sim = {sim:.4f}")

            if sims:
                unit_separation[model] = {
                    "n_units": len(unit_list),
                    "mean_similarity": float(np.mean(sims)),
                    "min_similarity": float(np.min(sims)),
                    "max_similarity": float(np.max(sims)),
                }

        all_results["test_3_individual_units"] = {
            "models_with_multiple_units": {m: len(u) for m, u in model_groups.items() if len(u) > 1},
            "unit_separation": unit_separation,
            "verdict": "Individual units distinguishable" if any(s["mean_similarity"] < 0.95 for s in unit_separation.values()) else "Units not distinguishable",
        }
    else:
        all_results["test_3_individual_units"] = {"status": "SKIPPED"}

    # ═══════════════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    os.makedirs(RESULTS_REMOTE, exist_ok=True)
    all_results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    json_path = f"{RESULTS_REMOTE}/proper_cross_dataset.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  [ok] saved {json_path}")

    # Markdown report
    md_path = f"{RESULTS_REMOTE}/proper_cross_dataset.md"
    with open(md_path, "w") as f:
        f.write("# IRIS — Proper Cross-Dataset + Artifact Tests (Fixed)\n\n")
        f.write(f"**Generated:** {all_results['timestamp']}\n\n")

        f.write("## Test 1: Cross-Dataset Transfer (RFUAV → DRFF-R2, proper STFT)\n\n")
        t1 = all_results.get("test_1_cross_dataset", {})
        f.write(f"**Status: {t1.get('status', 'UNKNOWN')}**\n\n")
        f.write(f"DRFF-R2 raw IQ converted through STFTEngine (log-power + phase, same as RFUAV).\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| AUC (RFUAV holdout vs BG) | {t1.get('auc_rfuav_holdout', 0):.4f} |\n")
        f.write(f"| AUC (DRFF-R2 vs BG) | {t1.get('auc_drffr2', 0):.4f} |\n")
        f.write(f"| RFUAV holdout mean dist | {t1.get('rfuav_mean_dist', 0):.2f} |\n")
        f.write(f"| DRFF-R2 drone mean dist | {t1.get('drffr2_mean_dist', 0):.2f} |\n")
        f.write(f"| DRFF-R2 detection rate | {t1.get('drffr2_detection_rate', 0):.3f} |\n\n")

        if "per_type" in t1:
            f.write("Per-type DRFF-R2:\n\n")
            f.write("| Type | N | AUC | Detection | Mean Dist |\n|---|---|---|---|---|\n")
            for t, r in t1["per_type"].items():
                f.write(f"| {t} | {r['n']} | {r['auc']:.4f} | {r['detection_rate']:.3f} | {r['mean_dist']:.2f} |\n")

        f.write("\n## Test 2: Realistic Spectral Artifact Check\n\n")
        t2 = all_results.get("test_2_spectral_artifact", {})
        f.write(f"**Verdict: {t2.get('verdict', 'UNKNOWN')}**\n\n")
        f.write("Real WiFi/BT captures spectrally shaped to match drone envelope.\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| AUC (drones vs shaped RF) | {t2.get('auc_hard_negatives', 0):.4f} |\n")
        f.write(f"| AUC (drones vs real RF) | {t2.get('auc_real_negatives', 0):.4f} |\n")
        f.write(f"| Hard neg FP rate | {t2.get('hard_neg_fp_rate', 0):.3f} |\n")
        f.write(f"| Real RF FP rate | {t2.get('real_neg_fp_rate', 0):.3f} |\n\n")

        f.write("## Test 3: DRFF-R2 Individual Unit Analysis\n\n")
        t3 = all_results.get("test_3_individual_units", {})
        f.write(f"**Verdict: {t3.get('verdict', 'UNKNOWN')}**\n\n")
        if "unit_separation" in t3:
            f.write("| Model | Units | Mean Sim | Min Sim | Max Sim |\n|---|---|---|---|---|\n")
            for m, s in t3["unit_separation"].items():
                f.write(f"| {m} | {s['n_units']} | {s['mean_similarity']:.4f} | {s['min_similarity']:.4f} | {s['max_similarity']:.4f} |\n")

    print(f"  [ok] saved {md_path}")
    RESULTS_VOL.commit()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
    for k, v in all_results.items():
        if isinstance(v, dict):
            status = v.get("status") or v.get("verdict") or "UNKNOWN"
            print(f"  {k}: {status}")

    return all_results


@app.local_entrypoint()
def main():
    run_proper_tests.remote()


if __name__ == "__main__":
    main()
