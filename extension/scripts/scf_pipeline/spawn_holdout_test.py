"""Modal script: Test the trained encoder on additional holdouts.

Test sets:
  1. holdout_original (3,256,256) — likely Iris-style holdout (could be drones)
  2. holdout_matched_bg (3,256,256) — known BG, used as baseline
  3. Mix: BG + DRFF-R2 at various ratios to simulate real-world deployment

Reuses the encoder from /models/rf_scf_real_encoder_seed42.pt
"""
import modal, os

app = modal.App("iris-cuas-holdout-test")

DATA_VOL = modal.Volume.from_name("iris-cuas-data", create_if_missing=True)
RESULTS_VOL = modal.Volume.from_name("iris-cuas-results", create_if_missing=True)
MODELS_VOL = modal.Volume.from_name("iris-cuas-models", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install("libgl1", "libglib2.0-0", "libhdf5-dev", "python3", "python3-pip", "python-is-python3")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "h5py==3.12.1", "numpy==1.26.4",
                 "scikit-learn==1.6.1", "scipy==1.14.1")
)

CORE = r'''
"""Test trained encoder on additional holdouts."""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py


# ============ Model components (must match training) ============

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


# ============ SCF (for DRFF-R2 IQ → SCF conversion) ============

def iq_to_scf_image(iq, out_size=256, n_fft=1<<12, alpha_max=0.5, window_len=128, n_alpha=128):
    z = np.asarray(iq, dtype=np.complex128)
    N = len(z)
    if N < n_fft:
        z = np.concatenate([z, np.zeros(n_fft - N, dtype=z.dtype)])
    else:
        z = z[:n_fft]
    N = n_fft
    z = z * np.hanning(N)
    X = np.fft.fftshift(np.fft.fft(z))
    alphas = np.linspace(0.0, alpha_max, n_alpha)
    n_freq = max(N // window_len, 1)
    win = np.hanning(window_len)
    SCF = np.zeros((n_alpha, n_freq), dtype=np.complex128)
    Sx = np.abs(X) ** 2
    for i, a in enumerate(alphas):
        shift = int(round(a * N / 2.0))
        scf_slice = np.roll(X, -shift) * np.conj(np.roll(X, shift))
        SCF[i, :] = np.convolve(scf_slice, win, mode="same")[::window_len][:n_freq]
    SCF[0, :] = 0
    eps = 1e-12 * (Sx.max() + 1e-30)
    COH = np.zeros((n_alpha, n_freq), dtype=np.float64)
    for i, a in enumerate(alphas):
        shift = int(round(a * len(X) / 2.0))
        Splus = np.convolve(np.roll(Sx, -shift), win, mode="same")[::window_len][:n_freq]
        Sminus = np.convolve(np.roll(Sx, shift), win, mode="same")[::window_len][:n_freq]
        denom = np.sqrt(Splus * Sminus) + eps
        COH[i, :] = np.abs(SCF[i, :]) / denom
    COH = np.clip(COH, 0.0, 1.0)
    ch0 = np.log10(np.abs(SCF) + 1e-12).astype(np.float64)
    ch1 = COH.astype(np.float64)
    img = np.stack([ch0, ch1], axis=0)
    C, H, W = img.shape
    if H != out_size or W != out_size:
        t = torch.from_numpy(img).float().unsqueeze(0)
        t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
        img = t.squeeze(0).numpy()
    for c in range(img.shape[0]):
        mu, sd = img[c].mean(), img[c].std() + 1e-8
        img[c] = (img[c] - mu) / sd
    return img.astype(np.float32)


# ============ Helpers ============

@torch.no_grad()
def encode(encoder, specs, device, bs=32):
    encoder.eval()
    all_embs = []
    for i in range(0, len(specs), bs):
        batch = torch.from_numpy(specs[i:i+bs]).float().to(device)
        all_embs.append(encoder(batch).cpu().numpy())
    return np.concatenate(all_embs)

def fit_mahal(embs, reg=1e-3):
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


def normalize_to_2ch(arr):
    """Take a (N, 3, H, W) or (N, 2, H, W) array and return (N, 2, 256, 256) normalized."""
    if arr.ndim == 3:
        arr = arr[None]
    N = arr.shape[0]
    # Take first 2 channels
    if arr.shape[1] >= 2:
        arr = arr[:, :2]
    # Resize if needed
    if arr.shape[2] != 256 or arr.shape[3] != 256:
        out = np.empty((N, 2, 256, 256), dtype=np.float32)
        t_arr = torch.from_numpy(arr).float().unsqueeze(0) if N == 1 else torch.from_numpy(arr).float()
        t_arr = F.interpolate(t_arr, size=(256, 256), mode="bilinear", align_corners=False)
        arr = t_arr.numpy()
    # Normalize per-channel
    for i in range(N):
        for c in range(2):
            mu, sd = arr[i, c].mean(), arr[i, c].std() + 1e-8
            arr[i, c] = (arr[i, c] - mu) / sd
    return arr.astype(np.float32)


def main():
    device = "cuda"
    ZENODO_H5 = "/data/zenodo_scf_samples.h5"
    DRFFR2_H5 = "/data/data/drffr2.h5"
    BG_H5 = "/data/data/iris_matched_bg.h5"
    MODEL_PATH = "/models/rf_scf_real_encoder_seed42.pt"

    print("="*70, flush=True)
    print("=== Holdout Testing ===", flush=True)
    print("="*70, flush=True)

    # 1. Load trained encoder
    print("\n[1] Loading trained encoder...", flush=True)
    encoder = CNNEncoder(in_ch=2, embed_dim=256).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    print(f"  Loaded from {MODEL_PATH}", flush=True)

    # 2. Reload Zenodo SCF for fitting Mahalanobis (same as training)
    print("\n[2] Loading Zenodo SCF for Mahalanobis fit...", flush=True)
    with h5py.File(ZENODO_H5, "r") as f:
        zenodo_scf = f["images"][:].astype(np.float32)
    print(f"  Zenodo: {len(zenodo_scf)} samples", flush=True)

    td_embs = encode(encoder, zenodo_scf, device)
    centroid, cov_inv = fit_mahal(td_embs)
    td_dists = mahal_l2(td_embs, centroid, cov_inv)
    thresh = float(np.percentile(td_dists, 99))
    print(f"  Threshold (99p): {thresh:.4f}", flush=True)
    print(f"  Train drone dist mean: {td_dists.mean():.4f}", flush=True)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model": MODEL_PATH,
        "threshold": thresh,
        "train_drone_dist_mean": float(td_dists.mean()),
        "tests": {}
    }

    # 3. Test 1: BG holdout_matched_bg (same BG distribution as training BG)
    print("\n[3] Test 1: BG holdout_matched_bg (in-distribution BG)...", flush=True)
    with h5py.File(BG_H5, "r") as f:
        bg_keys = sorted(list(f["holdout_matched_bg"].keys()),
                         key=lambda x: int(x) if x.isdigit() else 0)
        bg_test_keys = bg_keys[600:1100]  # 500 fresh BG samples
        bg_test_arr = np.stack([f["holdout_matched_bg"][k][:] for k in bg_test_keys])
    bg_test = normalize_to_2ch(bg_test_arr)
    bg_embs = encode(encoder, bg_test, device)
    bg_dists = mahal_l2(bg_embs, centroid, cov_inv)
    bg_fp = float((bg_dists <= thresh).mean())
    print(f"  N={len(bg_test)}  FP={bg_fp:.4f}  dist_mean={bg_dists.mean():.4f}", flush=True)
    results["tests"]["bg_holdout_matched"] = {
        "n_samples": len(bg_test),
        "fp_rate": bg_fp,
        "dist_mean": float(bg_dists.mean()),
        "dist_median": float(np.median(bg_dists)),
        "dist_p99": float(np.percentile(bg_dists, 99)),
    }

    # 4. Test 2: BG holdout_original (potentially different distribution)
    print("\n[4] Test 2: BG holdout_original (potentially OOD BG)...", flush=True)
    with h5py.File(BG_H5, "r") as f:
        ho_keys = sorted(list(f["holdout_original"].keys()),
                         key=lambda x: int(x) if x.isdigit() else 0)
        ho_test_keys = ho_keys[:500]  # first 500
        ho_test_arr = np.stack([f["holdout_original"][k][:] for k in ho_test_keys])
    ho_test = normalize_to_2ch(ho_test_arr)
    ho_embs = encode(encoder, ho_test, device)
    ho_dists = mahal_l2(ho_embs, centroid, cov_inv)
    ho_detection = float((ho_dists <= thresh).mean())  # if these are drones, high detection is good
    print(f"  N={len(ho_test)}  detection={ho_detection:.4f}  dist_mean={ho_dists.mean():.4f}", flush=True)
    results["tests"]["holdout_original"] = {
        "n_samples": len(ho_test),
        "detection_rate": ho_detection,  # could be FP if BG, or DET if drones
        "dist_mean": float(ho_dists.mean()),
        "dist_median": float(np.median(ho_dists)),
        "dist_p10": float(np.percentile(ho_dists, 10)),
        "dist_p90": float(np.percentile(ho_dists, 90)),
    }

    # 5. Test 3: DRFF-R2 (the main OOD drone holdout — re-confirm)
    print("\n[5] Test 3: DRFF-R2 (OOD drone holdout)...", flush=True)
    drffr2_scf = {}
    with h5py.File(DRFFR2_H5, "r") as f:
        drones = f["drones"]
        for tname in sorted(drones.keys()):
            type_grp = drones[tname]
            specs = []
            for sk in sorted(type_grp.keys()):
                raw = type_grp[sk][:]
                if raw.ndim == 2 and raw.shape[0] >= 2 and raw.shape[1] >= 4096:
                    iq = raw[0].astype(np.complex128) + 1j * raw[1].astype(np.complex128)
                    iq = iq / max(np.abs(iq).max(), 1.0)
                    specs.append(iq_to_scf_image(iq))
            if specs:
                drffr2_scf[tname] = np.stack(specs)

    all_drffr2_dists = []
    per_type = {}
    for tname in sorted(drffr2_scf.keys()):
        embs = encode(encoder, drffr2_scf[tname], device)
        dists = mahal_l2(embs, centroid, cov_inv)
        all_drffr2_dists.append(dists)
        det = float((dists <= thresh).mean())
        per_type[tname] = {"n": len(dists), "det": det, "dist_mean": float(dists.mean())}
        print(f"    {tname:20s}  n={len(dists):>3d}  det={det:.3f}  dist_mean={dists.mean():.2f}", flush=True)
    all_drffr2 = np.concatenate(all_drffr2_dists)
    drffr2_det = float((all_drffr2 <= thresh).mean())
    print(f"  Overall: N={len(all_drffr2)}  det={drffr2_det:.4f}  dist_mean={all_drffr2.mean():.4f}", flush=True)
    results["tests"]["drffr2"] = {
        "n_samples": int(len(all_drffr2)),
        "detection_rate": drffr2_det,
        "dist_mean": float(all_drffr2.mean()),
        "per_type": per_type,
    }

    # 6. Test 4: Mixed scenario (BG + drones at 50/50 ratio)
    print("\n[6] Test 4: Mixed scenario (500 BG + 500 DRFF-R2)...", flush=True)
    # Take 500 BG (different from test 1)
    with h5py.File(BG_H5, "r") as f:
        bg_keys = sorted(list(f["holdout_matched_bg"].keys()),
                         key=lambda x: int(x) if x.isdigit() else 0)
        bg_mix_keys = bg_keys[1100:1600]
        bg_mix_arr = np.stack([f["holdout_matched_bg"][k][:] for k in bg_mix_keys])
    bg_mix = normalize_to_2ch(bg_mix_arr)
    # Take 500 DRFF-R2 samples (balanced across types)
    drffr2_mix = []
    for tname in sorted(drffr2_scf.keys()):
        n_take = min(len(drffr2_scf[tname]), 500 // len(drffr2_scf))
        drffr2_mix.append(drffr2_scf[tname][:n_take])
    drffr2_mix = np.concatenate(drffr2_mix)[:500]

    bg_mix_embs = encode(encoder, bg_mix, device)
    drffr2_mix_embs = encode(encoder, drffr2_mix, device)
    bg_mix_dists = mahal_l2(bg_mix_embs, centroid, cov_inv)
    drffr2_mix_dists = mahal_l2(drffr2_mix_embs, centroid, cov_inv)

    # Compute precision/recall at threshold
    y_true = np.concatenate([np.zeros(len(bg_mix_dists)), np.ones(len(drffr2_mix_dists))])
    y_pred = np.concatenate([bg_mix_dists, drffr2_mix_dists]) <= thresh
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}", flush=True)
    print(f"  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}", flush=True)
    results["tests"]["mixed_50_50"] = {
        "n_bg": len(bg_mix),
        "n_drone": len(drffr2_mix),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # 7. Test 5: SNR-stress test — add noise to DRFF-R2 samples at various SNRs
    print("\n[7] Test 5: SNR stress test (DRFF-R2 + noise)...", flush=True)
    # Take a fixed set of 100 DRFF-R2 samples
    test_set = []
    test_labels = []
    for tname in sorted(drffr2_scf.keys()):
        n_take = min(len(drffr2_scf[tname]), 100 // len(drffr2_scf))
        test_set.append(drffr2_scf[tname][:n_take])
    test_set = np.concatenate(test_set)[:100]

    # Reload raw IQ to add noise BEFORE SCF
    snr_results = {}
    for snr_db in [0, 5, 10, 15, 20, 30]:
        # Generate noisy versions
        noisy_scf = []
        rng = np.random.RandomState(42)
        with h5py.File(DRFFR2_H5, "r") as f:
            drones = f["drones"]
            all_iq = []
            for tname in sorted(drones.keys()):
                type_grp = drones[tname]
                keys = sorted(type_grp.keys())
                for sk in keys[:13]:  # ~100 total
                    raw = type_grp[sk][:]
                    if raw.ndim == 2 and raw.shape[0] >= 2:
                        iq = raw[0].astype(np.complex128) + 1j * raw[1].astype(np.complex128)
                        iq = iq / max(np.abs(iq).max(), 1.0)
                        # Add noise at SNR
                        sig_pwr = np.mean(np.abs(iq)**2)
                        noise_pwr = sig_pwr / (10 ** (snr_db / 10))
                        noise = np.sqrt(noise_pwr / 2) * (rng.randn(len(iq)) + 1j * rng.randn(len(iq)))
                        noisy = iq + noise
                        noisy_scf.append(iq_to_scf_image(noisy))
        noisy_scf = np.stack(noisy_scf)
        embs = encode(encoder, noisy_scf, device)
        dists = mahal_l2(embs, centroid, cov_inv)
        det = float((dists <= thresh).mean())
        snr_results[f"snr_{snr_db}db"] = {
            "n": len(noisy_scf),
            "det": det,
            "dist_mean": float(dists.mean()),
        }
        print(f"  SNR {snr_db:>3d} dB: N={len(noisy_scf)}  det={det:.4f}  dist_mean={dists.mean():.2f}", flush=True)
    results["tests"]["snr_stress"] = snr_results

    # 8. Save
    print("\n[8] Saving...", flush=True)
    json_path = "/results/rf_scf_real_holdout_tests.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved {json_path}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("HOLDOUT TEST SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  BG (matched, fresh 500):       FP={bg_fp:.4f}  (target <0.05)", flush=True)
    print(f"  BG (holdout_original, 500):    det={ho_detection:.4f}  (interpretation depends on data)", flush=True)
    print(f"  DRFF-R2 (full 1000):           det={drffr2_det:.4f}  (target >0.50)", flush=True)
    print(f"  Mixed 50/50 (500+500):         F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}", flush=True)
    print(f"  SNR stress: 0dB={snr_results['snr_0db']['det']:.3f}  "
          f"10dB={snr_results['snr_10db']['det']:.3f}  "
          f"30dB={snr_results['snr_30db']['det']:.3f}", flush=True)

    return results
'''

CORE_PATH = "/tmp/holdout_test_core.py"
with open(CORE_PATH, "w") as f:
    f.write(CORE)

IMAGE = IMAGE.add_local_file(CORE_PATH, "/root/holdout_test_core.py")


@app.function(
    image=IMAGE,
    gpu="T4",
    volumes={"/data": DATA_VOL, "/results": RESULTS_VOL, "/models": MODELS_VOL},
    timeout=1800,
    memory=16384,
)
def launch():
    import sys
    sys.path.insert(0, "/root")
    from holdout_test_core import main
    return main()


if __name__ == "__main__":
    with app.run(detach=True):
        fc = launch.spawn()
        print(f"SPAWNED: {fc.object_id}")
        print(f"App: {app.name}")
        print(f"Track: https://modal.com/logs/call/{fc.object_id}")
        print("=" * 60)
