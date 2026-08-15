"""Modal: Extended OOD test on v1 encoder.

Tests:
  1. Cross-validation across Zenodo drone types (leave-one-type-out)
  2. ROC curve at many BG/drone ratios
  3. SNR sweep at finer granularity (0, 2, 4, 6, 8, 10, 15, 20, 25, 30 dB)
  4. Pure BG-only large-scale FP test (1000+ BG samples)
  5. BG distribution shift: holdout_original vs holdout_matched_bg
"""
import modal, os

app = modal.App("iris-cuas-extended-ood-test")

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
"""Extended OOD test on v1 encoder."""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from sklearn.metrics import roc_auc_score, roc_curve


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
    if arr.ndim == 3: arr = arr[None]
    N = arr.shape[0]
    if arr.shape[1] >= 2: arr = arr[:, :2]
    if arr.shape[2] != 256 or arr.shape[3] != 256:
        t = torch.from_numpy(arr).float()
        t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
        arr = t.numpy()
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
    print("=== Extended OOD Test on v1 encoder ===", flush=True)
    print("="*70, flush=True)

    # Load v1 encoder
    print("\n[1] Loading v1 encoder...", flush=True)
    encoder = CNNEncoder(in_ch=2, embed_dim=256).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    # Load Zenodo SCF (with type labels)
    print("\n[2] Loading Zenodo SCF...", flush=True)
    with h5py.File(ZENODO_H5, "r") as f:
        zenodo_scf = f["images"][:].astype(np.float32)
        zenodo_types = [t.decode() for t in f["types"][:]]
    print(f"  {len(zenodo_scf)} samples, {len(set(zenodo_types))} types", flush=True)

    # Fit Mahalanobis on all Zenodo
    td_embs = encode(encoder, zenodo_scf, device)
    centroid, cov_inv = fit_mahal(td_embs)
    td_dists = mahal_l2(td_embs, centroid, cov_inv)
    thresh = float(np.percentile(td_dists, 99))
    print(f"  Threshold (99p): {thresh:.4f}", flush=True)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model": MODEL_PATH,
        "threshold_99p": thresh,
        "tests": {}
    }

    # ============ TEST 1: Leave-one-type-out cross-validation ============
    print("\n[3] Test 1: Leave-one-type-out (LOTO) cross-validation on Zenodo...", flush=True)
    unique_types = sorted(set(zenodo_types))
    loto_results = {}
    for holdout_type in unique_types:
        train_idx = [i for i, t in enumerate(zenodo_types) if t != holdout_type]
        test_idx = [i for i, t in enumerate(zenodo_types) if t == holdout_type]
        train_embs = td_embs[train_idx]
        test_embs = td_embs[test_idx]
        # Fit Mahalanobis on training subset only
        c, ci = fit_mahal(train_embs)
        train_dists = mahal_l2(train_embs, c, ci)
        t = float(np.percentile(train_dists, 99))
        test_dists = mahal_l2(test_embs, c, ci)
        det = float((test_dists <= t).mean())
        loto_results[holdout_type] = {
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "threshold": t,
            "det_rate": det,
            "test_dist_mean": float(test_dists.mean()),
            "train_dist_mean": float(train_dists.mean()),
        }
        print(f"    Hold out {holdout_type:25s}: n_test={len(test_idx):>4d}  "
              f"det={det:.4f}  dist={test_dists.mean():.2f}", flush=True)
    avg_loto = float(np.mean([r["det_rate"] for r in loto_results.values()]))
    print(f"  LOTO avg det: {avg_loto:.4f}", flush=True)
    results["tests"]["loto_cross_val"] = {
        "avg_det_rate": avg_loto,
        "per_type": loto_results,
    }

    # ============ TEST 2: ROC curve at various BG/drone ratios ============
    print("\n[4] Test 2: ROC curve (BG vs DRFF-R2)...", flush=True)
    # Get BG eval distances
    with h5py.File(BG_H5, "r") as f:
        bg_keys = sorted(list(f["holdout_matched_bg"].keys()),
                         key=lambda x: int(x) if x.isdigit() else 0)
        bg_keys = bg_keys[1100:1600]  # 500 fresh BG
        bg_arr = np.stack([f["holdout_matched_bg"][k][:] for k in bg_keys])
    bg_test = normalize_to_2ch(bg_arr)
    bg_embs = encode(encoder, bg_test, device)
    bg_dists = mahal_l2(bg_embs, centroid, cov_inv)

    # Get DRFF-R2 distances
    with h5py.File(DRFFR2_H5, "r") as f:
        drones = f["drones"]
        all_drffr2_iq = []
        for tname in sorted(drones.keys()):
            type_grp = drones[tname]
            for sk in sorted(type_grp.keys()):
                raw = type_grp[sk][:]
                if raw.ndim == 2 and raw.shape[0] >= 2:
                    iq = raw[0].astype(np.complex128) + 1j * raw[1].astype(np.complex128)
                    iq = iq / max(np.abs(iq).max(), 1.0)
                    all_drffr2_iq.append(iq)
    # Compute SCF for first 200 (to keep time reasonable)
    drffr2_scf = np.stack([iq_to_scf_image(iq) for iq in all_drffr2_iq[:200]])
    drffr2_embs = encode(encoder, drffr2_scf, device)
    drffr2_dists = mahal_l2(drffr2_embs, centroid, cov_inv)

    # Compute ROC
    y_true = np.concatenate([np.zeros(len(bg_dists)), np.ones(len(drffr2_dists))])
    y_scores = -np.concatenate([bg_dists, drffr2_dists])  # higher = more drone-like
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = float(roc_auc_score(y_true, y_scores))

    # Find threshold at various operating points
    operating_points = {}
    for target_fpr in [0.001, 0.005, 0.01, 0.05, 0.1]:
        idx = np.argmin(np.abs(fpr - target_fpr))
        operating_points[f"fpr_{target_fpr}"] = {
            "threshold": float(thresholds[idx]),
            "fpr": float(fpr[idx]),
            "tpr": float(tpr[idx]),
        }
        print(f"    FPR={target_fpr}: TPR={tpr[idx]:.4f}  threshold={thresholds[idx]:.4f}", flush=True)

    print(f"  AUC: {auc:.4f}", flush=True)
    results["tests"]["roc_curve"] = {
        "auc": auc,
        "operating_points": operating_points,
        "n_bg": len(bg_dists),
        "n_drone": len(drffr2_dists),
    }

    # ============ TEST 3: Fine-grained SNR sweep ============
    print("\n[5] Test 3: Fine-grained SNR sweep (0-30 dB)...", flush=True)
    snr_results = {}
    rng = np.random.RandomState(42)
    # Use 50 DRFF-R2 samples per SNR
    test_iq_set = all_drffr2_iq[:50]
    for snr_db in [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30]:
        noisy_scf = []
        for iq in test_iq_set:
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
        print(f"    SNR {snr_db:>3d} dB: det={det:.4f}  dist_mean={dists.mean():.2f}", flush=True)
    results["tests"]["snr_sweep_fine"] = snr_results

    # ============ TEST 4: Large-scale pure BG FP test ============
    print("\n[6] Test 4: Large-scale pure BG test (1000 samples)...", flush=True)
    with h5py.File(BG_H5, "r") as f:
        bg_keys_all = sorted(list(f["holdout_matched_bg"].keys()),
                             key=lambda x: int(x) if x.isdigit() else 0)
        bg_big_keys = bg_keys_all[1600:2600]  # 1000 fresh BG
        bg_big_arr = np.stack([f["holdout_matched_bg"][k][:] for k in bg_big_keys])
    bg_big = normalize_to_2ch(bg_big_arr)
    bg_big_embs = encode(encoder, bg_big, device)
    bg_big_dists = mahal_l2(bg_big_embs, centroid, cov_inv)
    bg_big_fp = float((bg_big_dists <= thresh).mean())
    print(f"  N=1000  FP={bg_big_fp:.4f}  dist_mean={bg_big_dists.mean():.4f}", flush=True)
    results["tests"]["large_bg_pure"] = {
        "n_samples": 1000,
        "fp_rate": bg_big_fp,
        "dist_mean": float(bg_big_dists.mean()),
        "dist_p99": float(np.percentile(bg_big_dists, 99)),
        "dist_max": float(bg_big_dists.max()),
    }

    # ============ TEST 5: BG distribution shift ============
    print("\n[7] Test 5: BG distribution shift (holdout_original)...", flush=True)
    with h5py.File(BG_H5, "r") as f:
        ho_keys = sorted(list(f["holdout_original"].keys()),
                         key=lambda x: int(x) if x.isdigit() else 0)
        ho_keys = ho_keys[500:1000]  # 500 (different from earlier test)
        ho_arr = np.stack([f["holdout_original"][k][:] for k in ho_keys])
    ho_test = normalize_to_2ch(ho_arr)
    ho_embs = encode(encoder, ho_test, device)
    ho_dists = mahal_l2(ho_embs, centroid, cov_inv)
    ho_det = float((ho_dists <= thresh).mean())
    print(f"  N=500  det={ho_det:.4f}  dist_mean={ho_dists.mean():.4f}", flush=True)
    results["tests"]["bg_distribution_shift"] = {
        "n_samples": 500,
        "det_rate": ho_det,  # interpretation: low = good (looks like BG)
        "dist_mean": float(ho_dists.mean()),
    }

    # ============ SAVE ============
    print("\n[8] Saving...", flush=True)
    json_path = "/results/rf_scf_v1_extended_ood.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved {json_path}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"EXTENDED OOD TEST SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  LOTO cross-val avg:    {avg_loto:.4f}  (target >0.9)", flush=True)
    print(f"  ROC AUC:               {auc:.4f}", flush=True)
    print(f"  ROC @ FPR=0.01:        TPR={operating_points['fpr_0.01']['tpr']:.4f}", flush=True)
    print(f"  ROC @ FPR=0.001:       TPR={operating_points['fpr_0.001']['tpr']:.4f}", flush=True)
    print(f"  SNR sweep 0-30 dB:     min det = {min(r['det'] for r in snr_results.values()):.4f}", flush=True)
    print(f"  Large-scale BG FP:     {bg_big_fp:.4f}  (over 1000 samples)", flush=True)
    print(f"  BG distribution shift: {ho_det:.4f}  (target low)", flush=True)

    return results
'''

CORE_PATH = "/tmp/extended_ood_core.py"
with open(CORE_PATH, "w") as f:
    f.write(CORE)

IMAGE = IMAGE.add_local_file(CORE_PATH, "/root/extended_ood_core.py")


@app.function(
    image=IMAGE, gpu="T4",
    volumes={"/data": DATA_VOL, "/results": RESULTS_VOL, "/models": MODELS_VOL},
    timeout=1800, memory=16384,
)
def launch():
    import sys; sys.path.insert(0, "/root")
    from extended_ood_core import main
    return main()


if __name__ == "__main__":
    with app.run(detach=True):
        fc = launch.spawn()
        print(f"SPAWNED: {fc.object_id}")
        print(f"App: {app.name}")
        print("=" * 60)
