"""Modal training: train encoder on expanded Zenodo SCF (15k samples).
Evaluate on DRFF-R2 holdout.

If /data/zenodo_scf_samples_v2.h5 is corrupt/missing, regenerate it inline
from the raw .bin files (using same SCF function).
"""
import modal, os

app = modal.App("iris-cuas-rf-scf-real-v2")

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
"""Train encoder on expanded Zenodo SCF (15k samples); eval on DRFF-R2."""
import json, os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


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

class SIGRegLoss(nn.Module):
    def __init__(self, embed_dim=256, k=256, seed=42):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        W = torch.randn(k, embed_dim, generator=gen)
        W = W / W.norm(dim=1, keepdim=True)
        self.register_buffer("W", W)
    def forward(self, z):
        p = F.linear(z, self.W)
        return ((p.var(dim=0) - 1.0) ** 2).mean()

class Head(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


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

def collapse_metrics(embs):
    N, D = embs.shape
    var_per_dim = embs.var(axis=0)
    cov = np.cov(embs.T) + 1e-6 * np.eye(D)
    try:
        sv = np.linalg.svd(cov, compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float('inf')
    except: cond = float('inf')
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)
    s1, s2 = eigvals.sum(), (eigvals ** 2).sum()
    eff_dim = float(s1 ** 2 / s2) if s2 > 1e-12 else 0.0
    return {"var_mean": float(var_per_dim.mean()), "cond_number": cond, "eff_dim": eff_dim}


def load_or_recompute_scf(v2_h5_path, raw_dir, traces_per_file=1250, trace_len=4096):
    """Try loading pre-computed v2 H5; if corrupt, recompute from raw .bin files."""
    DRONE_MAP = {
        "DJI_matrice_100_2G.bin":         ("DJI Matrice 100",       "2.4GHz"),
        "DJI_matrice_210_2G.bin":         ("DJI Matrice 210",       "2.4GHz"),
        "DJI_inspire_2_2G.bin":           ("DJI Inspire 2",         "2.4GHz"),
        "DJI_mavic_pro_2G.bin":           ("DJI Mavic Pro",         "2.4GHz"),
        "DJI_mavic_mini_2G.bin":          ("DJI Mavic Mini",        "2.4GHz"),
        "DJI_phantom_4_2G.bin":           ("DJI Phantom 4",         "2.4GHz"),
        "DJI_phantom_4_pro_plus_2G.bin":  ("DJI Phantom 4 Pro+",    "2.4GHz"),
        "Parrot_disco_2G.bin":            ("Parrot Disco",          "2.4GHz"),
        "Parrot_mambo_control_2G.bin":    ("Parrot Mambo (ctrl)",   "2.4GHz"),
        "Parrot_mambo_video_2G.bin":      ("Parrot Mambo (video)",  "2.4GHz"),
        "Yuneec_typhoon_h_2G_1of2.bin":   ("Yuneec Typhoon H",      "2.4GHz"),
        "Yuneec_typhoon_h_5G.bin":        ("Yuneec Typhoon H",      "5.8GHz"),
    }
    # Try loading
    try:
        with h5py.File(v2_h5_path, "r") as f:
            n = len(f["images"])
            if n >= 14000:  # expected ~15000
                print(f"  Loaded {n} samples from {v2_h5_path}", flush=True)
                return f["images"][:].astype(np.float32), [s.decode() for s in f["sources"][:]]
            else:
                print(f"  H5 has only {n} samples (expected 15000) — regenerating", flush=True)
    except Exception as e:
        print(f"  Failed to load v2 H5: {e} — regenerating from raw .bin files", flush=True)

    # Recompute from raw
    print(f"  Recomputing SCF from raw .bin files in {raw_dir}", flush=True)
    all_imgs = []
    all_sources = []
    for fname, (drone_type, band) in DRONE_MAP.items():
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            print(f"    MISSING: {fname}", flush=True)
            continue
        print(f"    Processing {fname}...", flush=True)
        sz = os.path.getsize(path)
        n_total = sz // 4
        iq_mm = np.memmap(path, dtype="<i2", mode="r", shape=(n_total * 2,))
        n_possible = n_total // trace_len
        rng = np.random.RandomState(hash(fname) & 0xFFFF)
        trace_ids = sorted(rng.choice(n_possible, size=traces_per_file, replace=False))
        t0 = time.time()
        for i, tid in enumerate(trace_ids):
            s = tid * trace_len; e = s + trace_len
            raw = np.asarray(iq_mm[2*s:2*e], dtype=np.float64)
            iq = raw.view(np.complex128)
            img = iq_to_scf_image(iq)
            all_imgs.append(img)
            all_sources.append(fname)
            if (i+1) % 200 == 0:
                print(f"      {i+1}/{traces_per_file}  elapsed={time.time()-t0:.1f}s", flush=True)
        del iq_mm
    return np.stack(all_imgs).astype(np.float32), all_sources


def main(seed=42, n_epochs=30, n_bg=600, bg_eval_n=300):
    device = "cuda"
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); torch.cuda.manual_seed_all(seed)

    print("="*70, flush=True)
    print(f"=== RF SCF v2 Training (15k real samples, seed={seed}, epochs={n_epochs}) ===", flush=True)
    print("="*70, flush=True)

    ZENODO_H5_V2 = "/data/zenodo_scf_samples_v2.h5"
    RAW_DIR = "/data/raw_iq"
    DRFFR2_H5 = "/data/data/drffr2.h5"
    BG_H5 = "/data/data/iris_matched_bg.h5"

    # 1. Load/recompute Zenodo SCF v2 (15k samples)
    print("\n[1] Loading Zenodo SCF v2 (15k samples)...", flush=True)
    zenodo_scf, zenodo_sources = load_or_recompute_scf(ZENODO_H5_V2, RAW_DIR)
    print(f"  Total: {len(zenodo_scf)} samples, {len(set(zenodo_sources))} source files", flush=True)
    print(f"  Memory: {zenodo_scf.nbytes/1e9:.2f} GB", flush=True)

    # 2. Load BG
    print(f"\n[2] Loading BG...", flush=True)
    with h5py.File(BG_H5, "r") as f:
        bg_grp = f["holdout_matched_bg"]
        bg_keys_all = sorted(list(bg_grp.keys()), key=lambda x: int(x) if x.isdigit() else 0)
        bg_train_keys = bg_keys_all[:n_bg]
        bg_eval_keys = bg_keys_all[n_bg:n_bg+bg_eval_n]
        bg_specs_train = np.stack([bg_grp[k][:][:2].copy().astype(np.float32) for k in bg_train_keys])
        bg_specs_eval = np.stack([bg_grp[k][:][:2].copy().astype(np.float32) for k in bg_eval_keys])
    # Resize BG to (2,256,256) + normalize
    def _resize_bg(specs):
        if specs.shape[1:] == (2, 256, 256):
            return specs
        out = np.empty((specs.shape[0], 2, 256, 256), dtype=np.float32)
        for i in range(specs.shape[0]):
            for c in range(2):
                ch = specs[i, c]
                mu, sd = ch.mean(), ch.std() + 1e-8
                ch = (ch - mu) / sd
                t = torch.from_numpy(ch).float().unsqueeze(0).unsqueeze(0)
                t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
                out[i, c] = t.squeeze().numpy()
        return out
    bg_specs_train = _resize_bg(bg_specs_train)
    bg_specs_eval = _resize_bg(bg_specs_eval)
    print(f"  BG train: {len(bg_specs_train)}, BG eval: {len(bg_specs_eval)}", flush=True)

    # 3. DRFF-R2 SCF (holdout)
    print(f"\n[3] Computing DRFF-R2 SCF (holdout)...", flush=True)
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
                print(f"    {tname}: {len(specs)}", flush=True)
    n_drffr2 = sum(len(v) for v in drffr2_scf.values())
    print(f"  DRFF-R2: {n_drffr2}", flush=True)

    # 4. Combine
    print(f"\n[4] Assembling training set...", flush=True)
    all_specs = np.concatenate([zenodo_scf, bg_specs_train])
    all_labels = np.concatenate([
        np.ones(len(zenodo_scf), dtype=np.float32),
        np.zeros(len(bg_specs_train), dtype=np.float32)
    ])
    print(f"  Training: {len(all_specs)} ({int((all_labels==1).sum())} drones, "
          f"{int((all_labels==0).sum())} BG)", flush=True)

    # 5. Model
    print(f"\n[5] Building model...", flush=True)
    encoder = CNNEncoder(in_ch=2, embed_dim=256).to(device)
    sigreg = SIGRegLoss().to(device)
    head = Head().to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=1e-3, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    dl = DataLoader(
        TensorDataset(torch.from_numpy(all_specs).float(), torch.from_numpy(all_labels).float()),
        batch_size=32, shuffle=True, drop_last=True
    )

    # 6. Train
    print(f"\n[6] Training ({n_epochs} epochs, ~{len(dl)} batches/epoch)...", flush=True)
    train_history = []
    for epoch in range(n_epochs):
        encoder.train(); head.train()
        ep_sig = 0; ep_bce = 0; nb = 0
        for specs_b, labels_b in dl:
            specs_b = specs_b.to(device); labels_b = labels_b.to(device)
            z = encoder(specs_b)
            sig_loss = sigreg(z)
            bce_loss = F.binary_cross_entropy_with_logits(head(z), labels_b)
            total = sig_loss + bce_loss
            optimizer.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            ep_sig += sig_loss.item(); ep_bce += bce_loss.item(); nb += 1
        scheduler.step()
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: sig={ep_sig/nb:.4f} bce={ep_bce/nb:.4f}", flush=True)
        train_history.append({"epoch": epoch+1, "sig": ep_sig/nb, "bce": ep_bce/nb})

    # 7. Eval
    print(f"\n[7] Evaluation...", flush=True)
    encoder.eval()
    with torch.no_grad():
        td_embs = encode(encoder, zenodo_scf, device)
        centroid, cov_inv = fit_mahal(td_embs)
        td_dists = mahal_l2(td_embs, centroid, cov_inv)
        thresh = float(np.percentile(td_dists, 99))
        print(f"  Threshold (99p): {thresh:.4f}", flush=True)
        print(f"  Train drone dist: mean={td_dists.mean():.4f}", flush=True)

        be_embs = encode(encoder, bg_specs_eval, device)
        be_dists = mahal_l2(be_embs, centroid, cov_inv)
        bg_fp = float((be_dists <= thresh).mean())
        print(f"  BG eval: FP={bg_fp:.4f}  dist_mean={be_dists.mean():.4f}", flush=True)

        per_type_results = {}
        all_drffr2_dists = []
        for tname in sorted(drffr2_scf.keys()):
            embs = encode(encoder, drffr2_scf[tname], device)
            dists = mahal_l2(embs, centroid, cov_inv)
            all_drffr2_dists.append(dists)
            det_rate = float((dists <= thresh).mean())
            per_type_results[tname] = {
                "n_samples": int(len(dists)),
                "det_rate": det_rate,
                "dist_mean": float(dists.mean()),
            }
            print(f"    {tname:20s} n={len(dists):>3d}  det={det_rate:.3f}  "
                  f"dist_mean={dists.mean():.2f}", flush=True)

        all_drffr2 = np.concatenate(all_drffr2_dists) if all_drffr2_dists else np.array([])
        drffr2_det = float((all_drffr2 <= thresh).mean()) if len(all_drffr2) > 0 else 0.0
        print(f"\n  DRFF-R2 overall: det={drffr2_det:.4f}", flush=True)

        # AUC
        all_dists = np.concatenate([be_dists, all_drffr2])
        all_labels_auc = np.concatenate([np.zeros(len(be_dists)), np.ones(len(all_drffr2))])
        try:
            auc = float(roc_auc_score(all_labels_auc, -all_dists))
        except Exception:
            auc = -1.0
        print(f"  AUC (BG vs DRFF-R2): {auc:.4f}", flush=True)

        cm = collapse_metrics(td_embs)
        print(f"  Collapse: cond={cm['cond_number']:.0f}, eff_dim={cm['eff_dim']:.1f}", flush=True)

        # Source probe
        all_drffr2_embs = np.concatenate([encode(encoder, v, device) for v in drffr2_scf.values()])
        X_p = np.concatenate([td_embs, all_drffr2_embs])
        y_p = np.array([0]*len(td_embs) + [1]*len(all_drffr2_embs))
        X_pn = X_p / (np.linalg.norm(X_p, axis=1, keepdims=True) + 1e-8)
        X_ps = StandardScaler().fit_transform(X_pn)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        try:
            probe = float(cross_val_score(
                LogisticRegression(max_iter=2000, C=1.0), X_ps, y_p,
                cv=cv, scoring='accuracy'
            ).mean())
        except Exception:
            probe = -1.0
        print(f"  Source probe: {probe:.4f}", flush=True)

    # 8. Save
    print(f"\n[8] Saving...", flush=True)
    model_path = f"/models/rf_scf_real_v2_encoder_seed{seed}.pt"
    torch.save({"encoder": encoder.state_dict(), "head": head.state_dict()}, model_path)
    print(f"  Saved {model_path}", flush=True)

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "method": "RF SCF v2 — trained on 15k real Zenodo SCF (no aug), eval on DRFF-R2",
        "seed": seed,
        "n_epochs": n_epochs,
        "n_train_drone": int(len(zenodo_scf)),
        "n_train_bg": int(len(bg_specs_train)),
        "n_eval_bg": int(len(bg_specs_eval)),
        "n_eval_drffr2": int(n_drffr2),
        "threshold": thresh,
        "train_drone_dist_mean": float(td_dists.mean()),
        "bg_fp": bg_fp,
        "drffr2_det_overall": drffr2_det,
        "auc_bg_vs_drffr2": auc,
        "source_probe_acc": probe,
        "cond_number": cm['cond_number'],
        "eff_dim": cm['eff_dim'],
        "per_drffr2_type": per_type_results,
        "train_history": train_history,
    }

    json_path = f"/results/rf_scf_real_v2_eval_seed{seed}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved {json_path}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"v2 RESULTS (15k samples)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  BG FP:            {bg_fp:.4f}  (target <0.01)", flush=True)
    print(f"  DRFF-R2 det:      {drffr2_det:.4f}  (target >0.50)", flush=True)
    print(f"  AUC:              {auc:.4f}  (target >0.90)", flush=True)
    print(f"  Source probe:     {probe:.4f}  (~0.5=good)", flush=True)
    print(f"  Eff dim:          {cm['eff_dim']:.2f}  (target >10)", flush=True)

    return output
'''

CORE_PATH = "/tmp/train_rf_scf_v2_core.py"
with open(CORE_PATH, "w") as f:
    f.write(CORE)

IMAGE = IMAGE.add_local_file(CORE_PATH, "/root/train_rf_scf_v2_core.py")


@app.function(
    image=IMAGE,
    gpu="T4",
    volumes={"/data": DATA_VOL, "/results": RESULTS_VOL, "/models": MODELS_VOL},
    timeout=5400,  # 90 min (15k samples may need recompute)
    memory=16384,
)
def launch(seed=42, n_epochs=30):
    import sys
    sys.path.insert(0, "/root")
    from train_rf_scf_v2_core import main
    return main(seed=seed, n_epochs=n_epochs)


if __name__ == "__main__":
    with app.run(detach=True):
        fc = launch.spawn(seed=42, n_epochs=30)
        print(f"SPAWNED: {fc.object_id}")
        print(f"App: {app.name}")
        print(f"Track: https://modal.com/logs/call/{fc.object_id}")
        print("=" * 60)
