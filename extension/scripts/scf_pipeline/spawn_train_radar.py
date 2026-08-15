"""Modal pipeline: Train radar encoder on Open Radar Initiative dataset.

Dataset: Open Radar Initiative Outdoor Moving Object Dataset
  - Source: https://github.com/openradarinitiative/open_radar_datasets
  - License: CC-BY-NC-4.0
  - Citation: Gusland et al., 2021 IEEE International Radar Conference
  - 350 signatures: 50 UAV + 47 bicycle + 52 person + 201 vehicle
  - Each signature: complex Doppler spectrogram (44, 1008)

Strategy:
  - Drone (UAV) = positive class
  - Person + bicycle + vehicle = BG negatives
  - Compute (1, 256, 256) log-magnitude spectrograms from complex Doppler
  - Train with VICReg + SIGReg + BCE (same as RF v3 and acoustic)
  - 80/20 train/eval split

Output: 256-dim embedding compatible with IRIS FusionHead.
"""
import modal, os

app = modal.App("iris-cuas-radar")

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
"""Radar encoder training on Open Radar Initiative dataset."""
import json, os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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
    def __init__(self, in_ch=1, width=64, depth=6, embed_dim=256):
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


class VICRegLoss(nn.Module):
    def __init__(self, var_target=1.0, var_lambda=25.0, cov_lambda=1.0):
        super().__init__()
        self.var_target = var_target
        self.var_lambda = var_lambda
        self.cov_lambda = cov_lambda

    def variance_loss(self, z):
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.relu(self.var_target - std).mean()

    def covariance_loss(self, z):
        N, D = z.shape
        zc = z - z.mean(dim=0)
        cov = (zc.T @ zc) / (N - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / D

    def forward(self, z):
        return self.var_lambda * self.variance_loss(z) + self.cov_lambda * self.covariance_loss(z)


class Head(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def doppler_to_image(signature, target_size=256):
    """Convert complex Doppler signature to (1, 256, 256) image."""
    arr = 20 * np.log10(np.abs(signature) + 1e-12).T
    h, w = arr.shape
    if h != target_size or w != target_size:
        t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(target_size, target_size), mode="bilinear", align_corners=False)
        arr = t.squeeze().numpy()
    mu, sd = arr.mean(), arr.std() + 1e-8
    arr = (arr - mu) / sd
    return arr[np.newaxis, :, :].astype(np.float32)


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


def main(seed=42, n_epochs=30):
    device = "cuda"
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); torch.cuda.manual_seed_all(seed)

    print("="*70, flush=True)
    print(f"=== Radar Encoder Training (Open Radar Initiative, seed={seed}) ===", flush=True)
    print("="*70, flush=True)

    print("\n[1] Loading Open Radar dataset...", flush=True)
    data = np.load("/data/radar/sample_dataset.npy", allow_pickle=True)
    print(f"  Total signatures: {len(data)}", flush=True)

    print("\n[2] Converting signatures to images + splitting...", flush=True)
    drone_specs = []
    bg_specs = []
    for sig in data:
        cn = sig['class_name']
        if len(sig['snr_db']) < 10: continue
        try:
            img = doppler_to_image(sig['signature'])
            if cn == 'uav': drone_specs.append(img)
            else: bg_specs.append(img)
        except: pass

    print(f"  Drone (UAV) samples: {len(drone_specs)}", flush=True)
    print(f"  BG (person+bicycle+vehicle) samples: {len(bg_specs)}", flush=True)

    if len(drone_specs) < 10 or len(bg_specs) < 10:
        raise RuntimeError(f"Insufficient data: drones={len(drone_specs)}, bg={len(bg_specs)}")

    rng = np.random.RandomState(seed)
    n_drone_train = int(len(drone_specs) * 0.8)
    n_bg_train = int(len(bg_specs) * 0.8)
    drone_idx = rng.permutation(len(drone_specs))
    bg_idx = rng.permutation(len(bg_specs))
    drone_train = np.stack([drone_specs[i] for i in drone_idx[:n_drone_train]])
    drone_eval = np.stack([drone_specs[i] for i in drone_idx[n_drone_train:]])
    bg_train = np.stack([bg_specs[i] for i in bg_idx[:n_bg_train]])
    bg_eval = np.stack([bg_specs[i] for i in bg_idx[n_bg_train:]])
    print(f"\n  Train: drones={len(drone_train)}, bg={len(bg_train)}", flush=True)
    print(f"  Eval:  drones={len(drone_eval)}, bg={len(bg_eval)}", flush=True)

    print(f"\n[3] Building radar encoder (VICReg + SIGReg + BCE)...", flush=True)
    encoder = CNNEncoder(in_ch=1, embed_dim=256).to(device)
    sigreg = SIGRegLoss().to(device)
    vicreg = VICRegLoss(var_target=1.0, var_lambda=25.0, cov_lambda=1.0).to(device)
    head = Head().to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    all_specs = np.concatenate([drone_train, bg_train])
    all_labels = np.concatenate([np.ones(len(drone_train), dtype=np.float32), np.zeros(len(bg_train), dtype=np.float32)])
    dl = DataLoader(TensorDataset(torch.from_numpy(all_specs).float(), torch.from_numpy(all_labels).float()),
                    batch_size=16, shuffle=True, drop_last=True)

    print(f"\n[4] Training ({n_epochs} epochs, {len(dl)} batches/epoch)...", flush=True)
    for epoch in range(n_epochs):
        encoder.train(); head.train()
        ep_sig=0; ep_bce=0; ep_var=0; ep_cov=0; nb=0
        for specs_b, labels_b in dl:
            specs_b = specs_b.to(device); labels_b = labels_b.to(device)
            z = encoder(specs_b)
            sig_loss = sigreg(z)
            vic_loss = vicreg(z)
            bce_loss = F.binary_cross_entropy_with_logits(head(z), labels_b)
            total = sig_loss + vic_loss + bce_loss
            optimizer.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            ep_sig += sig_loss.item(); ep_bce += bce_loss.item()
            ep_var += vicreg.variance_loss(z).item()
            ep_cov += vicreg.covariance_loss(z).item()
            nb += 1
        scheduler.step()
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: sig={ep_sig/nb:.4f} bce={ep_bce/nb:.4f} var={ep_var/nb:.4f} cov={ep_cov/nb:.4f}", flush=True)

    print(f"\n[5] Evaluation...", flush=True)
    encoder.eval()
    with torch.no_grad():
        td_embs = encode(encoder, drone_train, device)
        centroid, cov_inv = fit_mahal(td_embs)
        td_dists = mahal_l2(td_embs, centroid, cov_inv)
        thresh = float(np.percentile(td_dists, 99))
        thresh_999 = float(np.percentile(td_dists, 99.9))

        be_embs = encode(encoder, bg_eval, device)
        be_dists = mahal_l2(be_embs, centroid, cov_inv)
        bg_fp = float((be_dists <= thresh).mean())
        bg_fp_999 = float((be_dists <= thresh_999).mean())

        de_embs = encode(encoder, drone_eval, device)
        de_dists = mahal_l2(de_embs, centroid, cov_inv)
        drone_det = float((de_dists <= thresh).mean())
        drone_det_999 = float((de_dists <= thresh_999).mean())

        all_eval_dists = np.concatenate([be_dists, de_dists])
        all_eval_labels = np.concatenate([np.zeros(len(be_dists)), np.ones(len(de_dists))])
        try: auc = float(roc_auc_score(all_eval_labels, -all_eval_dists))
        except: auc = -1.0

        N, D = td_embs.shape
        var_per_dim = td_embs.var(axis=0)
        cov = np.cov(td_embs.T) + 1e-6 * np.eye(D)
        try:
            sv = np.linalg.svd(cov, compute_uv=False)
            cond = float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float('inf')
        except: cond = float('inf')
        eigvals = np.maximum(np.linalg.eigvalsh(cov), 0)
        s1, s2 = eigvals.sum(), (eigvals ** 2).sum()
        eff_dim = float(s1 ** 2 / s2) if s2 > 1e-12 else 0.0

        print(f"  Drone det (99p):   {drone_det:.4f}", flush=True)
        print(f"  BG FP (99p):       {bg_fp:.4f}", flush=True)
        print(f"  AUC:               {auc:.4f}", flush=True)
        print(f"  Eff dim:           {eff_dim:.2f}", flush=True)

    print(f"\n[6] Saving...", flush=True)
    model_path = f"/models/radar_encoder_seed{seed}.pt"
    torch.save({"encoder": encoder.state_dict(), "head": head.state_dict()}, model_path)
    print(f"  Saved {model_path}", flush=True)

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "method": "Radar encoder — VICReg + SIGReg + BCE, Open Radar Initiative",
        "seed": seed, "n_epochs": n_epochs,
        "n_train_drone": int(len(drone_train)), "n_train_bg": int(len(bg_train)),
        "n_eval_drone": int(len(drone_eval)), "n_eval_bg": int(len(bg_eval)),
        "threshold_99p": thresh, "threshold_99_9p": thresh_999,
        "train_drone_dist_mean": float(td_dists.mean()),
        "bg_fp_99p": bg_fp, "bg_fp_99_9p": bg_fp_999,
        "drone_det_99p": drone_det, "drone_det_99_9p": drone_det_999,
        "auc_bg_vs_drone": auc, "cond_number": cond, "eff_dim": eff_dim,
        "var_mean": float(var_per_dim.mean()),
        "data_sources": {
            "drone": "Open Radar Initiative — UAV signatures (CC-BY-NC-4.0)",
            "bg": "Open Radar Initiative — person+bicycle+vehicle (CC-BY-NC-4.0)",
        }
    }
    json_path = f"/results/radar_encoder_eval_seed{seed}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved {json_path}", flush=True)
    return output
'''

CORE_PATH = "/tmp/train_radar_core.py"
with open(CORE_PATH, "w") as f:
    f.write(CORE)

IMAGE = IMAGE.add_local_file(CORE_PATH, "/root/train_radar_core.py")


@app.function(
    image=IMAGE, gpu="T4",
    volumes={"/data": DATA_VOL, "/results": RESULTS_VOL, "/models": MODELS_VOL},
    timeout=1800, memory=16384,
)
def launch(seed=42, n_epochs=30):
    import sys; sys.path.insert(0, "/root")
    from train_radar_core import main
    return main(seed=seed, n_epochs=n_epochs)


if __name__ == "__main__":
    with app.run(detach=True):
        fc = launch.spawn(seed=42, n_epochs=30)
        print(f"SPAWNED: {fc.object_id}")
        print(f"App: {app.name}")
        print("=" * 60)
