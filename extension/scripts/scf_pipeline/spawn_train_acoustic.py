"""Modal pipeline: Train acoustic encoder on DADS + ESC-50.

Steps:
  1. Download DADS (HuggingFace, 39 parquet shards, ~100K+ drone audio clips, MIT)
  2. Download ESC-50 (GitHub, 2000 environmental sound clips, CC-BY-NC-4.0) as BG
  3. Compute mel-spectrograms (1, 256, 256) per clip
  4. Train AcousticEncoder with VICReg + SIGReg + BCE loss
  5. Save encoder to /models/acoustic_encoder_seed42.pt

Acoustic data sources:
  - Drone positives: DADS (Drone Audio Detection Samples, MIT)
    URL: https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples
  - BG negatives: ESC-50 (environmental sounds, CC-BY-NC-4.0)
    URL: https://github.com/karolpiczak/ESC-50

Output: 256-dim embedding compatible with IRIS FusionHead.
"""
import modal, os

app = modal.App("iris-cuas-acoustic")

DATA_VOL = modal.Volume.from_name("iris-cuas-data", create_if_missing=True)
RESULTS_VOL = modal.Volume.from_name("iris-cuas-results", create_if_missing=True)
MODELS_VOL = modal.Volume.from_name("iris-cuas-models", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install("libgl1", "libglib2.0-0", "libhdf5-dev", "python3", "python3-pip",
                 "python-is-python3", "ffmpeg", "sox", "libsndfile1")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "h5py==3.12.1", "numpy==1.26.4",
                 "scikit-learn==1.6.1", "scipy==1.14.1", "librosa==0.10.2.post1",
                 "soundfile==0.12.1", "huggingface_hub==0.24.7", "pyarrow==15.0.2")
)


CORE = r'''
"""Acoustic encoder training on DADS + ESC-50."""
import json, os, sys, time, random, io, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score


# ============ Model components (mirror RF v3 architecture) ============

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


# ============ Audio processing ============

def audio_to_melspec(audio, sr=16000, n_fft=1024, hop_length=256,
                     n_mels=256, target_frames=256, fmin=0.0, fmax=8000.0):
    import librosa
    audio = audio.astype(np.float32)
    if audio.ndim > 1: audio = audio.mean(axis=0)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0
    )
    log_mel = np.log1p(mel)
    if log_mel.shape[1] != target_frames:
        if log_mel.shape[1] > target_frames: log_mel = log_mel[:, :target_frames]
        else:
            pad = np.tile(log_mel[:, -1:], (1, target_frames - log_mel.shape[1]))
            log_mel = np.concatenate([log_mel, pad], axis=1)
    if log_mel.shape[0] != 256:
        idx = np.linspace(0, log_mel.shape[0] - 1, 256).astype(int)
        log_mel = log_mel[idx, :]
    std = log_mel.std()
    log_mel = (log_mel - log_mel.mean()) / (std + 1e-8)
    return log_mel[np.newaxis, :, :].astype(np.float32)


def load_audio_from_bytes(audio_bytes, sr=16000, duration=4.0):
    import librosa
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True, duration=duration)
    target_len = int(sr * duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
    else: audio = audio[:target_len]
    return audio


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


def download_dads_shards(n_shards_to_download=1):
    """Try hf_hub_download with streaming fallback if HF Hub stalls."""
    import time
    print(f"  Downloading {n_shards_to_download} DADS shards from HuggingFace...", flush=True)
    shards = []
    for i in range(n_shards_to_download):
        shard_path = f"data/train-{i:05d}-of-00039.parquet"
        local = None
        # Try hf_hub_download with retries
        for attempt in range(3):
            try:
                from huggingface_hub import hf_hub_download
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("hf_hub_download hung")
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)  # 2 min per shard
                local = hf_hub_download(
                    repo_id="geronimobasso/drone-audio-detection-samples",
                    filename=shard_path, repo_type="dataset", local_dir="/tmp/dads",
                    resume_download=True,
                )
                signal.alarm(0)
                break
            except TimeoutError as e:
                print(f"    Shard {i} attempt {attempt+1} timeout, retrying...", flush=True)
                time.sleep(5)
            except Exception as e:
                print(f"    Shard {i} attempt {attempt+1} failed: {e}", flush=True)
                time.sleep(5)
        if local and os.path.exists(local):
            shards.append(local)
            print(f"    [{i+1}/{n_shards_to_download}] {shard_path} -> {os.path.getsize(local)/1e6:.0f} MB", flush=True)
        else:
            print(f"    [{i+1}/{n_shards_to_download}] {shard_path} FAILED after 3 attempts, will use streaming fallback", flush=True)
    # If hf_hub_download failed for all, fall back to datasets streaming for remaining
    if len(shards) < n_shards_to_download:
        print(f"  Falling back to datasets streaming for {n_shards_to_download - len(shards)} shards...", flush=True)
        try:
            from datasets import load_dataset
            stream = load_dataset("geronimobasso/drone-audio-detection-samples", split="train", streaming=True)
            # Collect needed samples via streaming (no parquet download)
            fallback_needed = (n_shards_to_download - len(shards)) * 300
            tmp_audio = []
            for idx, row in enumerate(stream):
                if len(tmp_audio) >= fallback_needed:
                    break
                audio_entry = row.get("audio")
                label = row.get("label", 1)
                if isinstance(audio_entry, dict) and "bytes" in audio_entry:
                    tmp_audio.append((audio_entry["bytes"], int(label) if label is not None else 1))
                elif isinstance(audio_entry, dict) and "array" in audio_entry:
                    # Some versions store as array + sampling_rate
                    import io, soundfile as sf
                    arr = np.array(audio_entry["array"], dtype=np.float32)
                    sr = audio_entry.get("sampling_rate", 16000)
                    buf = io.BytesIO()
                    sf.write(buf, arr, sr, format="WAV")
                    tmp_audio.append((buf.getvalue(), int(label) if label is not None else 1))
            print(f"    Streaming fallback collected {len(tmp_audio)} clips", flush=True)
            # Save fallback to a temp list that extract will handle as already-extracted
            # We stash it globally for extract to pick up
            global _streaming_fallback
            _streaming_fallback = tmp_audio
        except Exception as e:
            print(f"    Streaming fallback failed: {e}", flush=True)
    return shards

_streaming_fallback = []

def extract_dads_audio(shards, max_per_shard=300):
    import pyarrow.parquet as pq
    print(f"  Extracting audio from {len(shards)} shards...", flush=True)
    all_audio = []
    # Include streaming fallback if present
    global _streaming_fallback
    if _streaming_fallback:
        all_audio.extend(_streaming_fallback)
        print(f"    Streaming fallback: {len(_streaming_fallback)} clips pre-loaded", flush=True)
    for shard_path in shards:
        try:
            f = pq.ParquetFile(shard_path)
            n_rows = min(f.metadata.num_rows, max_per_shard)
            tbl = f.read_row_group(0, columns=['audio', 'label'])
            for i in range(min(n_rows, len(tbl))):
                audio_entry = tbl['audio'][i].as_py()
                label = tbl['label'][i].as_py() if 'label' in tbl.column_names else 1
                if isinstance(audio_entry, dict) and 'bytes' in audio_entry:
                    all_audio.append((audio_entry['bytes'], int(label) if label is not None else 1))
            print(f"    {shard_path}: {n_rows} samples extracted", flush=True)
        except Exception as e:
            print(f"    Error reading {shard_path}: {e}", flush=True)
    print(f"  Total DADS audio clips: {len(all_audio)}", flush=True)
    return all_audio


def download_esc50(max_clips=400):
    import urllib.request
    print(f"  Downloading {max_clips} ESC-50 clips from GitHub...", flush=True)
    os.makedirs("/tmp/esc50", exist_ok=True)
    clips = []
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv",
        "/tmp/esc50/meta.csv"
    )
    with open("/tmp/esc50/meta.csv") as fcsv:
        reader = csv.DictReader(fcsv)
        esc50_files = [(row['filename'], row['category']) for row in reader]
    print(f"    ESC-50 total files: {len(esc50_files)}", flush=True)
    for filename, category in esc50_files[:max_clips]:
        url = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{filename}"
        try:
            local_path = f"/tmp/esc50/{filename}"
            if not os.path.exists(local_path):
                urllib.request.urlretrieve(url, local_path)
            with open(local_path, 'rb') as f:
                audio_bytes = f.read()
            clips.append((audio_bytes, 0))
        except: pass
    print(f"  Total ESC-50 clips: {len(clips)}", flush=True)
    return clips


def main(seed=42, n_epochs=30, n_dads_shards=39, max_dads_per_shard=5000, max_esc50=2000):
    device = "cuda"
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); torch.cuda.manual_seed_all(seed)

    print("="*70, flush=True)
    print(f"=== Acoustic Encoder Training (DADS + ESC-50, seed={seed}) ===", flush=True)
    print("="*70, flush=True)

    print("\n[1] Downloading acoustic data...", flush=True)
    dads_shards = download_dads_shards(n_shards_to_download=n_dads_shards)
    dads_audio = extract_dads_audio(dads_shards, max_per_shard=max_dads_per_shard)
    esc50_audio = download_esc50(max_clips=max_esc50)

    print("\n[2] Computing mel-spectrograms...", flush=True)
    drone_specs = []
    for i, (audio_bytes, label) in enumerate(dads_audio):
        try:
            audio = load_audio_from_bytes(audio_bytes)
            spec = audio_to_melspec(audio)
            drone_specs.append(spec)
        except: pass
        if (i+1) % 100 == 0:
            print(f"    DADS mel-specs: {i+1}/{len(dads_audio)}  (success: {len(drone_specs)})", flush=True)
    print(f"  Drone mel-specs: {len(drone_specs)}/{len(dads_audio)}", flush=True)

    bg_specs = []
    for i, (audio_bytes, label) in enumerate(esc50_audio):
        try:
            audio = load_audio_from_bytes(audio_bytes)
            spec = audio_to_melspec(audio)
            bg_specs.append(spec)
        except: pass
        if (i+1) % 100 == 0:
            print(f"    ESC-50 mel-specs: {i+1}/{len(esc50_audio)}  (success: {len(bg_specs)})", flush=True)
    print(f"  BG mel-specs: {len(bg_specs)}/{len(esc50_audio)}", flush=True)

    if len(drone_specs) < 50 or len(bg_specs) < 50:
        raise RuntimeError(f"Insufficient data: drones={len(drone_specs)}, bg={len(bg_specs)}")

    n_drone_train = int(len(drone_specs) * 0.8)
    n_bg_train = int(len(bg_specs) * 0.8)
    drone_train = np.stack(drone_specs[:n_drone_train])
    drone_eval = np.stack(drone_specs[n_drone_train:])
    bg_train = np.stack(bg_specs[:n_bg_train])
    bg_eval = np.stack(bg_specs[n_bg_train:])
    print(f"\n  Train: drones={len(drone_train)}, bg={len(bg_train)}", flush=True)
    print(f"  Eval:  drones={len(drone_eval)}, bg={len(bg_eval)}", flush=True)

    print(f"\n[3] Building acoustic encoder (VICReg + SIGReg + BCE)...", flush=True)
    encoder = CNNEncoder(in_ch=1, embed_dim=256).to(device)
    sigreg = SIGRegLoss().to(device)
    vicreg = VICRegLoss(var_target=1.0, var_lambda=25.0, cov_lambda=1.0).to(device)
    head = Head().to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    all_specs = np.concatenate([drone_train, bg_train])
    all_labels = np.concatenate([np.ones(len(drone_train), dtype=np.float32), np.zeros(len(bg_train), dtype=np.float32)])
    dl = DataLoader(TensorDataset(torch.from_numpy(all_specs).float(), torch.from_numpy(all_labels).float()),
                    batch_size=32, shuffle=True, drop_last=True)

    print(f"\n[4] Training ({n_epochs} epochs)...", flush=True)
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
        print(f"  Drone det (99.9p): {drone_det_999:.4f}", flush=True)
        print(f"  BG FP (99p):       {bg_fp:.4f}", flush=True)
        print(f"  AUC:               {auc:.4f}", flush=True)
        print(f"  Eff dim:           {eff_dim:.2f}", flush=True)

    print(f"\n[6] Saving...", flush=True)
    model_path = f"/models/acoustic_encoder_seed{seed}.pt"
    torch.save({"encoder": encoder.state_dict(), "head": head.state_dict()}, model_path)
    print(f"  Saved {model_path}", flush=True)

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "method": "Acoustic encoder — VICReg + SIGReg + BCE, DADS+ESC-50",
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
            "drone": "DADS — Drone Audio Detection Samples (HuggingFace, MIT license)",
            "bg": "ESC-50 — Environmental Sound Classification (GitHub, CC-BY-NC-4.0)",
        }
    }
    json_path = f"/results/acoustic_encoder_eval_seed{seed}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved {json_path}", flush=True)
    return output
'''

CORE_PATH = "/tmp/train_acoustic_core.py"
with open(CORE_PATH, "w") as f:
    f.write(CORE)

IMAGE = IMAGE.add_local_file(CORE_PATH, "/root/train_acoustic_core.py")


@app.function(
    image=IMAGE, gpu="T4",
    volumes={"/data": DATA_VOL, "/results": RESULTS_VOL, "/models": MODELS_VOL},
    timeout=5400, memory=16384,
)
def launch(seed=42, n_epochs=30):
    import sys; sys.path.insert(0, "/root")
    from train_acoustic_core import main
    return main(seed=seed, n_epochs=n_epochs)


if __name__ == "__main__":
    with app.run(detach=True):
        fc = launch.spawn(seed=42, n_epochs=30)
        print(f"SPAWNED: {fc.object_id}")
        print(f"App: {app.name}")
        print("=" * 60)
