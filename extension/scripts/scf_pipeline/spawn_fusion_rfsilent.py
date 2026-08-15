"""Modal: Train fusion head with modality dropout, then run RF-Silent ablation.

Loads all 3 trained encoders (RF v3, Acoustic, Radar), computes embeddings
on a common eval set, trains FusionHead with modality dropout, then runs
ablations: full fusion, RF-silent, RF-only, acoustic-silent, radar-silent.

Outputs:
  /models/fusion_head_seed42.pt
  /results/rf_silent_ablation_seed42.json
"""
import modal, os

app = modal.App("iris-cuas-fusion-rfsilent")

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
"""Fusion training + RF Silent ablation."""
import json, os, sys, time, random, io, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
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


class ModalityDropout(nn.Module):
    def __init__(self, n_modalities=3, p=0.3):
        super().__init__()
        self.n_modalities = n_modalities
        self.p = p
    def forward(self, embeddings_list):
        if not self.training:
            return torch.cat(embeddings_list, dim=1)
        dropped = []
        for emb in embeddings_list:
            if torch.rand(1).item() < self.p:
                dropped.append(torch.zeros_like(emb))
            else:
                dropped.append(emb)
        return torch.cat(dropped, dim=1)


class FusionHead(nn.Module):
    def __init__(self, embed_dim=256, n_modalities=3, use_modality_dropout=True, dropout_p=0.3):
        super().__init__()
        self.input_dim = embed_dim * n_modalities
        self.embed_dim = embed_dim
        self.n_modalities = n_modalities
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        if use_modality_dropout:
            self.modality_dropout = ModalityDropout(n_modalities=n_modalities, p=dropout_p)
        else:
            self.modality_dropout = None
    def forward(self, embeddings_list):
        if self.modality_dropout is not None:
            x = self.modality_dropout(embeddings_list)
        else:
            x = torch.cat(embeddings_list, dim=1)
        return self.projection(x)


class Head(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


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


def iq_to_scf_image(iq, out_size=256, n_fft=1<<12, alpha_max=0.5, window_len=128, n_alpha=128):
    z = np.asarray(iq, dtype=np.complex128)
    N = len(z)
    if N < n_fft: z = np.concatenate([z, np.zeros(n_fft - N, dtype=z.dtype)])
    else: z = z[:n_fft]
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
    if img.shape[1] != out_size or img.shape[2] != out_size:
        t = torch.from_numpy(img).float().unsqueeze(0)
        t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
        img = t.squeeze(0).numpy()
    for c in range(img.shape[0]):
        mu, sd = img[c].mean(), img[c].std() + 1e-8
        img[c] = (img[c] - mu) / sd
    return img.astype(np.float32)


def doppler_to_image(signature, target_size=256):
    arr = 20 * np.log10(np.abs(signature) + 1e-12).T
    h, w = arr.shape
    if h != target_size or w != target_size:
        t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(target_size, target_size), mode="bilinear", align_corners=False)
        arr = t.squeeze().numpy()
    mu, sd = arr.mean(), arr.std() + 1e-8
    arr = (arr - mu) / sd
    return arr[np.newaxis, :, :].astype(np.float32)


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


@torch.no_grad()
def encode(encoder, specs, device, bs=32):
    encoder.eval()
    all_embs = []
    for i in range(0, len(specs), bs):
        batch = torch.from_numpy(specs[i:i+bs]).float().to(device)
        all_embs.append(encoder(batch).cpu().numpy())
    return np.concatenate(all_embs)


def main(seed=42, n_epochs=20, n_samples=150):
    device = "cuda"
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); torch.cuda.manual_seed_all(seed)

    print("="*70, flush=True)
    print(f"=== Fusion Training + RF Silent Ablation (seed={seed}) ===", flush=True)
    print("="*70, flush=True)

    # 1. Load all 3 trained encoders
    print("\n[1] Loading trained encoders...", flush=True)
    rf_encoder = CNNEncoder(in_ch=2, embed_dim=256).to(device)
    rf_ckpt = torch.load("/models/rf_scf_real_v3_encoder_seed42.pt", map_location=device)
    rf_encoder.load_state_dict(rf_ckpt["encoder"])
    rf_encoder.eval()
    print("  RF encoder loaded (v3 VICReg)", flush=True)

    acoustic_encoder = CNNEncoder(in_ch=1, embed_dim=256).to(device)
    try:
        ac_ckpt = torch.load("/models/acoustic_encoder_seed42.pt", map_location=device)
        acoustic_encoder.load_state_dict(ac_ckpt["encoder"])
        acoustic_encoder.eval()
        print("  Acoustic encoder loaded", flush=True)
        has_acoustic = True
    except Exception as e:
        print(f"  Acoustic encoder not available: {e}", flush=True)
        has_acoustic = False

    radar_encoder = CNNEncoder(in_ch=1, embed_dim=256).to(device)
    try:
        rd_ckpt = torch.load("/models/radar_encoder_seed42.pt", map_location=device)
        radar_encoder.load_state_dict(rd_ckpt["encoder"])
        radar_encoder.eval()
        print("  Radar encoder loaded", flush=True)
        has_radar = True
    except Exception as e:
        print(f"  Radar encoder not available: {e}", flush=True)
        has_radar = False

    # 2. Compute per-modality embeddings
    print(f"\n[2] Computing per-modality embeddings (n={n_samples} per class)...", flush=True)

    # RF: load pre-computed Zenodo SCF + BG
    print("  RF: Loading Zenodo SCF + BG...", flush=True)
    with h5py.File("/data/zenodo_scf_samples.h5", "r") as f:
        rf_drone_specs = f["images"][:n_samples].astype(np.float32)
    with h5py.File("/data/data/iris_matched_bg.h5", "r") as f:
        bg_keys = sorted(list(f["holdout_matched_bg"].keys()),
                         key=lambda x: int(x) if x.isdigit() else 0)
        bg_keys = bg_keys[:n_samples]
        rf_bg_arr = np.stack([f["holdout_matched_bg"][k][:][:2].copy().astype(np.float32) for k in bg_keys])
    if rf_bg_arr.shape[1:] != (2, 256, 256):
        out = np.empty((rf_bg_arr.shape[0], 2, 256, 256), dtype=np.float32)
        for i in range(rf_bg_arr.shape[0]):
            for c in range(2):
                ch = rf_bg_arr[i, c]
                mu, sd = ch.mean(), ch.std() + 1e-8
                ch = (ch - mu) / sd
                t = torch.from_numpy(ch).float().unsqueeze(0).unsqueeze(0)
                t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
                out[i, c] = t.squeeze().numpy()
        rf_bg_arr = out
    rf_drone_embs = encode(rf_encoder, rf_drone_specs, device)
    rf_bg_embs = encode(rf_encoder, rf_bg_arr, device)
    print(f"  RF drone embs: {rf_drone_embs.shape}", flush=True)

    # Acoustic: download DADS + ESC-50
    ac_drone_embs = None; ac_bg_embs = None
    if has_acoustic:
        print("  Acoustic: Downloading DADS + ESC-50...", flush=True)
        try:
            from huggingface_hub import hf_hub_download
            import pyarrow.parquet as pq
            import urllib.request, librosa

            shard_path = hf_hub_download(
                repo_id="geronimobasso/drone-audio-detection-samples",
                filename="data/train-00000-of-00039.parquet",
                repo_type="dataset", local_dir="/tmp/dads",
            )
            f = pq.ParquetFile(shard_path)
            tbl = f.read_row_group(0, columns=['audio', 'label'])
            ac_drone_specs = []
            for i in range(min(n_samples, len(tbl))):
                audio_entry = tbl['audio'][i].as_py()
                if isinstance(audio_entry, dict) and 'bytes' in audio_entry:
                    try:
                        audio, _ = librosa.load(io.BytesIO(audio_entry['bytes']), sr=16000, mono=True, duration=4.0)
                        target_len = int(16000 * 4.0)
                        if len(audio) < target_len:
                            audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
                        else: audio = audio[:target_len]
                        ac_drone_specs.append(audio_to_melspec(audio))
                    except: pass
            ac_drone_specs = np.stack(ac_drone_specs[:n_samples])

            urllib.request.urlretrieve("https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv", "/tmp/esc50_meta.csv")
            with open("/tmp/esc50_meta.csv") as fcsv:
                reader = csv.DictReader(fcsv)
                esc50_files = [row['filename'] for row in reader]
            ac_bg_specs = []
            for fn in esc50_files[:n_samples + 50]:
                url = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{fn}"
                try:
                    local_path = f"/tmp/esc50_{fn}"
                    if not os.path.exists(local_path):
                        urllib.request.urlretrieve(url, local_path)
                    audio, _ = librosa.load(local_path, sr=16000, mono=True, duration=4.0)
                    target_len = int(16000 * 4.0)
                    if len(audio) < target_len:
                        audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
                    else: audio = audio[:target_len]
                    ac_bg_specs.append(audio_to_melspec(audio))
                    if len(ac_bg_specs) >= n_samples: break
                except: pass
            ac_bg_specs = np.stack(ac_bg_specs[:n_samples])
            print(f"  Acoustic drone: {ac_drone_specs.shape}, BG: {ac_bg_specs.shape}", flush=True)
            ac_drone_embs = encode(acoustic_encoder, ac_drone_specs, device)
            ac_bg_embs = encode(acoustic_encoder, ac_bg_specs, device)
        except Exception as e:
            print(f"  Acoustic data prep failed: {e}", flush=True)
            has_acoustic = False

    # Radar: load Open Radar
    rd_drone_embs = None; rd_bg_embs = None
    if has_radar:
        print("  Radar: Loading Open Radar dataset...", flush=True)
        try:
            data = np.load("/data/radar/sample_dataset.npy", allow_pickle=True)
            rd_drone_specs = []; rd_bg_specs = []
            for sig in data:
                cn = sig['class_name']
                if len(sig['snr_db']) < 10: continue
                try:
                    img = doppler_to_image(sig['signature'])
                    if cn == 'uav': rd_drone_specs.append(img)
                    else: rd_bg_specs.append(img)
                except: pass
            if rd_drone_specs:
                rd_drone_specs = np.stack(rd_drone_specs)
                reps = max(1, n_samples // len(rd_drone_specs) + 1)
                rd_drone_specs = np.tile(rd_drone_specs, (reps, 1, 1, 1))[:n_samples]
                rd_bg_specs = np.stack(rd_bg_specs[:n_samples])
                print(f"  Radar drone: {rd_drone_specs.shape}, BG: {rd_bg_specs.shape}", flush=True)
                rd_drone_embs = encode(radar_encoder, rd_drone_specs, device)
                rd_bg_embs = encode(radar_encoder, rd_bg_specs, device)
        except Exception as e:
            print(f"  Radar data prep failed: {e}", flush=True)
            has_radar = False

    # 3. Build synthetic paired dataset
    print(f"\n[3] Building synthetic paired dataset...", flush=True)
    # Determine actual usable n per modality (use minimum to avoid out-of-bounds)
    n_rf = min(len(rf_drone_embs), len(rf_bg_embs))
    n_ac = min(len(ac_drone_embs), len(ac_bg_embs)) if has_acoustic and ac_drone_embs is not None else n_rf
    n_rd = min(len(rd_drone_embs), len(rd_bg_embs)) if has_radar and rd_drone_embs is not None else n_rf
    n = min(n_samples, n_rf, n_ac, n_rd)
    print(f"  Using n={n} per class (rf={n_rf}, ac={n_ac}, rd={n_rd})", flush=True)
    rng = np.random.RandomState(seed)
    rf_d = rf_drone_embs[:n]
    ac_d = ac_drone_embs[rng.permutation(len(ac_drone_embs))[:n]] if has_acoustic and ac_drone_embs is not None else np.zeros((n, 256), dtype=np.float32)
    rd_d = rd_drone_embs[rng.permutation(len(rd_drone_embs))[:n]] if has_radar and rd_drone_embs is not None else np.zeros((n, 256), dtype=np.float32)
    rf_b = rf_bg_embs[:n]
    ac_b = ac_bg_embs[rng.permutation(len(ac_bg_embs))[:n]] if has_acoustic and ac_bg_embs is not None else np.zeros((n, 256), dtype=np.float32)
    rd_b = rd_bg_embs[rng.permutation(len(rd_bg_embs))[:n]] if has_radar and rd_bg_embs is not None else np.zeros((n, 256), dtype=np.float32)

    all_rf = np.concatenate([rf_d, rf_b])
    all_ac = np.concatenate([ac_d, ac_b])
    all_rd = np.concatenate([rd_d, rd_b])
    all_labels = np.concatenate([np.ones(n), np.zeros(n)]).astype(np.float32)
    perm = rng.permutation(2 * n)
    all_rf = all_rf[perm]; all_ac = all_ac[perm]; all_rd = all_rd[perm]; all_labels = all_labels[perm]

    n_train = int(2 * n * 0.8)
    train_rf = torch.from_numpy(all_rf[:n_train]).float()
    train_ac = torch.from_numpy(all_ac[:n_train]).float()
    train_rd = torch.from_numpy(all_rd[:n_train]).float()
    train_labels = torch.from_numpy(all_labels[:n_train]).float()
    eval_rf = torch.from_numpy(all_rf[n_train:]).float()
    eval_ac = torch.from_numpy(all_ac[n_train:]).float()
    eval_rd = torch.from_numpy(all_rd[n_train:]).float()
    eval_labels = torch.from_numpy(all_labels[n_train:]).float()
    print(f"  Train: {n_train} ({int(train_labels.sum())} drone, {int((1-train_labels).sum())} BG)", flush=True)
    print(f"  Eval:  {2*n-n_train} ({int(eval_labels.sum())} drone, {int((1-eval_labels).sum())} BG)", flush=True)

    # 4. Train fusion head
    print(f"\n[4] Training fusion head ({n_epochs} epochs, dropout p=0.3)...", flush=True)
    fusion = FusionHead(embed_dim=256, n_modalities=3, use_modality_dropout=True, dropout_p=0.3).to(device)
    head = Head().to(device)
    sigreg = SIGRegLoss().to(device)
    optimizer = torch.optim.AdamW(list(fusion.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    dl = DataLoader(TensorDataset(train_rf, train_ac, train_rd, train_labels), batch_size=32, shuffle=True, drop_last=True)

    for epoch in range(n_epochs):
        fusion.train(); head.train()
        ep_loss = 0; nb = 0
        for rf_b, ac_b, rd_b, lb_b in dl:
            rf_b = rf_b.to(device); ac_b = ac_b.to(device); rd_b = rd_b.to(device); lb_b = lb_b.to(device)
            z = fusion([rf_b, ac_b, rd_b])
            sig_loss = sigreg(z)
            bce_loss = F.binary_cross_entropy_with_logits(head(z), lb_b)
            total = sig_loss + bce_loss
            optimizer.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
            optimizer.step()
            ep_loss += total.item(); nb += 1
        scheduler.step()
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={ep_loss/nb:.4f}", flush=True)

    # 5. Eval: full fusion + ablations
    print(f"\n[5] Evaluation (full fusion + ablations)...", flush=True)
    fusion.eval(); head.eval()
    labels_np = eval_labels.numpy().astype(int)
    results = {}

    with torch.no_grad():
        def eval_config(name, rf, ac, rd):
            z = fusion.projection(torch.cat([rf, ac, rd], dim=1))
            logits = head(z).cpu()
            probs = torch.sigmoid(logits).numpy()
            preds = (probs > 0.5).astype(int)
            acc = float((preds == labels_np).mean())
            try: auc = float(roc_auc_score(labels_np, probs))
            except: auc = -1.0
            print(f"  {name:20s}: acc={acc:.4f}  AUC={auc:.4f}", flush=True)
            return {"acc": acc, "auc": auc}

        e_rf = eval_rf.to(device); e_ac = eval_ac.to(device); e_rd = eval_rd.to(device)
        zero = torch.zeros_like(e_rf)

        results["full_fusion"] = eval_config("Full fusion", e_rf, e_ac, e_rd)
        results["rf_silent"] = eval_config("RF-silent", zero, e_ac, e_rd)
        results["acoustic_silent"] = eval_config("Acoustic-silent", e_rf, zero, e_rd)
        results["radar_silent"] = eval_config("Radar-silent", e_rf, e_ac, zero)
        results["rf_only"] = eval_config("RF-only", e_rf, zero, zero)
        results["acoustic_only"] = eval_config("Acoustic-only", zero, e_ac, zero)
        results["radar_only"] = eval_config("Radar-only", zero, zero, e_rd)

    # 6. Save
    print(f"\n[6] Saving...", flush=True)
    model_path = f"/models/fusion_head_seed{seed}.pt"
    torch.save({"fusion": fusion.state_dict(), "head": head.state_dict()}, model_path)
    print(f"  Saved {model_path}", flush=True)

    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "method": "Fusion training + RF Silent ablation",
        "seed": seed, "n_epochs": n_epochs,
        "n_train": int(n_train), "n_eval": int(2*n - n_train),
        "modalities": {"rf": True, "acoustic": has_acoustic, "radar": has_radar},
        "results": results,
        "rf_silent_retention": {
            "acc_ratio": results["rf_silent"]["acc"] / max(results["full_fusion"]["acc"], 1e-6),
            "auc_ratio": results["rf_silent"]["auc"] / max(results["full_fusion"]["auc"], 1e-6),
        }
    }
    json_path = f"/results/rf_silent_ablation_seed{seed}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved {json_path}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"RF SILENT ABLATION RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Full fusion:        acc={results['full_fusion']['acc']:.4f}  AUC={results['full_fusion']['auc']:.4f}", flush=True)
    print(f"  RF-silent:          acc={results['rf_silent']['acc']:.4f}  AUC={results['rf_silent']['auc']:.4f}", flush=True)
    print(f"  RF-only:            acc={results['rf_only']['acc']:.4f}  AUC={results['rf_only']['auc']:.4f}", flush=True)
    print(f"  Acoustic-only:      acc={results['acoustic_only']['acc']:.4f}  AUC={results['acoustic_only']['auc']:.4f}", flush=True)
    print(f"  Radar-only:         acc={results['radar_only']['acc']:.4f}  AUC={results['radar_only']['auc']:.4f}", flush=True)
    print(f"\n  RF-silent retention: {output['rf_silent_retention']['acc_ratio']:.1%} of full fusion acc", flush=True)
    return output
'''

CORE_PATH = "/tmp/fusion_rfsilent_core.py"
with open(CORE_PATH, "w") as f:
    f.write(CORE)

IMAGE = IMAGE.add_local_file(CORE_PATH, "/root/fusion_rfsilent_core.py")


@app.function(
    image=IMAGE, gpu="T4",
    volumes={"/data": DATA_VOL, "/results": RESULTS_VOL, "/models": MODELS_VOL},
    timeout=3600, memory=16384,
)
def launch(seed=42, n_epochs=20):
    import sys; sys.path.insert(0, "/root")
    from fusion_rfsilent_core import main
    return main(seed=seed, n_epochs=n_epochs)


if __name__ == "__main__":
    with app.run(detach=True):
        fc = launch.spawn(seed=42, n_epochs=20)
        print(f"SPAWNED: {fc.object_id}")
        print(f"App: {app.name}")
        print("=" * 60)
