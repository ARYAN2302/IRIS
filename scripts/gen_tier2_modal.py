#!/usr/bin/env python3
"""
IRIS v11 — Tier 2 Visual Generation (Modal)

Generates:
  #7: Training Evolution GIF — UMAP snapshots at each epoch (v11 checkpoints)
  #8: Signal Decomposition — Drone vs Matched BG vs Residual

Usage:
  modal run scripts/gen_tier2_modal.py
"""

import h5py
import os
import time

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

# ─── Modal setup ──────────────────────────────────────────────────────────────

app = modal.App("iris-tier2-visuals-v11")

VOL = modal.Volume.from_name("iris-data", create_if_missing=True)
V11_VOL = modal.Volume.from_name("iris-models-v11", create_if_missing=True)
MATCHED_VOL = modal.Volume.from_name("iris-matched-bg", create_if_missing=True)
RESULTS_VOL = modal.Volume.from_name("iris-results", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install("libgl1", "libglib2.0-0", "libhdf5-dev",
                 "python3", "python3-pip", "python-is-python3")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        "h5py==3.12.1", "numpy==1.26.4", "scikit-learn==1.6.1",
        "scipy==1.14.1", "umap-learn==0.5.7", "matplotlib==3.9.3",
        "Pillow==11.1.0",
    )
)

H5_REMOTE = "/data/iris_rfuav.h5"
MATCHED_REMOTE = "/matched/iris_matched_bg.h5"
OUTPUT_DIR = "/output"


# ─── Helper: resolve HDF5 type datasets ───────────────────────────────────────

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


def load_encoder(ckpt_path, device="cpu"):
    """Load encoder from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    encoder = CNNEncoder(
        in_ch=cfg.get("in_ch", 2),
        width=cfg.get("encoder_width", 64),
        depth=cfg.get("encoder_depth", 6),
        embed_dim=cfg.get("embed_dim", 256),
    ).to(device)
    state = ckpt["model"]
    encoder_state = {k.replace("encoder.", "", 1): v
                     for k, v in state.items()
                     if k.startswith("encoder.")}
    if encoder_state:
        encoder.load_state_dict(encoder_state)
    else:
        encoder.load_state_dict(state)
    epoch = ckpt.get("epoch", -1)
    encoder.eval()
    return encoder, epoch


# ─── Datasets ──────────────────────────────────────────────────────────────────

class SimpleEvalDS(Dataset):
    """Load spectrograms with per-channel normalization, no augmentation."""
    def __init__(self, h5_path, split_key="holdout"):
        self.f = h5py.File(h5_path, "r")
        grp = self.f[split_key]
        self.type_names = []
        self._resolved = {}
        self._sub_keys = {}
        self.type_to_label = {}
        label_idx = 0
        for key in sorted(grp.keys()):
            try:
                ds_or_grp, n_samples, is_multi = _resolve_type_dataset(grp, key)
                self.type_names.append(key)
                self._resolved[key] = (ds_or_grp, n_samples, is_multi)
                self.type_to_label[key] = label_idx
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
                label_idx += 1
            except ValueError:
                continue
        self.index = []
        for tname in self.type_names:
            _, n_samples, _ = self._resolved[tname]
            for i in range(n_samples):
                self.index.append((tname, i))

    def __len__(self):
        return len(self.index)

    def _read_sample(self, tname, local_idx):
        ds_or_grp, n_samples, is_multi = self._resolved[tname]
        if is_multi:
            sub_key = self._sub_keys[tname][local_idx]
            return ds_or_grp[sub_key][:]
        else:
            return ds_or_grp[local_idx]

    @staticmethod
    def _normalize_per_channel(x):
        for c in range(x.shape[0]):
            ch = x[c]
            ch_std = ch.std()
            if ch_std > 1e-6:
                x[c] = (ch - ch.mean()) / ch_std
            else:
                x[c] = ch - ch.mean()
        return x

    def __getitem__(self, idx):
        tname, local_idx = self.index[idx]
        sample = self._read_sample(tname, local_idx)
        if sample.shape[0] == 3:
            x = torch.from_numpy(sample[:2].copy()).float()
        elif sample.shape[0] == 2:
            x = torch.from_numpy(sample.copy()).float()
        else:
            x = torch.from_numpy(sample[:2].copy()).float()
        x = self._normalize_per_channel(x)
        return x, self.type_to_label[tname], tname


class MatchedBGDS(Dataset):
    """Load matched backgrounds with per-channel normalization."""
    def __init__(self, matched_path, split_key="holdout"):
        self.f = h5py.File(matched_path, "r")
        mbg_key = f"{split_key}_matched_bg"
        if mbg_key not in self.f:
            raise ValueError(f"No '{mbg_key}' in {matched_path}")
        self.grp = self.f[mbg_key]
        self.keys = sorted(list(self.grp.keys()),
                          key=lambda x: int(x) if x.isdigit() else 0)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        sample = self.grp[key][:]
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
        return x, idx


class RandomBGDS(Dataset):
    """Load random backgrounds with per-channel normalization."""
    def __init__(self, h5_path, max_negatives=2000, seed=42):
        self.f = h5py.File(h5_path, "r")
        neg_item = self.f["negatives"]
        if isinstance(neg_item, h5py.Dataset):
            n_total = neg_item.shape[0]
            self._is_multi = False
            self._ds = neg_item
        else:
            sub_keys = [sk for sk in neg_item.keys()
                        if isinstance(neg_item[sk], h5py.Dataset)
                        and len(neg_item[sk].shape) == 3]
            try:
                sub_keys.sort(key=lambda x: int(x))
            except ValueError:
                sub_keys.sort()
            self._is_multi = True
            self._grp = neg_item
            self._sub_keys = sub_keys
            n_total = len(sub_keys)
        n_load = min(n_total, max_negatives)
        rng = np.random.default_rng(seed)
        self._indices = rng.choice(n_total, n_load, replace=False).tolist()

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        local_idx = self._indices[idx]
        if self._is_multi:
            sub_key = self._sub_keys[local_idx]
            sample = self._grp[sub_key][:]
        else:
            sample = self._ds[local_idx]
        if sample.ndim == 2:
            sample = np.stack([sample, sample], axis=0)
        elif sample.shape[0] == 1:
            sample = np.concatenate([sample, sample], axis=0)
        if sample.shape[0] >= 3:
            x = torch.from_numpy(sample[:2].copy()).float()
        elif sample.shape[0] == 2:
            x = torch.from_numpy(sample.copy()).float()
        else:
            x = torch.from_numpy(sample[:2].copy()).float()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.shape[0] == 1:
            x = x.repeat(2, 1, 1)
        for c in range(x.shape[0]):
            ch = x[c]
            ch_std = ch.std()
            if ch_std > 1e-6:
                x[c] = (ch - ch.mean()) / ch_std
            else:
                x[c] = ch - ch.mean()
        return x, 0


def encode_dataset(encoder, dataset, device, batch_size=128):
    """Encode all samples in a dataset, return embeddings + metadata."""
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)
    all_embs = []
    all_labels = []
    all_types = []
    with torch.no_grad():
        for batch in dl:
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    x, label, tname = batch
                    all_labels.extend(label.tolist() if hasattr(label, 'tolist') else list(label))
                    all_types.extend(list(tname))
                elif len(batch) == 2:
                    x, label = batch
                    all_labels.extend(label.tolist() if hasattr(label, 'tolist') else list(label))
                    all_types.extend(["matched_bg"] * len(label))
                else:
                    x = batch[0]
                    all_labels.extend([0] * len(x))
                    all_types.extend(["random_bg"] * len(x))
            else:
                x = batch
                all_labels.extend([0] * len(x))
                all_types.extend(["random_bg"] * len(x))
            if x.ndim == 3:
                x = x.unsqueeze(1)
            z = encoder(x.to(device))
            all_embs.append(z.cpu().numpy())
    return np.concatenate(all_embs), np.array(all_labels), np.array(all_types)


# ─── Main ─────────────────────────────────────────────────────────────────────

@app.function(
    image=IMAGE,
    gpu="A100",
    volumes={
        "/data": VOL,
        "/models": V11_VOL,
        "/matched": MATCHED_VOL,
        "/output": RESULTS_VOL,
    },
    timeout=3600,
    memory=32768,
)
def generate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from umap import UMAP
    from PIL import Image as PILImage

    device = "cuda"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    # ═══════════════════════════════════════════════════════════════════════
    # #7: TRAINING EVOLUTION GIF (v11)
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("#7: TRAINING EVOLUTION GIF (v11)")
    print("=" * 70)

    # Discover checkpoint files
    V11_VOL.reload()
    ckpt_dir = "/models"
    all_files = os.listdir(ckpt_dir)
    print(f"  Files in iris-models-v11: {sorted(all_files)}")

    epoch_ckpts = {}
    for f in sorted(all_files):
        if f.startswith("lejepa_v11_epoch") and f.endswith(".pt"):
            try:
                ep_num = int(f.replace("lejepa_v11_epoch", "").replace(".pt", ""))
                epoch_ckpts[ep_num] = os.path.join(ckpt_dir, f)
            except ValueError:
                continue

    best_path = os.path.join(ckpt_dir, "lejepa_v11_best.pt")
    best_epoch = -1
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        best_epoch = best_ckpt.get("epoch", -1)
        print(f"  Best checkpoint is epoch {best_epoch}")

    sorted_epochs = sorted(epoch_ckpts.keys())
    print(f"  Found {len(sorted_epochs)} epoch checkpoints: {sorted_epochs}")

    gif_path = None
    if len(sorted_epochs) < 2:
        print("  WARNING: Need at least 2 checkpoints for GIF. Skipping GIF.")
    else:
        # ── Load datasets ──
        print("\n  Loading datasets...")
        rng = np.random.default_rng(42)

        # TRAINING drones — needed for proper Mahalanobis centroid
        print("  Loading training drones (500 subsample)...")
        train_ds = SimpleEvalDS(H5_REMOTE, "train")
        n_train = len(train_ds)
        train_sample_size = min(500, n_train)
        train_indices = rng.choice(n_train, train_sample_size, replace=False)

        # Holdout drones — 200 subsample
        print("  Loading holdout drones (200 subsample)...")
        holdout_ds = SimpleEvalDS(H5_REMOTE, "holdout")
        n_holdout = len(holdout_ds)
        holdout_sample_size = min(200, n_holdout)
        holdout_indices = rng.choice(n_holdout, holdout_sample_size, replace=False)

        # Matched BGs — 200 subsample
        print("  Loading matched backgrounds (200 subsample)...")
        MATCHED_VOL.reload()
        matched_ds = MatchedBGDS(MATCHED_REMOTE, "holdout")
        n_matched = len(matched_ds)
        matched_sample_size = min(200, n_matched)
        matched_indices = rng.choice(n_matched, matched_sample_size, replace=False)

        # Random BGs — 200
        print("  Loading random backgrounds (200)...")
        random_ds = RandomBGDS(H5_REMOTE, max_negatives=200)
        n_random = len(random_ds)

        # ── Encode through each checkpoint ──
        print("\n  Encoding through each checkpoint...")

        all_epoch_data = {}

        for ep in sorted_epochs:
            ckpt_path = epoch_ckpts[ep]
            print(f"\n  Loading epoch {ep}: {os.path.basename(ckpt_path)}")
            encoder, _ = load_encoder(ckpt_path, device)

            # Encode TRAINING drones (for centroid)
            train_dl = DataLoader(train_ds, batch_size=64, shuffle=False,
                                  sampler=torch.utils.data.SubsetRandomSampler(train_indices))
            train_embs_list = []
            with torch.no_grad():
                for batch in train_dl:
                    x, label, tname = batch
                    if x.ndim == 3:
                        x = x.unsqueeze(1)
                    z = encoder(x.to(device))
                    train_embs_list.append(z.cpu().numpy())
            train_embs = np.concatenate(train_embs_list)

            # Encode holdout drones
            holdout_dl = DataLoader(holdout_ds, batch_size=64, shuffle=False,
                                    sampler=torch.utils.data.SubsetRandomSampler(holdout_indices))
            holdout_embs_list = []
            holdout_types_list = []
            with torch.no_grad():
                for batch in holdout_dl:
                    x, label, tname = batch
                    if x.ndim == 3:
                        x = x.unsqueeze(1)
                    z = encoder(x.to(device))
                    holdout_embs_list.append(z.cpu().numpy())
                    holdout_types_list.extend(list(tname))
            holdout_embs = np.concatenate(holdout_embs_list)
            holdout_types = np.array(holdout_types_list)

            # Encode matched BGs
            matched_dl = DataLoader(matched_ds, batch_size=64, shuffle=False,
                                    sampler=torch.utils.data.SubsetRandomSampler(matched_indices))
            matched_embs_list = []
            with torch.no_grad():
                for batch in matched_dl:
                    x, _ = batch
                    if x.ndim == 3:
                        x = x.unsqueeze(1)
                    z = encoder(x.to(device))
                    matched_embs_list.append(z.cpu().numpy())
            matched_embs = np.concatenate(matched_embs_list)

            # Encode random BGs
            random_dl = DataLoader(random_ds, batch_size=64, shuffle=False, num_workers=2)
            random_embs_list = []
            with torch.no_grad():
                for batch in random_dl:
                    x, _ = batch
                    if x.ndim == 3:
                        x = x.unsqueeze(1)
                    z = encoder(x.to(device))
                    random_embs_list.append(z.cpu().numpy())
            random_embs = np.concatenate(random_embs_list)

            # Compute Mahalanobis from TRAINING centroid
            D = train_embs.shape[1]
            centroid = train_embs.mean(axis=0)
            cov = np.cov(train_embs.T) + 1e-3 * np.eye(D)
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov)

            def mahal(embs):
                diff = embs - centroid
                return np.sqrt(np.maximum(np.sum(diff @ cov_inv * diff, axis=1), 0))

            holdout_dists = mahal(holdout_embs)
            matched_dists = mahal(matched_embs)

            # Compute AUC
            if len(np.unique(np.concatenate([np.ones(len(holdout_dists)),
                                              np.zeros(len(matched_dists))]))) > 1:
                auc = roc_auc_score(
                    np.concatenate([np.ones(len(holdout_dists)), np.zeros(len(matched_dists))]),
                    np.concatenate([-holdout_dists, -matched_dists])
                )
            else:
                auc = 0.5

            all_epoch_data[ep] = {
                'train_embs': train_embs,
                'holdout_embs': holdout_embs,
                'matched_embs': matched_embs,
                'random_embs': random_embs,
                'holdout_types': holdout_types,
                'auc': auc,
            }

            is_best = (ep == best_epoch)
            print(f"    AUC (from TRAINING centroid) = {auc:.4f}{'  <-- BEST' if is_best else ''}")

            del encoder
            torch.cuda.empty_cache()

        # ── Fit UMAP on ALL embeddings combined ──
        print("\n  Fitting UMAP on all epochs combined...")
        all_embs_concat = []
        all_labels_concat = []
        all_epoch_ids = []

        for ep in sorted_epochs:
            data = all_epoch_data[ep]
            embs = np.concatenate([
                data['train_embs'][:100],
                data['holdout_embs'],
                data['matched_embs'],
                data['random_embs'],
            ])
            labels = np.concatenate([
                np.zeros(100),
                np.ones(len(data['holdout_embs'])),
                2 * np.ones(len(data['matched_embs'])),
                3 * np.ones(len(data['random_embs'])),
            ])
            all_embs_concat.append(embs)
            all_labels_concat.append(labels)
            all_epoch_ids.extend([ep] * len(embs))

        all_embs_concat = np.concatenate(all_embs_concat)
        all_labels_concat = np.concatenate(all_labels_concat)
        all_epoch_ids = np.array(all_epoch_ids)

        print(f"  Total points for UMAP: {len(all_embs_concat)}")

        reducer = UMAP(n_components=2, metric="cosine", n_neighbors=30,
                       min_dist=0.1, random_state=42, verbose=True)
        all_umap_coords = reducer.fit_transform(all_embs_concat)
        print("  UMAP done.")

        # ── Generate frames ──
        print("\n  Generating GIF frames...")

        all_x = all_umap_coords[:, 0]
        all_y = all_umap_coords[:, 1]
        x_min, x_max = all_x.min() - 1, all_x.max() + 1
        y_min, y_max = all_y.min() - 1, all_y.max() + 1

        COLORS = {0: '#2196F3', 1: '#58a6ff', 2: '#f85149', 3: '#555577'}
        NAMES = {0: 'Train Drones', 1: 'Holdout Drones', 2: 'Matched BG', 3: 'Random BG'}

        frames = []
        frame_paths = []

        for frame_idx, ep in enumerate(sorted_epochs):
            mask = all_epoch_ids == ep
            coords = all_umap_coords[mask]
            labels = all_labels_concat[mask]
            auc = all_epoch_data[ep]['auc']

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            fig.patch.set_facecolor('#0d1117')
            ax.set_facecolor('#0d1117')

            for cat in [3, 0, 2, 1]:
                cat_mask = labels == cat
                s = 5 if cat == 3 else (8 if cat == 0 else 14)
                alpha = 0.25 if cat == 3 else (0.35 if cat == 0 else 0.6)
                ax.scatter(coords[cat_mask, 0], coords[cat_mask, 1],
                          c=COLORS[cat], s=s, alpha=alpha,
                          label=NAMES[cat], edgecolors='none', zorder=4-cat)

            is_best = (ep == best_epoch)
            title_extra = " (BEST)" if is_best else ""
            ax.set_title(f'v11 Epoch {ep}{title_extra}  |  Matched AUC = {auc:.3f}',
                        fontsize=16, fontweight='bold', color='#e6edf3', pad=15)
            ax.legend(loc='best', fontsize=12, facecolor='#161b22',
                     edgecolor='#30363d', labelcolor='#e6edf3', markerscale=2.5)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.tick_params(colors='#484f58')
            ax.grid(True, alpha=0.08)
            ax.set_xlabel('UMAP-1', fontsize=11, color='#8b949e')
            ax.set_ylabel('UMAP-2', fontsize=11, color='#8b949e')

            plt.tight_layout()
            frame_path = f"{OUTPUT_DIR}/frame_ep{ep:02d}.png"
            plt.savefig(frame_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
            plt.close()
            frame_paths.append(frame_path)
            frames.append(PILImage.open(frame_path))

            print(f"    Frame {frame_idx+1}/{len(sorted_epochs)}: epoch {ep}, AUC={auc:.3f}")

        # ── Stitch into GIF ──
        print("\n  Stitching GIF...")
        gif_path = f"{OUTPUT_DIR}/07_training_evolution.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=800,
            loop=0,
        )
        print(f"  Saved: {gif_path}")

        for fp in frame_paths:
            if os.path.exists(fp):
                os.remove(fp)

    # ═══════════════════════════════════════════════════════════════════════
    # #8: SIGNAL DECOMPOSITION VISUAL
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("#8: SIGNAL DECOMPOSITION")
    print("=" * 70)

    # Close datasets from GIF section
    for ds_name in ['train_ds', 'holdout_ds', 'matched_ds', 'random_ds']:
        try:
            ds = locals().get(ds_name)
            if ds and hasattr(ds, 'f'):
                ds.f.close()
        except:
            pass

    VOL.reload()
    MATCHED_VOL.reload()

    drone_f = h5py.File(H5_REMOTE, "r")
    matched_f = h5py.File(MATCHED_REMOTE, "r")

    holdout_grp = drone_f["holdout"]
    type_names = sorted(holdout_grp.keys())
    print(f"  Holdout types: {type_names}")

    mbg_grp = matched_f["holdout_matched_bg"]
    mbg_keys = sorted(list(mbg_grp.keys()), key=lambda x: int(x) if x.isdigit() else 0)

    # Build holdout index -> type mapping
    holdout_index = []
    for tname in type_names:
        ds_or_grp, n_samples, is_multi = _resolve_type_dataset(holdout_grp, tname)
        if is_multi:
            sub_keys = [sk for sk in ds_or_grp.keys()
                        if isinstance(ds_or_grp[sk], h5py.Dataset) and len(ds_or_grp[sk].shape) == 3]
            try:
                sub_keys.sort(key=lambda x: int(x))
            except ValueError:
                sub_keys.sort()
        for i in range(n_samples):
            holdout_index.append((tname, i, is_multi,
                                  sub_keys if is_multi else None,
                                  ds_or_grp))

    # Pick a STRONG type (DJI) and WEAK type (FUTABA)
    strong_type = None
    strong_drone_idx = None
    strong_mbg_idx = None
    for i, (tname, local_idx, is_multi, sub_keys, ds_or_grp) in enumerate(holdout_index):
        if 'DJI' in tname.upper() and i < len(mbg_keys):
            strong_type = tname
            strong_drone_idx = i
            strong_mbg_idx = i
            break

    weak_type = None
    weak_drone_idx = None
    weak_mbg_idx = None
    for i, (tname, local_idx, is_multi, sub_keys, ds_or_grp) in enumerate(holdout_index):
        if ('FUTABA' in tname.upper() or 'JR' in tname.upper()) and i < len(mbg_keys):
            weak_type = tname
            weak_drone_idx = i
            weak_mbg_idx = i
            break

    if strong_type is None:
        strong_drone_idx = 0
        strong_mbg_idx = 0
        strong_type = holdout_index[0][0]

    if weak_type is None:
        weak_drone_idx = min(len(holdout_index) - 1, len(mbg_keys) - 1)
        weak_mbg_idx = weak_drone_idx
        weak_type = holdout_index[weak_drone_idx][0]

    def read_holdout_sample(idx):
        tname, local_idx, is_multi, sub_keys, ds_or_grp = holdout_index[idx]
        if is_multi:
            return ds_or_grp[sub_keys[local_idx]][:]
        else:
            return ds_or_grp[local_idx]

    drone_raw_strong = read_holdout_sample(strong_drone_idx)
    mbg_raw_strong = mbg_grp[mbg_keys[strong_mbg_idx]][:]
    drone_raw_weak = read_holdout_sample(weak_drone_idx)
    mbg_raw_weak = mbg_grp[mbg_keys[weak_mbg_idx]][:]

    min_ch_s = min(drone_raw_strong.shape[0], mbg_raw_strong.shape[0], 2)
    drone_raw_strong = drone_raw_strong[:min_ch_s]
    mbg_raw_strong = mbg_raw_strong[:min_ch_s]
    min_ch_w = min(drone_raw_weak.shape[0], mbg_raw_weak.shape[0], 2)
    drone_raw_weak = drone_raw_weak[:min_ch_w]
    mbg_raw_weak = mbg_raw_weak[:min_ch_w]

    residual_strong = drone_raw_strong - mbg_raw_strong
    residual_weak = drone_raw_weak - mbg_raw_weak

    drone_f.close()
    matched_f.close()

    # ── Plot ──
    plt.style.use('dark_background')

    fig, axes = plt.subplots(2, 3, figsize=(21, 14))

    def plot_spec_row(ax_row, drone, matched, resid, type_name, ch_idx=0):
        im0 = ax_row[0].imshow(drone[ch_idx], aspect='auto', cmap='inferno',
                                origin='lower', interpolation='bilinear')
        ax_row[0].set_title(f'{type_name}\nDrone Signal', fontsize=14,
                           fontweight='bold', color='#e6edf3', pad=10)
        ax_row[0].set_ylabel('Frequency', fontsize=11, color='#8b949e')
        ax_row[0].tick_params(colors='#484f58')
        plt.colorbar(im0, ax=ax_row[0], fraction=0.046, pad=0.04)

        im1 = ax_row[1].imshow(matched[ch_idx], aspect='auto', cmap='inferno',
                                origin='lower', interpolation='bilinear')
        ax_row[1].set_title('Matched Background\n(same recording, signal removed)', fontsize=14,
                           fontweight='bold', color='#f0883e', pad=10)
        ax_row[1].tick_params(colors='#484f58')
        plt.colorbar(im1, ax=ax_row[1], fraction=0.046, pad=0.04)

        vmax = max(abs(resid[ch_idx].min()), abs(resid[ch_idx].max()))
        if vmax < 1e-8:
            vmax = 1.0
        im2 = ax_row[2].imshow(resid[ch_idx], aspect='auto', cmap='RdBu_r',
                                origin='lower', interpolation='bilinear',
                                vmin=-vmax, vmax=vmax)
        signal_power = np.mean(resid[ch_idx] ** 2)
        noise_power = np.mean(matched[ch_idx] ** 2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        ax_row[2].set_title(f'Drone Signal Isolated (residual)\nSNR = {snr_db:.1f} dB', fontsize=14,
                           fontweight='bold', color='#3fb950', pad=10)
        ax_row[2].tick_params(colors='#484f58')
        plt.colorbar(im2, ax=ax_row[2], fraction=0.046, pad=0.04)

        for ax in ax_row:
            ax.set_xlabel('Time', fontsize=11, color='#8b949e')

    plot_spec_row(axes[0], drone_raw_strong, mbg_raw_strong, residual_strong, strong_type)
    plot_spec_row(axes[1], drone_raw_weak, mbg_raw_weak, residual_weak, weak_type)

    fig.suptitle('Signal Decomposition: What IRIS Actually Detects',
                 fontsize=20, fontweight='bold', color='#e6edf3', y=1.01)
    plt.tight_layout()

    sig_path = f"{OUTPUT_DIR}/08_signal_decomposition.png"
    plt.savefig(sig_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  Saved: {sig_path}")

    # ── Commit outputs ──
    RESULTS_VOL.commit()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"TIER 2 MODAL VISUALS GENERATED ({elapsed:.0f}s)")
    print(f"{'='*70}")
    if gif_path:
        print(f"  #7: {gif_path}")
    print(f"  #8: {sig_path}")
    print(f"\n  Download: modal volume get iris-results /output/ ./iris_output/")


@app.local_entrypoint()
def main():
    generate.remote()