#!/usr/bin/env python3
"""
Pull IRIS v11 checkpoint + data from Modal storage to local machine.

Downloads:
  1. lejepa_v11_best.pt     from volume "iris-models-v11"     (~13 MB)
  2. iris_rfuav.h5          from volume "iris-data"           (large — skip if too big)
  3. iris_matched_bg.h5     from volume "iris-matched-bg"     (large — skip if too big)

Then computes the Mahalanobis centroid from a sample of training drones
and saves it as models/drone_centroid.npz for fast inference.

Run on Mac after `modal token set`:
    python scripts/pull_from_modal.py

This script uses Modal's Volume API to read files directly without spawning
a remote container — it's just a remote file system.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
SAMPLES_DIR = Path.home() / ".iris_samples"

VOLUMES = {
    "iris-models-v11": {"remote_path": "/lejepa_v11_best.pt", "local_path": MODELS_DIR / "lejepa_v11_best.pt"},
    "iris-data": {"remote_path": "/iris_rfuav.h5", "local_path": DATA_DIR / "iris_rfuav.h5", "optional": True},
    "iris-matched-bg": {"remote_path": "/iris_matched_bg.h5", "local_path": DATA_DIR / "iris_matched_bg.h5", "optional": True},
}

# Number of training drone samples to use for fitting Mahalanobis centroid
N_CENTROID_SAMPLES = 1000
# Number of holdout samples to pull for local demos
N_HOLDOUT_SAMPLES = 200
N_MATCHED_BG_SAMPLES = 200


# ─────────────────────────────────────────────────────────────────────────────
# Modal volume helpers
# ─────────────────────────────────────────────────────────────────────────────


def list_volume_files(volume_name: str):
    """List all files in a Modal volume."""
    import modal
    vol = modal.Volume.from_name(volume_name, create_if_missing=False)
    files = []
    with vol.batch_upload() as batch:
        pass  # just to access the volume
    # Use a tiny Modal function to list files
    return files


def download_from_volume(volume_name: str, remote_path: str, local_path: Path, timeout: int = 600):
    """
    Download a file from a Modal volume to local.

    Spawns a tiny Modal container that reads the file and streams it back
    via modal.Volume.batch_download() (preferred) or by reading + returning bytes.
    """
    import modal

    local_path.parent.mkdir(parents=True, exist_ok=True)

    app = modal.App(f"iris-pull-{volume_name}")
    vol = modal.Volume.from_name(volume_name, create_if_missing=False)

    image = (
        modal.Image.from_registry("python:3.11-slim")
        .pip_install("h5py==3.12.1", "numpy==1.26.4")
    )

    @app.function(image=image, volumes={"/vol": vol}, timeout=timeout)
    def read_file() -> bytes:
        with open(f"/vol{remote_path}", "rb") as f:
            return f.read()

    print(f"  [info] downloading {volume_name}:{remote_path} → {local_path}")
    with app.run():
        # Reload volume to ensure we see latest commits
        vol.reload()
        data = read_file.remote()

    local_path.write_bytes(data)
    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"  [ok] downloaded {size_mb:.1f} MB → {local_path}")
    return local_path


def download_checkpoint_only():
    """Just download the v11 checkpoint (small, ~13 MB)."""
    print("\n" + "=" * 60)
    print("Downloading v11 checkpoint")
    print("=" * 60)
    return download_from_volume(
        volume_name="iris-models-v11",
        remote_path="/lejepa_v11_best.pt",
        local_path=MODELS_DIR / "lejepa_v11_best.pt",
        timeout=300,
    )


def download_data_files():
    """Download the HDF5 data files (large — may take a while)."""
    print("\n" + "=" * 60)
    print("Downloading HDF5 data files (this may take 5-20 minutes)")
    print("=" * 60)

    paths = {}
    for vol_name, cfg in VOLUMES.items():
        if "optional" in cfg and cfg.get("optional"):
            try:
                p = download_from_volume(
                    volume_name=vol_name,
                    remote_path=cfg["remote_path"],
                    local_path=cfg["local_path"],
                    timeout=1200,  # 20 min
                )
                paths[vol_name] = p
            except Exception as e:
                print(f"  [warn] could not download {vol_name}: {e}")
                print(f"         skipping — demos will use synthetic data instead")
        else:
            paths[vol_name] = download_from_volume(
                volume_name=vol_name,
                remote_path=cfg["remote_path"],
                local_path=cfg["local_path"],
                timeout=300,
            )
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Mahalanobis centroid computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_centroid_from_hdf5(h5_path: str, checkpoint_path: str, n_samples: int = 1000):
    """
    Compute Mahalanobis centroid from training drone spectrograms in HDF5.

    Pulls N random drone samples from the 'train' split, encodes them,
    and fits the Mahalanobis centroid + covariance.
    """
    import h5py
    from src.iris_inference import IRISDetector, fit_mahalanobis, compute_mahalanobis

    print(f"\n{'=' * 60}")
    print(f"Computing Mahalanobis centroid from {h5_path}")
    print(f"{'=' * 60}")

    if not os.path.exists(h5_path):
        print(f"  [warn] HDF5 not found at {h5_path}")
        print(f"         skipping centroid computation — IRISDetector will need on-the-fly fit")
        return None

    detector = IRISDetector(checkpoint_path=checkpoint_path)
    print(f"  [info] encoder loaded on {detector.device}")

    # Load training spectrograms
    print(f"  [info] loading HDF5...")
    with h5py.File(h5_path, "r") as f:
        if "train" not in f:
            print(f"  [error] no 'train' split in HDF5. Keys: {list(f.keys())}")
            return None

        train_grp = f["train"]
        type_names = sorted(list(train_grp.keys()))
        print(f"  [info] {len(type_names)} drone types in train split")

        # Collect drone samples — try to balance across types
        all_samples = []
        samples_per_type = max(1, n_samples // len(type_names))

        for tname in type_names:
            item = train_grp[tname]
            if isinstance(item, h5py.Dataset):
                if len(item.shape) == 4:
                    n = min(item.shape[0], samples_per_type)
                    for i in range(n):
                        all_samples.append((tname, ("ds", item, i)))
                elif len(item.shape) == 3:
                    all_samples.append((tname, ("ds", item, 0)))
            elif isinstance(item, h5py.Group):
                sub_keys = [k for k in item.keys()
                            if isinstance(item[k], h5py.Dataset) and len(item[k].shape) == 3]
                for sk in sub_keys[:samples_per_type]:
                    all_samples.append((tname, ("grp", item, sk)))

        # Cap at n_samples
        if len(all_samples) > n_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(all_samples), n_samples, replace=False)
            all_samples = [all_samples[i] for i in idx]

        print(f"  [info] encoding {len(all_samples)} drone samples...")

        # Encode in batches
        all_embs = []
        batch_size = 32
        for i in range(0, len(all_samples), batch_size):
            batch_items = all_samples[i:i + batch_size]
            specs = []
            for tname, (kind, parent, idx) in batch_items:
                if kind == "ds":
                    sample = parent[idx]
                else:
                    sample = parent[idx][:]
                # Convert to (2, 256, 256)
                if sample.shape[0] == 3:
                    x = sample[:2].copy()
                elif sample.shape[0] == 2:
                    x = sample.copy()
                else:
                    x = sample[:2].copy()
                x = x.astype(np.float32)
                # Per-channel normalize
                for c in range(x.shape[0]):
                    ch = x[c]
                    ch_std = ch.std()
                    if ch_std > 1e-6:
                        x[c] = (ch - ch.mean()) / ch_std
                    else:
                        x[c] = ch - ch.mean()
                specs.append(x)

            batch_tensor = torch.from_numpy(np.stack(specs)).float()
            embs = detector.encode(batch_tensor)
            all_embs.append(embs)

            if (i // batch_size) % 5 == 0:
                print(f"    encoded {i + len(batch_items)}/{len(all_samples)}")

        all_embs = np.concatenate(all_embs, axis=0)
        print(f"  [info] embeddings shape: {all_embs.shape}")

        # Fit Mahalanobis with L2 normalization (Mahalanobis++ 2025)
        centroid, cov_inv = fit_mahalanobis(all_embs, reg=1e-3, l2_normalize=True)

        # Compute training percentiles
        dists = compute_mahalanobis(all_embs, centroid, cov_inv, l2_normalize=True)
        percentiles = np.percentile(dists, [50, 75, 90, 95, 99])
        threshold = float(percentiles[-1])

        print(f"  [info] training distance percentiles:")
        print(f"          50th: {percentiles[0]:.2f}")
        print(f"          75th: {percentiles[1]:.2f}")
        print(f"          90th: {percentiles[2]:.2f}")
        print(f"          95th: {percentiles[3]:.2f}")
        print(f"          99th: {percentiles[4]:.2f} (threshold)")

    # Save centroid
    centroid_path = MODELS_DIR / "drone_centroid.npz"
    centroid_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        centroid_path,
        centroid=centroid,
        cov_inv=cov_inv,
        threshold=threshold,
        train_percentiles=percentiles,
    )
    print(f"  [ok] saved centroid to {centroid_path}")
    return centroid_path


def pull_demo_samples(h5_path: str, matched_path: str, n_drone: int = 200, n_bg: int = 200):
    """
    Pull a small set of holdout drone + matched BG spectrograms to ~/.iris_samples
    for offline demos (no need to load full HDF5 every time).
    """
    import h5py

    if not os.path.exists(h5_path):
        print(f"  [warn] HDF5 not found at {h5_path} — skipping demo sample extraction")
        return

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  [info] pulling {n_drone} drone + {n_bg} matched BG samples to {SAMPLES_DIR}...")

    with h5py.File(h5_path, "r") as f:
        # Holdout drones
        if "holdout" in f:
            holdout_grp = f["holdout"]
            type_names = sorted(list(holdout_grp.keys()))
            drone_samples = []
            for tname in type_names:
                item = holdout_grp[tname]
                if isinstance(item, h5py.Dataset) and len(item.shape) >= 3:
                    n = min(item.shape[0] if len(item.shape) == 4 else 1, max(1, n_drone // len(type_names)))
                    for i in range(n):
                        if len(item.shape) == 4:
                            drone_samples.append((tname, item[i]))
                        else:
                            drone_samples.append((tname, item[:]))
                elif isinstance(item, h5py.Group):
                    sub_keys = [k for k in item.keys()
                                if isinstance(item[k], h5py.Dataset) and len(item[k].shape) == 3]
                    for sk in sub_keys[:max(1, n_drone // len(type_names))]:
                        drone_samples.append((tname, item[sk][:]))

            # Cap
            if len(drone_samples) > n_drone:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(drone_samples), n_drone, replace=False)
                drone_samples = [drone_samples[i] for i in idx]

            # Save
            specs = []
            labels = []
            type_names_list = []
            for tname, sample in drone_samples:
                if sample.shape[0] == 3:
                    x = sample[:2].copy()
                elif sample.shape[0] == 2:
                    x = sample.copy()
                else:
                    x = sample[:2].copy()
                x = x.astype(np.float32)
                for c in range(x.shape[0]):
                    ch = x[c]
                    ch_std = ch.std()
                    if ch_std > 1e-6:
                        x[c] = (ch - ch.mean()) / ch_std
                    else:
                        x[c] = ch - ch.mean()
                specs.append(x)
                labels.append(1)
                type_names_list.append(tname)

            specs_arr = np.stack(specs)
            np.savez(SAMPLES_DIR / "drones.npz", specs=specs_arr, types=np.array(type_names_list))
            print(f"  [ok] saved {len(specs_arr)} drone samples")

        # Matched backgrounds
        if os.path.exists(matched_path):
            with h5py.File(matched_path, "r") as mf:
                mbg_key = "holdout_matched_bg"
                if mbg_key in mf:
                    mbg_grp = mf[mbg_key]
                    keys = sorted(list(mbg_grp.keys()),
                                  key=lambda x: int(x) if x.isdigit() else 0)
                    n_available = len(keys)
                    n_to_pull = min(n_bg, n_available)
                    rng = np.random.default_rng(123)
                    indices = rng.choice(n_available, n_to_pull, replace=False)

                    specs = []
                    for i in indices:
                        sample = mbg_grp[keys[i]][:]
                        if sample.shape[0] == 3:
                            x = sample[:2].copy()
                        elif sample.shape[0] == 2:
                            x = sample.copy()
                        else:
                            x = sample[:2].copy()
                        x = x.astype(np.float32)
                        for c in range(x.shape[0]):
                            ch = x[c]
                            ch_std = ch.std()
                            if ch_std > 1e-6:
                                x[c] = (ch - ch.mean()) / ch_std
                            else:
                                x[c] = ch - ch.mean()
                        specs.append(x)

                    specs_arr = np.stack(specs)
                    np.savez(SAMPLES_DIR / "matched_bg.npz", specs=specs_arr)
                    print(f"  [ok] saved {len(specs_arr)} matched BG samples")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("IRIS — Pull from Modal Storage")
    print("=" * 60)
    print(f"  Models dir:  {MODELS_DIR.resolve()}")
    print(f"  Data dir:    {DATA_DIR.resolve()}")
    print(f"  Samples dir: {SAMPLES_DIR.resolve()}")

    # Step 1: Download checkpoint (small, fast)
    ckpt_path = download_checkpoint_only()

    # Step 2: Download HDF5 data (large, slow — ask user)
    print("\n" + "-" * 60)
    print("Step 2: Download HDF5 data files?")
    print("-" * 60)
    print("  The full RFUAV HDF5 is ~1-5 GB. Matched BG HDF5 is smaller.")
    print("  Needed for centroid computation and demo samples.")
    print("  If you have a slow connection, you can skip and use synthetic data.")
    response = input("\n  Download HDF5 files? [y/N]: ").strip().lower()

    h5_path = DATA_DIR / "iris_rfuav.h5"
    matched_path = DATA_DIR / "iris_matched_bg.h5"

    if response == "y":
        try:
            download_data_files()
        except Exception as e:
            print(f"  [error] HDF5 download failed: {e}")
            print(f"          continuing with synthetic data only")
    else:
        print("  Skipping HDF5 download. Demos will use synthetic spectrograms.")

    # Step 3: Compute Mahalanobis centroid (if HDF5 available)
    if os.path.exists(h5_path):
        compute_centroid_from_hdf5(
            h5_path=str(h5_path),
            checkpoint_path=str(ckpt_path),
            n_samples=N_CENTROID_SAMPLES,
        )
    else:
        print("\n  [warn] no HDF5 — centroid not computed.")
        print(f"         IRISDetector will need to fit on-the-fly or use the default threshold (27.42)")

    # Step 4: Pull demo samples
    if os.path.exists(h5_path):
        pull_demo_samples(
            h5_path=str(h5_path),
            matched_path=str(matched_path),
        )

    # Step 5: Smoke test
    print("\n" + "=" * 60)
    print("Smoke test")
    print("=" * 60)
    from src.iris_inference import IRISDetector
    detector = IRISDetector(
        checkpoint_path=str(ckpt_path),
        centroid_path=str(MODELS_DIR / "drone_centroid.npz") if os.path.exists(MODELS_DIR / "drone_centroid.npz") else None,
    )
    print(f"  Encoder: {sum(p.numel() for p in detector.encoder.parameters()):,} params")
    print(f"  Device:  {detector.device}")
    print(f"  Threshold: {detector.threshold:.2f} ({detector.threshold_source})")

    if detector.centroid is not None:
        # Test with a real drone sample if available
        drone_path = SAMPLES_DIR / "drones.npz"
        if drone_path.exists():
            data = np.load(drone_path, allow_pickle=True)
            specs = data["specs"]
            tnames = data["types"]
            print(f"\n  Testing on real drone sample ({tnames[0]}):")
            spec = torch.from_numpy(specs[0]).float()
            result = detector.detect(spec)
            print(f"    verdict:    {result['verdict']}")
            print(f"    confidence: {result['confidence']:.3f}")
            print(f"    mahal_dist: {result['mahal_dist']:.2f} (threshold {result['threshold']:.2f})")

        bg_path = SAMPLES_DIR / "matched_bg.npz"
        if bg_path.exists():
            data = np.load(bg_path)
            specs = data["specs"]
            print(f"\n  Testing on matched BG sample:")
            spec = torch.from_numpy(specs[0]).float()
            result = detector.detect(spec)
            print(f"    verdict:    {result['verdict']}")
            print(f"    confidence: {result['confidence']:.3f}")
            print(f"    mahal_dist: {result['mahal_dist']:.2f} (threshold {result['threshold']:.2f})")

    print("\n" + "=" * 60)
    print("Pull complete. Next: run scripts/live_demo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
