"""
Download RFUAV spectrograms from HuggingFace and convert to HDF5.

Strategy: Download one drone type at a time → convert JPGs to tensors → 
append to HDF5 → delete JPGs. Disk never exceeds ~2 GB overhead.

The spectrograms are pre-processed by the authors (MATLAB pipeline).
They're RGB heatmaps ~1 MB each. We load as 3-channel tensors (256x256).

For zero-shot: we define a holdout set of 7 types that are EXCLUDED from
training. The model never sees any spectrogram from these types.
"""

import sys
sys.path.insert(0, ".")

import os
import h5py
import numpy as np
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files

# ──────────────────────────────────────────────
# ZERO-SHOT HOLDOUT SPLIT
# 7 types the model will NEVER see during training
# ──────────────────────────────────────────────
HOLDOUT_TYPES = {
    "DJI MINI3",        # Popular DJI — Armory.in cares about this
    "DJI AVATA2",       # FPV drone
    "FLYSKY NV 14",     # Non-DJI controller
    "FRSKY-X14",        # Non-DJI controller  
    "JUMPER-TProV2",    # Non-DJI controller
    "Radiolink AT9S Pro", # Non-DJI controller
    "YunZhuo-H30",      # Budget drone — different market
}

REPO_ID = "kitofrank/RFUAV"
HDF5_PATH = "data/processed/iris_rfuav.h5"
DOWNLOAD_DIR = "data/raw/RFUAV"


def get_drone_type_from_path(path: str) -> str:
    """Extract drone type from HuggingFace file path."""
    parts = path.split("/")
    # e.g., ImageSet-AllDrones-MatlabPipeline/train/DAUTEL EVO nano/DAUTEL0.jpg
    if len(parts) >= 4:
        return parts[2]  # The drone type folder name
    return ""


def get_split_from_path(path: str) -> str:
    """Extract train/valid split from path."""
    parts = path.split("/")
    if "train" in parts:
        return "train"
    elif "valid" in parts:
        return "valid"
    return ""


def download_and_convert_type(
    drone_type: str,
    split: str,
    hdf5_path: str,
    target_size: int = 256,
    max_per_type: int | None = None,
):
    """
    Download all spectrogram images for one drone type, convert to HDF5,
    then delete the downloaded JPGs to free space.
    """
    prefix = f"ImageSet-AllDrones-MatlabPipeline/{split}/{drone_type}/"
    
    # List all files for this type
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    type_files = sorted([f for f in all_files if f.startswith(prefix) and f.endswith(".jpg")])
    
    if not type_files:
        print(f"  No files found for {drone_type}/{split}")
        return 0
    
    if max_per_type:
        type_files = type_files[:max_per_type]
    
    print(f"  Downloading {len(type_files)} images for {drone_type}/{split}...")
    
    # Determine HDF5 split
    # If this drone type is in HOLDOUT_TYPES, it goes to /holdout/ regardless
    if drone_type in HOLDOUT_TYPES:
        hdf5_split = "holdout"
    else:
        hdf5_split = "train" if split == "train" else "train"  # both train+valid go to train for non-holdout
    
    count = 0
    downloaded_paths = []
    
    for i, file_path in enumerate(tqdm(type_files, desc=f"  {drone_type}/{split}")):
        # Download
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=file_path,
            repo_type="dataset",
            local_dir=DOWNLOAD_DIR,
        )
        downloaded_paths.append(local_path)
        
        # Load and resize
        try:
            img = Image.open(local_path).convert("RGB")
            img = img.resize((target_size, target_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)  # (256, 256, 3)
            arr = arr / 255.0  # normalize to [0, 1]
            arr = arr.transpose(2, 0, 1)  # (3, 256, 256)
            
            # Per-channel normalize
            for c in range(3):
                ch = arr[c]
                mean, std = ch.mean(), ch.std()
                if std > 1e-8:
                    arr[c] = (ch - mean) / std
                else:
                    arr[c] = ch - mean
            
            # Write to HDF5
            with h5py.File(hdf5_path, 'a') as f:
                group = f.require_group(f"/{hdf5_split}/{drone_type}")
                idx = len(group.keys())
                group.create_dataset(
                    f"sample_{idx:06d}",
                    data=arr,
                    compression="gzip",
                    chunks=(3, 256, 256),
                )
            
            count += 1
            
        except Exception as e:
            print(f"  ⚠️ Error processing {file_path}: {e}")
            continue
    
    # Delete downloaded JPGs
    for p in downloaded_paths:
        if os.path.exists(p):
            os.remove(p)
    
    # Try to remove empty directories
    for p in reversed(downloaded_paths):
        parent = Path(p).parent
        try:
            if parent.exists() and not list(parent.iterdir()):
                parent.rmdir()
        except:
            pass
    
    freed_mb = sum(os.path.getsize(p) for p in downloaded_paths if os.path.exists(p)) / 1e6
    print(f"  ✅ {count} samples → /{hdf5_split}/{drone_type}, JPGs cleaned up")
    
    return count


def main():
    # Create directories
    Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(HDF5_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Get all drone types
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    type_split_pairs = set()
    for f in all_files:
        if f.startswith("ImageSet-AllDrones-MatlabPipeline/") and f.endswith(".jpg"):
            parts = f.split("/")
            if len(parts) >= 4:
                drone_type = parts[2]
                split = parts[1]  # "train" or "valid"
                type_split_pairs.add((drone_type, split))
    
    # Sort: train first, then valid. Non-holdout first, holdout last.
    type_split_pairs = sorted(type_split_pairs, key=lambda x: (
        x[0] in HOLDOUT_TYPES,  # holdout types last
        x[1] == "valid",         # train before valid
        x[0],                    # alphabetical
    ))
    
    print(f"\n{'='*60}")
    print(f"RFUAV Download + Convert Pipeline")
    print(f"{'='*60}")
    print(f"Total types: {len(set(t for t, s in type_split_pairs))}")
    print(f"Holdout types: {sorted(HOLDOUT_TYPES)}")
    print(f"Train types: {sorted(set(t for t, s in type_split_pairs if t not in HOLDOUT_TYPES))}")
    print(f"Target HDF5: {HDF5_PATH}")
    print(f"{'='*60}\n")
    
    # Write manifest
    manifest = {
        "dataset": "RFUAV",
        "source": "kitofrank/RFUAV",
        "holdout_types": sorted(list(HOLDOUT_TYPES)),
        "train_types": sorted([t for t, s in type_split_pairs if t not in HOLDOUT_TYPES]),
        "total_type_split_pairs": len(type_split_pairs),
        "note": "Holdout types excluded from ALL training. Zero-shot evaluation only.",
    }
    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / "rfuav_split.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to: data/manifests/rfuav_split.json\n")
    
    # Process each type+split
    total = 0
    for drone_type, split in type_split_pairs:
        hdf5_split = "holdout" if drone_type in HOLDOUT_TYPES else "train"
        n = download_and_convert_type(drone_type, split, HDF5_PATH)
        total += n
    
    # Print final stats
    print(f"\n{'='*60}")
    with h5py.File(HDF5_PATH, 'r') as f:
        print(f"HDF5 Store: {HDF5_PATH}")
        print(f"File size: {Path(HDF5_PATH).stat().st_size / 1e9:.2f} GB")
        
        for h5_split in ['train', 'holdout']:
            if h5_split in f:
                print(f"\n  /{h5_split}/")
                split_total = 0
                for dtype in sorted(f[h5_split].keys()):
                    n = len(f[f"{h5_split}/{dtype}"].keys())
                    split_total += n
                    print(f"    {dtype}: {n} samples")
                print(f"    TOTAL: {split_total}")
    print(f"\n{'='*60}")
    print(f"✅ Done! {total} total spectrograms in HDF5")


if __name__ == "__main__":
    main()