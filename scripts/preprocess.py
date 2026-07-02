"""
Stream Preprocessor — the 30GB-aware pipeline.

Usage:
  python scripts/preprocess.py --dataset rfuav --file path/to/file.dat --drone-type DJI_Mavic --split train
  python scripts/preprocess.py --dataset drff --file path/to/file.mat --drone-type c1 --split train
  python scripts/preprocess.py --dataset auto --file path/to/file.dat --drone-type any --split train

Strategy: Download ONE file → STFT → Append to HDF5 → DELETE raw → Repeat.
Disk usage at any moment: ~2-3 GB raw + growing HDF5.
"""

import argparse
import os
import sys
sys.path.insert(0, ".")

from pathlib import Path
from src.loaders import load_iq
from src.hdf5_store import HDF5Store


def process_file(
    file_path: str,
    drone_type: str,
    split: str,
    source: str,
    hdf5_path: str = "data/processed/iris_spectrograms.h5",
    segment_len: int = 2048,
    stride: int | None = None,
    max_spectrograms: int | None = None,
    max_iq_samples: int | None = None,
    delete_raw: bool = True,
):
    """
    Process a single raw file into the HDF5 store.
    
    Args:
        file_path: path to the raw I/Q file (.dat, .mat, .bin)
        drone_type: label for this recording
        split: "train", "holdout", or "negatives"
        source: dataset name (e.g., "rfuav", "drff_r2")
        hdf5_path: path to the HDF5 store
        segment_len: I/Q samples per spectrogram
        stride: step between windows
        max_spectrograms: cap on spectrograms per file
        max_iq_samples: load only first N I/Q samples (saves RAM for huge files)
        delete_raw: delete the raw file after processing
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return 0

    file_size_gb = file_path.stat().st_size / 1e9
    print(f"\n{'─'*50}")
    print(f"File:   {file_path.name} ({file_size_gb:.2f} GB)")
    print(f"Drone:  {drone_type}")
    print(f"Split:  {split}")
    print(f"Source: {source}")
    print(f"{'─'*50}")

    # 1. Load I/Q
    print("Loading I/Q data...")
    iq = load_iq(file_path, max_samples=max_iq_samples)
    print(f"  Loaded: {len(iq):,} complex samples")

    # 2. STFT → HDF5
    with HDF5Store(hdf5_path) as store:
        n_written = store.add_iq_stream(
            iq_complex=iq,
            drone_type=drone_type,
            split=split,
            source=source,
            segment_len=segment_len,
            stride=stride,
            max_spectrograms=max_spectrograms,
        )

        # Update metadata
        meta = store.read_metadata()
        if 'processed_files' not in meta:
            meta['processed_files'] = []
        meta['processed_files'].append({
            'file': str(file_path),
            'drone_type': drone_type,
            'split': split,
            'source': source,
            'n_spectrograms': n_written,
        })
        store.write_metadata(meta)

    print(f"  Written: {n_written} spectrograms to HDF5")

    # 3. Delete raw file
    if delete_raw and file_size_gb > 0.1:  # only delete if > 100MB
        os.remove(file_path)
        print(f"  🗑  Deleted raw file (freed {file_size_gb:.2f} GB)")
    else:
        print(f"  Kept raw file (< 100MB, not worth deleting)")

    return n_written


def main():
    parser = argparse.ArgumentParser(description="IRIS Stream Preprocessor")
    parser.add_argument("--file", required=True, help="Path to raw I/Q file")
    parser.add_argument("--drone-type", required=True, help="Drone type label")
    parser.add_argument("--split", required=True, choices=["train", "holdout", "negatives"])
    parser.add_argument("--source", default="unknown", help="Dataset name")
    parser.add_argument("--hdf5", default="data/processed/iris_spectrograms.h5")
    parser.add_argument("--segment-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-spectrograms", type=int, default=None)
    parser.add_argument("--max-iq-samples", type=int, default=None,
                        help="Only load first N I/Q samples (for huge files)")
    parser.add_argument("--keep-raw", action="store_true", help="Don't delete raw file")

    args = parser.parse_args()

    n = process_file(
        file_path=args.file,
        drone_type=args.drone_type,
        split=args.split,
        source=args.source,
        hdf5_path=args.hdf5,
        segment_len=args.segment_len,
        stride=args.stride,
        max_spectrograms=args.max_spectrograms,
        max_iq_samples=args.max_iq_samples,
        delete_raw=not args.keep_raw,
    )

    print(f"\n✅ Done. {n} spectrograms processed.")


if __name__ == "__main__":
    main()