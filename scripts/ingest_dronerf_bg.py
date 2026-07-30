#!/usr/bin/env python3
"""Ingest DroneRF background RF .rar files into HDF5 as negatives.

Handles .csv files (comma-separated interleaved I/Q floats).

Usage:
    python scripts/ingest_dronerf_bg.py --rar-dir data/raw --hdf5 data/processed/iris_rfuav.h5
"""

import argparse
import os
import sys
import glob
import tempfile
import subprocess

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.stft_engine import STFTEngine


def extract_rar(rar_path, extract_dir):
    """Extract .rar using unar, unrar, or 7z."""
    try:
        result = subprocess.run(
            ['unar', '-o', extract_dir, '-f', rar_path],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass

    try:
        result = subprocess.run(
            ['unrar', 'x', '-o+', rar_path, extract_dir],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass

    try:
        result = subprocess.run(
            ['7z', 'x', '-y', f'-o{extract_dir}', rar_path],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass

    return False


def find_data_files(directory):
    """Find all .dat, .mat, and .csv files recursively."""
    files = []
    for ext in ('*.dat', '*.mat', '*.csv'):
        files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    return sorted(files)


def load_iq_csv(filepath):
    """Load I/Q from CSV file (comma-separated interleaved floats)."""
    raw = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
    if raw.ndim == 0:
        raw = np.array([raw], dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]
    return iq.astype(np.complex64)


def load_iq_file(filepath):
    """Auto-detect format and load I/Q data."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.csv':
        return load_iq_csv(filepath)
    elif ext in ('.dat', '.bin'):
        raw = np.fromfile(filepath, dtype=np.float32)
        return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    elif ext == '.mat':
        import h5py
        with h5py.File(filepath, 'r') as f:
            keys = list(f.keys())
            if len(keys) >= 2:
                i_data = f[keys[0]][()]
                q_data = f[keys[1]][()]
                return (i_data.astype(np.float32) + 1j * q_data.astype(np.float32)).astype(np.complex64)
            else:
                data = f[keys[0]][()]
                if np.iscomplexobj(data):
                    return data.astype(np.complex64)
                raw = data.flatten().astype(np.float32)
                return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    else:
        raise ValueError(f"Unknown file extension: {ext}")


def process_file(filepath, stft_engine, hdf5_path, source_label, segment_iq_len=4096):
    """Process a single file: load -> segment -> STFT -> zero-pad -> HDF5."""
    import h5py

    try:
        iq = load_iq_file(filepath)
    except Exception as e:
        print(f"  [SKIP] Cannot load {os.path.basename(filepath)}: {e}")
        return 0

    n_samples = len(iq)
    print(f"  Loaded {n_samples:,} I/Q samples from {os.path.basename(filepath)}")

    # Segment the long signal into overlapping chunks
    segments = stft_engine.segment_signal(iq, segment_len=segment_iq_len, stride=segment_iq_len // 2)
    print(f"  -> {len(segments)} segments")

    count = 0
    with h5py.File(hdf5_path, 'a') as f:
        if 'negatives' not in f:
            f.create_group('negatives')

        for seg_iq in segments:
            if len(seg_iq) < segment_iq_len:
                continue

            try:
                spec = stft_engine(seg_iq)  # (2, 256, 256), already normalized
            except Exception as e:
                continue

            # Zero-pad 2ch -> 3ch
            if spec.shape[0] == 2:
                pad = np.zeros((1, spec.shape[1], spec.shape[2]), dtype=np.float32)
                spec = np.concatenate([spec, pad], axis=0)

            sample_name = f"neg_{source_label}_{count:06d}"
            f['negatives'].create_dataset(sample_name, data=spec, compression='gzip')
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description='Ingest DroneRF background RF into HDF5')
    parser.add_argument('--rar-dir', help='Directory containing .rar files')
    parser.add_argument('--data-dir', help='Directory with already-extracted data files')
    parser.add_argument('--hdf5', required=True, help='Output HDF5 file path')
    parser.add_argument('--n-fft', type=int, default=1024, help='FFT size for STFT')
    parser.add_argument('--segment-len', type=int, default=4096,
                        help='I/Q samples per spectrogram segment')
    args = parser.parse_args()

    if not args.rar_dir and not args.data_dir:
        parser.error("Must specify --rar-dir or --data-dir")

    stft_engine = STFTEngine(n_fft=args.n_fft)
    total_count = 0

    if args.data_dir:
        data_files = find_data_files(args.data_dir)
        print(f"Found {len(data_files)} data files in {args.data_dir}")

        for fpath in tqdm(data_files, desc="Processing files"):
            label = os.path.basename(fpath).rsplit('.', 1)[0]
            n = process_file(fpath, stft_engine, args.hdf5, label, args.segment_len)
            total_count += n
            print(f"  -> {n} negative segments from {label}")

    elif args.rar_dir:
        rar_files = sorted(glob.glob(os.path.join(args.rar_dir, '*.rar')))
        print(f"Found {len(rar_files)} .rar files in {args.rar_dir}")

        for rar_path in tqdm(rar_files, desc="Processing .rar files"):
            rar_name = os.path.basename(rar_path).replace('.rar', '')
            print(f"\n--- {rar_name} ---")

            with tempfile.TemporaryDirectory() as tmpdir:
                print(f"  Extracting...")
                ok = extract_rar(rar_path, tmpdir)
                if not ok:
                    print(f"  [SKIP] Cannot extract. Install unar: brew install unar")
                    continue

                data_files = find_data_files(tmpdir)
                print(f"  Found {len(data_files)} data files")

                for fpath in data_files:
                    label = f"{rar_name}_{os.path.basename(fpath).rsplit('.', 1)[0]}"
                    n = process_file(fpath, stft_engine, args.hdf5, label, args.segment_len)
                    total_count += n
                    print(f"  -> {n} negative segments")

            # tmpdir auto-deleted here

    # Final stats
    import h5py
    with h5py.File(args.hdf5, 'r') as f:
        n_neg = len(f['negatives']) if 'negatives' in f else 0
        n_types = len(f['train']) if 'train' in f else 0
        n_train = sum(len(f['train'][t]) for t in f['train']) if 'train' in f else 0

    print(f"\n{'='*50}")
    print(f"Negative segments ingested: {total_count}")
    print(f"HDF5: {n_types} drone types ({n_train} samples) + {n_neg} negatives")
    print("Done!")


if __name__ == '__main__':
    main()