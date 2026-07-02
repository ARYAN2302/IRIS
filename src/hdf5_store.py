"""
HDF5 Store — structured storage for preprocessed spectrograms.

Layout:
  /train/{drone_type}/sample_000000  → (2, 256, 256) float32
  /holdout/{drone_type}/sample_000000 → (2, 256, 256) float32
  /negatives/{source}/sample_000000  → (2, 256, 256) float32
  /metadata                          → JSON string

The file grows incrementally. We never load everything into RAM.
Compression keeps it ~100x smaller than raw I/Q.
"""

import h5py
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from src.stft_engine import STFTEngine


class HDF5Store:
    def __init__(self, path: str | Path, stft_engine: STFTEngine | None = None):
        self.path = Path(path)
        self.engine = stft_engine or STFTEngine()
        self._file = None

    def open(self, mode: str = 'a') -> 'HDF5Store':
        self._file = h5py.File(self.path, mode)
        return self

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise RuntimeError("HDF5 not open. Use `with store:` or store.open()")
        return self._file

    # ──────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────

    def add_iq_stream(
        self,
        iq_complex: np.ndarray,
        drone_type: str,
        split: str,
        source: str = "",
        segment_len: int = 2048,
        stride: int | None = None,
        max_spectrograms: int | None = None,
    ) -> int:
        """
        Raw I/Q stream → segment → STFT → write to HDF5.
        
        Args:
            iq_complex: 1D complex64 array (one capture file)
            drone_type: e.g., "dji_mavic_pro"
            split: "train", "holdout", or "negatives"
            source: dataset name (for negatives path)
            segment_len: I/Q samples per spectrogram window
            stride: step between windows (default = segment_len // 2)
            max_spectrograms: cap on number of spectrograms to write
        
        Returns:
            number of spectrograms written
        """
        segments = self.engine.segment_signal(
            iq_complex, segment_len=segment_len, stride=stride
        )

        if max_spectrograms is not None:
            segments = segments[:max_spectrograms]

        # Build group path
        if split == "negatives":
            group_path = f"/{split}/{source}"
        else:
            group_path = f"/{split}/{drone_type}"

        group = self.file.require_group(group_path)
        start_idx = len(group.keys())

        count = 0
        for i, seg in enumerate(tqdm(segments, desc=f"STFT → {group_path}")):
            spec = self.engine(seg)  # (2, 256, 256) float32

            # Skip garbage
            if np.isnan(spec).any() or np.abs(spec).sum() < 1e-6:
                continue

            ds_name = f"sample_{start_idx + count:06d}"
            group.create_dataset(
                ds_name, data=spec,
                compression="gzip",
                chunks=(2, 256, 256),
            )
            count += 1

            if max_spectrograms is not None and count >= max_spectrograms:
                break

        return count

    # ──────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────

    def list_drone_types(self, split: str) -> list[str]:
        if split not in self.file:
            return []
        return sorted(self.file[split].keys())

    def count_samples(self, split: str, drone_type: str) -> int:
        group_path = f"/{split}/{drone_type}"
        if group_path not in self.file:
            return 0
        return len(self.file[group_path].keys())

    # ──────────────────────────────────────────
    # Metadata
    # ──────────────────────────────────────────

    def write_metadata(self, meta: dict):
        meta_json = json.dumps(meta, indent=2)
        if '/metadata' in self.file:
            del self.file['/metadata']
        self.file.create_dataset('/metadata', data=meta_json)

    def read_metadata(self) -> dict:
        if '/metadata' not in self.file:
            return {}
        return json.loads(self.file['/metadata'][()])

    # ──────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────

    def print_stats(self):
        print(f"\n{'='*60}")
        print(f"HDF5 Store: {self.path}")
        print(f"File size: {self.path.stat().st_size / 1e9:.2f} GB")
        print(f"{'='*60}")

        for split in ['train', 'holdout', 'negatives']:
            if split not in self.file:
                continue

            print(f"\n  /{split}/")
            split_total = 0
            for dtype in sorted(self.file[split].keys()):
                n = self.count_samples(split, dtype)
                split_total += n
                print(f"    {dtype}: {n} samples")
            print(f"    ────")
            print(f"    TOTAL: {split_total} samples")

        print(f"\n{'='*60}\n")