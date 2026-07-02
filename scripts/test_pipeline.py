"""End-to-end test: synthetic I/Q → STFT → HDF5 → read back."""
import sys
sys.path.insert(0, ".")

import numpy as np
from pathlib import Path
import shutil
from src.stft_engine import STFTEngine
from src.loaders import InterleavedBinaryLoader
from src.hdf5_store import HDF5Store

# Setup
test_dir = Path("data/test_pipeline")
test_dir.mkdir(parents=True, exist_ok=True)
hdf5_path = test_dir / "test_store.h5"

# Generate 3 fake drone captures
engine = STFTEngine()
drones = ["drone_alpha", "drone_beta", "drone_gamma"]

for drone in drones:
    # Create fake I/Q (different freq per drone so spectrograms differ)
    fs = 20e6
    duration = 0.005  # 5ms
    t = np.arange(int(fs * duration)) / fs
    freq = {"drone_alpha": 2e6, "drone_beta": 4e6, "drone_gamma": 6e6}[drone]
    iq = np.exp(2j * np.pi * freq * t).astype(np.complex64)
    iq += 0.05 * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
    
    # Save as interleaved binary (simulates RFUAV format)
    dat_path = test_dir / f"{drone}.dat"
    interleaved = np.empty(2 * len(iq), dtype=np.float32)
    interleaved[0::2] = iq.real
    interleaved[1::2] = iq.imag
    interleaved.tofile(dat_path)
    
    # Load through our loader
    loaded_iq = InterleavedBinaryLoader(dat_path).load()
    
    # Write to HDF5
    with HDF5Store(hdf5_path) as store:
        n = store.add_iq_stream(
            iq_complex=loaded_iq,
            drone_type=drone,
            split="train",
            source="test",
            segment_len=2048,
            stride=1024,
        )
        print(f"{drone}: {n} spectrograms written")

# Read back and verify
with HDF5Store(hdf5_path) as store:
    store.print_stats()
    
    # Read one sample
    sample = store.file["/train/drone_alpha/sample_000000"][:]
    print(f"Sample shape: {sample.shape}, dtype: {sample.dtype}")
    print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")

# Cleanup
shutil.rmtree(test_dir)

print("\n✅ Full pipeline test passed!")