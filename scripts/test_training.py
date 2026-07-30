"""Quick training test on synthetic data — verifies the whole pipeline runs."""
import sys
sys.path.insert(0, ".")

from pathlib import Path
import shutil
from src.train import train

# Use the HDF5 from the pipeline test if it exists, otherwise skip
hdf5_path = Path("data/processed/iris_spectrograms.h5")

if not hdf5_path.exists():
    # Create a tiny synthetic HDF5 for testing
    print("Creating synthetic HDF5 for training test...")
    import numpy as np
    from src.stft_engine import STFTEngine
    from src.hdf5_store import HDF5Store

    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    engine = STFTEngine()
    
    with HDF5Store(hdf5_path) as store:
        for drone in ["alpha", "beta", "gamma"]:
            fs = 20e6
            duration = 0.01
            t = np.arange(int(fs * duration)) / fs
            freq = {"alpha": 2e6, "beta": 4e6, "gamma": 6e6}[drone]
            iq = np.exp(2j * np.pi * freq * t).astype(np.complex64)
            iq += 0.05 * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
            
            store.add_iq_stream(
                iq_complex=iq,
                drone_type=drone,
                split="train",
                source="synthetic",
                segment_len=2048,
                stride=1024,
            )

# Run a quick 3-epoch training
train(
    hdf5_path=str(hdf5_path),
    output_dir="checkpoints_test",
    batch_size=4,
    num_epochs=3,
    lr=3e-3,
    embed_dim=768,
    device="auto",
)

print("\n✅ Training test complete!")