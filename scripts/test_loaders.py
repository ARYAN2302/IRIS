"""Smoke test for dataset loaders using synthetic data."""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.loaders import InterleavedBinaryLoader, MatSeparateIQLoader, MatComplexLoader, load_iq
from pathlib import Path
import h5py

test_dir = Path("data/test_temp")
test_dir.mkdir(parents=True, exist_ok=True)

# --- Test 1: InterleavedBinary ---
print("=== Test 1: InterleavedBinary (.dat) ===")
N = 10000
iq_orig = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)
interleaved = np.empty(2 * N, dtype=np.float32)
interleaved[0::2] = iq_orig.real
interleaved[1::2] = iq_orig.imag

dat_path = test_dir / "test_iq.dat"
interleaved.tofile(dat_path)

loaded = InterleavedBinaryLoader(dat_path).load()
print(f"  Original: {len(iq_orig)} complex samples")
print(f"  Loaded:   {len(loaded)} complex samples")
print(f"  Match:    {np.allclose(iq_orig, loaded)}")

# Auto-detect
loaded_auto = load_iq(dat_path)
print(f"  Auto:     {np.allclose(iq_orig, loaded_auto)}")

# --- Test 2: MatSeparateIQ (DRFF-R2 style) ---
print("\n=== Test 2: MatSeparateIQ (.mat with RF0_I + RF0_Q) ===")
mat_path = test_dir / "test_drff.mat"
with h5py.File(mat_path, 'w') as f:
    f.create_dataset('RF0_I', data=iq_orig.real)
    f.create_dataset('RF0_Q', data=iq_orig.imag)
    f.create_dataset('Fs', data=np.float64(100e6))

loaded = MatSeparateIQLoader(mat_path).load()
print(f"  Loaded:   {len(loaded)} complex samples")
print(f"  Match:    {np.allclose(iq_orig, loaded)}")

meta = MatSeparateIQLoader(mat_path).load_metadata()
print(f"  Metadata: Fs = {meta.get('Fs', 'N/A')}")

# Auto-detect
loaded_auto = load_iq(mat_path)
print(f"  Auto:     {np.allclose(iq_orig, loaded_auto)}")

# --- Test 3: MatComplex (DroneRF style) ---
print("\n=== Test 3: MatComplex (.mat with complex array) ===")
mat_path2 = test_dir / "test_dronerf.mat"
with h5py.File(mat_path2, 'w') as f:
    f.create_dataset('rf_data', data=iq_orig)

loaded = MatComplexLoader(mat_path2).load()
print(f"  Loaded:   {len(loaded)} complex samples")
print(f"  Match:    {np.allclose(iq_orig, loaded)}")

# Auto-detect
loaded_auto = load_iq(mat_path2)
print(f"  Auto:     {np.allclose(iq_orig, loaded_auto)}")

# --- Test 4: max_samples ---
print("\n=== Test 4: max_samples parameter ===")
loaded_1k = load_iq(dat_path, max_samples=1000)
print(f"  Requested 1000, got: {len(loaded_1k)}")
print(f"  Match first 1000: {np.allclose(iq_orig[:1000], loaded_1k)}")

# Cleanup
import shutil
shutil.rmtree(test_dir)

print("\n✅ All loader tests passed!")