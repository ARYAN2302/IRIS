#!/usr/bin/env python3
"""Verify Zenodo drone RF files load correctly and compute basic stats.

Each .bin file is interleaved int16 LE IQ data.
Expected: 480000000 bytes / 4 = 120,000,000 complex samples per 2.4 GHz file
Expected: 400000000 bytes / 4 = 100,000,000 complex samples per 5.8 GHz file
"""
import os, sys, time
import numpy as np

DIR = "/home/z/my-project/data/sources/zenodo_4264467"
DRONE_MAP = {
    # filename -> (drone_type_label, frequency_band)
    "DJI_matrice_100_2G.bin":         ("DJI Matrice 100",         "2.4GHz"),
    "DJI_matrice_210_2G.bin":         ("DJI Matrice 210",         "2.4GHz"),
    "DJI_inspire_2_2G.bin":           ("DJI Inspire 2",           "2.4GHz"),
    "DJI_phantom_4_pro_plus_2G.bin":  ("DJI Phantom 4 Pro+",      "2.4GHz"),
    "Yuneec_typhoon_h_2G_1of2.bin":   ("Yuneec Typhoon H",        "2.4GHz"),
    "Yuneec_typhoon_h_5G.bin":        ("Yuneec Typhoon H",        "5.8GHz"),
}

def verify_one(path, label, band):
    fname = os.path.basename(path)
    sz = os.path.getsize(path)
    n_complex = sz // 4  # int16 LE interleaved = 4 bytes per complex

    print(f"\n=== {fname} ===")
    print(f"  Drone: {label}    Band: {band}")
    print(f"  Size: {sz/1e6:.1f} MB")
    print(f"  Expected complex samples: {n_complex:,}")

    # Load first 1M samples for quick sanity check
    t0 = time.time()
    with open(path, "rb") as f:
        buf = f.read(4 * 1_000_000)  # 4MB = 1M complex
    arr = np.frombuffer(buf, dtype="<i2").astype(np.float64).view(np.complex128)
    dt = time.time() - t0

    mag = np.abs(arr)
    print(f"  Loaded 1M complex samples in {dt*1e3:.0f} ms")
    print(f"  Magnitude: min={mag.min():.1f}  max={mag.max():.1f}  "
          f"mean={mag.mean():.1f}  std={mag.std():.1f}")
    print(f"  Power (mean |x|^2): {np.mean(mag**2):.1f}")
    # Check it's not all zeros or all same value
    unique = len(np.unique(arr[:10000]))
    print(f"  Unique values in first 10k samples: {unique}")
    if unique < 10:
        print("  ⚠️  WARNING: very few unique values — possible corrupt file")
        return False
    return True

def main():
    print("="*70)
    print("Zenodo Drone RF File Verification")
    print("="*70)

    results = {}
    for fname, (label, band) in DRONE_MAP.items():
        path = os.path.join(DIR, fname)
        if not os.path.exists(path):
            print(f"\nMISSING: {fname}")
            results[fname] = False
            continue
        results[fname] = verify_one(path, label, band)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    total_samples = 0
    for fname, ok in results.items():
        path = os.path.join(DIR, fname)
        sz = os.path.getsize(path) if os.path.exists(path) else 0
        n = sz // 4
        total_samples += n if ok else 0
        print(f"  [{'OK' if ok else 'FAIL'}] {fname:42s}  {n:>12,} samples")

    print(f"\nTotal verified samples: {total_samples:,}")
    print(f"Total SCF traces @ 4096 samples each: {total_samples // 4096:,}")
    print(f"  (= potential SCF images from this corpus)")

if __name__ == "__main__":
    main()
