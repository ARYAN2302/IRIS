#!/usr/bin/env python3
"""
fix_hdf5_channel.py — Patch iris_rfuav.h5 IN-PLACE

Problem: DroneRF negatives are 2-channel STFT zero-padded to 3 channels.
Channel 2 (index 2) is all zeros, while RFUAV drones have real RGB data
in all 3 channels. This creates a trivial shortcut: model just checks
"does channel 2 have signal?" → instant 100% k-NN, zero actual learning.

Fix: Copy channel 0 → channel 2 for all 122K negatives.
This gives negatives the same 3-channel structure as drone spectrograms,
forcing the model to learn actual RF signal patterns.

Usage:
    cd /Users/adarshthakur/Desktop/IRIS
    python3 scripts/fix_hdf5_channel.py

Time: ~5-10 minutes for 122K samples on M1 MacBook Air
"""

import h5py
import numpy as np
import time
import sys
import os

HDF5_PATH = "data/processed/iris_rfuav.h5"
BATCH_SIZE = 500  # samples per batch (memory-safe for 8GB RAM)


def main():
    if not os.path.exists(HDF5_PATH):
        print(f"ERROR: {HDF5_PATH} not found. Run from IRIS project root.")
        sys.exit(1)

    file_size = os.path.getsize(HDF5_PATH) / 1e9
    print(f"Opening {HDF5_PATH} ({file_size:.1f} GB)...")

    f = h5py.File(HDF5_PATH, "r+")  # read-write mode
    neg_grp = f["negatives"]
    keys = list(neg_grp.keys())
    total = len(keys)
    print(f"Found {total:,} negatives to patch")

    # Verify the bug exists
    sample = neg_grp[keys[0]][:]
    ch2_mean = sample[2].mean()
    if ch2_mean > 1e-6:
        print(f"WARNING: Channel 2 already has data (mean={ch2_mean:.6f}).")
        print("  Skipping patch — negatives may already be fixed.")
        f.close()
        return

    print(f"Confirmed bug: channel 2 mean = {ch2_mean:.8f} (all zeros)")
    print(f"Patching: channel 0 → channel 2 for all {total:,} samples")
    print(f"Batch size: {BATCH_SIZE} samples")
    print()

    start = time.time()
    patched = 0

    for i in range(0, total, BATCH_SIZE):
        batch_keys = keys[i:i + BATCH_SIZE]

        for k in batch_keys:
            data = neg_grp[k]
            arr = data[:]  # read (3, 256, 256)
            arr[2] = arr[0]  # copy channel 0 → channel 2
            data[...] = arr  # write back in-place
            patched += 1

        elapsed = time.time() - start
        rate = patched / elapsed
        remaining = (total - patched) / rate if rate > 0 else 0
        bar = patched / total
        print(f"\r  [{bar:6.1%}] {patched:,}/{total:,} | "
              f"{rate:.0f} samples/s | ETA: {remaining:.0f}s", end="", flush=True)

    print()

    # Verify the fix
    elapsed = time.time() - start
    print(f"\nPatched {patched:,} samples in {elapsed:.1f}s")

    print("\nVerifying fix...")
    verify_keys = [keys[0], keys[total // 2], keys[-1]]
    for k in verify_keys:
        arr = neg_grp[k][:]
        ch0_mean = arr[0].mean()
        ch2_mean = arr[2].mean()
        ch0_ch2_diff = np.abs(arr[0] - arr[2]).max()
        print(f"  {k}: ch0_mean={ch0_mean:.6f}, ch2_mean={ch2_mean:.6f}, "
              f"max_diff_ch0_ch2={ch0_ch2_diff:.8f}")

    # Also verify drone data is unchanged
    train_types = list(f["train"].keys())
    t0 = train_types[0]
    k0 = list(f["train"][t0].keys())[0]
    drone_sample = f["train"][t0][k0][:]
    print(f"\n  Drone ({t0}/{k0}): ch0_mean={drone_sample[0].mean():.6f}, "
          f"ch2_mean={drone_sample[2].mean():.6f} (should be different from ch0)")

    f.close()
    print("\nDone! HDF5 patched in-place. Ready to re-upload to Modal.")


if __name__ == "__main__":
    main()