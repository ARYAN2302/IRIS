#!/usr/bin/env python3
"""
Pull IRIS artifacts from Modal to local Mac.

Downloads:
  1. lejepa_v11_best.pt     — trained v11 encoder (from iris-models-v11 volume)
  2. intent_head.pt         — trained intent head (from iris-intent volume, if exists)
  3. iris_rfuav.h5          — RFUAV dataset (from iris-data volume, optional, large)
  4. iris_matched_bg.h5     — matched backgrounds (from iris-matched-bg volume, optional)
  5. t4_pipeline_test.json  — T4 test results (from iris-results volume, if exists)
  6. honest_eval.md/json    — honest evaluation results (from iris-results volume, if exists)
  7. adversarial_robustness.md/json — robustness results (from iris-results volume, if exists)

Usage:
    python scripts/t4/pull_artifacts.py
    python scripts/t4/pull_artifacts.py --skip-hdf5   # skip large HDF5 files
    python scripts/t4/pull_artifacts.py --only intent # only pull intent head
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import modal


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# (volume_name, remote_path, local_path, is_optional, description)
ARTIFACTS = [
    ("iris-models-v11", "/lejepa_v11_best.pt", "models/lejepa_v11_best.pt", False,
     "Trained v11 encoder (~13 MB)"),
    ("iris-intent", "/intent_head.pt", "models/intent_head.pt", True,
     "Trained intent head (~170 KB, may not exist yet)"),
    ("iris-data", "/iris_rfuav.h5", "data/iris_rfuav.h5", True,
     "RFUAV dataset (large, ~1-5 GB)"),
    ("iris-matched-bg", "/iris_matched_bg.h5", "data/iris_matched_bg.h5", True,
     "Matched backgrounds (smaller)"),
    ("iris-results", "/t4_pipeline_test.json", "results/t4_pipeline_test.json", True,
     "T4 pipeline test results"),
    ("iris-results", "/honest_eval.json", "results/honest_eval.json", True,
     "Honest evaluation JSON"),
    ("iris-results", "/honest_eval.md", "results/honest_eval.md", True,
     "Honest evaluation markdown"),
    ("iris-results", "/intent_results.md", "results/intent_results.md", True,
     "Intent training results"),
    ("iris-results", "/adversarial_robustness.json", "results/adversarial_robustness.json", True,
     "Adversarial robustness JSON"),
    ("iris-results", "/adversarial_robustness.md", "results/adversarial_robustness.md", True,
     "Adversarial robustness markdown"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Download helper
# ─────────────────────────────────────────────────────────────────────────────


def download_from_volume(volume_name: str, remote_path: str, local_path: str, timeout: int = 600) -> bool:
    """Download a file from a Modal volume. Returns True on success."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    app = modal.App(f"iris-pull-{volume_name}")
    vol = modal.Volume.from_name(volume_name, create_if_missing=False)

    image = modal.Image.from_registry("python:3.11-slim").pip_install("h5py==3.12.1", "numpy==1.26.4")

    @app.function(image=image, volumes={"/vol": vol}, timeout=timeout)
    def read_file() -> bytes:
        path = f"/vol{remote_path}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found in volume")
        with open(path, "rb") as f:
            return f.read()

    try:
        with app.run():
            vol.reload()
            data = read_file.remote()
        local_path.write_bytes(data)
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  [ok] {local_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        if "not found" in str(e).lower() or "FileNotFoundError" in str(e):
            print(f"  [skip] {remote_path} not in {volume_name} (may not be generated yet)")
        else:
            print(f"  [error] {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pull IRIS artifacts from Modal")
    parser.add_argument("--skip-hdf5", action="store_true", help="Skip large HDF5 files")
    parser.add_argument("--only", choices=["encoder", "intent", "hdf5", "results"],
                        help="Only pull specific artifact type")
    args = parser.parse_args()

    print("=" * 60)
    print("IRIS — Pull Artifacts from Modal")
    print("=" * 60)

    pulled = []
    skipped = []

    for vol_name, remote_path, local_path, is_optional, desc in ARTIFACTS:
        # Filter by --only
        if args.only:
            if args.only == "encoder" and "lejepa" not in remote_path:
                continue
            elif args.only == "intent" and "intent_head" not in remote_path:
                continue
            elif args.only == "hdf5" and not remote_path.endswith(".h5"):
                continue
            elif args.only == "results" and vol_name != "iris-results":
                continue

        # Filter by --skip-hdf5
        if args.skip_hdf5 and remote_path.endswith(".h5"):
            skipped.append(local_path)
            continue

        print(f"\n  [{desc}]")
        success = download_from_volume(vol_name, remote_path, local_path)
        if success:
            pulled.append(local_path)
        elif not is_optional:
            print(f"  [warning] required artifact failed: {local_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Pulled:  {len(pulled)} files")
    for p in pulled:
        print(f"    ✓ {p}")
    if skipped:
        print(f"\n  Skipped: {len(skipped)} files")
        for p in skipped:
            print(f"    - {p}")

    # Next steps
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    if any("lejepa_v11" in p for p in pulled):
        print("  [ok] encoder downloaded — can run inference")
    if any("intent_head" in p for p in pulled):
        print("  [ok] intent head downloaded — live_demo will show intent")
    if any("drone_centroid" in p for p in pulled):
        print("  [ok] Mahalanobis centroid available")
    else:
        print("\n  [info] to compute Mahalanobis centroid, run:")
        print("           python scripts/pull_from_modal.py")
    print("\n  [info] to verify everything works locally:")
    print("           python scripts/live_demo.py --no-display")
    print("           python scripts/spoof_demo.py --synthetic")


if __name__ == "__main__":
    main()
