#!/usr/bin/env python3
"""
Experiment 1: DADS Loader Audit — fix acoustic data starvation.

Context: DADS on HF has 180,320 clips (163K drone / 16K no-drone) but
training used only 80 drone clips. This audit runs on Modal to verify
loader behavior and re-baseline acoustics.

Run: modal run extension/scripts/experiments/audit_dads_loader.py
Requires: Modal key (modal token new) + HF_TOKEN if needed.
"""
import modal

app = modal.App("iris-audit-dads")
VOL = modal.Volume.from_name("iris-cuas-data", create_if_missing=True)
IMAGE = (
    modal.Image.debian_slim()
    .pip_install("datasets==2.20.0", "huggingface_hub==0.24.0", "librosa==0.10.0",
                 "soundfile>=0.12.1", "numpy==1.26.4", "torch==2.5.1",
                 "scikit-learn==1.6.1", "scipy==1.14.1")
    .env({"HF_HUB_OFFLINE": "0"})
)

CORE = r'''
import os, json, collections
from datasets import load_dataset
import librosa, numpy as np, soundfile as sf
from pathlib import Path

# 1. Stream DADS and count what the loader actually ingests
print("Loading DADS streaming to count samples...")
ds = load_dataset("geronimobasso/drone-audio-detection-samples", split="train", streaming=True)
# DADS is parquet sharded — streaming avoids full download

counts = collections.Counter()
durations = []
try:
    for i, row in enumerate(ds):
        # Inspect schema on first sample
        if i == 0:
            print(f"Row keys: {list(row.keys())}")
            print(f"Row sample: {{k: type(v).__name__ if not isinstance(v,(str,int,float)) else v for k,v in row.items()}}")
            # Try to dump one audio field if present
            for k in row:
                if "audio" in k.lower() or isinstance(row[k], dict):
                    print(f"  {k}: {str(row[k])[:500]}")
        # Label field varies: try label / labels / target / class
        label = row.get("label", row.get("labels", row.get("target", row.get("class", None))))
        counts[str(label)] += 1
        if i >= 5000:
            break
        if i % 1000 == 0 and i>0:
            print(f"  counted {i}: {dict(counts)}")
except Exception as e:
    print(f"Streaming stopped at {i}: {e}")

print(f"Counts in first 5k: {dict(counts)}")

# 2. Try non-streaming small slice to check total
try:
    ds_small = load_dataset("geronimobasso/drone-audio-detection-samples", split="train", streaming=False)
    print(f"Non-streaming length: {len(ds_small)}")
    print(f"Features: {ds_small.features}")
except Exception as e:
    print(f"Non-streaming load failed (expected for large): {e}")

# 3. Check current loader in repo: does it slice [0:80]?
import importlib.util, pathlib
for p in pathlib.Path("extension/src/encoders").rglob("*.py"):
    if "acoustic" in p.name.lower():
        print(f"\n=== {p} ===")
        print(open(p).read()[:3000])
        if "[:80]" in open(p).read() or "n_samples.*80" in open(p).read() or "80" in open(p).read():
            print("  -> POSSIBLE HARDCODED 80 SAMPLE SLICE!")

# 4. Check ml split used in training scripts
for p in pathlib.Path("extension/scripts").rglob("*.py"):
    if "acoustic" in p.name.lower():
        txt = open(p).read()
        if "80" in txt or "limit" in txt.lower() or "max_samples" in txt.lower():
            snippet = [l for l in txt.splitlines() if "80" in l or "limit" in l.lower() or "max_samples" in l.lower()]
            if snippet:
                print(f"\n=== {p.name} acoustic limit snippet ===")
                print("\n".join(snippet[:15]))
'''

@app.function(image=IMAGE, volumes={"/data": VOL}, timeout=600, memory=4096)
def audit():
    import subprocess, textwrap, tempfile, os
    script = tempfile.mktemp(suffix=".py")
    open(script, "w").write(CORE)
    # Write the core to a file inside the container and run it
    subprocess.run(["python3", script], check=False)

@app.local_entrypoint()
def main():
    print("Auditing DADS loader on Modal...")
    audit.remote()
    print("Done. Check logs above for: total length, label distribution, and hardcoded 80 slice.")
