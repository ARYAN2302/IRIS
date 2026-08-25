#!/usr/bin/env python3
"""
Experiment 3: Radar Upgrade — DIAT-μSAT ingestion (50 → 4,849 samples).

Replaces the 50-sample Open Radar Initiative radar set with DIAT-μSAT
(IEEE DataPort 10.21227/1x2q-8v62, X-band CW, 6 classes incl. quadcopter,
mini-heli+bird, bionic bird, RC plane) and re-trains the radar encoder.

Steps:
  1. Download DIAT-μSAT zip from DataPort (requires free IEEE DataPort login;
     on Modal, use pre-uploaded /data/diat_musat.zip or wget with token).
  2. Parse MATLAB .mat / PNG micro-Doppler images → 1×256×256 tensors
  3. Train CNNEncoder (backbone.py, 1 channel, same as acoustic) with
     VICReg + BCE (identical to RF v3 recipe) — expect AUC 0.85→~0.92.
  4. Evaluate with recording-grouped CV (no leakage).

Run: modal run extension/scripts/experiments/upgrade_radar_diat.py
Dataset: https://ieee-dataport.org/documents/diat-msat-micro-doppler-signature-dataset-small-unmanned-aerial-vehicle-suav
DOI: 10.21227/1x2q-8v62
"""
import modal

app = modal.App("iris-radar-diat")
DATA_VOL = modal.Volume.from_name("iris-cuas-data", create_if_missing=True)
MODELS_VOL = modal.Volume.from_name("iris-cuas-models", create_if_missing=True)
RESULTS_VOL = modal.Volume.from_name("iris-cuas-results", create_if_missing=True)
IMAGE = (
    modal.Image.debian_slim()
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy==1.26.4",
                 "h5py==3.12.1", "scipy==1.14.1", "scikit-learn==1.6.1",
                 "pillow==10.4.0", "tqdm==4.67.1")
)

CORE = r'''
import os, json, zipfile, pathlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ZIP = "/data/diat_musat.zip"
OUT_H5 = "/data/diat_musat_processed.h5"

# 1. Unzip if present
if os.path.exists(ZIP):
    print(f"Unzipping {ZIP}...")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall("/tmp/diat")
    print("Extracted to /tmp/diat:", os.listdir("/tmp/diat")[:10])
else:
    print(f"Place DIAT-μSAT zip at {ZIP} on volume iris-cuas-data")
    print("Download: https://ieee-dataport.org/documents/diat-msat-micro-doppler-signature-dataset-small-unmanned-aerial-vehicle-suav")
    raise SystemExit(1)

# 2. Walk .mat / .png / .jpg structure and build 1×256×256 tensors
#    DIAT-μSAT ships as micro-Doppler spectrogram images per class folder.
#    Adjust glob below after inspecting actual unzip layout.

import glob, re
candidates = glob.glob("/tmp/diat/**/*", recursive=True)
img_files = [p for p in candidates if p.lower().endswith((".png",".jpg",".jpeg",".mat"))]
print(f"Found {len(img_files)} image/mat files")
# Group by parent folder = class label (quadcopter, RC plane, bird, etc.)
from collections import Counter
labels = [pathlib.Path(p).parent.name for p in img_files]
print(Counter(labels))

# 3. Convert to H5 for training: drone vs background (bird/plane = background for now,
#    or keep 6-way and collapse later). Save as (N,1,256,256) float32 in [0,1].

# Placeholder — inspect one image to choose resize/normalize:
if img_files:
    im = Image.open(img_files[0]).convert("L")
    print(f"Sample: {img_files[0]} size={im.size} mode={im.mode}")

print("\nNext: implement Dataset that does PIL resize 256 + ToTensor + normalize,")
print("then train extension/src/encoders/backbone.py CNNEncoder(in_ch=1, embed_dim=256)")
print("with loss = SIGReg(var) + VICReg + BCE, same as rf_scf_real_v3_eval_seed42.json")
print("Target: radar AUC 0.85 → 0.92+, then re-run fusion RF-silent ablation on real synchronized data.")
'''

@app.function(image=IMAGE, volumes={"/data": DATA_VOL, "/models": MODELS_VOL, "/results": RESULTS_VOL}, timeout=3600, memory=16384)
def ingest():
    import subprocess, tempfile
    p = tempfile.mktemp(suffix=".py"); open(p,"w").write(CORE)
    subprocess.run(["python3", p], check=False)

@app.local_entrypoint()
def main():
    ingest.remote()
