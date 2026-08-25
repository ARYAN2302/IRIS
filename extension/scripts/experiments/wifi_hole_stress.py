#!/usr/bin/env python3
"""
Experiment 2: WiFi-Hole Stress Test — the decisive credibility experiment.

Frozen v3 RF detector (SCF + VICReg + Mahalanobis) was trained on anechoic
Zenodo BG (not operational). Wi-Fi/LTE are also OFDM — this experiment
measures false alarms on dense urban Wi-Fi/LTE captures.

Protocol:
  - Load frozen rf_scf_real_v3_encoder_seed42.pt + centroid/cov
  - Score three BG sets: (a) Zenodo anechoic BG, (b) synthetic Wi-Fi/LTE
    captures OR real urban captures (if you collect 30 min on 2.4/5.8GHz),
    (c) ESC-50 environmental (as control)
  - Report: BG score distribution, threshold at 99p/99.9p, FP rate per set,
    ROC vs DRFF-R2 drones. If urban Wi-Fi FP >> 0%, the "CP=drone-ness"
    narrative breaks — reframe as "protocol-topology fingerprint".

Run: modal run extension/scripts/experiments/wifi_hole_stress.py
Optionally: --urban-h5 /path/to/urban_captures.h5  (local IQ → SCF via same pipeline)
"""
import modal

app = modal.App("iris-wifi-hole")
DATA_VOL = modal.Volume.from_name("iris-cuas-data", create_if_missing=True)
RESULTS_VOL = modal.Volume.from_name("iris-cuas-results", create_if_missing=True)
MODELS_VOL = modal.Volume.from_name("iris-cuas-models", create_if_missing=True)
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install("libhdf5-dev")
    .pip_install("torch==2.5.1", "numpy==1.26.4", "h5py==3.12.1",
                 "scikit-learn==1.6.1", "scipy==1.14.1")
)

# Minimal core: load frozen encoder + centroid, score BG sets.
# Full version will reuse extension/src/encoders/backbone.py + SCF loader.
CORE = r'''
import os, json, numpy as np, torch, h5py
from sklearn.metrics import roc_auc_score

# Paths on Modal volumes (mirror your existing volume layout)
CKPT = "/models/rf_scf_real_v3_encoder_seed42.pt"
CENTROID_NPZ = "/models/rf_scf_real_v3_centroid.npz"  # if you saved one; else refit on train

# 1. Load encoder (backbone.py: CNNEncoder)
import sys; sys.path.insert(0, "/")
from extension.src.encoders.backbone import CNNEncoder
ckpt = torch.load(CKPT, map_location="cpu")
enc = CNNEncoder(in_ch=2, embed_dim=256)
# Adapt key prefix if ckpt was saved as state_dict vs {"model": ...}
sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
sd = {k.replace("encoder.","").replace("backbone.",""):v for k,v in sd.items() if "encoder" in k or v.ndim>=2}
try:
    enc.load_state_dict(sd, strict=False)
except Exception as e:
    print(f"Strict load failed, trying non-strict: {e}")
    enc.load_state_dict(sd, strict=False)
enc.eval()

# 2. Load or refit centroid — prefer saved npz for reproducibility
if os.path.exists(CENTROID_NPZ):
    npz = np.load(CENTROID_NPZ)
    centroid, cov_inv, threshold_99p, threshold_999p = npz["centroid"], npz["cov_inv"], float(npz["t99"]), float(npz["t999"])
else:
    print("No centroid npz — refitting on train SCF (Zenodo) — THIS IS APPROXIMATE")
    # Replace with your actual train SCF path on volume
    h5 = h5py.File("/data/zenodo_scf.h5","r")
    train = h5["train"][:]
    with torch.no_grad():
        embs = enc(torch.from_numpy(train).float()).numpy()
    from src.iris_inference import fit_mahalanobis
    centroid, cov_inv = fit_mahalanobis(embs)
    # calibrate thresholds on train
    import scipy as _scipy
    dists = np.sqrt(((embs-centroid) @ cov_inv * (embs-centroid)).sum(1))
    threshold_99p, threshold_999p = np.percentile(dists, [99, 99.9])

from src.iris_inference import compute_mahalanobis

def score_h5(path, dataset_key):
    h = h5py.File(path,"r")
    X = h[dataset_key][:]
    with torch.no_grad():
        embs = enc(torch.from_numpy(X).float()).numpy()
    dists = compute_mahalanobis(embs, centroid, cov_inv, l2_normalize=True)
    return dists

# 3. Score each BG set at both thresholds
for name, path, key in [
    ("zenodo_bg_anechoic", "/data/zenodo_bg.h5", "bg"),
    ("urban_wifi_lte",     "/data/urban_wifi.h5", "bg"),   # collect or synthesize
    ("esc50_control",      "/data/esc50_bg.h5",   "bg"),
]:
    if not os.path.exists(path):
        print(f"SKIP {name}: {path} not found — create by capturing 30min urban IQ on 2.4/5.8GHz and converting via SCF pipeline")
        continue
    dists = score_h5(path, key)
    for th, label in [(threshold_99p,"99p"), (threshold_999p,"99.9p")]:
        fp = (dists < th).mean() if False else (dists > th).mean()  # check direction: Mahalanobis far = BG
        # NOTE: verify direction — in IRIS, small distance = drone. So FP = dist < threshold on BG is WRONG.
        # Correct: BG is far, so BG counted as drone if dist < threshold. Fix below:
        # fp = (dists <= th).mean()  # BG incorrectly called drone
        pass

# Correct scoring: distance small = drone, large = BG. So BG FP = dist <= threshold
# Recompute cleanly:
import pathlib
results = {}
for name, path, key in [("zenodo_bg_anechoic","/data/zenodo_bg.h5","bg"),("urban_wifi_lte","/data/urban_wifi.h5","bg"),("esc50_control","/data/esc50_bg.h5","bg")]:
    if not os.path.exists(path): continue
    dists = score_h5(path, key)
    for th, lab in [(threshold_99p,"99p"),(threshold_999p,"99.9p")]:
        fp = float((dists <= th).mean())
        results[f"{name}@{lab}"] = {"n": len(dists), "fp_rate": fp, "dists_mean": float(dists.mean()), "dists_p50": float(np.median(dists))}
# Also ROC vs drones
try:
    drone_dists = score_h5("/data/drff_r2.h5","drone")
    bg_dists = score_h5("/data/zenodo_bg.h5","bg")
    y = np.array([1]*len(drone_dists) + [0]*len(bg_dists))
    # Mahalanobis: lower = more drone-like, so negate for AUC
    scores = np.concatenate([-drone_dists, -bg_dists])
    auc = roc_auc_score(y, scores)
    results["auc_drone_vs_zenodo_bg"] = auc
except Exception as e:
    print(f"AUC failed: {e}")

print(json.dumps(results, indent=2))
open("/results/wifi_hole_results.json","w").write(json.dumps(results, indent=2))
print("Saved to /results/wifi_hole_results.json")
'''

@app.function(image=IMAGE, volumes={"/data": DATA_VOL, "/results": RESULTS_VOL, "/models": MODELS_VOL}, timeout=1800, gpu="T4", memory=16384)
def stress():
    import subprocess, tempfile
    p = tempfile.mktemp(suffix=".py"); open(p,"w").write(CORE)
    subprocess.run(["python3", p], check=False)

@app.local_entrypoint()
def main():
    stress.remote()
