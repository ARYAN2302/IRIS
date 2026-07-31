# IRIS — Identify, Recognize, Isolate, Spot

**Self-supervised drone detection on RF spectrograms with zero-shot generalization to unseen drone types.**

IRIS learns a general representation of "drone-ness" from RF spectrograms using LeJEPA (Learned Joint Embedding Predictive Architecture) + SIGReg + Hierarchical Supervised Contrastive Learning. It detects drone types it has never seen during training, via Mahalanobis distance in a regularized embedding space.

---

## What IRIS Does

- **Zero-shot drone detection**: AUC 0.978 on 7 drone types never seen during training
- **Self-supervised pretraining**: LeJEPA + SIGReg (no labels needed for representation learning)
- **Hierarchical contrastive learning**: Forces all drone types into a unified "drone" region in embedding space while preserving type-level structure
- **Mahalanobis OOD detection**: Principled detection via distance to drone centroid (with L2-normalization per Mahalanobis++ 2025)
- **Edge-deployable**: 3.4M params, ~13MB ONNX, runs real-time on M1 Mac / Jetson-class hardware

---

## Architecture

```
┌──────────────────┐    ┌─────────────────┐    ┌────────────────────┐
│ RF Spectrogram   │───▶│ CNNEncoder      │───▶│ 256-dim Embedding  │
│ (2, 256, 256)    │    │ 6-layer CNN     │    │ (L2-normalized)    │
│ log-mag + grad   │    │ 3.4M params     │    └─────────┬──────────┘
└──────────────────┘    └─────────────────┘              │
                                                         ▼
                                              ┌──────────────────────┐
                                              │ Mahalanobis Detector │
                                              │ dist to drone cluster│
                                              │ threshold = 99th pct │
                                              └──────────────────────┘
```

### Training losses (v11)
- **LeJEPA invariance loss**: predictor predicts target embedding from context embedding
- **SIGReg**: sketched isotropic Gaussian regularizer — forces embedding distribution toward N(0, I), guarantees linear identifiability (Klindt, LeCun, Balestriero 2026)
- **Hierarchical SupCon** (Salesforce CVPR 2022):
  - Fine-grained (weight 0.7): same drone type = strong positive
  - Coarse-grained (weight 0.3): all drone types = weak positive (drone-ness)

---

## Repository Structure

```
iris/
├── src/
│   ├── iris_inference.py         # Clean inference module + Mahalanobis detector
│   ├── intent_head.py            # RF-only intent classifier (3-class)
│   ├── remote_id_decoder.py      # DJI DroneID + ASTM F3411 decoder
│   ├── remote_id_auth.py         # RF fingerprint authentication
│   ├── encoder.py                # Original v7-era encoder (legacy)
│   ├── model.py                  # Original LeJEPA model (legacy)
│   ├── sigreg.py                 # SIGReg loss
│   └── ...
├── scripts/
│   ├── train_modal_v11.py        # Train v11 encoder on Modal A100
│   ├── pull_from_modal.py        # Download checkpoint + compute centroid
│   ├── export_onnx.py            # ONNX export for edge inference
│   ├── edge_benchmark.py         # SWaP-C benchmark
│   ├── live_demo.py              # Real-time waterfall + detection
│   ├── honest_eval.py            # Recording-grouped CV + SNR curve
│   ├── train_intent.py           # Train intent head on Modal
│   ├── spoof_demo.py             # Remote ID spoof detection demo
│   ├── adversarial_test.py       # FGSM/PGD/DRFM robustness
│   ├── detect.py                 # Original v11 detection script
│   └── ...
├── scripts/t4/
│   ├── test_pipeline_t4.py       # T4 pipeline test (all phases, cheap)
│   └── pull_artifacts.py         # Pull artifacts from Modal to local
├── configs/
│   ├── default.yaml
│   ├── split.json                # Train/holdout split (30 train, 7 holdout)
│   └── ...
├── data/manifests/               # Dataset manifests
├── notebooks/                    # UMAP visualization
├── build.md                      # Original build doc (architecture details)
├── requirements.txt              # Training dependencies
└── requirements_demo.txt         # Demo dependencies (M1 Mac)
```

---

## Installation

### For training (Modal)
```bash
pip install -r requirements.txt
```

### For demos / inference (local Mac)
```bash
pip install -r requirements_demo.txt
```

---

## Quick Start

### 1. Pull trained checkpoint from Modal
```bash
python scripts/pull_from_modal.py
```
Downloads `models/lejepa_v11_best.pt` and computes `models/drone_centroid.npz` (Mahalanobis centroid + threshold).

### 2. Run the live demo
```bash
# Synthetic mode (no files needed)
python scripts/live_demo.py

# Replay HDF5 holdout spectrograms
python scripts/live_demo.py --mode hdf5

# Playback I/Q file
python scripts/live_demo.py --mode iq_file --file path/to/recording.cf32
```

### 3. Test the full pipeline on T4 (cheap, ~$0.30)
```bash
modal run scripts/t4/test_pipeline_t4.py
```
Verifies inference, honest evaluation, intent training, spoof detection, and FGSM adversarial robustness all work end-to-end before running on your local machine.

---

## Datasets

| Dataset | Role | Drone Types | Public |
|---------|------|-------------|--------|
| **RFUAV** (kitofrank/RFUAV on HuggingFace) | Primary train/test | 37 types | Yes (Apache-2.0) |
| **DroneRF** (Mendeley) | Cross-dataset validation | 3 types | Yes |
| **DRFF-R2** (SciDB) | Per-transmitter fingerprinting | 26 DJI units / 8 models | Yes (CC-BY 4.0) |

Train/holdout split: 30 drone types for training, 7 completely unseen types for zero-shot evaluation. See `configs/split.json`.

---

## Evaluation Protocol

### Honest evaluation (Shulman 2026)

Drone RF benchmarks commonly inflate accuracy by 30+ points via segment-level cross-validation (segments of the same recording split across train/test). IRIS uses **recording-grouped CV** — a recording's segments are never split across train/test. This measures generalization to new recordings, not memorization.

### L2-normalized Mahalanobis (Mahalanobis++ 2025)

Embeddings are L2-normalized before Mahalanobis distance computation. This significantly improves OOD detection, especially for cross-dataset transfer.

### Metrics
- **AUC** on holdout drones vs matched backgrounds (spectrally-shaped noise — hard negatives)
- **SNR degradation curve** at +25 to -12 dB AWGN
- **Per-type breakdown** across 7 holdout drone types
- **Bootstrap 95% confidence intervals** (10,000 iterations)

---

## Key Results (v11)

| Metric | Value |
|---|---|
| Overall AUC (matched BG) | 0.978 |
| Per-pair drone-closer rate | 98.6% |
| Bootstrap 95% CI | [0.979, 0.984] |
| Threshold (Mahalanobis) | 27.42 |
| TPR @ threshold | 88.1% |
| FPR @ threshold | 5.7% |
| Encoder params | 3.4M |
| ONNX model size | ~13 MB |

Per-type breakdown, SNR curves, and cross-dataset transfer numbers are generated by `scripts/honest_eval.py` → `results/honest_eval.md`.

---

## Theoretical Foundation

IRIS is built on LeJEPA (Klindt, LeCun, Balestriero 2026) — "Linearly Identified JEPA." The core insight: RF hardware fingerprints (carrier frequency offset, oscillator phase noise, amplifier nonlinearities, thermal noise) are Gaussian-distributed per DSP physics, satisfying LeJEPA's core assumption (Theorem 1). This guarantees the encoder is linearly identifiable — meaning different drones get different embeddings.

SIGReg (Sketched Isotropic Gaussian Regularizer) uses the Cramér-Wold theorem to force the embedding distribution toward N(0, I) via K=256 random 1D projections, making Theorem 3's identifiability bound apply.

See `build.md` for the full theoretical justification.

---

## References

- Klindt, LeCun, Balestriero. "Linearly Identified JEPA." 2026.
- Zheng et al. "Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework." CVPR 2022.
- Shulman. "How Much Do RF Drone Benchmarks Overstate?" arXiv:2607.01025, 2026.
- Mahalanobis++ (L2-normalization for OOD detection). 2025.
- Lee et al. "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks." NeurIPS 2018.
- Shi et al. "RFUAV: A Benchmark Dataset for UAV Detection and Identification." arXiv:2503.09033, 2025.

---

## License

This project is for research and demonstration purposes. See dataset licenses (RFUAV: Apache-2.0, DRFF-R2: CC-BY 4.0) for data usage terms.

## Citation

If you use IRIS in your research, cite:
```bibtex
@software{iris_2026,
  title={IRIS: Self-supervised drone detection via LeJEPA on RF spectrograms},
  author={Aryan},
  year={2026},
  url={https://github.com/ARYAN2302/IRIS}
}
```
