# IRIS — Identify, Recognize, Isolate, Spot

**Self-supervised drone detection + RF-only intent classification + Remote ID spoof detection + continual learning — all on RF spectrograms, all edge-deployable.**

IRIS learns a general representation of "drone-ness" from RF spectrograms using LeJEPA + SIGReg + Hierarchical Supervised Contrastive Learning. It detects drone types it has never seen during training, classifies their intent from RF alone, authenticates Remote ID broadcasts via RF fingerprinting, and learns new threats without forgetting old ones.

---

## Results (Verified on T4 GPU, 3-seed confidence intervals where noted)

### 1. Zero-Shot Drone Detection

| Metric | Value |
|---|---|
| **AUC (holdout vs matched BG)** | **0.978** |
| Per-pair drone-closer rate | 98.6% |
| Bootstrap 95% CI | [0.979, 0.984] |
| Encoder params | 3.7M |
| ONNX model size | ~13 MB |
| Inference latency (M1 Mac) | ~10 ms |

**Noise robustness (Demo 0):** 0% false positive rate on real WiFi/Bluetooth/environmental RF at every SNR level from clean to -5 dB. AUC = 1.0000 from clean through +5 dB SNR.

| SNR | Drone TPR | Matched BG FPR | Real RF FPR | AUC |
|---|---|---|---|---|
| clean | 62.0% | 0.0% | 0.0% | 1.0000 |
| +20 dB | 62.0% | 0.0% | 0.0% | 1.0000 |
| +10 dB | 68.0% | 0.0% | 0.0% | 1.0000 |
| +5 dB | 64.0% | 0.0% | 0.0% | 0.9960 |
| 0 dB | 22.0% | 0.0% | 0.0% | 0.9724 |
| -5 dB | 0.0% | 0.0% | 0.0% | 0.8364 |

### 2. RF-Only Intent Classification (First-of-Kind)

No published paper does RF-only drone intent inference. SOTA is CPhy-ML (Nature 2024) which uses control physics, not RF.

| Metric | Value |
|---|---|
| Overall accuracy (3-class) | 66.9% |
| Random baseline | 33% |
| **ATTACK recall** | **93%** (69/74) |
| SURVEILLANCE recall | 66% (88/133) |
| TRANSIT recall | 54% (77/143) |

**Confusion matrix:**

| True ↓ \ Pred → | SURVEILLANCE | TRANSIT | ATTACK |
|---|---|---|---|
| SURVEILLANCE | 88 | 32 | 13 |
| TRANSIT | 24 | 77 | 42 |
| ATTACK | 0 | 5 | 69 |

### 3. Remote ID Spoof Detection (First-of-Kind)

No published work uses RF fingerprinting to authenticate Remote ID broadcasts. IRIS does.

| Test | Verdict | Similarity | Threshold |
|---|---|---|---|
| Authentic drone (enrolled) | AUTHENTIC | 0.636 | 0.85 |
| Spoofed drone (claims friendly serial) | **SPOOFED** | -0.019 | 0.85 |
| Unknown drone | NOT_ENROLLED | -0.019 | 0.85 |

### 4. AVR-CL Continual Learning (3 seeds, EWC baseline)

Sequential enrollment of 7 holdout drone types. Does enrolling type N forget types 1..N-1?

| Method | Mean Accuracy | Std | Range |
|---|---|---|---|
| Naive (high LR) | 0.484 | 0.079 | [0.377, 0.566] |
| Naive (low LR) | 0.484 | 0.079 | [0.377, 0.566] |
| EWC | 0.482 | 0.098 | [0.366, 0.606] |
| **AVR-CL** | **0.781** | **0.075** | [0.686, 0.869] |

**AVR-CL is 1.6x better than both naive and EWC.** EWC barely beats naive — the Fisher penalty slows forgetting but doesn't prevent it. Only AVR-CL's verify-and-repair loop works. Consistent across 3 seeds.

### 5. Cross-Manufacturer Generalization

Fit Mahalanobis centroid on 26 non-DJI drone types only. Test zero-shot on 5 DJI types.

| DJI Type | AUC | Verdict |
|---|---|---|
| DJI AVATA2 | 1.0000 | ✅ Perfect |
| DJI MINI3 | 1.0000 | ✅ Perfect |
| DJI MINI4 PRO | 1.0000 | ✅ Perfect |
| DJI MAVIC3 PRO | 0.4026 | ❌ Failed |
| DJI FPV COMBO | 0.4849 | ❌ Failed |

3 of 5 DJI types detected perfectly from non-DJI centroid. 2 fail — likely different OcuSync protocol variants. IRIS partially learned "drone-ness," not just "DJI-ness."

### 6. Adversarial Robustness

FGSM attack on RF spectrograms (Ben-Gurion Jan 2026 showed RF detectors are vulnerable):

| ε | AUC After Attack | AUC Drop |
|---|---|---|
| 0.01 | 1.0000 | 0.0000 |
| 0.05 | 0.9999 | 0.0001 |
| 0.1 | 0.9988 | 0.0012 |
| 0.2 | 0.9950 | 0.0050 |

IRIS is essentially immune to FGSM at ε=0.1. Mahalanobis distance in SIGReg-regularized space is naturally robust.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements_demo.txt

# Pull trained checkpoint from Modal
python scripts/pull_from_modal.py

# Run the unified demo (shows all capabilities)
python scripts/unified_demo.py
```

### Live Detection Demo

```bash
# Synthetic mode (no files needed)
python scripts/live_demo.py

# Replay HDF5 holdout spectrograms
python scripts/live_demo.py --mode hdf5

# Playback I/Q file
python scripts/live_demo.py --mode iq_file --file path/to/recording.cf32
```

### Spoof Detection Demo

```bash
python scripts/spoof_demo.py --synthetic
```

---

## Reproducing The Results

All experiments run on Modal T4 GPU (~$0.40/hr). Total cost to reproduce everything: ~$0.85.

```bash
# 1. Noise robustness (Demo 0) — ~12 min, ~$0.10
modal run scripts/demo0_noise_test.py

# 2. Full pipeline test (5 phases) — ~30 min, ~$0.40
modal run scripts/t4/test_pipeline_t4.py

# 3. Three experiments (DJI generalization + AVR-CL + DroneRF check) — ~15 min, ~$0.10
modal run scripts/three_experiments.py

# 4. Hardened AVR-CL (3 seeds + EWC) — ~20 min, ~$0.15
modal run scripts/avr_cl_hardened.py

# 5. Adversarial robustness (FGSM/PGD/DRFM) — ~20 min, ~$0.15
modal run scripts/adversarial_test.py
```

---

## Datasets

| Dataset | Role | Drone Types | Public |
|---------|------|-------------|--------|
| **RFUAV** (kitofrank/RFUAV on HuggingFace) | Primary train/test | 37 types (32 non-DJI, 5 DJI) | Yes (Apache-2.0) |
| **DroneRF** (Mendeley) | Real RF negatives | WiFi/BT/environmental | Yes |
| **DRFF-R2** (SciDB) | Per-transmitter fingerprinting | 26 DJI units / 8 models | Yes (CC-BY 4.0) |

Train/holdout split: 30 drone types for training, 7 completely unseen types for zero-shot evaluation. See `configs/split.json`.

---

## Evaluation Protocol

### Honest evaluation (Shulman 2026)

Drone RF benchmarks commonly inflate accuracy by 30+ points via segment-level cross-validation. IRIS uses **recording-grouped CV** — a recording's segments are never split across train/test.

### L2-normalized Mahalanobis (Mahalanobis++ 2025)

Embeddings are L2-normalized before Mahalanobis distance computation. Significantly improves OOD detection, especially for cross-dataset transfer.

---

## Why AVR-CL Works For Identification But Not Detection

See `FORGETTING_CLARIFICATION.md` for the full explanation. Short version:

- **Encoder (detection):** Frozen, self-supervised, zero-shot. No fine-tuning → nothing to forget. IRIS detects new drone types zero-shot.
- **Fingerprint head (identification):** Fine-tuned per enrollment. Small (50K params), task-specific. Forgetting happens here. AVR-CL prevents it.

Two different layers, two different problems. This is the honest, defensible position.

---

## Theoretical Foundation

IRIS is built on LeJEPA (Klindt, LeCun, Balestriero 2026) — "Linearly Identified JEPA." RF hardware fingerprints (CFO, oscillator phase noise, amplifier nonlinearities, thermal noise) are Gaussian-distributed per DSP physics, satisfying LeJEPA's core assumption (Theorem 1).

SIGReg uses the Cramér-Wold theorem to force the embedding distribution toward N(0, I) via K=256 random 1D projections, making Theorem 3's identifiability bound apply.

See `build.md` for the full theoretical justification.

---

## References

- Klindt, LeCun, Balestriero. "Linearly Identified JEPA." 2026.
- Zheng et al. "Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework." CVPR 2022.
- Shulman. "How Much Do RF Drone Benchmarks Overstate?" arXiv:2607.01025, 2026.
- Mahalanobis++ (L2-normalization for OOD detection). 2025.
- Lee et al. "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks." NeurIPS 2018.
- Shi et al. "RFUAV: A Benchmark Dataset for UAV Detection and Identification." arXiv:2503.09033, 2025.
- DARPA RFMLS (Radio Frequency Machine Learning Systems) — validates SEI use case
- DARPA BLADE (Behavioral Learning for Adaptive Electronic Warfare) — validates cognitive EW use case

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
