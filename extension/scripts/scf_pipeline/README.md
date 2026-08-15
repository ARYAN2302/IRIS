# SCF Pipeline — Real Drone RF Detection with Spectral Correlation Function

This subdirectory contains the production SCF (Spectral Correlation Function) pipeline for IRIS RF drone detection.

## TL;DR

**Production model**: `v3 VICReg` (trained on 6,000 real Zenodo SCF samples with VICReg loss)

| Metric | Value | Target |
|--------|-------|--------|
| DRFF-R2 holdout detection (99.9p threshold) | **99.70%** | >50% |
| BG false positive rate | **0.0000** | <1% |
| AUC (BG vs DRFF-R2) | **1.0000** | >90% |
| Effective embedding dimension | **216 / 256** | >10 |
| Source probe (domain invariance) | 0.9377 | ~0.5 |
| SNR robustness | **100% det at 0 dB SNR** | >80% |

## Pipeline Overview

```
Zenodo .bin files (12 drone types, ~5 GB raw IQ)
        │
        ▼
[1] slice_and_compute_scf_v2.py  (or spawn_zenodo_scf.py for Modal)
        │  Slices each .bin into N traces of 4096 samples
        │  Computes SCF + COH images (2, 256, 256) per trace
        ▼
zenodo_scf_samples.h5  (6,000 samples × 2 × 256 × 256 float32)
        │
        ▼
[2] spawn_train_v3_vicreg.py  ← PRODUCTION
        │  Trains CNN encoder with VICReg + SIGReg + BCE loss
        │  Fits Mahalanobis on training drone embeddings
        ▼
rf_scf_real_v3_encoder_seed42.pt  (encoder + head)
        │
        ▼
[3] spawn_extended_ood_test.py  (Option D - holdout evaluation)
        │  Tests on DRFF-R2 (OOD drones), BG, SNR sweep, LOTO cross-val
        ▼
rf_scf_v1_extended_ood.json  (full eval metrics)
```

## Files

### Training/eval scripts (Modal launchers)
- `spawn_zenodo_scf.py` — Modal: generate SCF samples from raw .bin files
- `spawn_train_real_scf.py` — Modal: train v1 (SIGReg loss, 6k samples)
- `spawn_train_real_scf_v2.py` — Modal: train v2 (SIGReg, 15k samples) — **REGRESSED, do not use**
- `spawn_train_v3_vicreg.py` — Modal: train v3 (VICReg + SIGReg, 6k samples) ← **PRODUCTION**
- `spawn_holdout_test.py` — Modal: holdout tests (Option A)
- `spawn_extended_ood_test.py` — Modal: extended OOD tests (Option D)

### Local scripts (for development)
- `slice_and_compute_scf_v2.py` — Local SCF generation (memory-mapped, slower than Modal)
- `download_zenodo_missing.py` — Download missing Zenodo files
- `verify_zenodo.py` — Sanity check Zenodo .bin files load correctly

### Results (in `results/`)
- `rf_scf_real_eval_seed42.json` — v1 eval (SIGReg, 6k samples)
- `rf_scf_real_v2_eval_seed42.json` — v2 eval (SIGReg, 15k samples, regressed)
- `rf_scf_real_v3_eval_seed42.json` — **v3 eval (VICReg, 6k samples) ← PRODUCTION**
- `rf_scf_real_holdout_tests.json` — Option A holdout test results
- `rf_scf_v1_extended_ood.json` — Option D extended OOD test results
- `ABCD_SUMMARY.md` — Full A/B/C/D comparison report

### Documentation (in `docs/`)
- `INTEGRATION_REPORT.md` — Zenodo data source documentation

## How to Reproduce

### 1. Generate SCF samples from Zenodo .bin files
Prerequisite: 12 Zenodo `.bin` files in Modal volume `/raw_iq/`.

```bash
python extension/scripts/scf_pipeline/spawn_zenodo_scf.py
```

Produces `/data/zenodo_scf_samples.h5` (6,000 samples) on Modal volume `iris-cuas-data`.

### 2. Train v3 encoder (production)
```bash
python extension/scripts/scf_pipeline/spawn_train_v3_vicreg.py
```

Produces:
- `/models/rf_scf_real_v3_encoder_seed42.pt`
- `/results/rf_scf_real_v3_eval_seed42.json`

### 3. Run extended OOD tests
```bash
python extension/scripts/scf_pipeline/spawn_extended_ood_test.py
```

Produces `/results/rf_scf_v1_extended_ood.json`.

## Comparison: v1 vs v2 vs v3

| Model | Loss | Train samples | DRFF-R2 det | BG FP | AUC | Eff dim |
|-------|------|---------------|-------------|-------|-----|---------|
| v1 | SIGReg + BCE | 6,000 | 0.9850 (99p) | 0.0000 | 1.0000 | 2.04 |
| v2 | SIGReg + BCE | 15,000 | 0.2880 (99p) | 0.0000 | 1.0000 | 1.98 |
| **v3** | **VICReg + SIGReg + BCE** | **6,000** | **0.9970 (99.9p)** | **0.0000** | **1.0000** | **216.26** |

### Why v2 regressed
v2 expanded training data 2.5× but used the same SIGReg loss. The encoder overfit to Zenodo-specific features, making the Mahalanobis threshold too tight for DRFF-R2 (OOD drones).

### Why v3 wins
v3 keeps v1's training set (6k samples) but adds VICReg loss:
- **Variance loss**: penalize per-dim std < 1.0 (prevents collapse)
- **Covariance loss**: decorrelate embedding dimensions
- Result: eff_dim jumps from 2 to 216, source probe drops from 0.99 to 0.94 (more domain-invariant)

## Data Source

**Zenodo 4264467** — Radio-Frequency Control and Video Signal Recordings of Drones
- Author: Karel Pärlin (2020)
- URL: https://zenodo.org/records/4264467
- DOI: 10.5281/zenodo.4264467
- License: CC-BY 4.0
- Format: Interleaved int16 LE IQ, 4 bytes/complex sample
- Sample rate: 120 MSps (2.4 GHz) / 200 MSps (5.8 GHz)
- 12 drone files covering 11 distinct types (DJI, Parrot, Yuneec)

See `docs/INTEGRATION_REPORT.md` for full source attribution.

## Citation

```bibtex
@dataset{parlin_2020_4264467,
  author       = {Pärlin, Karel},
  title        = {{Radio-Frequency Control and Video Signal Recordings of Drones}},
  month        = nov,
  year         = 2020,
  publisher    = {Zenodo},
  version      = 1,
  doi          = {10.5281/zenodo.4264467},
  url          = {https://doi.org/10.5281/zenodo.4264467}
}
```

## Next Step: RF Silent

The v3 encoder is the production RF modality for the IRIS multimodal fusion system. The next step is to run the RF Silent ablation:
1. Train acoustic encoder on drone audio data
2. Train radar encoder on drone radar data
3. Train fusion head with modality dropout (per `extension/scripts/train_pipeline.py`)
4. Run `python -m extension.scripts.train_pipeline --stage rf_silent`
