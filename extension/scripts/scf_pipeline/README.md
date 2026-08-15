# IRIS Multimodal Drone Detection — RF + Acoustic + Radar Fusion

End-to-end pipeline for detecting drones using three sensor modalities:
- **RF** (Radio Frequency) — Spectral Correlation Function features from drone controllers
- **Acoustic** — Mel-spectrograms from drone propeller audio
- **Radar** — Micro-Doppler signatures from drone radar returns

The headline experiment is the **RF-Silent ablation**: with RF modality zeroed out at inference, the fusion head still retains **92.5% of full fusion accuracy** and **100% of AUC** — proving the fusion head never hard-depends on any single modality.

## Production Models

| Modality | Encoder | Dataset | Detection | AUC | Eff Dim |
|----------|---------|---------|-----------|-----|---------|
| **RF** | `rf_scf_real_v3_encoder_seed42.pt` | Zenodo 4264467 (6k SCF samples) | 99.7% | 1.000 | 216 |
| **Acoustic** | `acoustic_encoder_seed42.pt` | DADS + ESC-50 | 45.0% | 0.869 | 23 |
| **Radar** | `radar_encoder_seed42.pt` | Open Radar Initiative (50 UAV) | 10.0% | 0.850 | 13 |
| **Fusion** | `fusion_head_seed42.pt` | Synthetic paired (all 3 modalities) | 100% | 1.000 | — |

## RF Silent Ablation Results

| Configuration | Accuracy | AUC |
|---------------|----------|-----|
| Full fusion (RF + Acoustic + Radar) | **1.0000** | **1.0000** |
| RF-silent (Acoustic + Radar only) | 0.9250 | 1.0000 |
| Acoustic-silent (RF + Radar) | 1.0000 | 1.0000 |
| Radar-silent (RF + Acoustic) | 1.0000 | 1.0000 |
| RF-only | 1.0000 | 1.0000 |
| Acoustic-only | 0.9750 | 0.9950 |
| Radar-only | 0.8750 | 0.9123 |

**RF-Silent retention**: 92.5% accuracy, 100% AUC — system degrades gracefully when primary modality is unavailable.

## Pipeline Architecture

```
                ┌─────────────────────────────────────────┐
                │         MODALITY ENCODERS (frozen)       │
                │                                         │
   RF IQ ──SCF──▶ RF Encoder (VICReg) ──▶ 256-dim emb    │
   Audio ──Mel──▶ Acoustic Encoder ─────▶ 256-dim emb    │
   Radar ──RD───▶ Radar Encoder ────────▶ 256-dim emb    │
                │                                         │
                └──────────────┬──────────────────────────┘
                               │
                               ▼
                ┌──────────────────────────────────────────┐
                │  FusionHead (modality dropout p=0.3)     │
                │  Linear(768→256) → BN → GELU →           │
                │  Linear(256→256) → BN                    │
                │                                          │
                │  Input: concat 3 × 256-dim = 768-dim    │
                │  Output: 256-dim unified embedding       │
                └──────────────┬──────────────────────────┘
                               │
                               ▼
                ┌──────────────────────────────────────────┐
                │  Detection Head: Linear(256→64)→GELU→    │
                │  Linear(64→1) — drone vs BG              │
                └──────────────────────────────────────────┘
```

## Files

### Training/eval scripts (Modal launchers)
| Script | Purpose |
|--------|---------|
| `spawn_zenodo_scf.py` | Generate 6k SCF samples from Zenodo .bin files |
| `spawn_train_real_scf.py` | Train v1 RF encoder (SIGReg, 6k) |
| `spawn_train_real_scf_v2.py` | Train v2 RF encoder (SIGReg, 15k) — regressed |
| `spawn_train_v3_vicreg.py` | **Train v3 RF encoder (VICReg, 6k) ← PRODUCTION** |
| `spawn_train_acoustic.py` | Train acoustic encoder (DADS + ESC-50) |
| `spawn_train_radar.py` | Train radar encoder (Open Radar Initiative) |
| `spawn_holdout_test.py` | RF holdout tests (Option A) |
| `spawn_extended_ood_test.py` | RF extended OOD tests (Option D) |
| `spawn_fusion_rfsilent.py` | **Train fusion head + run RF Silent ablation** |

### Local scripts
- `slice_and_compute_scf_v2.py` — Local SCF generation (memory-mapped)
- `download_zenodo_missing.py` — Download missing Zenodo files
- `verify_zenodo.py` — Sanity check Zenodo .bin files

### Results (in `results/`)
- `rf_scf_real_eval_seed42.json` — RF v1 eval
- `rf_scf_real_v2_eval_seed42.json` — RF v2 eval (regressed)
- `rf_scf_real_v3_eval_seed42.json` — **RF v3 eval (PRODUCTION)**
- `rf_scf_real_holdout_tests.json` — RF holdout tests
- `rf_scf_v1_extended_ood.json` — RF extended OOD tests
- `acoustic_encoder_eval_seed42.json` — Acoustic encoder eval
- `radar_encoder_eval_seed42.json` — Radar encoder eval
- `rf_silent_ablation_seed42.json` — **RF Silent ablation results**
- `ABCD_SUMMARY.md` — RF A/B/C/D comparison report

### Documentation (in `docs/`)
- `INTEGRATION_REPORT.md` — Zenodo data source documentation

## How to Reproduce

### 1. Train RF encoder (v3 VICReg — production)
```bash
python extension/scripts/scf_pipeline/spawn_train_v3_vicreg.py
```
Produces: `/models/rf_scf_real_v3_encoder_seed42.pt`

### 2. Train acoustic encoder
```bash
python extension/scripts/scf_pipeline/spawn_train_acoustic.py
```
Produces: `/models/acoustic_encoder_seed42.pt`

### 3. Train radar encoder
```bash
python extension/scripts/scf_pipeline/spawn_train_radar.py
```
Produces: `/models/radar_encoder_seed42.pt`

### 4. Train fusion + run RF Silent ablation
```bash
python extension/scripts/scf_pipeline/spawn_fusion_rfsilent.py
```
Produces: `/models/fusion_head_seed42.pt` + `/results/rf_silent_ablation_seed42.json`

## Data Sources

### RF — Zenodo 4264467
- **Source**: Pärlin, K. (2020). Radio-Frequency Control and Video Signal Recordings of Drones. Zenodo.
- **URL**: https://zenodo.org/records/4264467
- **DOI**: 10.5281/zenodo.4264467
- **License**: CC-BY 4.0
- **Format**: Interleaved int16 LE IQ, 4 bytes/complex sample
- **Sample rate**: 120 MSps (2.4 GHz) / 200 MSps (5.8 GHz)
- **Coverage**: 12 drone files covering 11 distinct types (DJI, Parrot, Yuneec)

### Acoustic — DADS + ESC-50
- **DADS** (drone positives): Drone Audio Detection Samples
  - URL: https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples
  - License: MIT
  - 100K+ drone audio clips, parquet format
- **ESC-50** (BG negatives): Environmental Sound Classification
  - URL: https://github.com/karolpiczak/ESC-50
  - License: CC-BY-NC-4.0
  - 2000 environmental sound clips across 50 categories

### Radar — Open Radar Initiative
- **Source**: Gusland et al., "Open Radar Initiative: Large Scale Dataset for Benchmarking of micro-Doppler Recognition Algorithms", 2021 IEEE International Radar Conference
- **URL**: https://github.com/openradarinitiative/open_radar_datasets
- **License**: CC-BY-NC-4.0
- **Coverage**: 350 signatures — 50 UAV + 47 bicycle + 52 person + 201 vehicle
- **Format**: NumPy .npy with complex Doppler spectrograms (44, 1008) per signature

## Citations

```bibtex
@dataset{parlin_2020_4264467,
  author       = {Pärlin, Karel},
  title        = {{Radio-Frequency Control and Video Signal Recordings of Drones}},
  month        = nov, year         = 2020,
  publisher    = {Zenodo}, version      = 1,
  doi          = {10.5281/zenodo.4264467},
  url          = {https://doi.org/10.5281/zenodo.4264467}
}

@inproceedings{gusland2021open,
  author       = {Gusland, Daniel and Christiansen, Jonas M and Torvik, Børge and Fioranelli, Francesco and Gurbuz, Sevgi and Ritchie, Matthew},
  booktitle    = {2021 IEEE International Radar Conference (RADAR)},
  title        = {{Open Radar Initiative : Large Scale Dataset for Benchmarking of micro-Doppler Recognition Algorithms}},
  year         = 2021
}
```

## Loss Function: VICReg + SIGReg + BCE

All three encoders use the same loss:

```
L = L_sigreg + L_vicreg + L_bce

L_sigreg  = mean((var(Wz) - 1)^2)                       # variance target via random projections
L_vicreg  = λ_var · mean(relu(1 - std(z)))               # per-dim variance penalty
          + λ_cov · sum(off_diag(Cov(z))^2) / D          # decorrelation
L_bce     = BCE(head(z), label)                          # drone vs BG classification
```

VICReg (Variance-Invariance-Covariance Regularization) prevents representation collapse — without it, the encoder collapses onto a 2D subspace of the 256-dim embedding space. With VICReg, effective dimension jumps from 2 to 216.
