# IRIS — Identify, Recognize, Isolate, Spot

**Self-supervised drone detection for RF and RF-silent drones + multi-sensor fusion + intelligence layer — all edge-deployable.**

IRIS learns "drone-ness" from RF. Production path is **SCF cyclostationary features → VICReg → Mahalanobis**: receiver-invariant by construction, **99.7%** on unseen DJI types, **100%** at SNR 0–30 dB, `3.7M` params, `~13MB` ONNX, `~10ms` on M1. Same `3.7M` backbone scales to acoustic (`mel`, `0.999` with 3900 clips) and radar — late fusion with modality dropout gives graceful RF-silent fallback. Intelligence layer (tracking, bearing, threat, SAPIENT-ready) turns detections into C2 decisions.

[![License: Research](https://img.shields.io/badge/license-research-blue.svg)](#license) [![Modal](https://img.shields.io/badge/modal-%3E%3D1.5.0-black)](https://modal.com)

---

## Results

### RF — SCF Production (Zenodo → DRFF-R2, T4)

| Metric | Value |
|---|---|
| **Unseen DJI detection (8 types)** | **99.7%** @99.9p, AUC 1.0 |
| BG false positives | **0%** |
| SNR 0–30 dB sweep | **100%** at every level |
| LOTO (11 Zenodo types) | 98.75% mean |
| Effective dim | 216/256 (VICReg fixed 2→216) |

v11 STFT `0.978` AUC was the research baseline; v3 SCF is the production detector. Jump came from **input × data, not loss**: `|COH|` cancels receiver gain (STFT does not) + OFDM-family training data.

### RF-Silent — Acoustic (DADS) + Fusion

| Component | Before | After | Method |
|---|---|---|---|
| **Acoustic** | 80 clips, AUC 0.869 | **3900 clips, AUC 0.999** | Same 3.7M CNN+VICReg, mel 256×256, proves loader fix (80→3900) and arch scales |
| **Fusion RF-silent (Ac+Rad, RF zeroed)** | 0.925 / 1.0 AUC | 0.925 synthetic — **bottleneck is synthetic pairing, not data** | Late fusion `768→256` + `p=0.3` dropout; needs real aligned data (TSMS-Drone) to be honest |

Full DADS is 180k clips (37GB RAM for in-memory, needs HDF5 streaming — wired, post-meet). Radar is 50 samples (`0.85`), needs DIAT-μSAT `4,849` X-band images.

---

## Architecture

```
RF v11 (research):  IQ → STFT → CNNEncoder(768-d) → LeJEPA + SIGReg(Cramér-Wold) + Hierarchical SupCon → Mahalanobis
RF v3 (production): IQ → SCF |COH| → CNNEncoder(256-d, 6 blocks) → VICReg(var+cov) + SIGReg(var) + BCE → Mahalanobis  [frozen, 99.7%]

RF-silent:          Acoustic mel / Radar range-Doppler → same CNNEncoder(256-d) → same loss → same Mahalanobis
                    Fusion: concat 3×256 → 256 + ModalityDropout(p=0.3) — graceful RF-silent

Intelligence:       MultiTrack (embedding cosine + Hungarian, stable ≥3 dets), Bearing (Doppler v_radial only),
                    ThreatScorer (policy-gated), SAPIENT-ready Detection, Fleet (cross-site correlation)
```

Canonical for new code: `extension/src/encoders/backbone.py:22`. Checkpoint-bound copies in `src/encoder.py` / `src/iris_inference.py:64` stay frozen.

---

## Quick Start

```bash
pip install -r requirements_demo.txt
python scripts/pull_from_modal.py          # checkpoint from volume
python scripts/unified_demo.py
python scripts/live_demo.py                # synthetic / HDF5 / IQ file
```

## Reproducing

```bash
# v11 full pipeline — ~$0.85 on T4
modal run scripts/demo0_noise_test.py
modal run scripts/t4/test_pipeline_t4.py
modal run scripts/three_experiments.py

# v3 production — SCF VICReg
modal run extension/scripts/scf_pipeline/spawn_train_v3_vicreg.py
modal run extension/scripts/scf_pipeline/spawn_holdout_test.py
modal run extension/scripts/scf_pipeline/spawn_fusion_rfsilent.py  # RF-silent ablation
```

## Repository Structure

```
src/                          # v11 stack (LeJEPA, SIGReg, Mahalanobis)
extension/
  src/encoders/backbone.py    # canonical 3.7M CNN + SIGReg + VICReg
  src/fusion.py               # late fusion + ModalityDropout
  src/intelligence/           # multi_track, bearing, threat, drone_id
  src/sapient/output_schema.py# SAPIENT-ready messages
  src/fleet/coordination.py   # cross-site correlation
  scripts/scf_pipeline/       # SCF generation + v3 training + evals
configs/split.json            # 30/7 split (portable)
tests/test_contracts.py       # 10 contract tests (P0 regression guards)
results/                      # MD reports + JSONs
```

## Datasets

| Dataset | Role | Scale | License |
|---|---|---|---|
| **RFUAV** (Shi arXiv:2503.09033) | v11 train/test | 37 types, 1.3TB raw / 9.5GB parquet (10k JPEG) | Apache-2.0 |
| **Zenodo 4264467** (Pärlin) | v3 train | 10 models, anechoic | CC-BY 4.0 |
| **DRFF-R2** | v3 OOD | 26 units / 8 DJI | CC-BY 4.0 |
| **DADS** (`geronimobasso`) | Acoustic | 180k clips (was 80) | MIT |
| **DIAT-μSAT** (IEEE DataPort) | Radar upgrade | 4,849 X-band | Academic |

## Evaluation

**Recording-grouped CV** (Shulman arXiv:2607.01025) — never split a recording's segments. **L2-Mahalanobis** (`src/iris_inference.py:113,141`) — normalize before distance, critical for cross-dataset.

---

## Future Research & Works

**What we have now is OFDM-family universal (DJI). True universal = any drone, any radio (OFDM + FHSS ELRS/Crossfire + FM 5.8GHz + silent fiber/dark) with ONE RF transform, no choosing SCF vs STFT at inference.**

RFUAV parquet on volume is already STFT JPEGs (`10k`, `35` labels, `0-34`), not raw IQ — perfect for this. The path that literature (RFUAV 2503.09033 + CageDroneRF 2601.03302) proves is **single STFT log-power + per-sample min-max + power-norm + MixStyle (p=0.5 α=0.1) + GRL (λ 0→1) + VICReg**, trained on 37 mixed FHSS+OFDM types. One STFT, one 3.7M backbone, receiver-invariant by training (not hand-crafted `|COH|`), `~13MB` stays edge-quick.

**Roadmap:**

1.  **Universal RF (single STFT):** Train one STFT model on RFUAV parquet 35 classes, hold out 5 balanced unseen (2 narrow FHSS + 3 wide OFDM via k-means) + **DRFF-R2 cross-dataset** as true heldout. This is the `universal_parquet_stft.py` run — proper research, not dual-head OR.

2.  **RF-silent at full:** Acoustic already `80→3900 → 0.999`; wire HDF5 streaming for full 180k (needs `f.num_row_groups` loop, 37GB → streaming, not `np.concatenate`). Radar `50 → 4,849` DIAT-μSAT X-band.

3.  **Real fusion:** Replace synthetic paired `0.925` with **TSMS-Drone** figshare (time-aligned RF+CW+FMCW, `10.25452/figshare.plus.30027313`) — transformer early fusion, honest shuffle/OR/AND baselines.

4.  **Full 180k acoustic + DIAT + TSMS fusion** — post-meet scaling, ~20 GPU-hrs, makes RF + RF-silent both SOTA with one arch that scales with data.

We ship what we have (`99.7%` RF + `0.999` acoustic proof) as the proto and the universal plan above as the thesis.

---

## Theoretical Foundation

LeJEPA (Klindt et al. 2026) linear identifiability via SIGReg (Cramér-Wold, `K=256`, `exp(-t²/2)`). VICReg whitening is what makes Mahalanobis work far-OOD. SCF `|COH|` is the physics-informed prior that makes OFDM OOD possible at the input. See `build.md` and `ABCD_SUMMARY.md` (v1 98.5% → v3 99.7%).

## References

- Klindt et al. "Linearly Identified JEPA." 2026. — Bardes et al. "VICReg." 2105.04906.
- Shi et al. "RFUAV." 2503.09033. — Shulman "How Much Do RF Benchmarks Overstate?" 2607.01025.
- Zhou et al. "MixStyle." ICLR21 2104.02008. — Ganin et al. "GRL." 1505.07818.
- BSI Flex 335 v2.0 (SAPIENT) — Dstl/MoD.

## License & Citation

Research/demonstration. See dataset licenses. Cite as in previous README.

```bibtex
@software{iris_2026, title={IRIS}, author={Aryan}, year={2026}, url={https://github.com/ARYAN2302/IRIS}}
```
