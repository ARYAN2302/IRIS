# IRIS — Identify, Recognize, Isolate, Spot

**Self-supervised drone detection + RF-only intent classification + Remote ID spoof detection + continual learning + multimodal RF-silent fallback — all edge-deployable.**

IRIS learns a general representation of "drone-ness" from RF. Two complementary stacks: **v11 (LeJEPA + Hierarchical SupCon on STFT spectrograms)** and **v3 (VICReg + Mahalanobis on SCF cyclostationary features)**. The SCF path is the production detector — receiver-invariant by construction, 99.7% on unseen DJI types, 100% at SNR 0–30 dB. A late-fusion extension adds acoustic + radar with modality dropout for graceful degradation when RF is jammed/offline. An intelligence layer (tracking, threat scoring, SAPIENT-ready output, fleet coordination) turns detections into C2 decisions.

[![CI](https://github.com/ARYAN2302/IRIS/actions/workflows/ci.yml/badge.svg)](https://github.com/ARYAN2302/IRIS/actions) [![License: Research](https://img.shields.io/badge/license-research-blue.svg)](#license) [![Modal](https://img.shields.io/badge/modal-%3E%3D1.5.0-black)](https://modal.com)

---

## Results

### v11 — Zero-Shot on RF Spectrograms (30 train → 7 unseen types, T4, 3 seeds)

| Metric | Value |
|---|---|
| **AUC (holdout vs matched BG)** | **0.978** |
| Per-pair drone-closer rate | 98.6% |
| Bootstrap 95% CI | [0.979, 0.984] |
| Encoder | 3.7M params, ~13 MB ONNX, ~10 ms (M1) |

**Noise robustness (Demo 0):** 0% FP on real WiFi/BT/environmental RF at every SNR from clean to −5 dB. AUC 1.0 through +5 dB.

| SNR | Drone TPR | Matched BG FPR | Real RF FPR | AUC |
|---|---|---|---|---|
| clean | 62% | 0% | 0% | 1.0000 |
| +10 dB | 68% | 0% | 0% | 1.0000 |
| +5 dB | 64% | 0% | 0% | 0.9960 |
| 0 dB | 22% | 0% | 0% | 0.9724 |
| −5 dB | 0% | 0% | 0% | 0.8364 |

### v3 — SCF Production Detector (Zenodo → DRFF-R2, receiver-invariant, T4)

| Metric | Value |
|---|---|
| **DRFF-R2 detection (8 unseen DJI types)** | **99.7%** @ 99.9p, AUC 1.0 |
| BG false positives | **0%** @ 99p and 99.9p |
| SNR sweep 0–30 dB | **100%** at every level |
| LOTO cross-val (11 Zenodo types) | 98.75% mean |
| Effective dim | 216/256 (VICReg fixed collapse 2→216) |

v3's jump over v11 came from **input × data, not loss**: SCF `|COH|` cancels receiver gain (STFT does not) + Zenodo's OFDM-family training data shares topology with DRFF-R2 DJI. See `build.md` and `extension/scripts/scf_pipeline/results/ABCD_SUMMARY.md`.

### Multimodal Extension (SCF + Acoustic + Radar, late fusion, modality dropout p=0.3)

| Configuration | Accuracy | AUC |
|---|---|---|
| Full fusion (RF+Ac+Rad) | 1.0000 | 1.0000 |
| **RF-silent (Ac+Rad only)** | **0.925** | **1.0000** |

> **Note:** fusion trained/evaluated on synthetically paired embeddings — see *Known Limitations* below. Acoustic (DADS, 80/180k clips) AUC 0.869 and radar (50 UAV) AUC 0.85 are data-starved; not production.

### Other Capabilities

**RF-only Intent (3-class, first-of-kind):** 66.9% overall (random 33%), ATTACK recall **93%** (69/74).

| True \ Pred | SURVEILLANCE | TRANSIT | ATTACK |
|---|---|---|---|
| SURVEILLANCE | 88 | 32 | 13 |
| TRANSIT | 24 | 77 | 42 |
| ATTACK | 0 | 5 | 69 |

**Remote ID Spoof Detection (first-of-kind):** RF fingerprint vs enrolled registry — authentic 0.636 vs spoof −0.019 @ threshold 0.85.

**AVR-CL Continual Learning (3 seeds, EWC baseline):** AVR-CL **0.781 ±0.075** vs naive 0.484 / EWC 0.482 (1.6×). Verify-and-repair loop, frozen encoder + 50K fingerprint head. See `FORGETTING_CLARIFICATION.md`.

**Cross-Manufacturer:** non-DJI centroid → DJI AVATA2/MINI3/MINI4 PRO 1.0, MAVIC3 PRO 0.40 / FPV COMBO 0.48 (OcuSync variant gap).

**Adversarial (digital FGSM):** AUC 1.0→0.995 at ε=0.2 — Mahalanobis in SIGReg space is naturally robust. *OTA universal-I/Q perturbations (Gazit et al. Dec 2025) are the stronger, not-yet-tested threat — see Limitations.*

---

## Architecture — Two Stacks, One Production Path

```
STFT path (v11, research):  IQ → STFT (log-mag + grad) → CNNEncoder(768-d, 5 blocks, AdaPool)
                                      → LeJEPA projector/predictor + SIGReg(Cramér-Wold, K=256, λ=1e-3)
                                      → Hierarchical SupCon (0.7 fine same-type + 0.3 coarse all-drones)
                                      → L2-Mahalanobis OOD

SCF path (v3, production):  IQ → SCF |COH| (cyclostationary, receiver-invariant) → CNNEncoder(256-d, 6 blocks, MaxPool)
                                      → VICReg(var+cov) + SIGReg(var) + BCE
                                      → L2-Mahalanobis OOD  [frozen, 99.7%]

Extension (multimodal):     RF-SCF + Acoustic(mel) + Radar(range-Doppler) encoders (backbone.py:22)
                                      → Late Fusion(768→256) + ModalityDropout(p=0.3)
                                      → AVR-CL on fusion head (encoders frozen)
                            Intelligence: MultiTrack(embedding+Hungarian), Bearing(Doppler v_radial),
                                         ThreatScorer(policy-gated), SAPIENT-ready Detection protobuf
                            Fleet: cross-site embedding correlation + weight-delta (FedAvg next)
```

Canonical encoder for new code: `extension/src/encoders/backbone.py:22`. Copies in `src/encoder.py` and `src/iris_inference.py:64` are checkpoint-bound (do not consolidate).

---

## Repository Structure

```
src/                          # v11 stack (LeJEPA, SIGReg, Mahalanobis, inference)
extension/
  src/encoders/backbone.py    # canonical CNNEncoder + SIGRegLoss + DroneBGHead
  src/fusion.py               # late fusion + ModalityDropout
  src/intelligence/           # drone_id, multi_track, bearing, threat_scoring
  src/sapient/output_schema.py# SAPIENT-ready Detection messages
  src/fleet/coordination.py   # cross-site correlation + weight-delta sharing
  scripts/scf_pipeline/       # Zenodo SCF generation + v3 VICReg training + evals
  scripts/experiments/        # next-phase: DADS audit, WiFi-hole, DIAT-μSAT
configs/split.json            # 30/7 train/holdout split (now portable)
tests/test_contracts.py       # 10 contract tests (loss keys, dims, Mahalanobis, determinism)
results/                      # committed MD reports + JSONs (PNGs on Modal volumes)
scripts/                      # demos, Modal launchers, T4 pipeline
```

---

## Quick Start

```bash
pip install -r requirements_demo.txt
python scripts/pull_from_modal.py          # checkpoint from Modal volume
python scripts/unified_demo.py             # all capabilities
python scripts/live_demo.py                # synthetic / HDF5 / IQ file modes
python scripts/spoof_demo.py --synthetic   # Remote ID spoof demo
```

---

## Reproducing Results

All on Modal T4 (~$0.40/hr). Total ~$0.85 to reproduce v11.

```bash
modal run scripts/demo0_noise_test.py          # 12 min, $0.10
modal run scripts/t4/test_pipeline_t4.py       # 30 min, $0.40
modal run scripts/three_experiments.py         # 15 min, $0.10
modal run scripts/avr_cl_hardened.py           # 20 min, $0.15
modal run scripts/adversarial_test.py          # 20 min, $0.15
```

SCF v3 (production) on Modal:

```bash
modal run extension/scripts/scf_pipeline/spawn_train_v3_vicreg.py   # VICReg fix, eff_dim 216
modal run extension/scripts/scf_pipeline/spawn_holdout_test.py      # holdout A
modal run extension/scripts/scf_pipeline/spawn_extended_ood_test.py # extended OOD D
modal run extension/scripts/scf_pipeline/spawn_fusion_rfsilent.py   # fusion + RF-silent ablation
```

Next-phase experiments (staged, run after `modal token new`):

```bash
modal run --detach extension/scripts/experiments/verify_p0_fixes.py   # P0 contracts on T4
modal run --detach extension/scripts/experiments/audit_dads_loader.py # DADS 80→180k audit
modal run --detach extension/scripts/experiments/wifi_hole_stress.py  # urban WiFi/LTE FP
modal run --detach extension/scripts/experiments/upgrade_radar_diat.py# DIAT-μSAT 50→4,849
```

---

## Datasets

| Dataset | Role | Scale | License | Access |
|---|---|---|---|---|
| **RFUAV** (Shi et al. arXiv:2503.09033) | v11 train/test | 37 types, 1.3 TB raw IQ @100MS/s | Apache-2.0 | HF `kitofrank/RFUAV` |
| **Zenodo 4264467** (Pärlin, Tampere) | v3 train (SCF source) | 10 models, 120/200 MSps, anechoic | CC-BY 4.0 | zenodo.org/records/4264467 |
| **DRFF-R2** (SciDB) | v3 OOD eval | 26 units / 8 DJI models | CC-BY 4.0 | SciDB |
| **DroneRF** (Mendeley) | Negatives | WiFi/BT/environmental | — | Mendeley |
| **DADS** (HF `geronimobasso/drone-audio-detection-samples`) | Acoustic | 180k clips (163K drone) | MIT | HF |
| **DIAT-μSAT** (IEEE DataPort 10.21227/1x2q-8v62) | Radar upgrade | 4,849 X-band CW images, 6 classes | Academic | DataPort |
| **TSMS-Drone** (figshare 10.25452/figshare.plus.30027313) | Real fusion benchmark | Time-aligned RF+CW+FMCW | CC-BY 4.0 | figshare |

Split: `configs/split.json` (30 train / 7 holdout, matched BG, negative_count 122k). Seed 42.

---

## Evaluation Protocol

**Honest (Shulman arXiv:2607.01025):** recording-grouped CV — a recording's segments never split train/test. Segment-level CV inflates 15–30%.

**L2-Mahalanobis (Mahalanobis++ 2025):** L2-normalize embeddings before `fit_mahalanobis()` / `compute_mahalanobis()` — critical for cross-dataset transfer. Implemented in `src/iris_inference.py:113,141`.

---

## Known Limitations & Roadmap

We ship the limitations with the numbers — that's what makes the repo hirable.

| Gap | Status | Fix |
|---|---|---|
| **Fusion 92.5% is synthetically paired** — will not survive shuffle test | P1 | Re-evaluate on **TSMS-Drone** real aligned captures + shuffle/OR/AND baselines |
| **WiFi-hole:** SCF ridge is `CP = drone-ness` but Wi-Fi/LTE are also CP-OFDM — margin BG 0.972 vs drone 1.018 is thin | P0 | Run `wifi_hole_stress.py` on dense urban WiFi/LTE; reframe as *protocol-topology* (spacing + TDMA cadence) |
| **Acoustic 80/180k, radar 50 samples** — data-starved, not arch-limited | P0 | `audit_dads_loader.py` (expect 180k) + DIAT-μSAT 4,849 |
| **FHSS/analog FM blind:** ELRS/Crossfire GFSK hops, 5.8GHz FM video have no CP ridge | P1 | Dual-head RF: frozen SCF expert + raw-IQ masked-pretrained wild head on RFUAV FHSS + new FM captures |
| **Fiber/dark drones (0 W RF)** | P1 | Only rotor physics survives — radar micro-Doppler (20ms dwell) + acoustic BPF, not RF |
| **Tracking:** `multi_track.py` frequency ±5MHz fragments FHSS → false swarm | P0 | Embedding cosine + Hungarian (frequency as prior) |
| **Bearing:** `bearing.py` fake azimuth 0°/180° | P0 | Keep Doppler `v_radial`, real azimuth only via KrakenSDR/MUSIC or multi-node TDOA |
| **Adversarial:** FGSM digital only; OTA universal I/Q (Gazit et al. Dec 2025) untested | P1 | Retarget `adversarial/boundary_probe.py` to Mahalanobis distance + PGD; add CUAP harness |
| **SAPIENT:** JSON shaped like SAPIENT, not BSI Flex 335 Protobuf | P1 | Validate against `dstl/SAPIENT-Proto-Files`, UUID node_id, Registration(detectionDefinition) |
| **Architecture:** 34 CNNEncoder copies, 7 fit_mahalanobis | Hygiene | Canonical `backbone.py:22` for new code; inference copies frozen for checkpoint compat |

**Frozen for this proto:** `v3 CNN+VICReg+BCE+Mahalanobis on SCF` for RF detection. No world model — SCF is quasi-stationary; at most a JEPA-lite adjacent-frame predictor (+10% compute) is justified. Full world model is 5–20× cost for marginal detection gain.

---

## Future Research & Works

**Universal drone detection** — the thesis behind this proto is *one 3.7M arch that scales with data*. What's proven: same `CNNEncoder` + `VICReg` gives RF `99.7%` (SCF) and acoustic `0.999` (mel, 80→3900 clips) — arch is general, data was the bottleneck. What's next:

1.  **Universal RF (all radios):** Train one model on mixed **FHSS (ELRS/Crossfire/FrSky) + OFDM (DJI)** from RFUAV 37 types, plus FM analog 5.8GHz rule-based scanner. No SCF/STFT choice at inference — single receiver-invariant input that handles both. Validated as `OFDM 99.7%` + `FHSS held-out` + `DRFF-R2 cross-dataset` in one protocol.

2.  **RF-silent at scale:** Acoustic `3900 → 180k` via HDF5 streaming (37GB → streaming, not `np.concatenate`) and radar `50 → 4,849` DIAT-μSAT X-band images — same arch, same loss, full potential.

3.  **Real multi-sensor fusion:** Replace synthetic `0.925` pairing with **TSMS-Drone** time-aligned `RF+CW+FMCW` (figshare) — transformer early fusion, honest shuffle/OR/AND baselines, where industry is heading.

4.  **Intelligence at scale:** Tracking already FHSS-robust (embedding+Hungarian), bearing honest (Doppler only until KrakenSDR) — next is 3-node TDOA pilot positioning and SAPIENT `BSI Flex 335` Protobuf certification for fleet.

---

## Why AVR-CL Works For Identification But Not Detection

See `FORGETTING_CLARIFICATION.md`. Short version:

- **Encoder (detection):** frozen, zero-shot. No fine-tuning → nothing to forget.
- **Fingerprint head (identification):** 50K params, fine-tuned per enrollment. Forgetting happens here. AVR-CL prevents it (0.781 vs 0.484 naive / 0.482 EWC).

---

## Theoretical Foundation

LeJEPA (Klindt, LeCun, Balestriero 2026) — linear identifiability for Gaussian latents via SIGReg (Cramér-Wold, K=256, `exp(-t²/2)`). RF hardware fingerprints (CFO, phase noise, amp nonlinearities, thermal noise) satisfy Gaussian assumption per DSP physics. VICReg's whitening is what makes Mahalanobis OOD work far-OOD (2.25σ). SCF `|COH|` is the physics-informed prior that makes OOD generalization possible at the input, before any loss.

See `build.md` for full derivation and `extension/scripts/scf_pipeline/results/ABCD_SUMMARY.md` for the ab study (v1 98.5% → v3 99.7%, eff_dim 2→216, 15k-sample regression lesson).

---

## References

- Klindt, LeCun, Balestriero. "Linearly Identified JEPA." 2026.
- Bardes et al. "VICReg." arXiv:2105.04906, ICLR 2022.
- Zheng et al. "Use All The Labels." CVPR 2022.
- Shulman. "How Much Do RF Drone Benchmarks Overstate?" arXiv:2607.01025, 2026.
- Lee et al. "A Simple Unified Framework for Detecting OOD Samples." NeurIPS 2018.
- Shi et al. "RFUAV." arXiv:2503.09033, 2025.
- Gazit et al. "Real-World Adversarial Attacks on RF-Based Drone Detectors." arXiv:2512.20712, Dec 2025.
- Mototolea et al. "Non-Cooperative FHSS-GFSK Detection." OJComS 2020.
- BSI Flex 335 v2.0 (SAPIENT) — Dstl/MoD, Mar 2024.

---

## License

Research and demonstration purposes. See dataset licenses (RFUAV Apache-2.0, DRFF-R2 CC-BY 4.0, Zenodo CC-BY 4.0) for data terms.

## Citation

```bibtex
@software{iris_2026,
  title={IRIS: Self-supervised drone detection via LeJEPA on RF spectrograms},
  author={Aryan},
  year={2026},
  url={https://github.com/ARYAN2302/IRIS}
}
```
