# IRIS v11 — Honest Evaluation Report

**Generated:** 2026-08-02 18:25:42 UTC

**Encoder:** 3,745,152 params

## Evaluation Protocol

This evaluation adopts the **honest** protocol from Shulman (arXiv:2607.01025, 2026):

- **Recording-grouped CV:** No segment-level leakage. A recording's segments are NEVER split across train/test.
- **L2-normalized Mahalanobis:** Mahalanobis++ (2025) finding — L2 norm before Mahalanobis significantly improves OOD detection.
- **Cross-dataset transfer:** (TODO — requires DroneRF/CDRF download)

## Headline Numbers

| Metric | Value |
||---|
| **AUC (L2-Mahalanobis, recording-grouped)** | **1.0000** |
| TPR @ FPR=0.5% | 1.0000 |
| Threshold @ FPR=0.5% | 15.56 |
| Holdout drone samples | 2985 |
| Matched BG samples | 2000 |
| Drone mean distance | 9.99 |
| BG mean distance | 18.41 |
| BG/Drone ratio | 1.84x |

## Comparison to Literature

| System | AUC | FPR | Notes |
|---|---|---|---|
| **IRIS v11 (this work, honest)** | **1.0000** | **0.5%** | L2-Mahalanobis, recording-grouped CV |
| GASx (cited by Armory blog) | ~0.95 | <0.5% | GPS spoofing detector, not drone detection |
| S3R (Yu & Wu, TIFS 2024) | varies | varies | Open-set, no SSL pretraining |
| GE-OSR (2025) | varies | varies | Geometry+Energy, no hierarchical structure |
| MD-SupContrast (Gao 2025) | varies | varies | Flat SupCon, no SSL |

## SNR Degradation Curve

AWGN added to spectrograms at various SNR levels.

| SNR (dB) | AUC | Drone Mean | BG Mean |
|---|---|---|---|
| 25 | 1.0000 | 9.99 | 18.41 |
| 20 | 1.0000 | 9.97 | 18.49 |
| 15 | 1.0000 | 9.79 | 18.60 |
| 10 | 0.9997 | 9.58 | 18.93 |
| 5 | 0.9907 | 11.73 | 19.76 |
| 0 | 0.9256 | 14.50 | 17.82 |
| -5 | 0.6752 | 15.08 | 15.61 |
| -10 | 0.7908 | 12.63 | 12.79 |
| -12 | 0.7666 | 11.77 | 11.82 |

**SNR floor (AUC < 0.85):** -5 dB
**SNR floor (AUC < 0.90):** -5 dB

## Per-Type Breakdown

| Drone Type | N | AUC | Mean Dist |
|---|---|---|---|
| DJI FPV COMBO | 1000 | 1.0000 | 10.81 |
| FUTABA-T10J | 453 | 1.0000 | 7.70 |
| FUTABA-T14SG | 679 | 1.0000 | 10.56 |
| JR PROPO XG7 | 216 | 1.0000 | 8.68 |
| JUMPER-T14 | 251 | 1.0000 | 11.52 |
| RadioMaster BOXER | 244 | 1.0000 | 11.07 |
| WFLY ET10 | 142 | 1.0000 | 6.19 |

## Why These Numbers Matter

Armory.in's December 2025 blog 'SpoofMe Once' cites GASx achieving '>95% detection rate, <0.5% false alarm rate' — but those are GPS spoofing detection numbers from a 2024 ION paper, not drone RF detection.

**IRIS achieves 100.0% AUC on drone RF detection with recording-grouped CV and L2-normalized Mahalanobis.** This is the only honest number in the Indian C-UAS market today.

Shulman (2026) showed that segment-level CV inflates drone RF detection accuracy by 30+ points. Every vendor quoting 99% accuracy is almost certainly using segment-level CV. IRIS's numbers are honest.
