# IRIS Adversarial Robustness Report

**Generated:** 2026-08-02 18:32:41 UTC

**Baseline AUC:** 1.0000
**Threshold:** 11.22

## Why This Matters

Ben-Gurion University published in January 2026 (arXiv:2512.20712) that RF drone detectors are vulnerable to over-the-air adversarial attacks. No defenses exist for RF spectrogram classifiers.

CISA flagged C-UAS cyber vulnerabilities in October 2025. This report documents IRIS's robustness profile.

## Attack 1: FGSM False Positive (BG → DRONE)

Attacker perturbs background RF to make IRIS think there's a drone (false alarm flooding).

| ε | AUC After | BG→Drone Rate | AUC Drop |
|---|---|---|---|
| 0.01 | 1.0000 | 0.000 | 0.0000 |
| 0.05 | 0.9996 | 0.000 | 0.0004 |
| 0.1 | 0.9924 | 0.012 | 0.0076 |
| 0.2 | 0.9677 | 0.035 | 0.0323 |

## Attack 2: FGSM Evasion (DRONE → BG)

Attacker perturbs drone RF to make IRIS think it's background (drone becomes invisible).

| ε | AUC After | Drone→BG Rate | AUC Drop |
|---|---|---|---|
| 0.01 | 1.0000 | 0.160 | 0.0000 |
| 0.05 | 1.0000 | 0.039 | 0.0000 |
| 0.1 | 1.0000 | 0.030 | 0.0000 |
| 0.2 | 1.0000 | 0.026 | 0.0000 |

## Attack 3: PGD False Positive (stronger)

Iterative version of FGSM. Stronger but slower.

| ε | AUC After | BG→Drone Rate | AUC Drop | N Test |
|---|---|---|---|---|
| 0.01 | 1.0000 | 0.000 | 0.0000 | 100 |
| 0.05 | 0.9972 | 0.000 | 0.0028 | 100 |
| 0.1 | 0.8842 | 0.180 | 0.1158 | 100 |
| 0.2 | 0.5709 | 0.440 | 0.4291 | 100 |

## Attack 4: DRFM Replay

Digital Radio Frequency Memory — attacker records drone RF and replays it. Tests if IRIS creates ghost detections.

| Metric | Value |
|---|---|
| Original detection rate | 0.600 |
| Replay detection rate | 0.592 |
| Ghost rate | 0.592 |
| Embedding spread | 2.2739 |
| Detection consistency | 0.760 |
| Distance variation | 0.22 |
| Verdict | PARTIAL — some DRFM replays create ghosts |

## Comparison to Literature

| System | FGSM ε=0.1 | PGD ε=0.1 | DRFM |
|---|---|---|---|
| **IRIS v11 (this work)** | AUC 0.992 | AUC 0.884 | PARTIAL — some DRFM replays create ghosts |
| Ben-Gurion (arXiv:2512.20712) | various | various | N/A |
| AdvShield-UAV | 92-96% acc | 92-96% acc | N/A (network traffic, not RF) |

## Interpretation

IRIS uses Mahalanobis distance in a SIGReg-regularized embedding space as the detection mechanism. This is fundamentally different from softmax classifiers:

- Softmax classifiers produce a probability distribution that can be directly attacked via gradient methods
- Mahalanobis distance is a geometric measure in a regularized space
- SIGReg forces the embedding distribution toward Gaussian, which should smooth the loss landscape

The results above show whether this theoretical advantage translates to actual robustness.

## Recommendations

- **IRIS shows natural robustness to FGSM at ε=0.1.** Embedding geometry provides some defense.
- **DRFM creates ghost detections.** Add a temporal de-duplication layer: if multiple detections have similar embeddings (cosine sim > 0.95) within 1 second, collapse to one track.
