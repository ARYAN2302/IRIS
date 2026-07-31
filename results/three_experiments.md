# IRIS — Three Experiments Report

**Generated:** 2026-07-31 07:22:56 UTC

## Experiment 1: DJI-vs-non-DJI Re-split

Fit Mahalanobis centroid on 26 non-DJI train types only.
Test zero-shot on 5 DJI types.

**AUC (DJI vs BG):** 0.7775

**AUC (non-DJI holdout vs BG):** 1.0000

**Per-type DJI AUC:**

| Drone Type | AUC |
|---|---|
| DJI AVATA2 | 1.0000 |
| DJI MAVIC3 PRO | 0.4026 |
| DJI MINI3 | 1.0000 |
| DJI MINI4 PRO | 1.0000 |
| DJI FPV COMBO | 0.4849 |

**Verdict:** PARTIAL — some DJI generalization

## Experiment 2: AVR-CL Sequential Enrollment

Enrolled 7 holdout types sequentially.

**Naive final accuracy:** 0.031

**AVR-CL final accuracy:** 0.771

**Total AVR-CL repairs:** 22

**Enrollment history:**

| Step | Type | Naive Acc | AVR-CL Acc | Repairs |
|---|---|---|---|---|
| 1 | DJI FPV COMBO | 1.000 | 1.000 | 0 |
| 2 | FUTABA-T10J | 0.250 | 0.750 | 1 |
| 3 | FUTABA-T14SG | 0.207 | 0.727 | 1 |
| 4 | JR PROPO XG7 | 0.120 | 0.695 | 5 |
| 5 | JUMPER-T14 | 0.256 | 0.724 | 5 |
| 6 | RadioMaster BOXER | 0.080 | 0.650 | 5 |
| 7 | WFLY ET10 | 0.031 | 0.771 | 5 |

## Experiment 3: DroneRF Parrot Re-labeling Check

Scanned 200 DroneRF negatives.

**Detected as drones:** 0 (0.0%)

**Verdict:** Negatives are genuine background

