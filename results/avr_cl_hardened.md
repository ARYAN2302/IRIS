# IRIS AVR-CL Hardened Experiment — 3 Seeds + EWC Baseline

**Generated:** 2026-08-02 18:34:57 UTC

**Holdout types:** 7
**Test samples:** 521

## Results (3 seeds each)

| Method | Mean | Std | Min | Max |
|---|---|---|---|---|
| naive_high_lr | 0.037 | 0.007 | 0.027 | 0.044 |
| naive_low_lr | 0.037 | 0.007 | 0.027 | 0.044 |
| ewc | 0.030 | 0.002 | 0.027 | 0.033 |
| avr_cl | 0.725 | 0.035 | 0.676 | 0.754 |

## Key Findings

- **Naive (high LR):** 0.037 ± 0.007
- **Naive (low LR):** 0.037 ± 0.007
- **EWC:** 0.030 ± 0.002
- **AVR-CL:** 0.725 ± 0.035

- **AVR-CL vs Naive:** 19.5x improvement
- **AVR-CL vs EWC:** 24.1x improvement
