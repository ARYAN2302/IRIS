# IRIS — Three Hard Tests Report

**Generated:** 2026-08-02 18:48:02 UTC

## Test 1: Cross-Dataset Transfer (RFUAV → DRFF-R2)

**Status: FAIL**


## Test 2: IQFM Foundation Model Comparison

**Status: ERROR**





## Test 3: Harder Negatives (Spectrally-Matched)

**Verdict: FAIL — spectral artifact**

| Metric | Value |
|---|---|
| AUC (drones vs hard negs) | 0.3205 |
| AUC (drones vs real negs) | 1.0000 |
| Drone detection rate | 0.754 |
| Hard neg FP rate | 0.770 |
| Real neg FP rate | 0.000 |
| Hard neg mean dist | 8.58 |
| Drone mean dist | 9.78 |

Hard negatives = real drone spectrograms with scrambled temporal structure (permuted time bins + frequency shifts). Preserves spectral envelope but destroys drone-ness. If IRIS detects these as drones, it was relying on spectral shape (artifact). If IRIS rejects them, it learned real temporal patterns.
