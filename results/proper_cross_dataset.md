# IRIS — Proper Cross-Dataset + Artifact Tests (Fixed)

**Generated:** 2026-08-02 19:06:59 UTC

## Test 1: Cross-Dataset Transfer (RFUAV → DRFF-R2, proper STFT)

**Status: FAIL**

DRFF-R2 raw IQ converted through STFTEngine (log-power + phase, same as RFUAV).

| Metric | Value |
|---|---|
| AUC (RFUAV holdout vs BG) | 1.0000 |
| AUC (DRFF-R2 vs BG) | 0.2454 |
| RFUAV holdout mean dist | 9.78 |
| DRFF-R2 drone mean dist | 18.89 |
| DRFF-R2 detection rate | 0.000 |

Per-type DRFF-R2:

| Type | N | AUC | Detection | Mean Dist |
|---|---|---|---|---|
| mavic3C_u1 | 200 | 0.2456 | 0.000 | 18.89 |
| mavic3S_u1 | 200 | 0.2452 | 0.000 | 18.89 |
| mavic3_u1 | 200 | 0.2457 | 0.000 | 18.89 |
| mavicAir2_u1 | 200 | 0.2454 | 0.000 | 18.89 |
| mavicAir2s_u1 | 50 | 0.2454 | 0.000 | 18.89 |
| mini3pro_u2 | 50 | 0.2457 | 0.000 | 18.89 |
| mini4PRO_u1 | 50 | 0.2452 | 0.000 | 18.89 |
| mini5PRO_u1 | 50 | 0.2446 | 0.000 | 18.89 |

## Test 2: Realistic Spectral Artifact Check

**Verdict: PASS — learned temporal patterns**

Real WiFi/BT captures spectrally shaped to match drone envelope.

| Metric | Value |
|---|---|
| AUC (drones vs shaped RF) | 1.0000 |
| AUC (drones vs real RF) | 1.0000 |
| Hard neg FP rate | 0.000 |
| Real RF FP rate | 0.000 |

## Test 3: DRFF-R2 Individual Unit Analysis

**Verdict: Units not distinguishable**

| Model | Units | Mean Sim | Min Sim | Max Sim |
|---|---|---|---|---|
