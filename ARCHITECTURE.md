# IRIS — Full System Architecture

Every block, what it does, why it exists, and where the code lives.
Companion to README §4. Read top-to-bottom: it mirrors a real detection pass.

---

## 0. The 10-second map

```
        ┌────────────────── SENSING ──────────────────┐
RF IQ ─►│ SCF |COH|                                   │
Audio ─►│ Mel spectrogram                             │
Radar ─►│ Range-Doppler                               │
        └──────────────────┬──────────────────────────┘
                           ▼
        ┌──── ONE BACKBONE (per modality head) ───────┐
        │ CNNEncoder(3.7M) → 256-d → VICReg space     │
        │ decision: L2-Mahalanobis / cosine centroid  │
        └──────────────────┬──────────────────────────┘
                           ▼
        ┌──────────── FUSION (graceful) ──────────────┐
        │ late concat + modality dropout p=0.3        │
        └──────────────────┬──────────────────────────┘
                           ▼
        ┌──── INTELLIGENCE (C2 products) ─────────────┐
        │ MultiTrack → Bearing → Threat → SAPIENT     │
        │ Fleet: cross-site correlation               │
        └─────────────────────────────────────────────┘
```

Design rule: **physics goes into the front-end; learning goes into one small backbone.**
Anything that can be an equation (gain cancellation, coherence normalization) is an
equation — the network only reads already-invariant images.

---

## 1. Signal front-ends

### 1.1 SCF |COH| — OFDM path  (`extension/src/scf_features.py`)

```
IQ[n] ── Hann ── FFT ── X(f)
S^α(f) = smooth_f[ X(f+α/2) · X*(f−α/2) ]          # spectral correlation (FSM)
|C^α(f)| = |S^α(f)| / sqrt(S(f+α/2)·S(f−α/2))       # coherence ∈ [0,1]
image = stack( log10|SCF| , |COH| ) → 2×256×256, per-channel z-norm
```

* **Why:** OFDM's cyclic prefix makes `X` correlate with itself at `α = k/T_symbol`
  for *every* OFDM transmitter. Stationary noise has no α≠0 content. Dividing by
  power makes the statistic exactly invariant to receiver gain/AGC/phase — the
  property STFT lacks (v11 learned receiver identity instead of drones).
* **Covers:** DJI OcuSync family, Parrot, Yuneec, WiFi/LTE *structurally* (hence the
  open WiFi-hole question — discrimination is protocol-topology, not CP presence).
* **Cannot cover:** FHSS (no CP ⇒ E[S^α]=0), analog FM, RF-silent drones.

### 1.2 Acoustic mel & Radar range-Doppler

Standard transforms; identical image size so the backbone treats them as peers.
Acoustic blade-pass harmonics and radar micro-Doppler are the *only* signatures that
survive total RF silence (dark/fiber-tethered drones).

---

## 2. One backbone  (`extension/src/encoders/backbone.py`)

```
in_ch∈{2,4} → [ConvBlock×6 with MaxPool 256→4] → Linear→256 → BN   (3.7M params)
L = SIGReg_var(z) + VICReg_var/cov(z) + BCE(head(z), y)
score = L2-Mahalanobis(z → drone centroid)        # or cosine when n_fit < d
```

| Piece | Job | Failure it prevents |
|---|---|---|
| BatchNorm | stable statistics | LeJEPA paper: 36% collapse without |
| SIGReg (var→1 via random projections Wz) | isotropy target | dead dims |
| VICReg var+cov | no-collapse + **whitening** | eff_dim 2 → 216; covariance ↓4.3× |
| BCE | discriminative signal | pure-SL nothing to separate |
| Mahalanobis (L2-normed) | OOD verdict | threshold fragility |

Why Mahalanobis works here: VICReg pre-whitened the space, so Euclidean/Mahalanobis
distance is meaningful without a learned metric head; unseen OFDM-family drones land
~2.25σ inside the boundary while background sits outside (AUC 1.0, FP 0%).
When fit-sample count < embed dim (small corpora), use the covariance-free cosine
centroid scorer instead — implemented in the hybrid experiment.

**Two-stack history (kept honest):** v11 = STFT + LeJEPA(predictor)+SIGReg(Cramér-Wold,
K=256)+Hierarchical SupCon on RFUAV → AUC 0.978 in-distribution but 0% cross-family:
STFT pixels carry receiver identity, and RFUAV is dominated by FHSS RC controllers
while DRFF-R2 is DJI OFDM — nothing shared to cluster. v3 = SCF+Zenodo fixed input×data;
the loss was tertiary (+1.2% from VICReg). Lesson encoded in docs: *front-ends carry
generality; losses polish.*

---

## 3. Fusion — graceful degradation  (`extension/src/fusion.py`)

Late concat of modality embeddings (768→256) trained with **ModalityDropout(p=0.3)**:
each batch randomly zeros one modality so no single sensor becomes load-bearing.

* Measured: full fusion 100%; **RF-silent (acoustic+radar) retains 92.5%, AUC 1.0** —
  on synthetically paired embeddings today (labeled limitation; TSMS-Drone is the
  real-aligned replacement).
* Design intent matches the direction of the field: any-sensor-missing operation,
  per-modality encoders frozen, only fusion trainable.

---

## 4. Intelligence layer  (`extension/src/intelligence/`, `sapient/`, `fleet/`)

| Module | What it does | Key design decision |
|---|---|---|
| `multi_track.py` | groups detections into persistent tracks | embedding-cosine + Hungarian primary; frequency as prior only. Swarm alarm needs ≥3 detections AND ≥2s — a single FHSS drone hopping 20 channels can no longer fake a swarm |
| `bearing.py` | Doppler radial velocity `v = Δf·c/2f₀` | azimuth = None until coherent array (KrakenSDR/MUSIC) or multi-node TDOA — never fake numbers |
| `threat_scoring.py` | 0–100 composite: type/intent/trajectory/RSSI/context | actions are policy-gated text (no autonomous jamming; licensing reality) |
| `sapient/output_schema.py` | Detection/Track JSON shaped on Dstl SAPIENT | information-level messages, not raw data; Protobuf conformance is roadmap |
| `fleet/coordination.py` | cross-site embedding correlation + weight deltas | privacy-preserving by construction (no raw IQ leaves site) |

Data flow per second of operation:

```
frames → encoder scores → track update (associate/maintain/expire)
       → bearing update (Doppler trend)
       → threat re-score (approach rate ↑ ⇒ score ↑)
       → SAPIENT Detection{track_id, class, confidence, score}
```

---

## 5. Training corpus map (what fed which model)

| Model | Positives | Negatives | Split discipline |
|---|---|---|---|
| v11 (STFT) | RFUAV 30 types | DroneRF real BG + matched synth | recording-grouped; 7 types held out |
| v3 (SCF) | Zenodo 10 models (SCF h5) | matched BG + fresh holdout BG | LOTO across types; DRFF-R2 fully unseen |
| Acoustic | DADS 3,900 clips (was 80!) | ESC-50 | 80/20 clip-level |

---

## 6. Failure modes this architecture explicitly does NOT solve

1. **RF-silent drones** — no front-end sees them; radar/acoustic layers must.
2. **Analog FM video** — different periodicity family; needs its own α/feature study.
3. **Adversarial OTA perturbations** (Gazit et al.) — untested against CUAP-style
   attacks; boundary probe currently digital-only.
4. **Urban WiFi-hole** — CP presence alone is not drone-ness; dense-city captures
   through frozen v3 are the decisive pending experiment.
5. **Small-corpus calibration** — percentile thresholds need hundreds of fit samples;
   conformal/calibrated thresholds are the upgrade path.

Each maps to a row in README §9 and a stage in Future Work.
