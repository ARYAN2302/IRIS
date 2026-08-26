# IRIS — Full System Architecture

Every stage of a real detection pass: what happens, why it's designed that way, and where the code lives.

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SENSING LAYER                            │
│                                                                 │
│  RF IQ ──► SCF |COH| image ──► (2, 256, 256) tensor            │
│  Audio ──► Mel spectrogram ──► (1, 256, 256) tensor            │
│  Radar ──► Range-Doppler map ► (1, 256, 256) tensor            │
│                                                                 │
│  Each transform extracts a physics invariant unique to drones  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ENCODING LAYER                              │
│                                                                 │
│  CNNEncoder(3.7M params) → 256-dim embedding                   │
│  Trained with: VICReg + SIGReg + BCE                           │
│  VICReg prevents collapse and whitens the space                │
│  Same architecture class for every modality                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DETECTION LAYER                             │
│                                                                 │
│  L2-Mahalanobis distance to drone centroid                     │
│  threshold @ 99th percentile of training distances             │
│  → DRONE or BACKGROUND verdict + confidence score              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FUSION LAYER (multi-sensor)                    │
│                                                                 │
│  Concat modality embeddings (RF+Ac+Rad = 768-d)                │
│  Project to unified 256-d via FusionHead                       │
│  ModalityDropout p=0.3 during training                         │
│  → graceful RF-silent fallback                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Core design rule:** anything that can be computed by an equation (gain cancellation,
coherence normalization) IS an equation — the neural network only reads images that are
already physics-invariant.

---

## Stage 1 — Signal Front-Ends

### 1.1 SCF |COH| — RF path

**File:** `extension/src/scf_features.py`

**What happens:**
Raw IQ samples from the SDR are transformed into a 2-channel image using cyclostationary signal analysis:

1. Take 16,384 IQ samples, apply Hann window
2. Compute FFT → frequency-domain representation X(f)
3. For each cycle frequency α ∈ [0, 0.5], correlate X(f+α/2) with X*(f−α/2)
4. Normalize by power to produce coherence |COH| ∈ [0,1]
5. Stack log10|SCF| and |COH| as two channels, resize to 256×256

**Why this works:**

DJI drones transmit video using OFDM. OFDM requires a cyclic prefix — the transmitter copies the tail of each symbol to its front. This creates periodic self-correlation at specific cycle frequencies α = k/T_symbol.

Background noise has no such internal repetition. So the SCF image has bright ridges at known α values for drones and flat noise elsewhere.

**Why the |COH| channel is critical:**

If the received signal is y = g·x (receiver gain g applied to true signal x), then both numerator AND denominator scale by g². They cancel exactly. This means the coherence value is identical regardless of which receiver captured the signal, what AGC setting was used, or how far away the drone is.

This is NOT learned invariance — it's mathematical identity. No amount of training can break it because gain information simply doesn't exist in the |COH| output.

**What happened without it:** Our v11 model used STFT spectrograms. STFT pixel brightness = received power = transmitter_power × channel × receiver_gain. The model learned to recognize *the receiver* rather than *the drone*. When tested on data from a different receiver, accuracy collapsed from 98% to ~40%. SCF |COH| fixed this permanently.

**What it covers:** All DJI OcuSync-family video links (Phantom, Mavic, Mini, Air, Inspire, Matrice), Parrot, Yuneec — any drone with an OFDM-based transmission protocol.

**What it cannot cover:** FHSS control signals (ExpressLRS, Crossfire — no cyclic prefix), analog FM video (continuous carrier, no packet structure), and drones emitting zero RF (autonomous waypoint, fiber-optic tethered). These require different front-ends or different sensing modalities entirely.

### 1.2 Acoustic mel-spectrogram

**File:** `extension/scripts/scf_pipeline/spawn_train_acoustic.py`

Microphone captures audio → librosa computes mel-spectrogram (128 mel bins × time frames) → resized/log-compressed/z-normalized to (1, 256, 256).

Multirotor propellers produce a fundamental blade-passing frequency (BPF = N_blades × RPM/60, typically 150–300 Hz) plus harmonics extending to ~2 kHz. This pattern is present regardless of whether the drone transmits radio.

Trained on DADS dataset (3,900 clips after loader fix; full corpus is 180k on HuggingFace). ESC-50 environmental sounds serve as background negatives.

### 1.3 Radar range-Doppler map

FMCW or pulse-Doppler radar returns are processed into range-Doppler surfaces. Multirotor blades produce characteristic micro-Doppler modulation — oscillation around the main body Doppler line at the blade rotation frequency. Optimal dwell time ≈ 20 ms (two blade rotation periods).

Currently trained on only 50 UAV signatures from Open Radar Initiative (AUC 0.85). DIAT-μSAT (4,849 X-band CW images, IEEE DataPort DOI 10.21227/1x2q-8v62) is identified as the upgrade path.

---

## Stage 2 — The Neural Encoder

### Architecture (`extension/src/encoders/backbone.py`)

```
Input: (B, in_ch, 256, 256)

Block 0: Conv(→64) + BN + GELU + Conv(64→64) + BN + GELU + MaxPool    # → 128×128
Block 1: same width                                                     # → 64×64
Block 2: width doubles to 128                                          # → 32×32
Block 3: same width                                                     # → 16×16
Block 4: width doubles to 256                                          # → 8×8
Block 5: same width                                                     # → 4×4

Flatten → Linear(4096→256) → BatchNorm1d(256)
Output: 256-dim embedding
```

Total: ~3.7M parameters. Exports to ~13 MB ONNX. Runs in ~10ms on Apple M1.

### Why each loss term exists

| Loss | Purpose | What happens without it |
|---|---|---|
| **BCE** | Teaches drone-vs-background discrimination | No learning signal; embeddings are random |
| **VICReg variance** | Forces minimum std per dimension | Complete collapse: all inputs → same output vector |
| **VICReg covariance** | Decorrelates dimensions | Dimensional collapse: eff_dim drops from 216 to 2 |
| **SIGReg** | Isotropy target via random projections | Redundant with VICReg var but stabilizes early epochs |

Measured impact (from ABCD ablation study):
- Without VICReg: eff_dim = 2.04 (only 2 useful dims out of 256!)
- With VICReg: eff_dim = 216.26 (84% utilization)
- Covariance penalty dropped 4.3× during training
- Detection improved from 98.5% to 99.7%

### Why BatchNorm is mandatory

BatchNorm normalizes activations per batch, preventing any single feature from saturating. The LeJEPA paper documents 36% training runs collapsing without BN when using self-supervised losses. We confirmed this empirically.

### What the encoder actually learns

Because the input already contains gain-invariant structure (the |COH| operation removed receiver effects before the network ever sees data), the CNN learns patterns that transfer across receivers and drone generations:

- Early layers detect local features: horizontal ridges, contrast edges, spectral lines
- Middle layers compose these: consistent ridge spanning multiple frequencies, burst timing patterns
- Final layers make global decisions: "this overall topology matches an OFDM transmitter"

The network never needs to learn gain compensation because the |COH| computation already handled it mathematically. This separation — signal processing handles invariance, neural network handles discrimination — is the core architectural principle.

---

## Stage 3 — Detection: Mahalanobis Distance

After training completes, the encoder weights are frozen. Detection doesn't use a neural network classifier — it uses geometric distance in embedding space.

### Fitting phase

Take all known-drone training images, encode them through the frozen encoder, compute:

```python
centroid = mean(all_drone_embeddings)          # center of the drone cluster
covariance = np.cov(all_drone_embeddings.T)     # natural spread of the cluster
cov_inv = np.linalg.inv(covariance + reg×I)     # inverted for fast distance computation
```

Also record the distribution of distances for calibration:
```python
distances = mahalanobis(train_embeddings, centroid, cov_inv)
threshold_99p  = percentile(distances, 99)      # 99% of drones fall inside this
threshold_99_9p = percentile(distances, 99.9)   # 99.9% inside
```

### Scoring phase

For each new incoming SCF image:
1. Encode through frozen encoder → 256-dim embedding
2. L2-normalize (project onto unit sphere)
3. Compute Mahalanobis distance to centroid
4. If distance ≤ threshold → DRONE; if > threshold → BACKGROUND

### Why this achieves perfect discrimination

Three conditions hold simultaneously:

1. **Drone embeddings are tightly clustered** — VICReg whitening ensures all 256 dims carry independent information, so the cluster is compact and spherical-ish
2. **Background embeddings are far outside** — SCF ensures background has no CP ridges → structurally different → far in whitened space. Measured: 2.25σ separation between nearest BG samples and drone centroid
3. **Hard negatives don't break it** — we tested removing the 100 closest BG samples and their neighbors; AUC stayed at 1.0

These three together place us firmly in the "far-OOD" regime where Mahalanobis distance is theoretically expected to achieve near-perfect discrimination.

### Why L2-normalization matters

Before computing Mahalanobis distance, embeddings are projected onto the unit hypersphere (L2 norm = 1). This removes magnitude information — which correlates with signal strength, i.e., range to target. After normalization, only directional information (= signal structure) remains.

Without L2-norm: a strong signal from far away might have larger magnitude than a weak signal from close up, confounding the distance metric.
With L2-norm: direction alone determines classification, making range-independent detection possible.

---

## Stage 4 — Multi-Sensor Fusion

### Why fuse?

Some threats emit no radio:
- Autonomous waypoint drones (pre-programmed GPS route, no pilot link)
- Fiber-optic tethered FPVs (signal travels in glass fiber, not air)
- Military drones with emission-control

No single sensor covers everything. IRIS adds acoustic and radar using the SAME backbone architecture — proving the arch generalizes beyond RF.

### How fusion works

```python
# Each encoder runs independently (frozen)
rf_emb   = rf_encoder(scf_image)         # (256,)
acEmb   = acoustic_encoder(mel_spec)     # (256,)
rad_emb = radar_encoder(rd_map)          # (256,)

# Late fusion: concatenate and project
fused = fusion_head([rf_emb, acEmb, rad_emb])   # (768,) → (256,)

# Detection on fused embedding
score = mahalanobis(fused, fused_centroid, fused_cov_inv)
```

### Modality Dropout

During fusion head training, each modality has a 30% chance of being replaced with zeros. This forces the network to never depend on any single modality being present.

At inference: if RF is jammed/offline, zero it out and run acoustic+radar only. Measured result: 92.5% accuracy retained (vs 100% with all three).

**Known limitation:** fusion was trained on synthetically paired embeddings — we paired drone-RF with drone-acoustic arbitrarily, not from temporally synchronized recordings. Real deployment needs TSMS-Drone (time-aligned multi-sensor captures from figshare) to be defensible.

---

## Stage 5 — Continual Learning (AVR-CL)

When a new drone type appears in the field, you need to add it to the known-types registry without forgetting previously enrolled types.

Traditional approaches (naive fine-tuning, EWC regularization) suffer catastrophic forgetting — accuracy on old types drops to ~48% after learning new ones.

AVR-CL (Adaptive Verify-and-Repair Continual Learning) solves this:

1. **Enroll:** Extract embedding centroid for new type from a few captures
2. **Verify:** Evaluate fingerprint-head accuracy on ALL previously enrolled types
3. **Repair:** If old-type accuracy dropped below threshold, reduce learning rate and retry
4. **Accept:** Only commit updates that maintain ≥95% accuracy on ALL types

Key insight: the ENCODER stays completely frozen. Only a small (50K parameter) classification head updates. Since the encoder is frozen, there's nothing to forget at the representation level. Only the thin head can forget, and AVR-CL catches and repairs it.

Result: 0.781 mean accuracy vs 0.484 (naive) / 0.482 (EWC). Consistent across 3 seeds.
