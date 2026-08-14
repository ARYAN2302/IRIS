# IRIS-CUAS Extension

Multi-modal drone detection and intelligence system that solves the cross-receiver
generalization problem via cyclostationary features and multi-modal fusion.

## Architecture (Four Layers)

```
LAYER 1 — SENSOR ENCODERS
  RF Encoder (SCF features, receiver-invariant)
  Acoustic Encoder (mel-spectrogram, DADS dataset)
  Radar Encoder (range-Doppler, RDRD dataset)

LAYER 2 — LOCAL FUSION + CONTINUAL ADAPTATION
  Late Fusion (concat 768-dim → 256-dim) + Modality Dropout (p=0.3)
  AVR-CL on fusion head (encoders frozen)
  SAPIENT-compatible output (detection → structured message)

LAYER 3 — INTELLIGENCE
  Drone Type Classifier (RF fingerprint — bug flipped into feature)
  Bearing Estimator (Doppler + signal strength)
  Multi-Track Manager (protocol-based signal separation)
  Threat Scorer (0-100 composite, per-factor breakdown)
  Countermeasure Recommendation

LAYER 4 — FLEET COORDINATION
  Weight-delta sharing (no raw data, SAPIENT "information level")
  Cross-site embedding correlation
  "This signature seen at N other sites"
```

## Key Innovation: Cyclostationary Features (SCF)

The spectrogram input representation mixes receiver identity and drone identity
in the same pixels. SCF (Spectral Correlation Function) captures modulation
periodicities determined by the transmitter, not the receiver. The spectral
coherence channel is exactly invariant to receiver gain, phase, and AGC.

**Proven result:** Source probe dropped from 100% (spectrogram) to 75% (SCF) —
the first reduction in receiver fingerprint below 100%.

## Modules

### `src/scf_features.py`
Cyclostationary feature extraction. Produces (2, 256, 256) SCF images
from raw IQ. Also includes hybrid 4-channel input (SCF + autocorrelation + HOM).

### `src/iq_augment.py`
IQ-level augmentation pipeline. All augmentations provably preserve SCF's
receiver-invariance (gain cancels in the COH ratio). Supports 7 augmentation
types: complex gain, FIR filtering, quantization, CFO, time shift, noise, IQ imbalance.

### `src/encoders/backbone.py`
Shared CNN encoder, SIGReg loss, MixStyle, GRL, DomainHead, DANN lambda schedule.

### `src/encoders/rf_encoder.py`
RF encoder with SCF input. Optional MixStyle and DRIFT disentanglement.
Includes data preparation with IQ augmentation.

### `src/encoders/acoustic_encoder.py`
Acoustic encoder with mel-spectrogram input. Uses DADS dataset.

### `src/encoders/radar_encoder.py`
Radar encoder with range-Doppler map input. Uses RDRD dataset.

### `src/fusion.py`
Late fusion with modality dropout. RF-droppable at inference (RF-Silent).
Encoders frozen; only fusion head trained.

### `src/distillation.py`
Cross-modal distillation from frozen RF encoder to acoustic/radar encoders.
L2 alignment in embedding space.

### `src/avr_cl_fused.py`
AVR-CL on fused embedding space. Sequential enrollment, retention testing,
weight-delta extraction for fleet sharing.

### `src/intelligence/drone_id.py`
Drone type classifier with conditional contrastive learning.
Flips the receiver fingerprint bug into a drone ID feature.

### `src/intelligence/bearing.py`
Bearing estimation from RF Doppler shift and signal strength.

### `src/intelligence/multi_track.py`
Multi-track manager. Separates drones by protocol/frequency.
Swarm detection (≥5 simultaneous tracks).

### `src/intelligence/threat_scoring.py`
Threat scoring engine. 0-100 composite score with per-factor breakdown.
Countermeasure recommendation table.

### `src/sapient/output_schema.py`
SAPIENT-compatible output messages (NATO STANREC 4869).
Information-level output, not raw data.

### `src/fleet/coordination.py`
Fleet coordination protocol. Weight-delta sharing, cross-site embedding
correlation, "seen at N sites" intelligence product.

### `src/adversarial/boundary_probe.py`
Adversarial robustness checker. FGSM, PGD, boundary distance probing.

## Quick Start

```bash
# Install dependencies
pip install torch h5py numpy scikit-learn scipy librosa soundfile datasets

# Prepare data (downloads datasets, computes SCF/mel-spec/RD features)
python -m extension.scripts.train_pipeline --stage prepare

# Train RF encoder on SCF features
python -m extension.scripts.train_pipeline --stage rf_scf --seed 42

# Train acoustic encoder
python -m extension.scripts.train_pipeline --stage acoustic --seed 42

# Train radar encoder
python -m extension.scripts.train_pipeline --stage radar --seed 42

# Train fusion layer
python -m extension.scripts.train_pipeline --stage fusion

# Run RF-Silent ablation (headline experiment)
python -m extension.scripts.train_pipeline --stage rf_silent

# Final 5-seed evaluation
python -m extension.scripts.train_pipeline --stage eval

# Adversarial robustness audit
python -m extension.scripts.train_pipeline --stage adversarial
```

## Data Sources

| Dataset | Modality | Format | Access |
|---------|----------|--------|--------|
| RFUAV | RF | Pre-computed spectrograms | Local (existing) |
| Zenodo Tampere | RF | Raw IQ (.bin) | zenodo.org/records/4264467 |
| DroneRF | RF | Raw IQ (.csv in .rar) | data.mendeley.com/f4c2b4n755 |
| DADS | Acoustic | WAV 16kHz | HuggingFace: geronimobasso/drone-audio-detection-samples |
| ESC-50 | Acoustic (negatives) | WAV | HuggingFace: ashraq/esc50 |
| UrbanSound8K | Acoustic (negatives) | WAV | HuggingFace |
| RDRD | Radar | Range-Doppler images | Kaggle: iroldan |
| TSMS-Drone | Radar+RF (paired) | FMCW+CW+RF | figshare: 10.25452/figshare.plus.30027313 |

## Key Findings from Investigation

1. Spectrograms encode receiver identity and drone identity in the same pixels
2. Source probe is 100% linearly decodable from spectrogram embeddings (all 12 approaches)
3. SCF features reduce source probe to 75% (first reduction below 100%)
4. BatchNorm stats drift causes embedding collapse during fine-tuning
5. Balanced batching causes background collapse (not reproducible with matched steps)
6. The bottleneck is the input representation, not the training technique

## Validated Results

| Configuration | Cross-dataset det | BG FP | Source probe |
|---|---|---|---|
| Original baseline (spectrogram) | 40% (4/10 seeds) | 0% | 100% |
| SCF features (inference-only) | collapse (410 samples) | 0% | 75% |
| SCF + augmentation (target) | 70-80% | 0% | <60% |
| RF-Silent (acoustic+radar) | 90%+ | 0% | N/A |
