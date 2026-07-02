# IRIS — Identify, Recognize, Isolate, Spot

## Proving "Drone-ness": Zero-Shot Detection of Unseen UAVs via LeJEPA on RF Spectrograms

---

## 0. Project Overview

**Hypothesis:** "Drone-ness" — the property distinguishing UAV RF emissions from ambient RF noise — is a learnable latent signal. A LeJEPA model trained on RF spectrograms from known drone types will cluster unseen drone types into a distinct region of the latent space, enabling zero-shot detection of zero-day drones.

**End Goal:** Demo to Armory.in CTO → Internship

**Constraints:** <$30 compute, single GPU, <2 weeks to result

**Theoretical Foundation:** LeJEPA (Klindt, LeCun, Balestriero 2026) — linear identifiability guaranteed for Gaussian latent variables under OU transitions via SIGReg. RF hardware fingerprints (CFO, phase noise, amplifier nonlinearities, thermal noise) are Gaussian per DSP physics. Theorem 3 provides graceful degradation bound for non-Gaussian modulation symbols.

---

## 1. Architecture

### 1.1 LeJEPA Loss

```
L = lambda * L_SIG + (1 - lambda) * L_inv
```

- **Alignment loss** L_inv = (1/B) * sum(||f(x_i) - f(x'_i)||^2)
  - Positive pairs: adjacent time windows of same RF transmission
  - rho controlled by time gap between windows (target: 0.9-0.95)
- **SIGReg** L_SIG = Sketched Isotropic Gaussian Regularizer
  - Project embeddings onto K=256 random 1D directions
  - Match empirical sliced characteristic function to standard Gaussian
  - Cramer-Wold trick: O(n^6) -> O(log n)
- **lambda** = 1e-3 (SIGReg weight). Sweep: 1e-4 to 1e-2

### 1.2 Encoder: CNN with BatchNorm

```
Input: STFT spectrogram (1-channel, H x W)
  -> Conv(1->32, 4x4, stride=2) -> BatchNorm -> GELU
  -> Conv(32->64, 3x3) -> BatchNorm -> GELU
  -> Conv(64->128, 3x3) -> BatchNorm -> GELU
  -> Conv(128->256, 3x3) -> BatchNorm -> GELU
  -> AdaptiveAvgPool -> Flatten
  -> Linear(256, 256) -> BatchNorm -> GELU
  -> Linear(256, embed_dim)
Output: embed_dim-dimensional embedding (e.g., 128 or 256)
```

**BatchNorm is mandatory** — 36% of runs collapse without it (per LeJEPA paper).

Total: ~5-30M params depending on embed_dim and input size.

### 1.3 Positive Pair Generation

For RF spectrograms, positive pairs come from physics:
- **Source:** Same RF transmission, temporally adjacent windows
- **Window size:** Determined by STFT parameters (see Section 2)
- **Time gap (delta_t):** Controls rho
  - Small gap (1-2 spectrogram frames) -> rho ~0.99 (too high, risk of collapse)
  - Medium gap (5-20 frames) -> rho ~0.9-0.95 (sweet spot per paper)
  - Large gap (50+ frames) -> rho <0.8 (weak alignment signal)
- **Implementation:** For each spectrogram, sample two windows at positions t and t+delta_t

### 1.4 Why Not ViT

- CNN with BatchNorm is proven in the paper's Reacher experiment
- ViT requires more data and compute
- At our budget, CNN is the safer choice
- ViT/A-JEPA masking is a Phase 2 upgrade if CNN works

---

## 2. Data Pipeline

### 2.1 Datasets

| Dataset | Role | Drone Types | Size | Format |
|---------|------|-------------|------|--------|
| **RAUAV** | Primary (train/test) | 37 types | ~1.19 GB | Binary I/Q |
| **CDRF (CageDroneRF)** | Cross-dataset validation | Multiple | Large | Standardized spectrograms |
| **DRFF-R2** | Individual device fingerprinting | 26 DJI units (8 models) | Medium | STFT spectrograms |
| **DroneRFa** | Augmentation/ablation | 6 models | Medium | CSV amplitude |
| **DroneDetect** | Non-drone negatives (WiFi/BT) | 7 drones + clutter | 66 GB | Raw I/Q |

### 2.2 Train/Test Split (RAUAV)

- **Train:** 30 drone types (all flight modes, no labels used)
- **Hold-out:** 7 drone types (COMPLETELY unseen during training)
- **Non-drone negatives:** WiFi + Bluetooth from DroneDetect

### 2.3 I/Q to Spectrogram Pipeline

```
Raw I/Q data (2 channels: In-phase, Quadrature)
  -> Compute complex signal: s = I + jQ
  -> STFT with parameters:
       - n_fft: 256 or 512 (swept)
       - hop_length: n_fft // 4
       - window: hann
  -> Log-magnitude: 20 * log10(|STFT(s)| + eps)
  -> Normalize per-dataset:
       - Mean-center each frequency bin
       - Unit variance per frequency bin
       - DO NOT over-normalize (preserves hardware fingerprints)
  -> Reshape to fixed size (e.g., 128 x 128) via resize or crop
  -> Save as single-channel float32 HDF5
```

**STFT vs Wavelet:** Start with STFT (simpler, standard in DRFF-R2 baselines). Wavelet scattering is a Phase 2 upgrade.

### 2.4 HDF5 Schema

```
/dataset_name/
  /spectrograms/     # shape: (N, 1, H, W), float32
  /drone_type/       # shape: (N,), int (class label for eval only)
  /drone_unit/       # shape: (N,), int (individual device ID for DRFF-R2)
  /flight_mode/      # shape: (N,), int (hover, cruise, takeoff, etc.)
  /snr_db/           # shape: (N,), float (for SNR degradation tests)
  /split/            # shape: (N,), int (0=train, 1=hold-out, 2=negative)
```

### 2.5 Per-Dataset Normalization Strategy

Critical: over-normalization kills hardware fingerprints.

- **Level 0 (minimal):** Global mean/std across entire dataset
- **Level 1 (per-frequency-bin):** Mean/std per frequency bin (RECOMMENDED)
- **Level 2 (per-sample):** Mean/std per spectrogram (RISKY — may remove fingerprints)
- **Level 3 (per-dataset):** Separate normalization for each dataset (REQUIRED for cross-dataset)

Start with Level 1. If zero-shot fails, try Level 0. Never Level 2 without testing Level 1 first.

---

## 3. Training Protocol

### 3.1 Hyperparameters (from LeJEPA paper)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | |
| Learning rate | 3e-3 | |
| LR schedule | Warmup (10% steps) + cosine decay to 0 | |
| Weight decay | 0 | Paper uses 0 for controlled experiments |
| Batch size | 256 | Or gradient accumulate to 256 |
| SIGReg weight (lambda) | 1e-3 | Sweep: 1e-4, 5e-4, 1e-3, 5e-3 |
| SIGReg slices (K) | 256 | Random projection directions |
| Embedding dim | 128 | Start here, try 256 if compute allows |
| Gradient clipping | 1.0 | MANDATORY per paper |
| Precision | bfloat16 | Mixed precision |
| Epochs | 100 | Or until loss plateaus |

### 3.2 Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # batch shape: (B, 1, H, W)
        
        # Generate positive pairs
        x1, x2 = create_positive_pairs(batch, delta_t=optimal_gap)
        
        # Forward pass
        z1 = encoder(x1)  # (B, embed_dim)
        z2 = encoder(x2)  # (B, embed_dim)
        
        # Alignment loss
        l_inv = F.mse_loss(z1, z2)
        
        # SIGReg loss
        l_sig = sigreg_loss(z1)  # + z2 if using both views
        
        # Total loss
        loss = lambda_sig * l_sig + (1 - lambda_sig) * l_inv
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
```

### 3.3 SIGReg Implementation

```python
def sigreg_loss(z, K=256):
    """
    Sketched Isotropic Gaussian Regularizer.
    Forces embedding distribution toward N(0, I).
    
    Uses Cramer-Wold: project embeddings onto K random 1D directions,
    then match the empirical characteristic function to Gaussian target.
    """
    B, D = z.shape
    
    # Random projection directions (fixed per forward pass, or use rolling buffer)
    theta = F.normalize(torch.randn(K, D, device=z.device), dim=1)  # (K, D)
    
    # Project embeddings onto random directions
    projections = z @ theta.T  # (B, K)
    
    # Evaluate empirical characteristic function at grid points t
    # For standard Gaussian: phi(t) = exp(-t^2 / 2)
    t_grid = torch.linspace(-3, 3, steps=32, device=z.device)  # (T,)
    
    # Empirical characteristic function: (1/B) * sum(exp(i * t * projection))
    # Use real part only (cosine)
    # projections: (B, K), t_grid: (T,)
    # cos(t * projection): (T, B, K)
    emp_cf = torch.mean(
        torch.cos(t_grid.unsqueeze(1).unsqueeze(2) * projections.unsqueeze(0)),
        dim=1
    )  # (T, K)
    
    # Gaussian target characteristic function
    gauss_cf = torch.exp(-t_grid.unsqueeze(1)**2 / 2)  # (T, 1) broadcast to (T, K)
    
    # MSE between empirical and target
    loss = F.mse_loss(emp_cf, gauss_cf.expand_as(emp_cf))
    
    return loss
```

### 3.4 Compute Budget on Modal

| Phase | Hardware | Duration | Est. Cost |
|-------|----------|----------|-----------|
| Data prep | Local/CPU | 1-2 days | $0 |
| Training run 1 (default params) | T4 16GB | 12-24 hrs | $3-5 |
| Hyperparameter sweep (4 lambda values) | T4 16GB | 24-48 hrs | $6-10 |
| Rho sweep (3 values) | T4 16GB | 12-24 hrs | $3-5 |
| Evaluation + visualization | T4 16GB | 4-6 hrs | $1-2 |
| Fallback: V-JEPA + EMA teacher | T4 16GB | 12-24 hrs | $3-5 |
| Buffer | — | — | $5 |
| **Total** | | | **$21-32** |

Modal T4 pricing: ~$0.40/hr. A10G: ~$0.75/hr.

---

## 4. Evaluation Framework

### 4.1 Experiment 1: Linear Probing (Known Classes)

- Freeze LeJEPA encoder
- Train a single linear layer on top using labeled data from 30 known types
- Metric: classification accuracy on held-out samples of known types
- **Go/No-Go gate:** If linear probe accuracy <80%, representations are bad. Stop and debug.
- This is a NECESSARY condition, not sufficient — we care about zero-shot, not supervised accuracy

### 4.2 Experiment 2: Zero-Shot Drone Type Detection (The Money Experiment)

- Feed spectrograms from 7 held-out drone types through frozen encoder
- Compute embeddings for all samples (known + held-out + non-drone negatives)
- **K-means clustering:** Do held-out types form distinct clusters?
- **GMM fitting:** Fit Gaussian Mixture Model, check if held-out types are assigned to drone clusters
- **Distance-based detection:** For each held-out sample, compute distance to nearest known-drone embedding centroid. Small distance = high confidence detection.
- **Visualization:** UMAP projection colored by drone type. The "aha" moment: held-out drones land inside the drone super-cluster.

### 4.3 Experiment 3: Individual Device Fingerprinting (DRFF-R2)

- Train on some units of each DJI model, hold out specific physical units
- Same encoder, different dataset
- Metric: Can Unit #3 and Unit #12 of the same DJI Mavic model be separated?
- If yes: hardware fingerprint isolation works (protocol-independent signal)
- If no: model learned protocol signatures, not hardware fingerprints

### 4.4 Experiment 4: False Positive Rate (Non-Drone RF)

- Pass WiFi and Bluetooth spectrograms through frozen encoder
- Measure percentage that fall within the drone cluster
- Target: <5% FPR
- If FPR is high, the model learned "man-made RF transmitter", not "drone"

### 4.5 Experiment 5: SNR Degradation Curves

- Add synthetic AWGN to test spectrograms at SNR levels: +25, +20, +15, +10, +5, 0, -5, -10, -12 dB
- Plot detection accuracy vs SNR for both known and held-out types
- Compare with supervised baselines
- Operational relevance: real-world detection happens at low SNR

### 4.6 Theorem 3 Bound Computation

```python
def compute_theorem3_bound(encoder, dataloader, rho):
    """
    Compute approximate identifiability bound from Theorem 3.
    
    Recovery error <= D + (epsilon + D)^2
    where D = delta / (2 * rho * (1 - rho))
    """
    all_z = []
    all_z_prime = []
    
    for batch in dataloader:
        x1, x2 = create_positive_pairs(batch)
        z1 = encoder(x1)
        z2 = encoder(x2)
        all_z.append(z1)
        all_z_prime.append(z2)
    
    z = torch.cat(all_z)
    z_prime = torch.cat(all_z_prime)
    
    # Alignment loss (actual)
    L_actual = F.mse_loss(z, z_prime) * z.shape[1]  # per-sample * dim
    
    # Optimal alignment loss for linear map
    L_optimal = 2 * (1 - rho) * z.shape[1]
    
    # Alignment gap
    delta = max(0, L_actual.item() - L_optimal)
    
    # Covariance deviation from identity
    cov = torch.cov(z.T)
    epsilon = torch.norm(cov - torch.eye(z.shape[1]), p='fro').item()
    
    # Normalized alignment gap
    D = delta / (2 * rho * (1 - rho))
    
    # Bound
    bound = D + (epsilon + D) ** 2
    
    return {
        'delta': delta,
        'epsilon': epsilon,
        'D': D,
        'bound': bound,
        'actual_recovery_error': None  # measured separately
    }
```

### 4.7 Visualization (The Demo)

```
UMAP Projection of Latent Space
  - Color by drone type (37 colors for 37 RAUAV types)
  - Mark 30 known types with circles
  - Mark 7 held-out types with stars
  - Mark non-drone negatives with X
  - Goal: stars cluster with circles, away from X marks
  
Interactive version for CTO demo:
  1. Show known drone cluster (30 types)
  2. Animate: held-out drones appear and land inside cluster
  3. Show non-drone signals landing outside
  4. Show SNR degradation (noisy signals still cluster correctly)
```

---

## 5. Supervised Baselines (Must Beat These on Zero-Shot)

| Baseline | Architecture | Expected Known-Class Acc | Zero-Shot? |
|----------|-------------|------------------------|------------|
| Wavelet + ResNet-50 | Wavelet scattering + 2D ResNet-50 | ~99% | No |
| EfficientNetB0 | 2D CNN on spectrograms | ~97% | No |
| CV-CNN | Complex-valued 1D CNN on raw I/Q | ~99% | No |
| MC-DNN + FEG | Feature engineering + multi-channel DNN | ~98.4% | No |

Our model will NOT beat these on known-class accuracy. Our value is: any accuracy at all on unknown classes, where these models score 0%.

---

## 6. Fallback Plan

### 6.1 If SIGReg Doesn't Converge

- Drop SIGReg, add EMA teacher (momentum=0.996) -> standard V-JEPA
- Simpler, empirically proven, loses theoretical guarantee but keeps the architecture

### 6.2 If V-JEPA Also Fails

- Drop JEPA entirely, use SimCLR with temporal proximity positive pairs
- Proven for RF in existing literature (81% linear probing accuracy)

### 6.3 If Self-Supervised Entirely Fails

- Train supervised ResNet-50 baseline
- Extract penultimate layer features
- Show t-SNE/UMAP of held-out types
- Even this can demonstrate "drone-ness" as a signal, just not as elegantly

### 6.4 Fallback Order

```
LeJEPA (SIGReg) -> V-JEPA (EMA teacher) -> SimCLR -> Supervised + UMAP
```

Each step loses theoretical elegance but preserves the ability to produce a result.

---

## 7. Code Repository Structure

```
iris/
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml          # Default hyperparameters
│   ├── sweep_lambda.yaml     # Lambda sweep config
│   └── modal.yaml            # Modal deployment config
├── data/
│   ├── download.py           # Dataset download scripts
│   ├── preprocess.py         # I/Q -> STFT -> HDF5 pipeline
│   ├── dataset.py            # PyTorch Dataset + positive pair generation
│   └── normalize.py          # Per-dataset normalization strategies
├── models/
│   ├── encoder.py            # CNN encoder with BatchNorm
│   ├── sigreg.py             # SIGReg loss implementation
│   └── lejepa.py             # Full LeJEPA model (encoder + losses)
├── train/
│   ├── trainer.py            # Main training loop
│   └── modal_train.py        # Modal deployment script
├── eval/
│   ├── linear_probe.py       # Supervised linear probe on frozen encoder
│   ├── zeroshot.py           # Zero-shot detection evaluation
│   ├── fingerprint.py        # Individual device fingerprinting (DRFF-R2)
│   ├── fpr.py                # False positive rate computation
│   ├── snr_curve.py          # SNR degradation curve generation
│   ├── theorem3.py           # Theorem 3 bound computation
│   └── visualize.py          # UMAP/t-SNE visualization + demo animation
├── baselines/
│   ├── resnet50_supervised.py
│   ├── efficientnet_supervised.py
│   └── simclr.py             # SimCLR fallback
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_monitor.ipynb
│   └── 03_results_visualization.ipynb
└── scripts/
    ├── run_train.sh
    ├── run_eval.sh
    └── run_sweep.sh
```

---

## 8. Timeline

| Day | Milestone | Deliverable |
|-----|-----------|-------------|
| 1 | Download RAUAV + DroneDetect datasets | Raw data on disk |
| 2 | I/Q -> STFT pipeline, HDF5 conversion | Processed data ready |
| 3 | Data splits, normalization, positive pair testing | Verified data pipeline |
| 4 | CNN encoder + SIGReg + training loop | Code compiles and runs |
| 5 | First training run on Modal (default params) | Loss curves, checkpoint |
| 6 | Linear probe evaluation | Go/no-go on representation quality |
| 7 | Zero-shot evaluation on 7 held-out types | **THE RESULT** |
| 8 | FPR evaluation + SNR curves | Robustness data |
| 9 | Hyperparameter sweep (lambda, rho) | Optimized model |
| 10 | DRFF-R2 individual fingerprinting | Hardware fingerprint proof |
| 11 | Cross-dataset validation (CDRF) | Generalization evidence |
| 12 | Theorem 3 bound computation | Theoretical validation |
| 13 | UMAP visualization + demo prep | Demo-ready materials |
| 14 | Fallback runs if needed (V-JEPA, SimCLR) | Safety net results |

---

## 9. Success Criteria

| Result | Meaning | Next Step |
|--------|---------|-----------|
| Held-out drones cluster separately from non-drone | Drone-ness is real | Demo to Armory.in CTO |
| Held-out drones cluster by type | Hardware fingerprint isolation works | Paper submission |
| Individual units within same model separate | Full device fingerprinting | This is a company |
| Zero-shot + low FPR + SNR robust | Operationally viable | Production deployment discussion |
| None of the above | Drone-ness is not a separable signal in RF | Honest result, 2 weeks not 2 months |

---

## 10. Theoretical Justification (For Paper/Demo)

### Why LeJEPA Works for RF

1. **RF hardware fingerprints are Gaussian.** CFO drift, oscillator phase noise, thermal noise, amplifier nonlinearities are modeled as Gaussian random variables in every DSP textbook. This satisfies LeJEPA's core assumption (Theorem 1).

2. **Adjacent RF windows are OU-like.** The hardware fingerprint persists across time windows (same transmitter) while noise varies — this is an Ornstein-Uhlenbeck process: z' = rho*z + sqrt(1-rho^2)*eta.

3. **Theorem 3 covers non-Gaussian modulation.** Discrete QAM symbols and packet headers violate pure Gaussianity, but Theorem 3's approximate bound shows graceful degradation. The physical fingerprints (Gaussian) dominate because they're present in every sample, while discrete symbols appear sporadically.

4. **Linear identifiability = hardware fingerprint recovery.** Theorem 1 guarantees that LeJEPA recovers the true latent variables (up to rotation). For RF, those latent variables ARE the hardware fingerprints. The model has no choice but to learn them.

### Why This Is Novel

- Nobody has applied LeJEPA to RF signals
- Nobody has proven zero-shot drone detection via self-supervised RF representation learning
- The theoretical alignment between LeJEPA's Gaussian assumption and RF hardware physics is a new contribution
- Theorem 3 bound provides a quantitative prediction of representation quality — testable on RF data

---

## 11. Key Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SIGReg doesn't converge on RF data | Medium | High | Fall back to V-JEPA with EMA teacher |
| Model learns protocol signatures, not hardware fingerprints | Medium | Critical | DRFF-R2 same-model-different-unit test |
| Over-normalization kills fingerprints | Medium | High | Start with Level 1, test Level 0 |
| "Drone-ness" isn't a coherent signal across all drone types | Low-Medium | Critical | That's what the experiment tests. If it fails, we know. |
| FPR too high (model detects "RF transmitter", not "drone") | Medium | High | Include diverse non-drone negatives; if FPR is high, refine with hard negative mining |
| rho (autocorrelation) wrong for positive pairs | Low | Medium | Sweep delta_t to find optimal rho range |
| BatchNorm collapse despite using it | Low | Medium | Try LayerNorm or GroupNorm as alternative |
| Compute budget insufficient | Low | Medium | Use smaller model, fewer epochs; fall back to SimCLR which trains faster |

---

## 12. Reference Implementations

| Repo | URL | Use |
|------|-----|-----|
| LeJEPA identifiability | github.com/klindtlab/lejepa-identifiability | SIGReg theory + Lean proofs |
| galilai-group/lejepa | github.com/galilai-group/lejepa | Reference LeJEPA implementation |
| lucas-maes/le-wm | github.com/lucas-maes/le-wm | LeWorldModel + LanceDB data loaders |
| lejepa-playground | github.com/ssenthilnathan3/lejepa-playground | Easiest to fork and hack |
| S3R (open-set drone) | github.com/DaftJun/S3R | Open set learning for drone RF |
| dji_droneid | github.com/proto17/dji_droneid | DJI DroneID GNU Radio simulation |

---

## 13. What "Done" Looks Like

**Minimum viable result (for internship demo):**
- UMAP plot showing 7 held-out drone types clustering with 30 known types
- FPR <10% against WiFi/Bluetooth negatives
- One-paragraph explanation that a non-technical CTO can understand

**Full result (for paper):**
- All of above
- Linear probe accuracy on known types (>80%)
- FPR <5%
- SNR degradation curve showing robustness to -5 dB
- DRFF-R2 individual device separation
- Theorem 3 bound computation matching empirical recovery error
- Cross-dataset validation on CDRF
- Ablation: lambda sweep, rho sweep, normalization strategy

**Stretch result (for company):**
- All of above
- Individual unit fingerprinting with >90% accuracy
- Real-time inference benchmark (<5ms on edge hardware)
- ONNX export for deployment on SDR-equipped device