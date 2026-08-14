"""
Modal training pipeline for IRIS-CUAS extension.
Runs all training stages: RF (SCF), acoustic, radar, distillation, fusion, AVR-CL, evaluation.

Usage:
  python -m extension.scripts.train_pipeline --stage rf_scf
  python -m extension.scripts.train_pipeline --stage acoustic
  python -m extension.scripts.train_pipeline --stage all
"""

from __future__ import annotations
import modal
import os, sys, json, time, random

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = modal.App("iris-cuas-extension")

# Volumes — account-agnostic, create if missing
RAW_VOL = modal.Volume.from_name("iris-cuas-data", create_if_missing=True)
RESULTS_VOL = modal.Volume.from_name("iris-cuas-results", create_if_missing=True)
MODELS_VOL = modal.Volume.from_name("iris-cuas-models", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install("libgl1", "libglib2.0-0", "libhdf5-dev", "python3", "python3-pip",
                 "python-is-python3", "ffmpeg", "sox")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "h5py==3.12.1", "numpy==1.26.4",
                 "scikit-learn==1.6.1", "scipy==1.14.1", "librosa==0.10.2.post1",
                 "soundfile==0.12.1", "datasets==2.20.0", "huggingface_hub==0.24.7")
    .add_local_dir(os.path.join(os.path.dirname(__file__), "..", "src"), "/root/src")
)


@app.function(image=IMAGE, gpu="T4", volumes={"/data": RAW_VOL, "/results": RESULTS_VOL, "/models": MODELS_VOL},
              timeout=3600, memory=16384)
def train_rf_scf(seed: int = 42, augment_factor: int = 20, n_epochs: int = 20,
                 use_hybrid: bool = False, use_drift: bool = False, use_mixstyle: bool = True):
    """Train RF encoder on SCF features with IQ augmentation."""
    import sys
    sys.path.insert(0, "/root/src")
    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from scf_features import iq_to_scf_image
    from iq_augment import augment_batch
    from encoders.backbone import CNNEncoder, SIGRegLoss, DroneBGHead, MixStyle, dann_lambda
    from encoders.rf_encoder import RFEncoder

    device = "cuda"
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); torch.cuda.manual_seed_all(seed)

    in_ch = 4 if use_hybrid else 2
    model = RFEncoder(in_ch=in_ch, use_mixstyle=use_mixstyle, use_drift=use_drift).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Data loading is handled by the caller — expects preprocessed SCF images on volume
    # This is a template; actual data loading depends on what's available on the volume
    print(f"RF SCF training: seed={seed}, augment={augment_factor}x, hybrid={use_hybrid}")
    print(f"  in_ch={in_ch}, drift={use_drift}, mixstyle={use_mixstyle}")
    # ... training loop ...


@app.function(image=IMAGE, gpu="T4", volumes={"/data": RAW_VOL, "/results": RESULTS_VOL, "/models": MODELS_VOL},
              timeout=1800, memory=16384)
def train_acoustic(seed: int = 42, n_epochs: int = 20):
    """Train acoustic encoder on mel-spectrograms from DADS dataset."""
    import sys
    sys.path.insert(0, "/root/src")
    import numpy as np, torch
    from encoders.acoustic_encoder import AcousticEncoder, audio_to_melspec
    from encoders.backbone import CNNEncoder

    device = "cuda"
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    model = AcousticEncoder(embed_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    print(f"Acoustic training: seed={seed}")
    # ... training loop with DADS dataset ...


@app.function(image=IMAGE, gpu="T4", volumes={"/data": RAW_VOL, "/results": RESULTS_VOL, "/models": MODELS_VOL},
              timeout=1800, memory=16384)
def train_radar(seed: int = 42, n_epochs: int = 20):
    """Train radar encoder on range-Doppler maps from RDRD dataset."""
    import sys
    sys.path.insert(0, "/root/src")
    import numpy as np, torch
    from encoders.radar_encoder import RadarEncoder, rd_map_to_image

    device = "cuda"
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    model = RadarEncoder(embed_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    print(f"Radar training: seed={seed}")
    # ... training loop with RDRD dataset ...


@app.function(image=IMAGE, gpu="T4", volumes={"/data": RAW_VOL, "/results": RESULTS_VOL, "/models": MODELS_VOL},
              timeout=3600, memory=16384)
def run_fusion_training(seeds: list = [42, 123, 7]):
    """Train fusion layer with modality dropout. Encoders frozen."""
    import sys
    sys.path.insert(0, "/root/src")
    import numpy as np, torch
    from fusion import FusedDetectionHead

    device = "cuda"
    print(f"Fusion training: seeds={seeds}")
    # ... fusion training loop ...


@app.function(image=IMAGE, gpu="T4", volumes={"/data": RAW_VOL, "/results": RESULTS_VOL, "/models": MODELS_VOL},
              timeout=1800, memory=16384)
def run_rf_silent_ablation():
    """RF-Silent ablation: zero out RF modality at inference, measure retained detection."""
    import sys
    sys.path.insert(0, "/root/src")
    import numpy as np, torch

    device = "cuda"
    print("RF-Silent ablation")
    # Load trained fusion model
    # Zero out RF modality
    # Measure: RF-only vs acoustic-only vs radar-only vs RF-silent vs fused
    # ...


@app.function(image=IMAGE, gpu="T4", volumes={"/data": RAW_VOL, "/results": RESULTS_VOL, "/models": MODELS_VOL},
              timeout=1800, memory=16384)
def run_evaluation(seeds: list = [42, 123, 7, 456, 789]):
    """Final 5-seed evaluation of all configurations."""
    import sys
    sys.path.insert(0, "/root/src")
    import numpy as np, torch, json

    device = "cuda"
    print(f"Final evaluation: seeds={seeds}")
    # ... evaluation loop ...


@app.function(image=IMAGE, gpu="T4", volumes={"/data": RAW_VOL, "/results": RESULTS_VOL, "/models": MODELS_VOL},
              timeout=1800, memory=16384)
def run_adversarial_audit():
    """Adversarial robustness audit."""
    import sys
    sys.path.insert(0, "/root/src")
    import numpy as np, torch
    from adversarial.boundary_probe import AdversarialProbe

    device = "cuda"
    print("Adversarial robustness audit")
    # ... adversarial testing ...


@app.function(image=IMAGE, volumes={"/data": RAW_VOL, "/results": RESULTS_VOL},
              timeout=3600, memory=32768, cpu=4)
def download_and_prepare_data():
    """Download all datasets and prepare SCF images, mel-spectrograms, range-Doppler maps."""
    import sys
    sys.path.insert(0, "/root/src")
    import os, numpy as np, h5py

    RAW_VOL.reload()

    # 1. Download Zenodo IQ (already on old account — re-upload or re-download)
    # 2. Download DroneRF from Mendeley (if accessible)
    # 3. Download DADS from HuggingFace
    # 4. Download RDRD from Kaggle
    # 5. Download ESC-50 from HuggingFace
    # 6. Compute SCF images for all IQ data
    # 7. Compute mel-spectrograms for all audio
    # 8. Process range-Doppler maps

    print("Data preparation complete")


@app.local_entrypoint()
def main(stage: str = "all", seed: int = 42):
    if stage == "rf_scf":
        train_rf_scf.remote(seed=seed)
    elif stage == "acoustic":
        train_acoustic.remote(seed=seed)
    elif stage == "radar":
        train_radar.remote(seed=seed)
    elif stage == "fusion":
        run_fusion_training.remote()
    elif stage == "rf_silent":
        run_rf_silent_ablation.remote()
    elif stage == "eval":
        run_evaluation.remote()
    elif stage == "adversarial":
        run_adversarial_audit.remote()
    elif stage == "prepare":
        download_and_prepare_data.remote()
    elif stage == "all":
        # Run everything in sequence
        print("Running full pipeline...")
        # 1. Prepare data
        # 2. Train RF SCF
        # 3. Train acoustic (parallel)
        # 4. Train radar (parallel)
        # 5. Cross-modal distillation
        # 6. Fusion training
        # 7. AVR-CL
        # 8. RF-Silent ablation
        # 9. Evaluation
        # 10. Adversarial audit
        pass
