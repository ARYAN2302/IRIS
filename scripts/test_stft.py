"""Quick smoke test for the STFT engine."""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.stft_engine import STFTEngine

# Generate fake I/Q data — a chirp signal (frequency sweep)
fs = 20e6  # 20 MHz sample rate
duration = 0.01  # 10 ms
t = np.arange(int(fs * duration)) / fs

# Chirp from 1 MHz to 5 MHz
f0, f1 = 1e6, 5e6
freq = f0 + (f1 - f0) * t / duration
iq = np.exp(2j * np.pi * freq * t).astype(np.complex64)

# Add some noise (realistic)
iq += 0.1 * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq))).astype(np.complex64)

# Run STFT engine
engine = STFTEngine(n_fft=1024, hop_len=256, win_len=1024, target_height=256, target_width=256)
spec = engine(iq)

print(f"Input:  {len(iq)} complex I/Q samples ({len(iq)/fs*1000:.1f} ms)")
print(f"Output: shape={spec.shape}, dtype={spec.dtype}")
print(f"  Ch0 (log-power):  min={spec[0].min():.3f}, max={spec[0].max():.3f}, mean={spec[0].mean():.3f}")
print(f"  Ch1 (phase):      min={spec[1].min():.3f}, max={spec[1].max():.3f}, mean={spec[1].mean():.3f}")

# Test segmentation
segments = engine.segment_signal(iq, segment_len=2048, stride=1024)
print(f"\nSegmentation: {len(segments)} segments of 2048 samples (50% overlap)")
specs = [engine(seg) for seg in segments[:3]]
print(f"3 spectrograms: shapes = {[s.shape for s in specs]}")

print("\n✅ STFT engine works!")