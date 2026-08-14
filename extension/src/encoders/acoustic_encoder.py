"""
Acoustic Encoder — drone detection from audio.

Input: mel-spectrogram (1 channel, 256×256)
Data: DADS (drone-audio-detection-samples) + ESC-50 + UrbanSound8K as negatives

NOTE: DroneAudioSet (ahlab-drone-project) is audio FROM drone-mounted mics
for SAR/human-presence detection — NOT drone propeller signatures.
Use DADS (geronimobasso/drone-audio-detection-samples) instead.
"""

import numpy as np
import librosa
from .backbone import CNNEncoder, SIGRegLoss, DroneBGHead
import torch
import torch.nn as nn


class AcousticEncoder(nn.Module):
    """Acoustic drone detection encoder using mel-spectrograms."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.encoder = CNNEncoder(in_ch=1, embed_dim=embed_dim)  # single channel
        self.sigreg = SIGRegLoss(embed_dim=embed_dim)
        self.bg_head = DroneBGHead(d=embed_dim)

    def forward(self, x):
        return self.encoder(x)

    def compute_loss(self, z, labels):
        sig_loss = self.sigreg(z)
        bg_logits = self.bg_head(z)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(bg_logits, labels)
        return sig_loss + bce_loss, sig_loss, bce_loss


def audio_to_melspec(audio, sr=16000, n_fft=1024, hop_length=256,
                     n_mels=256, target_frames=256, fmin=0.0, fmax=8000.0):
    """
    Convert raw audio to (1, 256, 256) mel-spectrogram for CNN input.

    Parameters:
        audio: (N,) float32 audio samples
        sr: sample rate
        n_fft: FFT size
        hop_length: hop size
        n_mels: number of mel bins
        target_frames: target time frames (resize if needed)

    Returns: (1, 256, 256) float32
    """
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)  # stereo → mono

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0
    )
    log_mel = np.log1p(mel)  # (n_mels, n_frames)

    # Resize to (256, 256)
    if log_mel.shape[1] != target_frames:
        if log_mel.shape[1] > target_frames:
            log_mel = log_mel[:, :target_frames]
        else:
            pad = np.tile(log_mel[:, -1:], (1, target_frames - log_mel.shape[1]))
            log_mel = np.concatenate([log_mel, pad], axis=1)

    if log_mel.shape[0] != 256:
        idx = np.linspace(0, log_mel.shape[0] - 1, 256).astype(int)
        log_mel = log_mel[idx, :]

    # Z-score
    std = log_mel.std()
    log_mel = (log_mel - log_mel.mean()) / (std + 1e-8)

    return log_mel[np.newaxis, :, :].astype(np.float32)


def load_audio_file(path, sr=16000, duration=4.0):
    """Load audio file, resample, pad/truncate to fixed duration."""
    audio, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    target_len = int(sr * duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
    else:
        audio = audio[:target_len]
    return audio
