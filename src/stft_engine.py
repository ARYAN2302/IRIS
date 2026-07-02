"""
STFT Engine — I/Q samples → 2-channel spectrogram tensor
Channel 1: log-power spectrogram (spectral shape + modulation)
Channel 2: normalized phase (carrier offset + phase noise = hardware fingerprint)
"""

import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann


class STFTEngine:
    def __init__(
        self,
        n_fft: int = 1024,
        hop_len: int = 256,
        win_len: int = 1024,
        target_height: int = 256,
        target_width: int = 256,
    ):
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.target_height = target_height
        self.target_width = target_width

        # Hann window — standard for RF spectral analysis
        # Good frequency resolution, moderate sidelobe suppression
        self.window = hann(win_len, sym=True)

        self.stft = ShortTimeFFT(
            win=self.window,
            hop=hop_len,
            fs=1.0,
            mfft=n_fft,
            fft_mode='twosided',
        )
    def __call__(self, iq_complex: np.ndarray) -> np.ndarray:
        # S shape: (n_fft, time_frames) where n_fft = 1024 for twosided FFT
        """
        Args:
            iq_complex: 1D complex64 array of raw I/Q samples

        Returns:
            spec: float32 array of shape (2, target_height, target_width)
                  Channel 0 = log-power (normalized)
                  Channel 1 = phase (normalized)
        """
        # --- STFT ---
        # scipy ShortTimeFFT returns complex STFT matrix
        # shape: (n_fft // 2 + 1, num_time_frames) for onesided
        S = self.stft.stft(iq_complex)
        # S shape: (freq_bins, time_frames) where freq_bins = n_fft//2 + 1 = 513

        # --- Channel 1: Log-power spectrogram ---
        power = np.abs(S) ** 2
        log_power = np.log1p(power)  # log(1 + |X|²), avoids log(0)

        # --- Channel 2: Normalized phase ---
        phase = np.angle(S) / np.pi  # maps to [-1, 1]

        # --- Resize to target dimensions ---
        log_power = self._resize(log_power, self.target_height, self.target_width)
        phase = self._resize(phase, self.target_height, self.target_width)

        # --- Per-channel normalization ---
        log_power = self._normalize(log_power)
        phase = self._normalize(phase)

        # --- Stack into 2-channel tensor ---
        spec = np.stack([log_power, phase], axis=0).astype(np.float32)
        return spec

    def _resize(self, img: np.ndarray, h: int, w: int) -> np.ndarray:
        """Bilinear resize — handles any input shape to (h, w)."""
        orig_h, orig_w = img.shape
        if orig_h == h and orig_w == w:
            return img

        # Row indices for target height
        row_idx = np.linspace(0, orig_h - 1, h)
        col_idx = np.linspace(0, orig_w - 1, w)

        # Nearest-neighbor interpolation (fast, sufficient for spectrograms)
        row_idx = np.round(row_idx).astype(int)
        col_idx = np.round(col_idx).astype(int)

        # Clamp to valid range
        row_idx = np.clip(row_idx, 0, orig_h - 1)
        col_idx = np.clip(col_idx, 0, orig_w - 1)

        return img[np.ix_(row_idx, col_idx)]

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Zero-mean, unit-variance normalization. Handles flat images."""
        mean = img.mean()
        std = img.std()
        if std < 1e-8:
            return img - mean
        return (img - mean) / std

    def segment_signal(
        self, iq_complex: np.ndarray, segment_len: int, stride: int | None = None
    ) -> list[np.ndarray]:
        """
        Chop a long I/Q capture into overlapping segments.
        Each segment becomes one spectrogram sample.

        Args:
            iq_complex: 1D complex64 array (full capture)
            segment_len: number of I/Q samples per segment
            stride: step between segments (default = segment_len // 2 for 50% overlap)

        Returns:
            list of 1D complex64 arrays, each of length segment_len
        """
        if stride is None:
            stride = segment_len // 2  # 50% overlap

        segments = []
        for start in range(0, len(iq_complex) - segment_len + 1, stride):
            seg = iq_complex[start : start + segment_len]
            segments.append(seg)

        return segments