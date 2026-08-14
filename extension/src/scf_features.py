"""
Cyclostationary Feature Extraction (SCF) for RF Drone Detection.

Computes the Spectral Correlation Function (SCF) and Spectral Coherence (COH)
from raw complex IQ samples. The COH channel is exactly invariant to receiver
gain, phase, and AGC by construction.

Output: (2, 256, 256) float32 image
  ch0 = log10(|SCF| + eps)  — cyclic power texture
  ch1 = |COH| in [0, 1]     — receiver-invariant spectral coherence

References:
  Gardner 1986 (IEEE Trans Comm, 531 citations)
  PySDR Cyclostationary Processing chapter
  Roberts, Brown, Loomis 1991 (FAM method, 448 citations)
"""

import numpy as np
import torch
import torch.nn.functional as F


def _to_complex(iq):
    """Accept (N,), (N,2), (2,N) and return complex128 (N,)."""
    x = np.asarray(iq)
    if np.iscomplexobj(x):
        return x.astype(np.complex128)
    if x.ndim == 2 and x.shape[0] == 2:
        return x[0].astype(np.complex128) + 1j * x[1].astype(np.complex128)
    if x.ndim == 2 and x.shape[1] == 2:
        return x[:, 0].astype(np.complex128) + 1j * x[:, 1].astype(np.complex128)
    return x.astype(np.complex128)


def scf_frequency_smoothing(iq, n_fft=1<<14, n_alpha=256, alpha_max=0.5, window_len=128):
    """
    Estimate SCF via the Frequency-Smoothing Method (FSM).

    S^a(f) = smooth_f [ X(f + a/2) X*(f - a/2) ]

    Complexity: O(N log N + N_alpha * N)
    """
    z = _to_complex(iq)
    N = len(z)
    if N < n_fft:
        z = np.concatenate([z, np.zeros(n_fft - N, dtype=z.dtype)])
    else:
        z = z[:n_fft]
    N = n_fft

    z = z * np.hanning(N)
    X = np.fft.fftshift(np.fft.fft(z))

    alphas = np.linspace(0.0, alpha_max, n_alpha)
    n_freq = max(N // window_len, 1)
    win = np.hanning(window_len)

    SCF = np.zeros((n_alpha, n_freq), dtype=np.complex128)
    Sx = np.abs(X) ** 2

    for i, a in enumerate(alphas):
        shift = int(round(a * N / 2.0))
        scf_slice = np.roll(X, -shift) * np.conj(np.roll(X, shift))
        SCF[i, :] = np.convolve(scf_slice, win, mode="same")[::window_len][:n_freq]

    SCF[0, :] = 0  # null PSD (alpha=0) so cyclic features are visible
    return SCF, alphas, np.linspace(-0.5, 0.5, n_freq, endpoint=False)


def spectral_coherence(SCF, iq, n_fft=1<<14, window_len=128):
    """
    Spectral Coherence |C^a(f)| — exactly invariant to complex gain.

    |C^a(f)|^2 = |S^a(f)|^2 / ( S(f+a/2) * S(f-a/2) )

    Returns: (n_alpha, n_freq) real float in [0, 1]
    """
    z = _to_complex(iq)
    N = len(z)
    if N < n_fft:
        z = np.concatenate([z, np.zeros(n_fft - N, dtype=z.dtype)])
    else:
        z = z[:n_fft]
    z = z * np.hanning(len(z))
    X = np.fft.fftshift(np.fft.fft(z))
    Sx = np.abs(X) ** 2

    n_alpha, n_freq = SCF.shape
    alphas = np.linspace(0.0, 0.5, n_alpha)
    win = np.hanning(window_len)
    eps = 1e-12 * (Sx.max() + 1e-30)

    COH = np.zeros((n_alpha, n_freq), dtype=np.float64)
    for i, a in enumerate(alphas):
        shift = int(round(a * len(X) / 2.0))
        Splus = np.convolve(np.roll(Sx, -shift), win, mode="same")[::window_len][:n_freq]
        Sminus = np.convolve(np.roll(Sx, shift), win, mode="same")[::window_len][:n_freq]
        denom = np.sqrt(Splus * Sminus) + eps
        COH[i, :] = np.abs(SCF[i, :]) / denom

    return np.clip(COH, 0.0, 1.0)


def iq_to_scf_image(iq, out_size=256, n_fft=1<<14, alpha_max=0.5, window_len=128):
    """
    Convert raw IQ to (2, 256, 256) SCF image for CNN input.

    ch0: log10(|SCF| + eps) — cyclic power texture
    ch1: |COH| in [0, 1]    — exactly receiver-invariant

    Both channels are resized to out_size × out_size and standardized
    to zero-mean / unit-variance per channel.
    """
    z = _to_complex(iq)
    SCF, alphas, f = scf_frequency_smoothing(z, n_fft=n_fft, n_alpha=out_size,
                                              alpha_max=alpha_max, window_len=window_len)
    COH = spectral_coherence(SCF, z, n_fft=n_fft, window_len=window_len)

    ch0 = np.log10(np.abs(SCF) + 1e-12).astype(np.float64)
    ch1 = COH.astype(np.float64)
    img = np.stack([ch0, ch1], axis=0)

    # Resize
    C, H, W = img.shape
    if H != out_size or W != out_size:
        t = torch.from_numpy(img).float().unsqueeze(0)
        t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
        img = t.squeeze(0).numpy()

    # Standardize per channel
    for c in range(img.shape[0]):
        mu, sd = img[c].mean(), img[c].std() + 1e-8
        img[c] = (img[c] - mu) / sd

    return img.astype(np.float32)


def normalized_autocorrelation(iq, max_lag=256):
    """
    r[tau] = sum x[n] x*[n-tau] / sum |x[n]|^2

    Gain-invariant (|g|^2 cancels). Returns (max_lag,) complex.
    """
    z = _to_complex(iq)
    z = z - z.mean()
    norm = np.sum(np.abs(z) ** 2) + 1e-12
    r = np.empty(max_lag, dtype=np.complex128)
    for tau in range(max_lag):
        if tau == 0:
            r[tau] = np.sum(np.abs(z) ** 2)
        else:
            r[tau] = np.sum(z[tau:] * np.conj(z[:-tau]))
    return r / norm


def higher_order_moments(iq, orders=(2, 4)):
    """
    M_p = E[|x|^p] / E[|x|^2]^(p/2) — scale-invariant modulation descriptors.
    """
    mags = np.abs(_to_complex(iq))
    p2 = np.mean(mags ** 2) + 1e-12
    return {p: float(np.mean(mags ** p) / (p2 ** (p / 2.0))) for p in orders}


def iq_to_hybrid_image(iq, out_size=256, n_fft=1<<14, alpha_max=0.5, window_len=128):
    """
    4-channel hybrid input: 2 channels SCF + 1 channel autocorrelation + 1 channel HOM.

    All channels are receiver-invariant by construction.
    Output: (4, 256, 256) float32
    """
    z = _to_complex(iq)

    # SCF (2 channels)
    SCF, alphas, f = scf_frequency_smoothing(z, n_fft=n_fft, n_alpha=out_size,
                                              alpha_max=alpha_max, window_len=window_len)
    COH = spectral_coherence(SCF, z, n_fft=n_fft, window_len=window_len)
    ch0 = np.log10(np.abs(SCF) + 1e-12).astype(np.float64)
    ch1 = COH.astype(np.float64)

    # Normalized autocorrelation (1 channel)
    r = normalized_autocorrelation(z, max_lag=out_size)
    ch2 = np.abs(r).astype(np.float64).reshape(1, -1)
    ch2 = np.tile(ch2, (1, out_size))  # (1, out_size) → (out_size, out_size)

    # Higher-order moment M4 as a constant channel (1 channel)
    hom = higher_order_moments(z, orders=(4,))
    ch3 = np.full((out_size, out_size), hom[4], dtype=np.float64)

    img = np.stack([ch0, ch1, ch2, ch3], axis=0)

    # Resize all channels
    C, H, W = img.shape
    if H != out_size or W != out_size:
        t = torch.from_numpy(img).float().unsqueeze(0)
        t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
        img = t.squeeze(0).numpy()

    for c in range(img.shape[0]):
        mu, sd = img[c].mean(), img[c].std() + 1e-8
        img[c] = (img[c] - mu) / sd

    return img.astype(np.float32)
