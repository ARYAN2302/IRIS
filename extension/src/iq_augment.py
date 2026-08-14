"""
IQ-Level Augmentation Pipeline for SCF Features.

All augmentations provably preserve the COH channel's receiver-invariance
because COH = |SCF| / sqrt(S+ * S-) cancels any complex gain g:
  - g multiplies both numerator and denominator
  - The ratio is unchanged

Supported augmentations:
  1. Random complex gain (simulates AGC variation)
  2. FIR filtering (simulates front-end frequency response)
  3. Quantization noise (simulates ADC bit depth)
  4. Carrier frequency offset (simulates LO drift)
  5. Time shift (simulates sampling offset)
  6. Additive noise (simulates noise floor variation)
  7. IQ imbalance (simulates analog front-end asymmetry)
"""

import numpy as np
from scipy.signal import fftconvolve


def augment_complex_gain(iq, gain_range=(0.1, 10.0)):
    """Multiply by random complex gain g = |g| * exp(j*theta).
    COH invariance: g cancels in |SCF|^2 / (S+ * S-)."""
    magnitude = np.random.uniform(*gain_range)
    phase = np.random.uniform(0, 2 * np.pi)
    g = magnitude * np.exp(1j * phase)
    return iq * g


def augment_fir_filter(iq, n_taps=9):
    """Convolve with random complex Gaussian FIR filter.
    COH invariance: linear filtering multiplies X(f) by H(f),
    so |SCF|^2 gets |H|^2 in both numerator and denominator."""
    h = (np.random.randn(n_taps) + 1j * np.random.randn(n_taps)) / n_taps
    return fftconvolve(iq, h, mode='same')


def augment_quantization(iq, bits=None):
    """Simulate ADC quantization at random bit depth.
    COH invariance: quantization noise is additive and uncorrelated
    with the signal, so it adds to both S+ and S- equally."""
    if bits is None:
        bits = np.random.randint(8, 15)
    max_val = np.max(np.abs(iq))
    if max_val < 1e-10:
        return iq
    scale = (2 ** (bits - 1) - 1) / max_val
    quantized = np.round(iq.real * scale) / scale + 1j * np.round(iq.imag * scale) / scale
    return quantized


def augment_cfo(iq, max_offset=0.01):
    """Apply random carrier frequency offset.
    COH invariance: CFO shifts the spectrum, but SCF is computed
    over the full band so the cyclic structure is preserved."""
    offset = np.random.uniform(-max_offset, max_offset)
    n = np.arange(len(iq))
    return iq * np.exp(2j * np.pi * offset * n)


def augment_time_shift(iq, max_shift=100):
    """Circular time shift.
    COH invariance: time shift adds linear phase in frequency domain,
    which cancels in |SCF|^2."""
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(iq, shift)


def augment_noise(iq, snr_db_range=(0, 30)):
    """Add white Gaussian noise at random SNR.
    COH invariance: noise adds to both S+ and S- equally."""
    snr_db = np.random.uniform(*snr_db_range)
    signal_power = np.mean(np.abs(iq) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(iq)) + 1j * np.random.randn(len(iq))
    )
    return iq + noise


def augment_iq_imbalance(iq, max_imbalance=0.1):
    """Simulate IQ imbalance (amplitude + phase mismatch).
    COH invariance: IQ imbalance is a linear transformation,
    multiplies X(f) by a 2x2 matrix, cancels in ratio."""
    amp_imb = 1 + np.random.uniform(-max_imbalance, max_imbalance)
    phase_imb = np.random.uniform(-max_imbalance, max_imbalance)
    i = iq.real * amp_imb
    q = iq.imag * np.cos(phase_imb) - iq.real * np.sin(phase_imb) * amp_imb
    return i + 1j * q


def augment_batch(iq, augmentations=None, n_augments=20):
    """
    Apply random combinations of augmentations to generate n_augments
    variants of a single IQ sample.

    Returns: list of n_augments augmented IQ arrays
    """
    if augmentations is None:
        augmentations = [
            augment_complex_gain,
            augment_fir_filter,
            augment_quantization,
            augment_cfo,
            augment_time_shift,
            augment_noise,
            augment_iq_imbalance,
        ]

    augmented = []
    for _ in range(n_augments):
        x = iq.copy()
        # Apply 2-4 random augmentations per sample
        n_to_apply = np.random.randint(2, min(5, len(augmentations) + 1))
        chosen = np.random.choice(augmentations, size=n_to_apply, replace=False)
        for aug_fn in chosen:
            x = aug_fn(x)
        augmented.append(x)

    return augmented


def augment_dataset(iq_samples, labels, n_augments=20, seed=42):
    """
    Augment an entire dataset of IQ samples.

    Parameters:
        iq_samples: list of complex IQ arrays
        labels: list of labels (1=drone, 0=BG)
        n_augments: number of augmented copies per original sample
        seed: random seed

    Returns: (augmented_iq, augmented_labels) lists
    """
    np.random.seed(seed)
    all_iq = list(iq_samples)
    all_labels = list(labels)

    for iq, label in zip(iq_samples, labels):
        augmented = augment_batch(iq, n_augments=n_augments)
        all_iq.extend(augmented)
        all_labels.extend([label] * n_augments)

    return all_iq, all_labels
