"""
Bearing Estimator — estimates drone direction from RF signal characteristics.

Uses:
  - Doppler shift from RF spectrogram (approach/depart direction)
  - Signal strength over time (crude bearing curve as drone moves)
  - Phase information (if multi-antenna, for angle-of-arrival)

Even with single antenna: Doppler shift gives approach/depart direction,
and signal strength over time gives a crude bearing curve.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class BearingEstimate:
    """Estimated bearing to a detected drone."""
    azimuth: Optional[float] = None  # degrees, 0=North, 90=East
    elevation: Optional[float] = None  # degrees, 0=horizon, 90=overhead
    approach_rate: Optional[float] = None  # m/s, positive=approaching
    confidence: float = 0.0  # 0-1
    method: str = 'single_antenna'  # or 'multi_antenna', 'acoustic_doa'


class BearingEstimator:
    """
    Estimate bearing from RF spectrogram features.

    For single-antenna systems:
      1. Doppler shift → approach/depart rate
      2. Signal strength over time → crude bearing curve
      3. Phase pattern → directional hint

    For multi-antenna systems (future):
      1. Phase difference between antennas → angle of arrival
      2. MUSIC/ESPRIT algorithms for high-resolution DOA
    """
    def __init__(self, sample_rate=1e6, center_freq=2.4e9):
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.c = 3e8  # speed of light

    def estimate_from_spectrogram(self, spectrogram_log_power: np.ndarray,
                                   time_axis: np.ndarray,
                                   freq_axis: np.ndarray) -> BearingEstimate:
        """
        Estimate bearing from a log-power spectrogram.

        Parameters:
            spectrogram_log_power: (freq_bins, time_frames) — channel 0 of STFT
            time_axis: (time_frames,) time values in seconds
            freq_axis: (freq_bins,) frequency values in Hz

        Returns: BearingEstimate
        """
        # 1. Find the dominant signal frequency over time
        peak_freq_idx = np.argmax(spectrogram_log_power, axis=0)
        peak_freqs = freq_axis[peak_freq_idx]

        # 2. Compute Doppler shift (deviation from center frequency)
        doppler_shift = peak_freqs - self.center_freq

        # 3. Estimate approach rate from Doppler trend
        if len(doppler_shift) > 10:
            # Linear fit of Doppler over time
            coeffs = np.polyfit(time_axis, doppler_shift, 1)
            doppler_rate = coeffs[0]  # Hz/s

            # Convert to approach rate: v = (doppler_rate * c) / (2 * f_center)
            # Positive doppler_rate = frequency increasing = approaching
            approach_rate = (doppler_rate * self.c) / (2 * self.center_freq)
        else:
            approach_rate = 0.0

        # 4. Signal strength over time (crude bearing)
        signal_strength = np.max(spectrogram_log_power, axis=0)  # (time_frames,)
        if len(signal_strength) > 10:
            # Strength trend: increasing = getting closer
            strength_trend = np.polyfit(time_axis, signal_strength, 1)[0]
            confidence = min(1.0, abs(strength_trend) / 10.0)
        else:
            confidence = 0.3

        # 5. Estimate azimuth from Doppler + strength pattern
        # This is very crude for single-antenna — just approach/depart
        if approach_rate > 0:
            azimuth_hint = 0.0  # approaching from front
        else:
            azimuth_hint = 180.0  # departing

        return BearingEstimate(
            azimuth=azimuth_hint,
            approach_rate=float(approach_rate),
            confidence=float(confidence),
            method='single_antenna'
        )

    def estimate_from_signal_strength(self, rssi_over_time: np.ndarray,
                                       timestamps: np.ndarray) -> BearingEstimate:
        """
        Estimate bearing from received signal strength indicator (RSSI) over time.

        As a drone moves, RSSI changes. A peak in RSSI indicates the drone
        is closest to the receiver. The shape of the curve gives bearing info.
        """
        if len(rssi_over_time) < 5:
            return BearingEstimate(confidence=0.1, method='rssi')

        # Find peak RSSI time
        peak_idx = np.argmax(rssi_over_time)
        peak_time = timestamps[peak_idx]

        # Before peak: approaching. After peak: departing.
        # Rate of change gives speed estimate
        pre_peak = rssi_over_time[:peak_idx+1]
        post_peak = rssi_over_time[peak_idx:]

        if len(pre_peak) > 2 and len(post_peak) > 2:
            pre_rate = np.mean(np.diff(pre_peak))
            post_rate = np.mean(np.diff(post_peak))

            # Symmetric approach/depart = drone passing nearby
            # Asymmetric = drone approaching or departing
            symmetry = abs(pre_rate + post_rate) / (abs(pre_rate) + abs(post_rate) + 1e-8)
            confidence = 1.0 - symmetry  # more symmetric = more confident
        else:
            confidence = 0.2

        return BearingEstimate(
            approach_rate=float(-np.mean(np.diff(rssi_over_time))),  # positive = approaching
            confidence=float(confidence),
            method='rssi'
        )
