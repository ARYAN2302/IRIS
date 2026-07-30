#!/usr/bin/env python3
"""
IRIS Live Demo — Real-Time Drone Detection on M1 Mac

This is the meeting-day demo. Open laptop, run this script, watch IRIS
detect drones in real-time on a spectrogram waterfall.

Three input modes (auto-detected or via --mode):
  1. iq_file    — playback of a recorded I/Q file (.cf32, .complex64, .sigmf-data)
  2. hdf5       — replay holdout spectrograms from IRIS HDF5 (simulates streaming)
  3. synthetic  — inject synthetic drone-like bursts into background noise (default, no files needed)

Display:
  - Top panel: rolling spectrogram waterfall (most recent N windows)
  - Bottom panel: Mahalanobis distance over time + threshold line
  - Alert banner: "⚠ DRONE DETECTED" when above threshold

Usage:
    # Default — synthetic mode, no files needed
    python scripts/live_demo.py

    # Replay HDF5 holdout spectrograms
    python scripts/live_demo.py --mode hdf5

    # Playback I/Q file
    python scripts/live_demo.py --mode iq_file --file path/to/recording.cf32

    # Adjust speed
    python scripts/live_demo.py --fps 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.iris_inference import IRISDetector, iq_to_spectrogram

# Optional: intent classifier (Build 3)
try:
    from src.intent_head import IntentClassifier, INTENT_CLASSES
    INTENT_AVAILABLE = True
except ImportError:
    INTENT_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "models/lejepa_v11_best.pt"
CENTROID_PATH = "models/drone_centroid.npz"
INTENT_HEAD_PATH = "models/intent_head.pt"
SAMPLES_DIR = Path.home() / ".iris_samples"
HDF5_PATH = "data/iris_rfuav.h5"
MATCHED_BG_PATH = "data/iris_matched_bg.h5"

WATERFALL_HISTORY = 64  # number of spectrogram columns to show in waterfall
DEFAULT_FPS = 8         # spectrograms per second
ALERT_HOLD_SECONDS = 3.0  # how long to keep alert banner visible after detection


# ─────────────────────────────────────────────────────────────────────────────
# Input sources
# ─────────────────────────────────────────────────────────────────────────────


class SyntheticSource:
    """
    Generates synthetic drone-like RF bursts injected into background noise.
    No files needed — for demo when no I/Q or HDF5 is available.

    Simulates:
      - Background: white Gaussian noise (the IRIS background cluster)
      - Drone bursts: every ~5 seconds, inject a synthetic drone-like
        pattern (frequency-hopping tone + sidebands) for ~2 seconds
    """

    def __init__(self, sample_rate: float = 20e6, n_fft: int = 1024, seed: int = 42):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.rng = np.random.default_rng(seed)
        self.step = 0
        self.next_drone_at = 30  # steps before first drone
        self.drone_active_until = 0

    def next_spectrogram(self) -> torch.Tensor:
        """Generate the next spectrogram (256x256, 2-channel)."""
        self.step += 1

        # Decide if drone is active
        if self.step >= self.next_drone_at and self.step < self.drone_active_until:
            drone_active = True
        elif self.step >= self.next_drone_at:
            # Start drone burst
            self.drone_active_until = self.step + 16  # ~2 sec at 8 fps
            self.next_drone_at = self.drone_active_until + 40  # next drone in ~5 sec
            drone_active = True
        else:
            drone_active = self.step < self.drone_active_until

        # Generate I/Q samples
        n_samples = self.n_fft * 4  # 4 STFT windows per spectrogram
        if drone_active:
            # Drone-like signal: carrier + harmonics + noise
            t = np.arange(n_samples) / self.sample_rate
            # Carrier frequency (random within 2.4 GHz band simulation)
            fc = self.rng.uniform(0.1, 0.4) * self.sample_rate / 2
            # Frequency hopping pattern
            hop_pattern = self.rng.uniform(-0.05, 0.05, size=n_samples // 100)
            hop = np.repeat(hop_pattern, 100)[:n_samples]
            carrier = np.exp(2j * np.pi * (fc + hop * self.sample_rate / 4) * t)
            # Sidebands (rotor modulation)
            rotor_freq = self.rng.uniform(50, 200)
            sideband = 0.3 * (np.exp(2j * np.pi * (fc + rotor_freq) * t) +
                              np.exp(2j * np.pi * (fc - rotor_freq) * t))
            # Add harmonics
            harmonics = 0.2 * np.exp(2j * np.pi * fc * 2 * t)
            # Noise floor
            noise = (self.rng.standard_normal(n_samples) +
                     1j * self.rng.standard_normal(n_samples)) * 0.5
            iq = carrier + sideband + harmonics + noise
        else:
            # Pure background noise
            iq = (self.rng.standard_normal(n_samples) +
                  1j * self.rng.standard_normal(n_samples)).astype(np.complex64)

        # Convert to spectrogram
        spec = iq_to_spectrogram(iq, n_fft=self.n_fft, target_size=256)
        return spec, drone_active


class HDF5ReplaySource:
    """
    Replays holdout spectrograms from the IRIS HDF5 file.
    Simulates a real-time stream by playing back pre-recorded spectrograms.

    Alternates between drone and matched-BG samples so the demo shows
    detection firing on real drones and not firing on real backgrounds.
    """

    def __init__(self, h5_path: str = HDF5_PATH, matched_path: str = MATCHED_BG_PATH):
        import h5py

        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f"HDF5 not found: {h5_path}\n"
                f"Run scripts/pull_from_modal.py to download, or use --mode synthetic"
            )

        # Load all holdout drones into memory (small enough)
        self.drone_specs = []
        self.drone_types = []
        with h5py.File(h5_path, "r") as f:
            if "holdout" not in f:
                raise ValueError(f"No 'holdout' split in {h5_path}")
            holdout = f["holdout"]
            for tname in sorted(holdout.keys()):
                item = holdout[tname]
                if isinstance(item, h5py.Dataset) and len(item.shape) >= 3:
                    n = item.shape[0] if len(item.shape) == 4 else 1
                    for i in range(min(n, 20)):  # cap at 20 per type
                        if len(item.shape) == 4:
                            sample = item[i]
                        else:
                            sample = item[:]
                        x = self._prep(sample)
                        self.drone_specs.append(x)
                        self.drone_types.append(tname)

        # Load matched BGs
        self.bg_specs = []
        if os.path.exists(matched_path):
            with h5py.File(matched_path, "r") as mf:
                key = "holdout_matched_bg"
                if key in mf:
                    grp = mf[key]
                    keys = sorted(list(grp.keys()),
                                  key=lambda x: int(x) if x.isdigit() else 0)
                    for k in keys[:200]:
                        sample = grp[k][:]
                        x = self._prep(sample)
                        self.bg_specs.append(x)

        self.rng = np.random.default_rng(42)
        self.step = 0

        print(f"  [info] loaded {len(self.drone_specs)} drone samples "
              f"({len(set(self.drone_types))} types) + {len(self.bg_specs)} matched BGs")

    @staticmethod
    def _prep(sample: np.ndarray) -> torch.Tensor:
        if sample.shape[0] == 3:
            x = sample[:2].copy()
        elif sample.shape[0] == 2:
            x = sample.copy()
        else:
            x = sample[:2].copy()
        x = x.astype(np.float32)
        for c in range(x.shape[0]):
            ch = x[c]
            ch_std = ch.std()
            if ch_std > 1e-6:
                x[c] = (ch - ch.mean()) / ch_std
            else:
                x[c] = ch - ch.mean()
        return torch.from_numpy(x)

    def next_spectrogram(self) -> tuple:
        """Return next (spectrogram, is_drone)."""
        self.step += 1
        # Alternate: 60% drone, 40% BG (or all BGs if no drones)
        if len(self.drone_specs) > 0 and (len(self.bg_specs) == 0 or self.rng.random() < 0.6):
            idx = self.rng.integers(0, len(self.drone_specs))
            return self.drone_specs[idx], True
        else:
            idx = self.rng.integers(0, len(self.bg_specs))
            return self.bg_specs[idx], False


class IQFileSource:
    """
    Plays back a recorded I/Q file at real-time rate.

    Supports:
      - .cf32 / .complex64  — raw complex float32
      - .sigmf-data         — SigMF recording (ignores metadata for simplicity)
      - .npy                — numpy array of complex samples
    """

    def __init__(self, file_path: str, sample_rate: float = 20e6, n_fft: int = 1024):
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.n_fft = n_fft

        # Load I/Q
        ext = Path(file_path).suffix.lower()
        if ext in [".cf32", ".complex64"]:
            self.iq = np.fromfile(file_path, dtype=np.complex64)
        elif ext == ".npy":
            self.iq = np.load(file_path)
            if not np.iscomplexobj(self.iq):
                if self.iq.ndim == 2 and self.iq.shape[1] == 2:
                    self.iq = self.iq[:, 0] + 1j * self.iq[:, 1]
                else:
                    raise ValueError("npy must be complex or (N, 2) real")
        elif ext == ".sigmf-data":
            self.iq = np.fromfile(file_path, dtype=np.complex64)
        else:
            # Try complex64 as fallback
            print(f"  [warn] unknown extension {ext}, trying complex64...")
            self.iq = np.fromfile(file_path, dtype=np.complex64)

        print(f"  [info] loaded {len(self.iq)} samples ({len(self.iq)/sample_rate:.2f}s at {sample_rate/1e6:.0f} MHz)")
        self.pos = 0
        self.samples_per_window = n_fft * 4

    def next_spectrogram(self) -> tuple:
        if self.pos + self.samples_per_window > len(self.iq):
            # Loop
            self.pos = 0

        iq_chunk = self.iq[self.pos:self.pos + self.samples_per_window]
        self.pos += self.samples_per_window

        spec = iq_to_spectrogram(iq_chunk, n_fft=self.n_fft, target_size=256)
        # We don't know ground truth — assume drone present most of the time
        return spec, True


# ─────────────────────────────────────────────────────────────────────────────
# Live display
# ─────────────────────────────────────────────────────────────────────────────


def run_live_demo(source, detector: IRISDetector, fps: int = 8, no_display: bool = False,
                  intent_classifier=None):
    """
    Run the live demo with matplotlib animation.

    Args:
        source: instance of SyntheticSource / HDF5ReplaySource / IQFileSource
        detector: IRISDetector instance
        fps: target spectrograms per second
        no_display: if True, just print to console (for headless testing)
        intent_classifier: optional IntentClassifier instance for intent display
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import Normalize

    # State
    waterfall_history = deque(maxlen=WATERFALL_HISTORY)
    distance_history = deque(maxlen=WATERFALL_HISTORY * 2)
    truth_history = deque(maxlen=WATERFALL_HISTORY * 2)
    alert_until = 0.0
    stats = {"detections": 0, "misses": 0, "false_alarms": 0, "true_negatives": 0, "total": 0}

    if no_display:
        # Console-only mode
        print("\n" + "=" * 60)
        print("IRIS Live Demo — Console Mode")
        print("=" * 60)
        print(f"  Threshold: {detector.threshold:.2f}")
        print(f"  FPS:       {fps}")
        print(f"  Source:    {source.__class__.__name__}")
        print("-" * 60)

        for i in range(50):  # 50 iterations
            spec, is_drone = source.next_spectrogram()
            result = detector.detect(spec)

            stats["total"] += 1
            detected = result["verdict"] == "DRONE"
            if is_drone and detected:
                stats["detections"] += 1
            elif is_drone and not detected:
                stats["misses"] += 1
            elif not is_drone and detected:
                stats["false_alarms"] += 1
            else:
                stats["true_negatives"] += 1

            marker = "🚁" if is_drone else "  "
            alert = "⚠ DRONE" if detected else "  BG   "
            intent_str = ""
            if intent_classifier is not None and detected:
                intent_result = intent_classifier.classify(spec)
                intent_str = f" | INTENT: {intent_result['intent']} ({intent_result['confidence']:.2f})"
            print(f"  [{i:3d}] {marker} {alert} | mahal={result['mahal_dist']:6.2f} "
                  f"thresh={result['threshold']:.2f} conf={result['confidence']:.2f}{intent_str}")

            time.sleep(1.0 / fps)

        print("-" * 60)
        print(f"  Total: {stats['total']}")
        print(f"  Detections (correct):    {stats['detections']}")
        print(f"  Misses (drone, no fire): {stats['misses']}")
        print(f"  False alarms:            {stats['false_alarms']}")
        print(f"  True negatives:          {stats['true_negatives']}")
        if stats['detections'] + stats['misses'] > 0:
            tpr = stats['detections'] / (stats['detections'] + stats['misses'])
            print(f"  TPR: {tpr:.3f}")
        if stats['false_alarms'] + stats['true_negatives'] > 0:
            fpr = stats['false_alarms'] / (stats['false_alarms'] + stats['true_negatives'])
            print(f"  FPR: {fpr:.3f}")
        return

    # Matplotlib live display
    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8),
                                         gridspec_kw={"height_ratios": [3, 1, 0.3]})
    fig.suptitle("IRIS v11 — Real-Time Drone Detection", fontsize=14, color="white")
    fig.canvas.manager.set_window_title("IRIS Live Demo")

    # Waterfall axis
    waterfall_img = np.zeros((256, WATERFALL_HISTORY))
    im = ax1.imshow(waterfall_img, aspect="auto", cmap="viridis",
                     norm=Normalize(vmin=-3, vmax=3), origin="lower",
                     extent=[0, WATERFALL_HISTORY, 0, 256])
    ax1.set_title("RF Spectrogram Waterfall (most recent → right)", color="white")
    ax1.set_xlabel("Time (frames)")
    ax1.set_ylabel("Frequency bin")
    ax1.tick_params(colors="white")

    # Distance axis
    dist_line, = ax2.plot([], [], color="cyan", linewidth=1.5, label="Mahalanobis distance")
    thresh_line = ax2.axhline(y=detector.threshold, color="red", linestyle="--",
                               label=f"Threshold ({detector.threshold:.1f})")
    fill_above = ax2.fill_between([], [], color="red", alpha=0.2)
    fill_below = ax2.fill_between([], [], color="green", alpha=0.2)
    ax2.set_xlim(0, WATERFALL_HISTORY * 2)
    ax2.set_ylim(0, max(detector.threshold * 2, 50))
    ax2.set_title("Mahalanobis Distance → Drone Cluster", color="white")
    ax2.set_xlabel("Time (frames)")
    ax2.set_ylabel("Distance")
    ax2.legend(loc="upper right", facecolor="black", edgecolor="white", labelcolor="white")
    ax2.tick_params(colors="white")

    # Alert banner axis (no axes, just text)
    ax3.axis("off")
    alert_text = ax3.text(0.5, 0.5, "", transform=ax3.transAxes,
                           fontsize=24, ha="center", va="center", color="white",
                           fontweight="bold")
    stats_text = ax3.text(0.02, 0.5, "", transform=ax3.transAxes,
                           fontsize=10, ha="left", va="center", color="white")

    plt.tight_layout()

    def update(frame):
        nonlocal alert_until

        # Get next spectrogram
        spec, is_drone = source.next_spectrogram()

        # Detect
        t0 = time.time()
        result = detector.detect(spec)
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000

        # Update stats
        stats["total"] += 1
        detected = result["verdict"] == "DRONE"
        if is_drone and detected:
            stats["detections"] += 1
            alert_until = time.time() + ALERT_HOLD_SECONDS
        elif is_drone and not detected:
            stats["misses"] += 1
        elif not is_drone and detected:
            stats["false_alarms"] += 1
            alert_until = time.time() + ALERT_HOLD_SECONDS
        else:
            stats["true_negatives"] += 1

        # Update waterfall (push new column)
        spec_np = spec[0].numpy() if isinstance(spec, torch.Tensor) else spec[0]
        waterfall_history.append(spec_np)
        waterfall_img = np.stack(list(waterfall_history), axis=1)
        if waterfall_img.shape[1] < WATERFALL_HISTORY:
            pad = np.zeros((256, WATERFALL_HISTORY - waterfall_img.shape[1]))
            waterfall_img = np.concatenate([pad, waterfall_img], axis=1)
        im.set_data(waterfall_img)

        # Update distance plot
        distance_history.append(result["mahal_dist"])
        truth_history.append(int(is_drone))
        x = list(range(len(distance_history)))
        dist_line.set_data(x, list(distance_history))
        ax2.set_xlim(0, max(WATERFALL_HISTORY * 2, len(distance_history)))

        # Color the area based on truth (green if BG, red if drone)
        # Simple: just show threshold line
        artists = [im, dist_line, thresh_line]

        # Update alert banner
        is_alert = time.time() < alert_until
        if is_alert:
            if detected:
                # Get intent if classifier available
                intent_text = ""
                if intent_classifier is not None:
                    try:
                        intent_result = intent_classifier.classify(spec)
                        intent_text = f" — INTENT: {intent_result['intent']}"
                    except Exception:
                        pass
                alert_text.set_text(f"⚠  DRONE DETECTED{intent_text}  ⚠")
                # Color by intent
                if intent_classifier is not None and "intent_result" in dir():
                    color_map = {"SURVEILLANCE": "#ffaa00", "TRANSIT": "#ff8800", "ATTACK": "#ff0000"}
                    alert_text.set_color(color_map.get(intent_result.get("intent", ""), "#ff4444"))
                else:
                    alert_text.set_color("#ff4444")
            else:
                alert_text.set_text("")
        else:
            alert_text.set_text("Monitoring...")
            alert_text.set_color("#44ff44")

        # Update stats text
        tpr = stats["detections"] / max(1, stats["detections"] + stats["misses"])
        fpr = stats["false_alarms"] / max(1, stats["false_alarms"] + stats["true_negatives"])
        stats_text.set_text(
            f"Frames: {stats['total']}\n"
            f"Latency: {latency_ms:.1f} ms\n"
            f"TPR: {tpr:.2f}\n"
            f"FPR: {fpr:.2f}\n"
            f"Dist: {result['mahal_dist']:.1f} / {result['threshold']:.1f}"
        )

        return artists + [alert_text, stats_text]

    # Run animation
    interval_ms = int(1000 / fps)
    ani = animation.FuncAnimation(
        fig, update, interval=interval_ms, blit=False, cache_frame_data=False
    )

    print("\n" + "=" * 60)
    print("IRIS Live Demo — Close window to exit")
    print("=" * 60)
    print(f"  Source:    {source.__class__.__name__}")
    print(f"  FPS:       {fps}")
    print(f"  Threshold: {detector.threshold:.2f}")
    print(f"  Device:    {detector.device}")
    print("-" * 60)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n  [info] interrupted by user")

    # Print final stats
    print("\n" + "=" * 60)
    print("Final Stats")
    print("=" * 60)
    print(f"  Total frames:        {stats['total']}")
    print(f"  Correct detections:  {stats['detections']}")
    print(f"  Misses:              {stats['misses']}")
    print(f"  False alarms:        {stats['false_alarms']}")
    print(f"  True negatives:      {stats['true_negatives']}")
    if stats['detections'] + stats['misses'] > 0:
        tpr = stats['detections'] / (stats['detections'] + stats['misses'])
        print(f"  TPR (recall):        {tpr:.3f}")
    if stats['false_alarms'] + stats['true_negatives'] > 0:
        fpr = stats['false_alarms'] / (stats['false_alarms'] + stats['true_negatives'])
        print(f"  FPR:                 {fpr:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="IRIS Live Demo")
    parser.add_argument(
        "--mode", choices=["synthetic", "hdf5", "iq_file"], default="synthetic",
        help="Input source mode (default: synthetic — no files needed)"
    )
    parser.add_argument("--file", help="I/Q file path (for iq_file mode)")
    parser.add_argument("--sample-rate", type=float, default=20e6, help="I/Q sample rate (Hz)")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Target frames per second")
    parser.add_argument("--no-display", action="store_true", help="Console-only mode (no GUI)")
    parser.add_argument("--threshold", type=float, default=None, help="Override Mahalanobis threshold")
    parser.add_argument("--no-intent", action="store_true",
                        help="Disable intent classifier even if available")
    args = parser.parse_args()

    print("=" * 60)
    print("IRIS v11 — Live Drone Detection Demo")
    print("=" * 60)

    # Load detector
    centroid_path = CENTROID_PATH if os.path.exists(CENTROID_PATH) else None
    print(f"\n  [info] loading IRIS detector...")
    print(f"    checkpoint: {CHECKPOINT_PATH}")
    print(f"    centroid:   {centroid_path or '(will use default threshold)'}")

    detector = IRISDetector(
        checkpoint_path=CHECKPOINT_PATH,
        centroid_path=centroid_path,
        threshold=args.threshold,
    )
    print(f"  [ok] encoder loaded: {sum(p.numel() for p in detector.encoder.parameters()):,} params")
    print(f"  [ok] device: {detector.device}")
    print(f"  [ok] threshold: {detector.threshold:.2f} ({detector.threshold_source})")

    # Load intent classifier if available
    intent_classifier = None
    if INTENT_AVAILABLE and not args.no_intent and os.path.exists(INTENT_HEAD_PATH):
        print(f"\n  [info] loading intent classifier...")
        try:
            intent_classifier = IntentClassifier(
                encoder_checkpoint=CHECKPOINT_PATH,
                intent_head_checkpoint=INTENT_HEAD_PATH,
            )
            print(f"  [ok] intent classifier loaded")
        except Exception as e:
            print(f"  [warn] intent classifier failed to load: {e}")
            intent_classifier = None
    elif not INTENT_AVAILABLE:
        print(f"\n  [info] intent_head module not available (run scripts/train_intent.py first)")
    elif args.no_intent:
        print(f"\n  [info] intent classifier disabled (--no-intent)")
    else:
        print(f"\n  [info] intent head checkpoint not found at {INTENT_HEAD_PATH}")
        print(f"         run scripts/train_intent.py to enable intent classification")

    # Create input source
    print(f"\n  [info] creating input source (mode={args.mode})...")
    if args.mode == "synthetic":
        source = SyntheticSource()
    elif args.mode == "hdf5":
        source = HDF5ReplaySource()
    elif args.mode == "iq_file":
        if not args.file:
            print("  [error] --mode iq_file requires --file path")
            sys.exit(1)
        source = IQFileSource(args.file, sample_rate=args.sample_rate)
    else:
        print(f"  [error] unknown mode: {args.mode}")
        sys.exit(1)

    print(f"  [ok] source ready: {source.__class__.__name__}")

    # Run
    run_live_demo(source, detector, fps=args.fps, no_display=args.no_display,
                  intent_classifier=intent_classifier)


if __name__ == "__main__":
    main()
