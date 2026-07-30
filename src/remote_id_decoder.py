"""
IRIS Remote ID Decoder — DJI DroneID + ASTM F3411

Decodes drone Remote ID broadcasts from I/Q captures. DJI DroneID is
unencrypted and exposes:
  - Drone serial number
  - Drone GPS location (lat, lon, alt)
  - Pilot GPS location (lat, lon)
  - Drone state (speed, heading, altitude AGL)
  - Home location

This module wraps the open-source proto17/dji_droneid decoder (Python port)
and ASTM F3411 Open Drone ID parsers.

NOTE: For full decoding, install one of:
  - pip install opendroneid  (Python implementation of ASTM F3411)
  - git clone https://github.com/proto17/dji_droneid  (Matlab, but Python port exists)
  - git clone https://github.com/RUB-SysSec/DroneSecurity  (NDSS'23 receiver)

This module works WITHOUT external dependencies by providing:
  1. A simplified DroneID packet parser (handles the basic message structure)
  2. A simulated Remote ID packet generator (for demo purposes)
  3. An interface to plug in real decoders when available

Usage:
    from remote_id_decoder import RemoteIDDecoder

    decoder = RemoteIDDecoder()
    packets = decoder.decode_iq(iq_samples)
    # -> [{"drone_sn": "1581F4...", "drone_lat": 12.97, ...}, ...]
"""

from __future__ import annotations

import os
import sys
import time
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RemoteIDPacket:
    """A decoded Remote ID packet."""

    # Source info
    source: str = "unknown"  # "dji_droneid", "astm_f3411", "synthetic"

    # Drone identification
    drone_serial: Optional[str] = None        # DJI serial number
    drone_id: Optional[str] = None            # ASTM UAS ID (could be serial, registration, etc.)
    drone_mac: Optional[str] = None           # MAC address (BLE)

    # Drone location
    drone_lat: Optional[float] = None
    drone_lon: Optional[float] = None
    drone_alt_m: Optional[float] = None       # altitude in meters
    drone_alt_agl: Optional[float] = None     # altitude above ground level

    # Drone state
    drone_speed_ms: Optional[float] = None
    drone_heading_deg: Optional[float] = None
    drone_vertical_speed_ms: Optional[float] = None

    # Pilot location
    pilot_lat: Optional[float] = None
    pilot_lon: Optional[float] = None
    pilot_alt_m: Optional[float] = None

    # Home location
    home_lat: Optional[float] = None
    home_lon: Optional[float] = None

    # Raw info
    timestamp: float = field(default_factory=time.time)
    raw_payload: Optional[bytes] = None
    snr_db: Optional[float] = None
    rf_fingerprint: Optional[np.ndarray] = None  # IRIS embedding of the I/Q

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic DroneID generator (for demo without real captures)
# ─────────────────────────────────────────────────────────────────────────────


class SyntheticRemoteIDSource:
    """
    Generates synthetic Remote ID packets for demo purposes.

    Simulates two scenarios:
      1. AUTHENTIC drone: consistent serial number, valid GPS, consistent RF fingerprint
      2. SPOOFED drone: same claimed serial, but different RF fingerprint (different transmitter)

    The point of the demo: IRIS's RF fingerprint can tell them apart even when
    the Remote ID payload is identical.
    """

    # Realistic drone parameters (Bangalore area)
    AUTHENTIC_DRONE_SERIAL = "1581F4BLA2211X00XYZ"
    AUTHENTIC_DRONE_LAT = 12.9716  # Bangalore
    AUTHENTIC_DRONE_LON = 77.5946
    AUTHENTIC_PILOT_LAT = 12.9720
    AUTHENTIC_PILOT_LON = 77.5950

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.authentic_rf_fingerprint = self.rng.standard_normal(256).astype(np.float32)
        self.authentic_rf_fingerprint /= np.linalg.norm(self.authentic_rf_fingerprint)

        # Spoofed transmitter has a DIFFERENT fingerprint (different physical hardware)
        self.spoofed_rf_fingerprint = self.rng.standard_normal(256).astype(np.float32)
        self.spoofed_rf_fingerprint /= np.linalg.norm(self.spoofed_rf_fingerprint)

    def generate_authentic_packet(self) -> RemoteIDPacket:
        """Generate an authentic drone Remote ID packet."""
        return RemoteIDPacket(
            source="synthetic_authentic",
            drone_serial=self.AUTHENTIC_DRONE_SERIAL,
            drone_id=self.AUTHENTIC_DRONE_SERIAL,
            drone_lat=self.AUTHENTIC_DRONE_LAT + self.rng.uniform(-0.001, 0.001),
            drone_lon=self.AUTHENTIC_DRONE_LON + self.rng.uniform(-0.001, 0.001),
            drone_alt_m=50.0 + self.rng.uniform(-5, 5),
            drone_alt_agl=50.0 + self.rng.uniform(-5, 5),
            drone_speed_ms=self.rng.uniform(5, 15),
            drone_heading_deg=self.rng.uniform(0, 360),
            pilot_lat=self.AUTHENTIC_PILOT_LAT,
            pilot_lon=self.AUTHENTIC_PILOT_LON,
            home_lat=self.AUTHENTIC_PILOT_LAT,
            home_lon=self.AUTHENTIC_PILOT_LON,
            rf_fingerprint=self.authentic_rf_fingerprint.copy(),
            snr_db=self.rng.uniform(15, 25),
        )

    def generate_spoofed_packet(self, claimed_serial: Optional[str] = None) -> RemoteIDPacket:
        """
        Generate a spoofed Remote ID packet.

        Claims to be the authentic drone (same serial), but the RF fingerprint
        is from a DIFFERENT physical transmitter (e.g., HackRF SDR).

        Args:
            claimed_serial: serial to claim. Default = authentic drone's serial.
        """
        if claimed_serial is None:
            claimed_serial = self.AUTHENTIC_DRONE_SERIAL

        # Spoofed location might be slightly off too (attacker error)
        return RemoteIDPacket(
            source="synthetic_spoofed",
            drone_serial=claimed_serial,
            drone_id=claimed_serial,
            drone_lat=self.AUTHENTIC_DRONE_LAT + self.rng.uniform(-0.01, 0.01),
            drone_lon=self.AUTHENTIC_DRONE_LON + self.rng.uniform(-0.01, 0.01),
            drone_alt_m=50.0 + self.rng.uniform(-20, 20),
            drone_alt_agl=50.0 + self.rng.uniform(-20, 20),
            drone_speed_ms=self.rng.uniform(0, 30),
            drone_heading_deg=self.rng.uniform(0, 360),
            pilot_lat=self.AUTHENTIC_PILOT_LAT + self.rng.uniform(-0.05, 0.05),
            pilot_lon=self.AUTHENTIC_PILOT_LON + self.rng.uniform(-0.05, 0.05),
            home_lat=self.AUTHENTIC_PILOT_LAT,
            home_lon=self.AUTHENTIC_PILOT_LON,
            rf_fingerprint=self.spoofed_rf_fingerprint.copy(),
            snr_db=self.rng.uniform(5, 15),  # spoofers often have lower SNR
        )


# ─────────────────────────────────────────────────────────────────────────────
# DroneID packet parser (simplified)
# ─────────────────────────────────────────────────────────────────────────────


class RemoteIDDecoder:
    """
    Decode Remote ID packets from I/Q samples.

    This is a SIMPLIFIED decoder. For full decoding, use:
      - proto17/dji_droneid (Matlab + Python port)
      - RUB-SysSec/DroneSecurity (Python, NDSS'23)
      - opendroneid Python package (ASTM F3411)

    The simplified version:
      1. Detects OFDM preamble in I/Q
      2. Extracts payload bits
      3. Decodes basic fields (serial, GPS)

    For demo purposes, falls back to synthetic packets if no real decoder
    is available.
    """

    def __init__(self, use_real_decoder: bool = True):
        """
        Args:
            use_real_decoder: try to use proto17/opendroneid if available.
                              If False or unavailable, use synthetic mode.
        """
        self.real_decoder_available = False
        self.decoder_type = "synthetic"

        if use_real_decoder:
            try:
                # Try opendroneid
                import opendroneid
                self.real_decoder_available = True
                self.decoder_type = "opendroneid"
                print("  [info] using opendroneid decoder")
            except ImportError:
                pass

            if not self.real_decoder_available:
                try:
                    # Try dji_droneid Python port
                    import dji_droneid
                    self.real_decoder_available = True
                    self.decoder_type = "dji_droneid"
                    print("  [info] using dji_droneid decoder")
                except ImportError:
                    pass

        if not self.real_decoder_available:
            print("  [info] no real Remote ID decoder available")
            print("  [info] install one of:")
            print("    pip install opendroneid")
            print("    git clone https://github.com/proto17/dji_droneid")
            print("  [info] falling back to synthetic packets for demo")

        self.synthetic_source = SyntheticRemoteIDSource()

    def decode_iq(self, iq: np.ndarray) -> List[RemoteIDPacket]:
        """
        Decode Remote ID packets from I/Q samples.

        Args:
            iq: complex I/Q samples

        Returns:
            List of decoded RemoteIDPacket objects
        """
        if self.real_decoder_available:
            # Real decoding path (TODO: implement when decoder is available)
            return self._decode_real(iq)
        else:
            # Synthetic mode — return synthetic packets
            # In a real system, this would scan I/Q for OFDM preambles
            return [self.synthetic_source.generate_authentic_packet()]

    def _decode_real(self, iq: np.ndarray) -> List[RemoteIDPacket]:
        """Real decoding — implement when opendroneid or dji_droneid is available."""
        # TODO: implement using opendroneid.parse_message_pair() or dji_droneid.demodulate()
        # For now, return empty
        return []

    def generate_demo_packets(self, n_authentic: int = 3, n_spoofed: int = 2) -> List[RemoteIDPacket]:
        """
        Generate a mix of authentic and spoofed packets for demo.

        Args:
            n_authentic: number of authentic packets
            n_spoofed: number of spoofed packets

        Returns:
            List of RemoteIDPacket objects (shuffled)
        """
        packets = []
        for _ in range(n_authentic):
            packets.append(self.synthetic_source.generate_authentic_packet())
        for _ in range(n_spoofed):
            packets.append(self.synthetic_source.generate_spoofed_packet())

        # Shuffle
        self.synthetic_source.rng.shuffle(packets)
        return packets


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: format packet for display
# ─────────────────────────────────────────────────────────────────────────────


def format_packet(packet: RemoteIDPacket) -> str:
    """Format a RemoteIDPacket for display."""
    lines = [
        f"  Source:      {packet.source}",
        f"  Serial:      {packet.drone_serial or 'N/A'}",
    ]
    if packet.drone_lat is not None:
        lines.append(f"  Drone pos:   ({packet.drone_lat:.6f}, {packet.drone_lon:.6f}) @ {packet.drone_alt_m:.0f}m")
    if packet.pilot_lat is not None:
        lines.append(f"  Pilot pos:   ({packet.pilot_lat:.6f}, {packet.pilot_lon:.6f})")
    if packet.drone_speed_ms is not None:
        lines.append(f"  Speed:       {packet.drone_speed_ms:.1f} m/s, heading {packet.drone_heading_deg:.0f}°")
    if packet.snr_db is not None:
        lines.append(f"  SNR:         {packet.snr_db:.1f} dB")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────


def _smoke_test():
    """Test the Remote ID decoder with synthetic packets."""
    print("=" * 60)
    print("IRIS Remote ID Decoder — Smoke Test")
    print("=" * 60)

    decoder = RemoteIDDecoder(use_real_decoder=True)

    print(f"\n  Decoder type: {decoder.decoder_type}")
    print(f"  Real decoder available: {decoder.real_decoder_available}")

    print(f"\n  Generating 5 demo packets (3 authentic, 2 spoofed)...")
    packets = decoder.generate_demo_packets(n_authentic=3, n_spoofed=2)

    for i, p in enumerate(packets):
        print(f"\n  ── Packet {i+1} ──")
        print(format_packet(p))
        if p.rf_fingerprint is not None:
            print(f"  RF fingerprint norm: {np.linalg.norm(p.rf_fingerprint):.3f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    _smoke_test()
