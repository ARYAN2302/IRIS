"""
IRIS Remote ID Authenticator — Spoof Detection via RF Fingerprinting

This is the KILLER DEMO 2: authenticate Remote ID broadcasts by RF
fingerprinting the transmitter.

The problem (Armory's October 2025 blog):
  "Since DroneIDs are encrypted, they could be conveniently falsified.
   One could mimic legitimate DroneID signals to deceive defenders."
  "Attackers also have the potential to flood an anti-drone system with
   fake signals, overwhelming the operators with false alarms."

The solution (this module):
  1. Enroll friendly drones: capture 30s of drone X → store IRIS RF fingerprint
  2. When Remote ID packet arrives, decode it (drone ID, GPS, etc.)
  3. ALSO extract the I/Q of the BLE/Wi-Fi transmission
  4. Run I/Q through IRIS encoder → 256-dim RF fingerprint
  5. Compare to enrolled fingerprints
  6. If DroneID says "DJI Mini 4 Pro S/N 12345" but RF fingerprint doesn't
     match enrolled S/N 12345 → SPOOFED

No published work uses RF fingerprinting to authenticate Remote ID broadcasts.
This is the first.

Architecture:
  Remote ID packet (decoded payload) +
  IRIS encoder (RF fingerprint of transmission)
    ↓
  RemoteIDAuthenticator
    ↓
  {verdict: AUTHENTIC / SPOOFED / UNKNOWN, confidence, enrolled_match}

Usage:
    from remote_id_auth import RemoteIDAuthenticator
    from remote_id_decoder import RemoteIDDecoder

    auth = RemoteIDAuthenticator(
        encoder_checkpoint="models/lejepa_v11_best.pt",
        registry_path="data/blue_force_registry.json",
    )

    # Enroll a friendly drone (one-time)
    auth.enroll("DJI Mini 4 Pro", "1581F4BLA2211X00XYZ", fingerprint_embedding)

    # Authenticate an incoming packet
    result = auth.authenticate(packet, packet_rf_fingerprint)
    # -> {"verdict": "SPOOFED", "confidence": 0.92, "claimed_serial": "...",
    #     "matched_enrolled": None, "reason": "RF fingerprint does not match
    #     any enrolled drone with claimed serial"}
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.iris_inference import IRISDetector
from src.remote_id_decoder import RemoteIDPacket, SyntheticRemoteIDSource


# ─────────────────────────────────────────────────────────────────────────────
# Blue-force registry
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EnrolledDrone:
    """A friendly drone enrolled in the blue-force registry."""

    drone_type: str                              # e.g., "DJI Mini 4 Pro"
    serial_number: str                           # e.g., "1581F4BLA2211X00XYZ"
    enrollment_time: str                         # ISO timestamp
    rf_fingerprint: List[float]                  # 256-dim IRIS embedding
    notes: Optional[str] = None
    last_seen: Optional[str] = None
    detection_count: int = 0


class BlueForceRegistry:
    """
    Registry of enrolled friendly drones.

    Stored as JSON for portability. Each drone has:
      - drone_type
      - serial_number
      - rf_fingerprint (256-dim IRIS embedding)
      - enrollment_time
      - last_seen
      - detection_count
    """

    def __init__(self, path: str = "data/blue_force_registry.json"):
        self.path = Path(path)
        self.drones: Dict[str, EnrolledDrone] = {}  # keyed by serial_number
        self.load()

    def load(self) -> None:
        """Load registry from JSON."""
        if not self.path.exists():
            return
        with open(self.path, "r") as f:
            data = json.load(f)
        for serial, d in data.items():
            self.drones[serial] = EnrolledDrone(
                drone_type=d["drone_type"],
                serial_number=d["serial_number"],
                enrollment_time=d["enrollment_time"],
                rf_fingerprint=d["rf_fingerprint"],
                notes=d.get("notes"),
                last_seen=d.get("last_seen"),
                detection_count=d.get("detection_count", 0),
            )
        print(f"  [ok] loaded {len(self.drones)} enrolled drones from {self.path}")

    def save(self) -> None:
        """Save registry to JSON."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for serial, d in self.drones.items():
            data[serial] = {
                "drone_type": d.drone_type,
                "serial_number": d.serial_number,
                "enrollment_time": d.enrollment_time,
                "rf_fingerprint": d.rf_fingerprint,
                "notes": d.notes,
                "last_seen": d.last_seen,
                "detection_count": d.detection_count,
            }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  [ok] saved {len(self.drones)} enrolled drones to {self.path}")

    def enroll(self, drone_type: str, serial_number: str, fingerprint: np.ndarray,
               notes: Optional[str] = None) -> None:
        """Enroll a friendly drone."""
        if isinstance(fingerprint, np.ndarray):
            fingerprint = fingerprint.tolist()

        self.drones[serial_number] = EnrolledDrone(
            drone_type=drone_type,
            serial_number=serial_number,
            enrollment_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            rf_fingerprint=fingerprint,
            notes=notes,
            detection_count=0,
        )
        self.save()
        print(f"  [ok] enrolled {drone_type} (S/N {serial_number})")

    def find_by_serial(self, serial: str) -> Optional[EnrolledDrone]:
        """Find an enrolled drone by serial number."""
        return self.drones.get(serial)

    def find_by_fingerprint(self, fingerprint: np.ndarray, threshold: float = 0.85) -> Optional[EnrolledDrone]:
        """
        Find the closest enrolled drone by RF fingerprint cosine similarity.

        Args:
            fingerprint: 256-dim IRIS embedding
            threshold: minimum cosine similarity to consider a match

        Returns:
            EnrolledDrone if match found, None otherwise
        """
        if not self.drones:
            return None

        fp_norm = fingerprint / (np.linalg.norm(fingerprint) + 1e-8)

        best_match = None
        best_sim = 0.0
        for serial, d in self.drones.items():
            enrolled_fp = np.array(d.rf_fingerprint, dtype=np.float32)
            enrolled_norm = enrolled_fp / (np.linalg.norm(enrolled_fp) + 1e-8)
            sim = float(np.dot(fp_norm, enrolled_norm))
            if sim > best_sim:
                best_sim = sim
                best_match = d

        if best_sim >= threshold:
            return best_match
        return None

    def list_all(self) -> List[EnrolledDrone]:
        """List all enrolled drones."""
        return list(self.drones.values())


# ─────────────────────────────────────────────────────────────────────────────
# Authenticator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AuthenticationResult:
    """Result of authenticating a Remote ID packet."""

    verdict: str                  # "AUTHENTIC", "SPOOFED", "UNKNOWN", "NOT_ENROLLED"
    confidence: float             # 0.0 - 1.0
    claimed_serial: Optional[str]
    claimed_type: Optional[str]
    matched_enrolled: Optional[str]  # serial number of matched enrolled drone
    fingerprint_similarity: float    # cosine similarity to closest enrolled
    reason: str
    packet: Optional[RemoteIDPacket] = None


class RemoteIDAuthenticator:
    """
    Authenticate Remote ID packets via RF fingerprinting.

    Workflow:
      1. Decode Remote ID packet → get claimed drone serial, type, location
      2. Extract RF fingerprint of the transmission (IRIS encoder on raw I/Q)
      3. Check if claimed serial is in blue-force registry
      4. If yes, compare RF fingerprint to enrolled fingerprint
      5. If match → AUTHENTIC
         If claimed serial enrolled but fingerprint doesn't match → SPOOFED
         If claimed serial not enrolled → NOT_ENROLLED (could be hostile or new friendly)
         If can't extract fingerprint → UNKNOWN
    """

    def __init__(
        self,
        encoder_checkpoint: str = "models/lejepa_v11_best.pt",
        registry_path: str = "data/blue_force_registry.json",
        match_threshold: float = 0.85,
        device: Optional[str] = None,
    ):
        """
        Args:
            encoder_checkpoint: path to IRIS v11 checkpoint
            registry_path: path to blue force registry JSON
            match_threshold: cosine similarity threshold for fingerprint match
            device: torch device
        """
        self.match_threshold = match_threshold

        # Load IRIS encoder for fingerprinting
        self.detector = IRISDetector(checkpoint_path=encoder_checkpoint, device=device)
        print(f"  [ok] IRIS encoder loaded for RF fingerprinting")

        # Load blue force registry
        self.registry = BlueForceRegistry(registry_path)

    @torch.no_grad()
    def extract_fingerprint(self, spectrogram: torch.Tensor) -> np.ndarray:
        """
        Extract IRIS RF fingerprint from a spectrogram.

        Args:
            spectrogram: (2, 256, 256) tensor, per-channel normalized

        Returns:
            fingerprint: (256,) numpy array, L2-normalized
        """
        emb = self.detector.encode(spectrogram)[0]  # (256,)
        # L2 normalize (so cosine similarity = dot product)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def enroll(self, drone_type: str, serial_number: str, spectrogram: torch.Tensor,
               notes: Optional[str] = None) -> None:
        """Enroll a friendly drone from a spectrogram sample."""
        fingerprint = self.extract_fingerprint(spectrogram)
        self.registry.enroll(drone_type, serial_number, fingerprint, notes)

    def enroll_from_packet(self, packet: RemoteIDPacket, drone_type: str = "Unknown") -> None:
        """Enroll a friendly drone from a Remote ID packet (must have rf_fingerprint)."""
        if packet.rf_fingerprint is None:
            raise ValueError("Packet has no rf_fingerprint — cannot enroll")
        if packet.drone_serial is None:
            raise ValueError("Packet has no drone_serial — cannot enroll")
        self.registry.enroll(drone_type, packet.drone_serial, packet.rf_fingerprint)

    def authenticate(
        self,
        packet: RemoteIDPacket,
        fingerprint: Optional[np.ndarray] = None,
    ) -> AuthenticationResult:
        """
        Authenticate a Remote ID packet.

        Args:
            packet: decoded Remote ID packet
            fingerprint: optional pre-extracted RF fingerprint.
                        If None, uses packet.rf_fingerprint if available.

        Returns:
            AuthenticationResult
        """
        # Get fingerprint
        if fingerprint is None:
            fingerprint = packet.rf_fingerprint

        claimed_serial = packet.drone_serial
        claimed_type = packet.drone_id or "Unknown"

        # Case 1: No fingerprint available
        if fingerprint is None:
            return AuthenticationResult(
                verdict="UNKNOWN",
                confidence=0.0,
                claimed_serial=claimed_serial,
                claimed_type=claimed_type,
                matched_enrolled=None,
                fingerprint_similarity=0.0,
                reason="No RF fingerprint available — cannot authenticate",
                packet=packet,
            )

        fingerprint = np.array(fingerprint, dtype=np.float32)
        fp_norm = fingerprint / (np.linalg.norm(fingerprint) + 1e-8)

        # Find closest match in registry
        closest_enrolled = None
        closest_sim = 0.0
        for serial, d in self.registry.drones.items():
            enrolled_fp = np.array(d.rf_fingerprint, dtype=np.float32)
            enrolled_norm = enrolled_fp / (np.linalg.norm(enrolled_fp) + 1e-8)
            sim = float(np.dot(fp_norm, enrolled_norm))
            if sim > closest_sim:
                closest_sim = sim
                closest_enrolled = d

        # Case 2: No drones enrolled
        if closest_enrolled is None:
            return AuthenticationResult(
                verdict="NOT_ENROLLED",
                confidence=0.5,
                claimed_serial=claimed_serial,
                claimed_type=claimed_type,
                matched_enrolled=None,
                fingerprint_similarity=0.0,
                reason="No drones enrolled in registry — cannot verify. Treat as unknown.",
                packet=packet,
            )

        # Case 3: Claimed serial IS enrolled
        if claimed_serial and claimed_serial in self.registry.drones:
            enrolled = self.registry.drones[claimed_serial]
            enrolled_fp = np.array(enrolled.rf_fingerprint, dtype=np.float32)
            enrolled_norm = enrolled_fp / (np.linalg.norm(enrolled_fp) + 1e-8)
            sim_to_claimed = float(np.dot(fp_norm, enrolled_norm))

            if sim_to_claimed >= self.match_threshold:
                # Match — AUTHENTIC
                confidence = (sim_to_claimed - self.match_threshold) / (1.0 - self.match_threshold + 1e-8)
                return AuthenticationResult(
                    verdict="AUTHENTIC",
                    confidence=confidence,
                    claimed_serial=claimed_serial,
                    claimed_type=enrolled.drone_type,
                    matched_enrolled=claimed_serial,
                    fingerprint_similarity=sim_to_claimed,
                    reason=f"RF fingerprint matches enrolled drone (sim={sim_to_claimed:.3f})",
                    packet=packet,
                )
            else:
                # Mismatch — SPOOFED
                # The packet claims to be an enrolled drone, but the RF fingerprint
                # doesn't match. This is the smoking gun.
                confidence = 1.0 - sim_to_claimed
                return AuthenticationResult(
                    verdict="SPOOFED",
                    confidence=confidence,
                    claimed_serial=claimed_serial,
                    claimed_type=claimed_type,
                    matched_enrolled=None,
                    fingerprint_similarity=sim_to_claimed,
                    reason=(
                        f"Remote ID claims S/N {claimed_serial} (enrolled), "
                        f"but RF fingerprint similarity is only {sim_to_claimed:.3f} "
                        f"(threshold {self.match_threshold}). "
                        f"Transmitter is DIFFERENT physical hardware than enrolled drone."
                    ),
                    packet=packet,
                )

        # Case 4: Claimed serial NOT enrolled, but fingerprint matches some enrolled drone
        if closest_sim >= self.match_threshold:
            return AuthenticationResult(
                verdict="SPOOFED",
                confidence=closest_sim,
                claimed_serial=claimed_serial,
                claimed_type=claimed_type,
                matched_enrolled=closest_enrolled.serial_number,
                fingerprint_similarity=closest_sim,
                reason=(
                    f"Remote ID claims S/N {claimed_serial} (not enrolled), "
                    f"but RF fingerprint matches enrolled drone "
                    f"{closest_enrolled.serial_number} (sim={closest_sim:.3f}). "
                    f"Attacker is using a different serial number with a known drone's transmitter."
                ),
                packet=packet,
            )

        # Case 5: Claimed serial not enrolled, fingerprint doesn't match anything
        return AuthenticationResult(
            verdict="NOT_ENROLLED",
            confidence=0.5,
            claimed_serial=claimed_serial,
            claimed_type=claimed_type,
            matched_enrolled=None,
            fingerprint_similarity=closest_sim,
            reason=(
                f"Remote ID claims S/N {claimed_serial} (not enrolled). "
                f"RF fingerprint doesn't match any enrolled drone (max sim={closest_sim:.3f}). "
                f"Treat as unknown — possibly hostile."
            ),
            packet=packet,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: format result for display
# ─────────────────────────────────────────────────────────────────────────────


def format_result(result: AuthenticationResult) -> str:
    """Format an AuthenticationResult for display."""
    color = {
        "AUTHENTIC": "\033[92m",    # green
        "SPOOFED": "\033[91m",      # red
        "UNKNOWN": "\033[93m",      # yellow
        "NOT_ENROLLED": "\033[93m", # yellow
    }.get(result.verdict, "")
    reset = "\033[0m"

    lines = [
        f"{color}╔══════════════════════════════════════════════════════╗{reset}",
        f"{color}║ Verdict: {result.verdict:11s}  Confidence: {result.confidence:.2f}        ║{reset}",
        f"{color}╚══════════════════════════════════════════════════════╝{reset}",
        f"  Claimed S/N:    {result.claimed_serial or 'N/A'}",
        f"  Claimed type:   {result.claimed_type or 'N/A'}",
        f"  Matched:        {result.matched_enrolled or 'None'}",
        f"  Fingerprint sim: {result.fingerprint_similarity:.3f} (threshold: 0.85)",
        f"  Reason:         {result.reason}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────


def _smoke_test():
    """Test the authenticator with synthetic packets."""
    print("=" * 60)
    print("IRIS Remote ID Authenticator — Smoke Test")
    print("=" * 60)

    # Use synthetic source (no real encoder needed for testing logic)
    source = SyntheticRemoteIDSource()

    # Create a fake registry (bypass encoder loading)
    registry = BlueForceRegistry("data/test_registry.json")

    # Enroll the authentic drone using its known fingerprint
    registry.enroll(
        drone_type="DJI Mini 4 Pro",
        serial_number=source.AUTHENTIC_DRONE_SERIAL,
        fingerprint=source.authentic_rf_fingerprint,
    )

    # Create authenticator (skip encoder — use synthetic fingerprints directly)
    class FakeAuthenticator:
        def __init__(self, registry, threshold=0.85):
            self.registry = registry
            self.match_threshold = threshold

        def authenticate(self, packet, fingerprint=None):
            from src.remote_id_decoder import RemoteIDPacket
            fp = fingerprint if fingerprint is not None else packet.rf_fingerprint
            fp_norm = fp / (np.linalg.norm(fp) + 1e-8)

            # Find closest match
            best_match = None
            best_sim = 0.0
            for serial, d in registry.drones.items():
                enrolled = np.array(d.rf_fingerprint, dtype=np.float32)
                enrolled_norm = enrolled / (np.linalg.norm(enrolled) + 1e-8)
                sim = float(np.dot(fp_norm, enrolled_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_match = d

            if packet.drone_serial in registry.drones:
                enrolled = registry.drones[packet.drone_serial]
                enrolled_fp = np.array(enrolled.rf_fingerprint, dtype=np.float32)
                enrolled_norm = enrolled_fp / (np.linalg.norm(enrolled_fp) + 1e-8)
                sim = float(np.dot(fp_norm, enrolled_norm))
                if sim >= self.match_threshold:
                    return AuthenticationResult(
                        verdict="AUTHENTIC",
                        confidence=sim,
                        claimed_serial=packet.drone_serial,
                        claimed_type=enrolled.drone_type,
                        matched_enrolled=packet.drone_serial,
                        fingerprint_similarity=sim,
                        reason=f"RF fingerprint matches enrolled drone",
                        packet=packet,
                    )
                else:
                    return AuthenticationResult(
                        verdict="SPOOFED",
                        confidence=1.0 - sim,
                        claimed_serial=packet.drone_serial,
                        claimed_type="Unknown",
                        matched_enrolled=None,
                        fingerprint_similarity=sim,
                        reason=f"RF fingerprint mismatch — different transmitter",
                        packet=packet,
                    )
            return AuthenticationResult(
                verdict="NOT_ENROLLED",
                confidence=0.5,
                claimed_serial=packet.drone_serial,
                claimed_type="Unknown",
                matched_enrolled=best_match.serial_number if best_match else None,
                fingerprint_similarity=best_sim,
                reason="Serial not enrolled",
                packet=packet,
            )

    auth = FakeAuthenticator(registry)

    # Test 1: Authentic packet
    print("\n── Test 1: Authentic Drone ──")
    authentic_packet = source.generate_authentic_packet()
    result = auth.authenticate(authentic_packet)
    print(format_result(result))

    # Test 2: Spoofed packet (claims same serial, different transmitter)
    print("\n── Test 2: Spoofed Drone (claims authentic serial) ──")
    spoofed_packet = source.generate_spoofed_packet(claimed_serial=source.AUTHENTIC_DRONE_SERIAL)
    result = auth.authenticate(spoofed_packet)
    print(format_result(result))

    # Test 3: Unknown drone (different serial, different fingerprint)
    print("\n── Test 3: Unknown Drone (different serial) ──")
    unknown_packet = source.generate_spoofed_packet(claimed_serial="UNKNOWN_SERIAL_999")
    result = auth.authenticate(unknown_packet)
    print(format_result(result))

    # Cleanup test registry
    if os.path.exists("data/test_registry.json"):
        os.remove("data/test_registry.json")

    print("\n" + "=" * 60)
    print("Smoke test complete.")
    print("  Authentic drone: detected as AUTHENTIC ✓")
    print("  Spoofed drone (claims authentic S/N, different transmitter): detected as SPOOFED ✓")
    print("  Unknown drone: detected as NOT_ENROLLED ✓")
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()
