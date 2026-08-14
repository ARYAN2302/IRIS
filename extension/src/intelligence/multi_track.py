"""
Multi-Track Manager — tracks multiple simultaneous drone signals.

Separates drones by:
  1. Protocol (DJI OcuSync 5.8 GHz, FPV 2.4 GHz, etc.)
  2. Modulation fingerprint (micro-characteristics)
  3. Frequency channel assignment

Maintains separate tracks per detected signal, correlates across modalities.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class DroneTrack:
    """A single tracked drone signal."""
    track_id: int
    frequency: float  # Hz
    protocol: str  # 'DJI_OcuSync', 'FPV_2.4G', 'unknown', etc.
    first_detected: float  # timestamp
    last_detected: float  # timestamp
    detection_count: int = 0
    rssi_history: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None  # last known embedding
    drone_type: Optional[str] = None
    bearing: Optional[Dict] = None
    intent: Optional[str] = None
    threat_score: Optional[float] = None

    @property
    def is_active(self) -> bool:
        """Track is active if detected in the last 5 seconds."""
        return (time.time() - self.last_detected) < 5.0

    @property
    def duration(self) -> float:
        return self.last_detected - self.first_detected


# Protocol frequency bands
PROTOCOL_BANDS = {
    'DJI_OcuSync_2.4': (2400e6, 2483e6),
    'DJI_OcuSync_5.8': (5725e6, 5875e6),
    'FPV_2.4': (2400e6, 2500e6),
    'FPV_5.8': (5645e6, 5945e6),
    'Parrot_2.4': (2400e6, 2483e6),
    'WiFi_2.4': (2412e6, 2472e6),
    'WiFi_5': (5180e6, 5825e6),
    'Bluetooth': (2402e6, 2480e6),
}


def identify_protocol(frequency: float) -> str:
    """Identify drone protocol from frequency."""
    for protocol, (low, high) in PROTOCOL_BANDS.items():
        if low <= frequency <= high:
            return protocol
    return 'unknown'


class MultiTrackManager:
    """
    Manages multiple drone tracks from RF signal separation.

    Tracks are separated by frequency + protocol. Each track maintains
    its own detection history, embedding, and bearing estimate.

    Usage:
        manager = MultiTrackManager()
        manager.update(frequency=2.45e9, rssi=-65, embedding=z, timestamp=time.time())
        active_tracks = manager.get_active_tracks()
        if len(active_tracks) >= 5:
            print("SWARM DETECTED")
    """
    def __init__(self, track_timeout=5.0, swarm_threshold=5):
        self.tracks: Dict[int, DroneTrack] = {}
        self.next_track_id = 0
        self.track_timeout = track_timeout  # seconds
        self.swarm_threshold = swarm_threshold
        self.frequency_tolerance = 5e6  # 5 MHz tolerance for track matching

    def update(self, frequency: float, rssi: float, timestamp: float,
               embedding: Optional[np.ndarray] = None,
               drone_type: Optional[str] = None,
               bearing: Optional[Dict] = None,
               intent: Optional[str] = None) -> DroneTrack:
        """
        Update or create a track for a detected signal.

        Parameters:
            frequency: signal frequency in Hz
            rssi: received signal strength in dBm
            timestamp: detection timestamp
            embedding: drone embedding (optional)
            drone_type: identified drone type (optional)
            bearing: bearing estimate dict (optional)
            intent: classified intent (optional)

        Returns: the updated or created DroneTrack
        """
        # Find matching track (same frequency band)
        matching_track = None
        for track in self.tracks.values():
            if abs(track.frequency - frequency) < self.frequency_tolerance:
                matching_track = track
                break

        if matching_track is None:
            # Create new track
            protocol = identify_protocol(frequency)
            track = DroneTrack(
                track_id=self.next_track_id,
                frequency=frequency,
                protocol=protocol,
                first_detected=timestamp,
                last_detected=timestamp,
            )
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1
            matching_track = track

        # Update track
        matching_track.last_detected = timestamp
        matching_track.detection_count += 1
        matching_track.rssi_history.append(rssi)
        matching_track.timestamps.append(timestamp)

        # Keep history bounded
        if len(matching_track.rssi_history) > 1000:
            matching_track.rssi_history = matching_track.rssi_history[-500:]
            matching_track.timestamps = matching_track.timestamps[-500:]

        if embedding is not None:
            matching_track.embedding = embedding
        if drone_type is not None:
            matching_track.drone_type = drone_type
        if bearing is not None:
            matching_track.bearing = bearing
        if intent is not None:
            matching_track.intent = intent

        return matching_track

    def get_active_tracks(self) -> List[DroneTrack]:
        """Get all currently active tracks."""
        current_time = time.time()
        return [t for t in self.tracks.values()
                if (current_time - t.last_detected) < self.track_timeout]

    def is_swarm(self) -> bool:
        """Check if the number of active tracks indicates a swarm."""
        return len(self.get_active_tracks()) >= self.swarm_threshold

    def get_swarm_count(self) -> int:
        """Get the number of active tracks (for swarm detection)."""
        return len(self.get_active_tracks())

    def cleanup_stale_tracks(self):
        """Remove tracks that haven't been seen recently."""
        current_time = time.time()
        stale_ids = [tid for tid, t in self.tracks.items()
                     if (current_time - t.last_detected) > self.track_timeout * 10]
        for tid in stale_ids:
            del self.tracks[tid]

    def get_summary(self) -> Dict:
        """Get summary of all tracks for display/reporting."""
        active = self.get_active_tracks()
        return {
            'total_tracks': len(self.tracks),
            'active_tracks': len(active),
            'is_swarm': self.is_swarm(),
            'swarm_count': len(active),
            'tracks': [{
                'track_id': t.track_id,
                'frequency_mhz': t.frequency / 1e6,
                'protocol': t.protocol,
                'drone_type': t.drone_type,
                'intent': t.intent,
                'duration_s': t.duration,
                'detection_count': t.detection_count,
                'last_rssi': t.rssi_history[-1] if t.rssi_history else None,
                'bearing': t.bearing,
                'threat_score': t.threat_score,
            } for t in active]
        }
