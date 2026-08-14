"""
SAPIENT-Compatible Output Schema.

SAPIENT (Sensing for Asset Protection with Integrated Electronic Networked Technology)
is the UK Dstl / NATO STANREC 4869 interoperability protocol for CUAS systems.

Key principle: sensors send summary MESSAGES (information level), not raw data.
Each sensor node uses AI to make detections locally and sends only information
to the C2 system.

This module defines the output schema for IRIS-CUAS detections, compatible
with the SAPIENT philosophy: structured messages, not raw tensors.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import json
import time
import uuid


@dataclass
class SAPIENTDetection:
    """
    SAPIENT-compatible detection message.

    This is what IRIS-CUAS sends to the C2 system — a structured summary,
    not raw spectrograms or embeddings.
    """
    # Message metadata
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    sensor_id: str = 'IRIS-CUAS-001'
    sensor_type: str = 'multi_modal'  # 'rf', 'acoustic', 'radar', 'multi_modal'
    message_type: str = 'detection'  # 'detection', 'track_update', 'threat_alert', 'status'

    # Detection info
    detected: bool = False
    confidence: float = 0.0  # 0-1
    modalities_used: List[str] = field(default_factory=list)  # ['rf', 'acoustic', 'radar']
    rf_silent: bool = False  # True if RF modality was not used

    # Classification
    drone_type: Optional[str] = None
    drone_type_confidence: float = 0.0
    intent: Optional[str] = None  # 'attack', 'surveillance', 'delivery', 'recreational'
    intent_confidence: float = 0.0

    # Location/Bearing
    azimuth: Optional[float] = None  # degrees
    elevation: Optional[float] = None
    approach_rate: Optional[float] = None  # m/s
    range_estimate: Optional[float] = None  # meters (if available)

    # Signal info
    frequency_mhz: Optional[float] = None
    protocol: Optional[str] = None
    rssi: Optional[float] = None  # dBm

    # Threat assessment
    threat_score: Optional[float] = None  # 0-100
    threat_level: Optional[str] = None  # 'low', 'medium', 'high', 'critical'
    recommended_action: Optional[str] = None

    # Track info
    track_id: Optional[int] = None
    is_swarm: bool = False
    swarm_count: int = 0

    # System health
    adaptation_triggered: bool = False  # AVR-CL triggered
    adaptation_reason: Optional[str] = None  # 'drift_detected', 'novelty_detected', 'manual'

    def to_json(self) -> str:
        """Serialize to JSON for SAPIENT-compatible transmission."""
        return json.dumps(asdict(self), indent=2)

    def to_dict(self) -> Dict:
        """Return as dictionary."""
        return asdict(self)

    @classmethod
    def from_detection(cls, detected: bool, confidence: float,
                       modalities: List[str], drone_type: str = None,
                       intent: str = None, threat_assessment=None,
                       bearing=None, track=None, rf_silent: bool = False,
                       adaptation_triggered: bool = False):
        """Create from detection results."""
        msg = cls(
            detected=detected,
            confidence=confidence,
            modalities_used=modalities,
            rf_silent=rf_silent,
            drone_type=drone_type,
            intent=intent,
        )

        if threat_assessment:
            msg.threat_score = threat_assessment.threat_score
            msg.threat_level = threat_assessment.threat_level
            msg.recommended_action = threat_assessment.recommended_action

        if bearing:
            msg.azimuth = bearing.get('azimuth')
            msg.approach_rate = bearing.get('approach_rate')

        if track:
            msg.track_id = track.track_id
            msg.frequency_mhz = track.frequency / 1e6 if track.frequency else None
            msg.protocol = track.protocol
            msg.rssi = track.rssi_history[-1] if track.rssi_history else None
            msg.is_swarm = False  # Set by multi-track manager
            msg.swarm_count = 0

        msg.adaptation_triggered = adaptation_triggered

        return msg


class SAPIENTStream:
    """
    Streams SAPIENT-compatible detection messages.
    Used as the output interface for the IRIS-CUAS system.
    """
    def __init__(self, sensor_id: str = 'IRIS-CUAS-001'):
        self.sensor_id = sensor_id
        self.message_history: List[SAPIENTDetection] = []

    def emit(self, detection: SAPIENTDetection) -> str:
        """Emit a detection message and return JSON."""
        detection.sensor_id = self.sensor_id
        self.message_history.append(detection)
        return detection.to_json()

    def emit_detection(self, detected: bool, confidence: float,
                       modalities: List[str], **kwargs) -> str:
        """Convenience method to emit a detection."""
        msg = SAPIENTDetection.from_detection(detected, confidence, modalities, **kwargs)
        return self.emit(msg)

    def get_history(self) -> List[Dict]:
        """Get all emitted messages."""
        return [m.to_dict() for m in self.message_history]

    def get_stats(self) -> Dict:
        """Get summary statistics."""
        total = len(self.message_history)
        detections = sum(1 for m in self.message_history if m.detected)
        rf_silent = sum(1 for m in self.message_history if m.rf_silent)
        threats = [m for m in self.message_history if m.threat_level in ('high', 'critical')]

        return {
            'total_messages': total,
            'detections': detections,
            'rf_silent_detections': rf_silent,
            'high_threat_alerts': len(threats),
            'detection_rate': detections / total if total > 0 else 0,
        }
