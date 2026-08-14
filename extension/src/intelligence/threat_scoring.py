"""
Threat Scoring Engine — fuses detection, ID, intent, bearing, and trajectory
into a composite 0-100 threat score with per-factor breakdown.

This is the intelligence product that transforms IRIS from a detection system
into a CUAS command input.
"""

from typing import Optional, Dict, List
from dataclasses import dataclass, field


@dataclass
class ThreatAssessment:
    """Complete threat assessment for a detected drone."""
    threat_score: float  # 0-100 composite
    threat_level: str  # 'low', 'medium', 'high', 'critical'
    factors: Dict[str, float]  # per-factor scores
    factors_detail: Dict[str, str]  # human-readable factor descriptions
    recommended_action: str  # countermeasure recommendation
    confidence: float  # 0-1
    timestamp: str  # ISO format


# Threat weights (sum to 1.0)
THREAT_WEIGHTS = {
    'drone_type': 0.25,
    'intent': 0.30,
    'trajectory': 0.20,
    'signal_strength': 0.10,
    'context': 0.15,
}

# Drone type threat levels
DRONE_TYPE_THREAT = {
    'FPV': 90,  # First-person-view, typically weaponized
    'kamikaze': 95,  # One-way attack
    'DJI_Mavic': 40,  # Commercial, dual-use
    'DJI_Phantom': 35,
    'DJI_Mini': 25,  # Small, limited payload
    'DJI_Inspire': 50,  # Larger, more capable
    'Parrot': 30,
    'Yuneec': 30,
    'unknown': 50,  # Unknown = medium threat by default
}

# Intent threat levels
INTENT_THREAT = {
    'attack': 95,
    'surveillance': 60,
    'delivery': 20,
    'recreational': 15,
    'unknown': 50,
}

# Context threat multipliers
CONTEXT_MULTIPLIERS = {
    'night': 1.3,
    'restricted_airspace': 1.5,
    'near_critical_infrastructure': 1.4,
    'near_airport': 1.3,
    'near_military': 1.4,
    'near_power_plant': 1.5,
    'urban_dense': 1.2,
    'normal': 1.0,
}


class ThreatScorer:
    """
    Computes composite threat score from multiple factors.

    Usage:
        scorer = ThreatScorer()
        assessment = scorer.assess(
            drone_type='FPV',
            intent='attack',
            approach_rate=15.0,  # m/s, approaching
            rssi=-55,  # dBm, strong signal (close)
            context={'time_of_day': 'night', 'location': 'near_power_plant'},
            swarm=False,
            confidence=0.85
        )
    """
    def assess(self, drone_type: str = 'unknown',
               intent: str = 'unknown',
               approach_rate: float = 0.0,
               rssi: float = -80.0,
               context: Optional[Dict] = None,
               swarm: bool = False,
               confidence: float = 0.5,
               bearing: Optional[Dict] = None) -> ThreatAssessment:
        """
        Compute threat assessment.

        Parameters:
            drone_type: identified drone type ('FPV', 'DJI_Mavic', etc.)
            intent: classified intent ('attack', 'surveillance', etc.)
            approach_rate: m/s, positive = approaching
            rssi: signal strength in dBm (higher = closer)
            context: dict with time_of_day, location, etc.
            swarm: True if part of a swarm
            confidence: detection confidence 0-1
            bearing: bearing estimate dict
        """
        if context is None:
            context = {}

        # 1. Drone type factor
        type_score = DRONE_TYPE_THREAT.get(drone_type, 50)
        type_detail = f"Drone type: {drone_type} (base threat: {type_score})"

        # 2. Intent factor
        intent_score = INTENT_THREAT.get(intent, 50)
        intent_detail = f"Intent: {intent} (threat: {intent_score})"

        # 3. Trajectory factor (approach rate + bearing)
        if approach_rate > 10:
            traj_score = 85
            traj_detail = f"Fast approach ({approach_rate:.1f} m/s) — high threat"
        elif approach_rate > 0:
            traj_score = 60
            traj_detail = f"Approaching ({approach_rate:.1f} m/s) — moderate threat"
        elif approach_rate > -5:
            traj_score = 30
            traj_detail = f"Slow/stationary ({approach_rate:.1f} m/s) — low threat"
        else:
            traj_score = 15
            traj_detail = f"Departing ({approach_rate:.1f} m/s) — minimal threat"

        # 4. Signal strength factor (proxy for range)
        if rssi > -50:
            sig_score = 80
            sig_detail = f"Very strong signal ({rssi} dBm) — very close (<100m)"
        elif rssi > -65:
            sig_score = 60
            sig_detail = f"Strong signal ({rssi} dBm) — close (~100-300m)"
        elif rssi > -80:
            sig_score = 40
            sig_detail = f"Moderate signal ({rssi} dBm) — medium range (~300-800m)"
        else:
            sig_score = 20
            sig_detail = f"Weak signal ({rssi} dBm) — far (>800m)"

        # 5. Context factor
        context_score = 50
        context_details = []
        multiplier = 1.0

        time_of_day = context.get('time_of_day', 'day')
        if time_of_day == 'night':
            multiplier *= CONTEXT_MULTIPLIERS['night']
            context_details.append("Night operation (+30%)")

        location = context.get('location', 'normal')
        if location in CONTEXT_MULTIPLIERS:
            multiplier *= CONTEXT_MULTIPLIERS[location]
            context_details.append(f"{location} (+{int((CONTEXT_MULTIPLIERS[location]-1)*100)}%)")

        context_score = min(100, int(50 * multiplier))
        context_detail = "; ".join(context_details) if context_details else "Normal context"

        # Swarm boost
        if swarm:
            type_score = min(100, type_score + 20)
            type_detail += " [SWARM DETECTED +20]"

        # Compute weighted composite
        composite = (
            THREAT_WEIGHTS['drone_type'] * type_score +
            THREAT_WEIGHTS['intent'] * intent_score +
            THREAT_WEIGHTS['trajectory'] * traj_score +
            THREAT_WEIGHTS['signal_strength'] * sig_score +
            THREAT_WEIGHTS['context'] * context_score
        )

        # Determine threat level
        if composite >= 80:
            level = 'critical'
        elif composite >= 60:
            level = 'high'
        elif composite >= 35:
            level = 'medium'
        else:
            level = 'low'

        # Recommend countermeasure
        action = self._recommend_action(level, drone_type, intent, approach_rate, swarm)

        import time as _time
        return ThreatAssessment(
            threat_score=round(composite, 1),
            threat_level=level,
            factors={
                'drone_type': type_score,
                'intent': intent_score,
                'trajectory': traj_score,
                'signal_strength': sig_score,
                'context': context_score,
            },
            factors_detail={
                'drone_type': type_detail,
                'intent': intent_detail,
                'trajectory': traj_detail,
                'signal_strength': sig_detail,
                'context': context_detail,
            },
            recommended_action=action,
            confidence=confidence,
            timestamp=_time.strftime('%Y-%m-%dT%H:%M:%SZ', _time.gmtime()),
        )

    def _recommend_action(self, level: str, drone_type: str,
                          intent: str, approach_rate: float,
                          swarm: bool) -> str:
        """Recommend countermeasure based on threat assessment."""
        if swarm and level in ('high', 'critical'):
            return "HPM / AREA DENIAL RECOMMENDED — swarm detected, individual jamming insufficient"

        if level == 'critical':
            if intent == 'attack':
                return "JAM IMMEDIATELY (directional, specific frequency) — attack intent, critical threat"
            elif drone_type == 'FPV':
                return "JAM IMMEDIATELY — FPV drone, critical threat level"
            else:
                return "JAM (directional) — critical threat level"

        elif level == 'high':
            if approach_rate > 5:
                return "PREPARE TO JAM — high threat, approaching target"
            else:
                return "TRACK AND PREPARE COUNTERMEASURES — high threat level"

        elif level == 'medium':
            if intent == 'surveillance':
                return "MONITOR AND LOG — surveillance intent, medium threat"
            else:
                return "TRACK AND ASSESS — medium threat, monitor for escalation"

        else:  # low
            if intent == 'recreational':
                return "LOG ONLY — recreational activity, low threat"
            else:
                return "MONITOR — low threat level"

    def assess_track(self, track) -> ThreatAssessment:
        """Assess threat from a DroneTrack object."""
        return self.assess(
            drone_type=track.drone_type or 'unknown',
            intent=track.intent or 'unknown',
            approach_rate=track.bearing.get('approach_rate', 0) if track.bearing else 0,
            rssi=track.rssi_history[-1] if track.rssi_history else -90,
            context={},
            swarm=False,
            confidence=0.5,
        )
