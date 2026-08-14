"""
Fleet Coordination Protocol — privacy-preserving multi-site intelligence.

Sites share WEIGHT DELTAS (not raw signal) via AVR-CL gate output.
This is compatible with SAPIENT's "information level, not raw data" philosophy.

Cross-site embedding correlation: compare embeddings across sites to detect
"This signature has been seen at N other sites." This requires the cross-receiver
fix (SCF features) to work — you can't compare embeddings across sites if they
encode receiver identity instead of drone identity.

Architecture:
  Site A (AVR-CL) → weight delta → Fleet Coordinator → weight delta → Site B
  Site A (embeddings) → similarity check → Fleet Coordinator → "seen at 3 sites"
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import time
import json
import hashlib


@dataclass
class SiteUpdate:
    """Weight delta + metadata from a deployed site."""
    site_id: str
    timestamp: str
    weight_delta: Dict[str, np.ndarray]  # fusion + detection head weights
    new_enrollments: List[Dict]  # new drone types enrolled
    embedding_samples: Dict[str, np.ndarray]  # {drone_type: (N, D)} — for cross-site correlation
    adaptation_triggers: List[Dict]  # what caused adaptations
    site_stats: Dict  # detection rate, FP rate, etc.


@dataclass
class FleetCorrelation:
    """Result of cross-site embedding correlation."""
    drone_type: str
    sites_seen: List[str]
    correlation_confidence: float
    first_seen: str
    last_seen: str
    total_detections: int


class FleetCoordinator:
    """
    Coordinates weight-delta sharing and cross-site correlation.

    Does NOT receive raw RF data. Only receives:
      1. Weight deltas (for applying learned patterns to other sites)
      2. Embedding samples (for "have we seen this before?" correlation)
      3. Adaptation logs (for fleet-wide intelligence)

    Usage:
        coordinator = FleetCoordinator()
        coordinator.register_site('site_alpha')
        coordinator.receive_update(site_update)
        correlations = coordinator.correlate_across_sites()
        delta = coordinator.get_weight_delta_for_site('site_beta')
    """
    def __init__(self):
        self.sites: Dict[str, SiteUpdate] = {}  # latest update per site
        self.update_history: List[SiteUpdate] = []
        self.embedding_registry: Dict[str, Dict[str, np.ndarray]] = {}  # {site_id: {drone_type: embeddings}}
        self.correlations: List[FleetCorrelation] = []

    def register_site(self, site_id: str):
        """Register a new deployed site."""
        if site_id not in self.sites:
            self.sites[site_id] = None
            self.embedding_registry[site_id] = {}

    def receive_update(self, update: SiteUpdate):
        """Receive a weight-delta update from a site."""
        self.sites[update.site_id] = update
        self.update_history.append(update)

        # Store embedding samples for cross-site correlation
        for drone_type, embs in update.embedding_samples.items():
            if drone_type not in self.embedding_registry[update.site_id]:
                self.embedding_registry[update.site_id][drone_type] = embs
            else:
                # Append new embeddings
                existing = self.embedding_registry[update.site_id][drone_type]
                self.embedding_registry[update.site_id][drone_type] = np.concatenate([existing, embs])

        # Re-compute correlations
        self._compute_correlations()

    def _compute_correlations(self):
        """Find drone types seen at multiple sites."""
        self.correlations = []

        # Collect all drone types across all sites
        all_types = set()
        for site_embs in self.embedding_registry.values():
            all_types.update(site_embs.keys())

        for drone_type in all_types:
            sites_with_type = []
            total_detections = 0

            for site_id, site_embs in self.embedding_registry.items():
                if drone_type in site_embs:
                    sites_with_type.append(site_id)
                    total_detections += len(site_embs[drone_type])

            if len(sites_with_type) >= 2:
                # Cross-site similarity check
                # Compare embeddings from different sites
                similarities = []
                for i, site_a in enumerate(sites_with_type):
                    for site_b in sites_with_type[i+1:]:
                        embs_a = self.embedding_registry[site_a][drone_type]
                        embs_b = self.embedding_registry[site_b][drone_type]

                        # Compute centroid similarity
                        centroid_a = embs_a.mean(axis=0)
                        centroid_b = embs_b.mean(axis=0)

                        # Cosine similarity
                        sim = np.dot(centroid_a, centroid_b) / (
                            np.linalg.norm(centroid_a) * np.linalg.norm(centroid_b) + 1e-8
                        )
                        similarities.append(float(sim))

                avg_sim = np.mean(similarities) if similarities else 0.0

                self.correlations.append(FleetCorrelation(
                    drone_type=drone_type,
                    sites_seen=sites_with_type,
                    correlation_confidence=avg_sim,
                    first_seen=self.update_history[0].timestamp if self.update_history else '',
                    last_seen=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    total_detections=total_detections,
                ))

    def get_weight_delta_for_site(self, site_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Get the latest weight delta to apply to a specific site.
        Aggregates deltas from all other sites."""
        if site_id not in self.sites:
            return None

        # Simple: return the latest delta from any other site
        # Production: would aggregate/federate multiple deltas
        for other_id, update in self.sites.items():
            if other_id != site_id and update is not None:
                return update.weight_delta

        return None

    def get_correlations(self) -> List[FleetCorrelation]:
        """Get all cross-site correlations — the intelligence product."""
        return self.correlations

    def get_fleet_summary(self) -> Dict:
        """Get fleet-wide summary."""
        return {
            'n_sites': len(self.sites),
            'n_active_sites': sum(1 for u in self.sites.values() if u is not None),
            'n_total_enrollments': sum(
                len(u.new_enrollments) for u in self.update_history if u
            ),
            'n_cross_site_correlations': len(self.correlations),
            'correlations': [
                {
                    'drone_type': c.drone_type,
                    'sites': c.sites_seen,
                    'confidence': c.correlation_confidence,
                    'total_detections': c.total_detections,
                }
                for c in self.correlations
            ],
        }


def create_site_update(site_id: str, avr_cl, weight_delta: Dict[str, np.ndarray],
                       site_stats: Dict = None) -> SiteUpdate:
    """Create a SiteUpdate from an AVR-CL instance."""
    embedding_samples = {}
    for entry in avr_cl.registry:
        # Share a few embedding samples per type for correlation
        embedding_samples[entry.drone_type] = entry.centroid[np.newaxis, :]  # just the centroid

    return SiteUpdate(
        site_id=site_id,
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        weight_delta=weight_delta,
        new_enrollments=[e.to_dict() for e in avr_cl.registry],
        embedding_samples=embedding_samples,
        adaptation_triggers=avr_cl.get_adaptation_log(),
        site_stats=site_stats or {},
    )
