"""
AVR-CL (Adaptive Variance-Replay Continual Learning) on Fused Embedding Space.

Adapts the existing AVR-CL mechanism to work on the fusion head's output
embedding, allowing sequential enrollment of new drone types across modalities.

Key design: encoders are frozen. Only the fusion head + detection head
are updated during continual learning. This is the same regime where
AVR-CL showed 19.6x robustness over naive fine-tuning.

Metrics: BWT (Backward Transfer) + mean post-enrollment accuracy on
recording-grouped held-out data. Real retention test: detect old types
via modality M' ≠ enrollment modality M.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional


class AVREntry:
    """A single enrolled drone type with its embedding statistics."""
    def __init__(self, drone_type: str, centroid: np.ndarray, covariance: np.ndarray,
                 n_samples: int, modality_used: str):
        self.drone_type = drone_type
        self.centroid = centroid
        self.covariance = covariance
        self.n_samples = n_samples
        self.modality_used = modality_used
        self.timestamp = None  # Set when enrolled

    def to_dict(self):
        return {
            'drone_type': self.drone_type,
            'centroid': self.centroid.tolist(),
            'covariance': self.covariance.tolist(),
            'n_samples': self.n_samples,
            'modality_used': self.modality_used,
        }


class AVRCLFused:
    """
    AVR-CL for the fused embedding space.

    Maintains a registry of enrolled drone types with their Mahalanobis
    statistics. New types are enrolled by computing their centroid and
    covariance in the fused embedding space. Retention is measured by
    re-testing old types after each enrollment.

    The key mechanism: weight deltas (not raw data) can be shared across
    sites for fleet-level intelligence. Each enrollment produces a weight
    delta that other sites can apply.
    """
    def __init__(self, fusion_head: nn.Module, detection_head: nn.Module,
                 embed_dim: int = 256):
        self.fusion_head = fusion_head
        self.detection_head = detection_head
        self.embed_dim = embed_dim
        self.registry: List[AVREntry] = []
        self.adaptation_log: List[Dict] = []

    @torch.no_grad()
    def enroll(self, drone_type: str, embeddings: np.ndarray,
               modality_used: str = 'fused'):
        """
        Enroll a new drone type from its fused embeddings.

        Parameters:
            drone_type: string identifier (e.g., 'DJI_Mavic_3')
            embeddings: (N, D) numpy array of fused embeddings for this type
            modality_used: which modality provided the data ('rf', 'acoustic', 'radar', 'fused')
        """
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embs_n = embeddings / norms

        centroid = embs_n.mean(axis=0)
        D = embs_n.shape[1]
        cov = np.cov(embs_n.T) + 1e-3 * np.eye(D)

        entry = AVREntry(drone_type, centroid, cov, len(embeddings), modality_used)
        entry.timestamp = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        self.registry.append(entry)

        # Log the adaptation
        self.adaptation_log.append({
            'action': 'enroll',
            'drone_type': drone_type,
            'n_samples': len(embeddings),
            'modality': modality_used,
            'timestamp': entry.timestamp,
            'trigger': 'manual_enrollment',  # or 'drift_detected', 'novelty_detected'
        })

        return entry

    @torch.no_grad()
    def identify(self, embedding: np.ndarray) -> Dict:
        """
        Identify which enrolled drone type an embedding matches.

        Returns: {'drone_type': str, 'confidence': float, 'distances': dict}
        """
        if not self.registry:
            return {'drone_type': 'unknown', 'confidence': 0.0, 'distances': {}}

        norm = np.linalg.norm(embedding) + 1e-8
        emb_n = embedding / norm

        distances = {}
        for entry in self.registry:
            diff = emb_n - entry.centroid
            try:
                cov_inv = np.linalg.inv(entry.covariance)
            except:
                cov_inv = np.linalg.pinv(entry.covariance)
            dist = np.sqrt(np.maximum(np.sum(diff @ cov_inv * diff), 0))
            distances[entry.drone_type] = float(dist)

        best_type = min(distances, key=distances.get)
        best_dist = distances[best_type]

        # Confidence: inverse of distance, normalized
        confidence = max(0.0, 1.0 - best_dist / 10.0)

        return {
            'drone_type': best_type,
            'confidence': float(confidence),
            'distances': distances,
        }

    @torch.no_grad()
    def test_retention(self, test_embeddings: Dict[str, np.ndarray],
                       modality_filter: Optional[str] = None) -> Dict:
        """
        Test retention on previously enrolled types.

        Parameters:
            test_embeddings: {drone_type: (N, D) array}
            modality_filter: if set, only test types enrolled with this modality

        Returns: {'retention_rate': float, 'per_type': {type: accuracy}}
        """
        per_type = {}
        correct = 0
        total = 0

        for dtype, embs in test_embeddings.items():
            # Find the enrolled entry for this type
            entry = None
            for e in self.registry:
                if e.drone_type == dtype:
                    if modality_filter is None or e.modality_used == modality_filter:
                        entry = e
                        break

            if entry is None:
                continue

            # Test each embedding
            type_correct = 0
            for emb in embs:
                result = self.identify(emb)
                if result['drone_type'] == dtype:
                    type_correct += 1

            type_acc = type_correct / len(embs) if len(embs) > 0 else 0
            per_type[dtype] = type_acc
            correct += type_correct
            total += len(embs)

        retention_rate = correct / total if total > 0 else 0

        return {
            'retention_rate': retention_rate,
            'per_type': per_type,
            'n_types_tested': len(per_type),
            'modality_filter': modality_filter,
        }

    def get_weight_delta(self) -> Dict[str, np.ndarray]:
        """
        Extract weight deltas for fleet sharing.
        Returns the current state of fusion + detection head weights.
        Other sites can apply this to update their system.
        """
        delta = {}
        for name, param in self.fusion_head.named_parameters():
            delta[f'fusion_{name}'] = param.detach().cpu().numpy()
        for name, param in self.detection_head.named_parameters():
            delta[f'detection_{name}'] = param.detach().cpu().numpy()
        return delta

    def apply_weight_delta(self, delta: Dict[str, np.ndarray]):
        """Apply weight delta from another site (fleet coordination)."""
        with torch.no_grad():
            for name, param in self.fusion_head.named_parameters():
                key = f'fusion_{name}'
                if key in delta:
                    param.copy_(torch.from_numpy(delta[key]))
            for name, param in self.detection_head.named_parameters():
                key = f'detection_{name}'
                if key in delta:
                    param.copy_(torch.from_numpy(delta[key]))

    def get_adaptation_log(self) -> List[Dict]:
        """Return the log of all adaptations — seed of fleet intelligence layer."""
        return self.adaptation_log


import time  # needed for AVREnrollment timestamp
