"""
Adversarial Robustness Checker — tests Mahalanobis boundary against crafted perturbations.

An adversary who knows the detection boundary could craft perturbations to evade it.
This module tests the system against adversarial evasion attempts, using the same
collapse-monitoring rigor as the cross-receiver experiments.

Reference: SVM-based GPS spoofing detector at 99.9% accuracy dropped to 20.4%
against crafted positional-shift attack (NIH documented result).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple


class AdversarialProbe:
    """
    Test Mahalanobis detection boundary against adversarial perturbations.

    Methods:
      1. FGSM (Fast Gradient Sign Method) — single-step perturbation
      2. PGD (Projected Gradient Descent) — iterative perturbation
      3. Boundary probing — find the minimum perturbation to cross the boundary
      4. Noise floor injection — simulate adversarial noise floor manipulation
    """
    def __init__(self, encoder, centroid, cov_inv, device='cuda'):
        self.encoder = encoder
        self.centroid = centroid  # numpy (D,)
        self.cov_inv = cov_inv    # numpy (D, D)
        self.device = device

    def mahalanobis_distance(self, embeddings):
        """Compute Mahalanobis distance for a batch of embeddings."""
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        norms = torch.norm(embeddings, dim=1, keepdim=True) + 1e-8
        embs_n = embeddings / norms
        centroid_t = torch.from_numpy(self.centroid).float().to(embeddings.device)
        cov_inv_t = torch.from_numpy(self.cov_inv).float().to(embeddings.device)
        diff = embs_n - centroid_t
        dists = torch.sqrt(torch.clamp(
            torch.sum(diff @ cov_inv_t * diff, dim=1), min=0
        ))
        return dists

    def fgsm_attack(self, specs, labels, epsilon=0.01):
        """
        FGSM: x' = x + epsilon * sign(grad_x(loss))

        Perturbs the INPUT (spectrogram/SCF image) to maximize detection loss.
        """
        specs = specs.clone().detach().requires_grad_(True).to(self.device)
        labels = labels.to(self.device)

        z = self.encoder(specs)
        loss = F.binary_cross_entropy_with_logits(
            z.mean(dim=1), labels  # crude: use mean as logit
        )

        grad = torch.autograd.grad(loss, specs)[0]
        perturbed = specs + epsilon * grad.sign()
        return perturbed.detach()

    def pgd_attack(self, specs, labels, epsilon=0.03, alpha=0.01, n_steps=20):
        """
        PGD: iterative FGSM with projection back to epsilon-ball.
        """
        specs = specs.clone().detach().to(self.device)
        original = specs.clone()
        labels = labels.to(self.device)

        for _ in range(n_steps):
            specs.requires_grad_(True)
            z = self.encoder(specs)
            loss = F.binary_cross_entropy_with_logits(z.mean(dim=1), labels)
            grad = torch.autograd.grad(loss, specs)[0]

            specs = specs.detach() + alpha * grad.sign()
            # Project back to epsilon-ball
            delta = torch.clamp(specs - original, -epsilon, epsilon)
            specs = (original + delta).detach()

        return specs

    def boundary_distance(self, specs, threshold, n_steps=50, epsilon=0.1):
        """
        Find minimum perturbation to cross the Mahalanobis boundary.

        Uses binary search along the gradient direction.
        """
        specs = specs.clone().detach().to(self.device)

        with torch.no_grad():
            z = self.encoder(specs)
            original_dist = self.mahalanobis_distance(z).cpu().numpy()

        # If already across boundary, no perturbation needed
        already_evaded = original_dist > threshold

        # Binary search for minimum epsilon
        lo, hi = 0.0, epsilon
        for _ in range(n_steps):
            mid = (lo + hi) / 2
            perturbed = self.fgsm_attack(specs, torch.ones(specs.size(0), device=self.device), mid)

            with torch.no_grad():
                z_p = self.encoder(perturbed)
                perturbed_dist = self.mahalanobis_distance(z_p).cpu().numpy()

            if np.any(perturbed_dist > threshold):
                hi = mid  # perturbation too large
            else:
                lo = mid  # need more perturbation

        return {
            'min_epsilon_to_evade': hi,
            'original_distance': original_dist.mean(),
            'threshold': threshold,
            'already_evaded': already_evaded.any(),
        }

    def run_robustness_audit(self, specs, labels, threshold, device='cuda'):
        """
        Full adversarial robustness audit.

        Returns a report with:
          - FGSM evasion rate at multiple epsilon values
          - PGD evasion rate at multiple epsilon values
          - Boundary distance (minimum perturbation to evade)
          - Comparison to random noise baseline
        """
        results = {}

        # Test at multiple epsilon values
        for eps in [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]:
            # FGSM
            fgsm_specs = self.fgsm_attack(specs, labels, epsilon=eps)
            with torch.no_grad():
                z_fgsm = self.encoder(fgsm_specs)
                fgsm_dists = self.mahalanobis_distance(z_fgsm).cpu().numpy()
                fgsm_evasion = (fgsm_dists > threshold).mean()

            # PGD
            pgd_specs = self.pgd_attack(specs, labels, epsilon=eps, alpha=eps/3, n_steps=10)
            with torch.no_grad():
                z_pgd = self.encoder(pgd_specs)
                pgd_dists = self.mahalanobis_distance(z_pgd).cpu().numpy()
                pgd_evasion = (pgd_dists > threshold).mean()

            # Random noise baseline
            noise = torch.randn_like(specs) * eps
            random_specs = specs + noise.to(specs.device)
            with torch.no_grad():
                z_rand = self.encoder(random_specs)
                rand_dists = self.mahalanobis_distance(z_rand).cpu().numpy()
                rand_evasion = (rand_dists > threshold).mean()

            results[f'eps_{eps}'] = {
                'fgsm_evasion_rate': float(fgsm_evasion),
                'pgd_evasion_rate': float(pgd_evasion),
                'random_noise_evasion_rate': float(rand_evasion),
                'fgsm_mean_distance': float(fgsm_dists.mean()),
                'pgd_mean_distance': float(pgd_dists.mean()),
                'random_mean_distance': float(rand_dists.mean()),
            }

        # Boundary distance
        boundary = self.boundary_distance(specs[:10], threshold)
        results['boundary_analysis'] = boundary

        # Summary
        results['summary'] = {
            'threshold': threshold,
            'fgsm_at_eps_0.01': results['eps_0.01']['fgsm_evasion_rate'],
            'pgd_at_eps_0.01': results['eps_0.01']['pgd_evasion_rate'],
            'min_eps_to_evade': boundary['min_epsilon_to_evade'],
            'assessment': self._assess_robustness(results),
        }

        return results

    def _assess_robustness(self, results):
        """Human-readable robustness assessment."""
        fgsm_001 = results.get('eps_0.01', {}).get('fgsm_evasion_rate', 0)
        pgd_001 = results.get('eps_0.01', {}).get('pgd_evasion_rate', 0)
        min_eps = results.get('boundary_analysis', {}).get('min_epsilon_to_evade', 0)

        if pgd_001 > 0.5:
            return "VULNERABLE — PGD at ε=0.01 evades >50% of detections. Adversarial training needed."
        elif fgsm_001 > 0.3:
            return "MODERATE RISK — FGSM at ε=0.01 evades >30%. Consider adversarial training."
        elif min_eps < 0.005:
            return "LOW RISK — minimum perturbation to evade is small but non-trivial."
        else:
            return "ROBUST — system withstands standard adversarial perturbations at tested epsilon values."
