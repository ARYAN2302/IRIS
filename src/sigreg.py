"""
SIGReg — Sketched Isotropic Gaussian Regularizer

From: "Linearly Identified JEPA" (Klindt, LeCun, Balestriero 2026)

The core idea:
  - Project embeddings onto K random 1D directions
  - Match the empirical characteristic function of each projection
    to the standard Gaussian characteristic function φ(t) = exp(-t²/2)
  - This uses the Cramér-Wold theorem: matching all 1D marginals
    is equivalent to matching the full multivariate distribution

Why this works for RF drone data:
  - RF hardware fingerprints are Gaussian-distributed (CFO, phase noise, amp nonlinearities)
  - LeJEPA's Theorem 3 proves that if latents are Gaussian, the encoder
    is linearly identifiable — meaning different drones get different embeddings
  - SIGReg forces the embedding space to be Gaussian, making Theorem 3 apply

Key hyperparameters from the paper:
  K = 256 random projection directions
  λ = 10⁻³ (SIGReg weight in total loss)
  Projections are FIXED (not learned) — sampled once, never updated
"""

import torch
import torch.nn as nn
import math


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularizer.
    
    Uses Cramér-Wold theorem with K random projections.
    For each projection direction w_k:
      1. Project: p_k = z @ w_k  (1D scalar per sample)
      2. For t values, compute empirical char function: φ̂_k(t) = mean(exp(j·t·p_k))
      3. Match to Gaussian char function: φ(t) = exp(-t²/2)
      4. Loss = |φ̂_k(t) - φ(t)|²
    
    Average over all K projections and T t-values.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        K: int = 256,           # number of random projections
        T: int = 8,             # number of t-values for char function matching
        t_max: float = 3.0,     # max value of t (range: [-t_max, t_max])
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.K = K
        self.T = T
        
        # FIXED random projection directions: (K, embed_dim)
        # Sampled from standard Gaussian, then normalized
        # These are NOT learned — registered as buffers, not parameters
        projections = torch.randn(K, embed_dim)
        projections = projections / projections.norm(dim=1, keepdim=True)
        self.register_buffer('projections', projections)
        
        # FIXED t-values for characteristic function matching: (T,)
        # Evenly spaced in [-t_max, t_max]
        t_values = torch.linspace(-t_max, t_max, T)
        self.register_buffer('t_values', t_values)
        
        # Target: standard Gaussian characteristic function at each t
        # φ(t) = exp(-t²/2) for real part, 0 for imaginary part
        # This is precomputed since it never changes
        gaussian_cf_real = torch.exp(-t_values ** 2 / 2)
        gaussian_cf_imag = torch.zeros_like(t_values)
        self.register_buffer('gaussian_cf_real', gaussian_cf_real)
        self.register_buffer('gaussian_cf_imag', gaussian_cf_imag)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss for a batch of embeddings.
        
        Args:
            z: (B, embed_dim) embedding vectors from encoder
        
        Returns:
            loss: scalar — average |φ̂(t) - φ(t)|² over all K projections and T t-values
        """
        B = z.shape[0]
        
        # Project embeddings onto K directions: (B, K)
        projections = z @ self.projections.T
        
        # For each projection direction k, compute empirical characteristic function
        # φ̂_k(t) = (1/B) * Σ exp(j * t * p_k^{(b)})
        # Shape operations:
        #   projections: (B, K) → (B, K, 1)
        #   t_values: (T,) → (1, 1, T)
        #   phase = t * p: (B, K, T)
        
        p = projections.unsqueeze(-1)                   # (B, K, 1)
        t = self.t_values.view(1, 1, -1)               # (1, 1, T)
        phase = t * p                                    # (B, K, T)
        
        # Empirical characteristic function: mean over batch dimension
        # Real part: mean(cos(phase)), Imag part: mean(sin(phase))
        emp_cf_real = torch.cos(phase).mean(dim=0)      # (K, T)
        emp_cf_imag = torch.sin(phase).mean(dim=0)      # (K, T)
        
        # Target characteristic function (standard Gaussian)
        target_real = self.gaussian_cf_real.unsqueeze(0)  # (1, T)
        target_imag = self.gaussian_cf_imag.unsqueeze(0)  # (1, T)
        
        # MSE between empirical and target
        loss_real = (emp_cf_real - target_real) ** 2     # (K, T)
        loss_imag = (emp_cf_imag - target_imag) ** 2     # (K, T)
        
        # Average over K projections and T t-values
        loss = (loss_real + loss_imag).mean()
        
        return loss


class InvarianceLoss(nn.Module):
    """
    Alignment/invariance loss for LeJEPA.
    
    L_inv = ||predictor(z_ctx) - z_target||²
    
    The predictor tries to predict the target embedding from the
    context embedding. No stop-gradient, no EMA teacher.
    Just direct MSE between predicted and target.
    
    This is different from BYOL/VICReg — LeJEPA doesn't use
    stop-gradients or EMA. The identifiability comes from SIGReg,
    not from architectural asymmetry.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: (B, 768) predicted target from predictor
            z_target: (B, 768) actual target embedding from encoder
        
        Returns:
            loss: scalar MSE
        """
        return ((y_pred - z_target) ** 2).mean()


class LeJEPALoss(nn.Module):
    """
    Total LeJEPA loss:
    
    L = λ · L_SIG + (1 - λ) · L_inv
    
    where:
      L_SIG = SIGReg loss (Gaussian distribution matching)
      L_inv = invariance loss (prediction alignment)
      λ = 10⁻³ (paper default)
    
    The λ is small because SIGReg is a regularizer — the main
    learning signal comes from L_inv. SIGReg just prevents
    representation collapse and ensures identifiability.
    """
    
    def __init__(self, embed_dim: int = 768, lam: float = 1e-3, K: int = 256):
        super().__init__()
        self.lam = lam
        self.sigreg = SIGReg(embed_dim=embed_dim, K=K)
        self.invariance = InvarianceLoss()
    
    def forward(
        self, z_ctx: torch.Tensor, z_target: torch.Tensor, y_pred: torch.Tensor
    ) -> dict:
        """
        Args:
            z_ctx: (B, 768) context embedding
            z_target: (B, 768) target embedding
            y_pred: (B, 768) predicted target from predictor
        
        Returns:
            dict with 'total', 'sigreg', 'invariance' losses
        """
        l_sig = self.sigreg(z_ctx)
        l_inv = self.invariance(y_pred, z_target)
        total = self.lam * l_sig + (1 - self.lam) * l_inv
        
        return {
            'total': total,
            'sigreg': l_sig,
            'invariance': l_inv,
        }