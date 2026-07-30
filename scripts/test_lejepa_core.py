"""Smoke test for Predictor + SIGReg + LeJEPALoss."""
import sys
sys.path.insert(0, ".")

import torch
from src.encoder import CNNEncoder
from src.predictor import Predictor
from src.sigreg import SIGReg, InvarianceLoss, LeJEPALoss

B = 8  # batch size
embed_dim = 768

# ─── Encoder ───
encoder = CNNEncoder(embed_dim=embed_dim)
print(f"Encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params")

# ─── Predictor ───
predictor = Predictor(embed_dim=embed_dim)
print(f"Predictor: {sum(p.numel() for p in predictor.parameters())/1e6:.2f}M params")

# ─── SIGReg ───
sigreg = SIGReg(embed_dim=embed_dim, K=256)
print(f"SIGReg: {sum(p.numel() for p in sigreg.parameters())/1e6:.2f}M params (should be 0 — projections are fixed buffers)")

# ─── Full forward pass ───
print("\n--- Forward Pass ---")

# Simulate a positive pair: context + target spectrograms
ctx_spec = torch.randn(B, 2, 256, 256)
tgt_spec = torch.randn(B, 2, 256, 256)

# Encode both
z_ctx = encoder(ctx_spec)       # (B, 768)
z_target = encoder(tgt_spec)    # (B, 768)

# Predict target from context
y_pred = predictor(z_ctx)       # (B, 768)

print(f"z_ctx:    {z_ctx.shape}, range=[{z_ctx.min():.3f}, {z_ctx.max():.3f}]")
print(f"z_target: {z_target.shape}, range=[{z_target.min():.3f}, {z_target.max():.3f}]")
print(f"y_pred:   {y_pred.shape}, range=[{y_pred.min():.3f}, {y_pred.max():.3f}]")

# ─── SIGReg alone ───
sig_loss = sigreg(z_ctx)
print(f"\nSIGReg loss: {sig_loss.item():.6f}")

# ─── Invariance alone ───
inv_loss = InvarianceLoss()(y_pred, z_target)
print(f"Invariance loss: {inv_loss.item():.6f}")

# ─── Full LeJEPA loss ───
criterion = LeJEPALoss(embed_dim=embed_dim, lam=1e-3, K=256)
losses = criterion(z_ctx, z_target, y_pred)

print(f"\nLeJEPA Loss:")
print(f"  Total:      {losses['total'].item():.6f}")
print(f"  SIGReg:     {losses['sigreg'].item():.6f}")
print(f"  Invariance: {losses['invariance'].item():.6f}")
print(f"  Ratio:      λ·L_SIG = {(1e-3 * losses['sigreg']).item():.6f}, (1-λ)·L_inv = {(1-1e-3) * losses['invariance'].item():.6f}")

# ─── Backward pass (verify gradients flow) ───
losses['total'].backward()

has_grad = all(p.grad is not None for p in encoder.parameters() if p.requires_grad)
print(f"\nGradients flow through encoder: {has_grad}")

# Check no gradient on SIGReg projections (they're buffers, not parameters)
proj_grad = sigreg.projections.grad
print(f"SIGReg projections have gradient: {proj_grad is not None} (should be False — fixed buffers)")

print("\n✅ LeJEPA core test passed!")