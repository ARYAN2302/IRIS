"""
Shared CNN encoder backbone for all modalities (RF, acoustic, radar).
Same architecture as IRIS v11 — 6-layer CNN with ConvBlocks + MaxPool.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
        )
    def forward(self, x): return self.block(x)


class CNNEncoder(nn.Module):
    """
    6-layer CNN encoder producing 256-dim embedding.
    Works with any input channel count (RF=2, acoustic=1, radar=1, hybrid=4).

    Architecture: 6 ConvBlocks with MaxPool (256→128→64→32→16→8→4)
    + Flatten + Linear(256) + BatchNorm
    """
    def __init__(self, in_ch=2, width=64, depth=6, embed_dim=256):
        super().__init__()
        layers, ch = [], in_ch
        for i in range(depth):
            out_ch = min(width * (2 ** (i // 2)), 512)
            layers.append(ConvBlock(ch, out_ch))
            layers.append(nn.MaxPool2d(2))
            ch = out_ch
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 256, 256)
            out = self.conv(dummy)
            flat = out.numel() // out.shape[0]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )

    def forward(self, x):
        return self.head(self.conv(x))


class SIGRegLoss(nn.Module):
    """Spectrally Invariant Regularization — forces unit variance per dim."""
    def __init__(self, embed_dim=256, k=256, seed=42):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        W = torch.randn(k, embed_dim, generator=gen)
        W = W / W.norm(dim=1, keepdim=True)
        self.register_buffer("W", W)

    def forward(self, z):
        p = torch.nn.functional.linear(z, self.W)
        return ((p.var(dim=0) - 1.0) ** 2).mean()


class DroneBGHead(nn.Module):
    """Binary classifier: drone vs background."""
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


class MixStyle(nn.Module):
    """
    MixStyle — randomly mix instance-level feature statistics across samples.
    Insert after early conv blocks. Parameter-free.

    Paper: Zhou et al. ICLR 2021 (1463 citations)
    Applied to RF by GAN-RXA (Zhao et al. 2024, IEEE TCCN)
    """
    def __init__(self, p=0.5, alpha=0.1):
        super().__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        sig = x.std(dim=[2, 3], keepdim=True)
        x_norm = (x - mu) / (sig + 1e-6)
        perm = torch.randperm(B, device=x.device)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]
        return x_norm * sig_mix + mu_mix


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = 0.0
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainHead(nn.Module):
    """Linear domain classifier for GRL (RFUAV vs Zenodo etc.)."""
    def __init__(self, d=256, n_domains=2):
        super().__init__()
        self.grl = GRL()
        self.fc = nn.Linear(d, n_domains)
    def forward(self, x):
        return self.fc(self.grl(x))


def dann_lambda(p):
    """DANN sigmoid ramp: λ = 2/(1+exp(-10·p)) − 1"""
    import math
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
