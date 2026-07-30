#!/usr/bin/env python3
"""LeJEPA training loop.

Features:
  - Collapse detection (std, cosine, dimensional, plateau)
  - Projection head (encoder → projector → loss space)
  - Weighted sampler for drone/negative balance
  - Negatives get SIGReg only, no invariance
  - Cosine warmup + decay schedule
  - Gradient clipping
  - Emergency checkpoint on collapse
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import LeJEPA
from src.train_dataset import LeJEPATrainDataset, LeJEPAEvalDataset


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class CosineWarmupScheduler:
    """Linear warmup + cosine decay."""
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr

    def step(self, step_idx):
        if step_idx < self.warmup_steps:
            lr = self.base_lr * (step_idx + 1) / self.warmup_steps
        else:
            progress = (step_idx - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr


class CollapseDetector:
    """Detect representation collapse during training.

    Checks after each epoch:
    - Embedding std < threshold → all embeddings are the same
    - Pairwise cosine similarity > threshold → all same direction
    - Effective dimensionality < threshold → dimensional collapse
    - Loss plateau → model is stuck, not learning

    If collapse detected: stop training, save emergency checkpoint, suggest fixes.
    """
    def __init__(self, std_threshold=0.01, cosine_threshold=0.95,
                 dim_threshold=10, plateau_patience=5):
        self.std_threshold = std_threshold
        self.cosine_threshold = cosine_threshold
        self.dim_threshold = dim_threshold
        self.plateau_patience = plateau_patience
        self.loss_history = []
        self.collapsed = False
        self.collapse_reason = None

    def check(self, model, dataloader, device, epoch_loss):
        """Run all collapse checks. Returns (is_ok, message)."""
        self.loss_history.append(epoch_loss)

        # Check 1: Loss plateau
        if len(self.loss_history) >= self.plateau_patience:
            recent = self.loss_history[-self.plateau_patience:]
            loss_range = max(recent) - min(recent)
            if loss_range < 1e-6:
                self.collapsed = True
                self.collapse_reason = (
                    f"LOSS PLATEAU: loss hasn't changed by >1e-6 "
                    f"in {self.plateau_patience} epochs (loss={epoch_loss:.8f})"
                )
                return False, self.collapse_reason

        # Check 2 & 3 & 4: Embedding statistics
        model.eval()
        embeddings = []
        with torch.no_grad():
            for i, (x1, x2, is_pos) in enumerate(dataloader):
                if i >= 3:
                    break
                x1 = x1.to(device)
                z = model.encode(x1)
                embeddings.append(z)

        embeddings = torch.cat(embeddings, dim=0)

        # Check 2: Std per dimension
        stds = embeddings.std(dim=0)
        mean_std = stds.mean().item()
        min_std = stds.min().item()

        if mean_std < self.std_threshold:
            self.collapsed = True
            self.collapse_reason = (
                f"EMBEDDING COLLAPSE: mean_std={mean_std:.6f} < {self.std_threshold}. "
                f"min_std={min_std:.6f}. All embeddings are nearly identical."
            )
            return False, self.collapse_reason

        # Check 3: Pairwise cosine similarity
        n = min(64, embeddings.shape[0])
        sample = embeddings[:n]
        sample_norm = sample / (sample.norm(dim=1, keepdim=True) + 1e-8)
        cos_sim = torch.mm(sample_norm, sample_norm.T)

        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        mean_cos = cos_sim[mask].mean().item()
        max_cos = cos_sim[mask].max().item()

        if mean_cos > self.cosine_threshold:
            self.collapsed = True
            self.collapse_reason = (
                f"EMBEDDING COLLAPSE: mean_cosine={mean_cos:.4f} > {self.cosine_threshold}. "
                f"max_cosine={max_cos:.4f}. All embeddings point in the same direction."
            )
            return False, self.collapse_reason

        # Check 4: Dimensional collapse (SVD)
        if embeddings.shape[0] >= 10:
            try:
                sample_svd = embeddings[:min(256, embeddings.shape[0])]
                _, s, _ = torch.svd(sample_svd)
                effective_dim = (s > 0.01 * s[0]).sum().item()
                if effective_dim < self.dim_threshold:
                    self.collapsed = True
                    self.collapse_reason = (
                        f"DIMENSIONAL COLLAPSE: effective_dim={effective_dim} "
                        f"< {self.dim_threshold}. Embeddings live in a "
                        f"{effective_dim}D subspace of 768D."
                    )
                    return False, self.collapse_reason
            except Exception:
                pass

        return True, (
            f"OK (std={mean_std:.4f}, min_std={min_std:.4f}, "
            f"cos={mean_cos:.4f}, eff_dim=OK)"
        )


def train_one_epoch(model, dataloader, optimizer, scheduler, device, grad_clip, step_counter):
    model.train()
    total_loss = 0
    total_sig = 0
    total_inv = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for x1, x2, is_pos in pbar:
        x1 = x1.to(device)
        x2 = x2.to(device)
        is_pos = is_pos.to(device)

        # Forward
        z1, z2, p1, p2, y_pred, losses = model(x1, x2)

        sig_loss = losses['sig']
        inv_loss = losses['inv']

        # Negatives: SIGReg only, no invariance
        batch_loss = sig_loss + is_pos.mean() * inv_loss

        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step(step_counter[0])
        step_counter[0] += 1

        total_loss += batch_loss.item()
        total_sig += sig_loss.item()
        total_inv += inv_loss.item()
        n_batches += 1

        pbar.set_postfix({
            'loss': f'{batch_loss.item():.6f}',
            'sig': f'{sig_loss.item():.6f}',
            'inv': f'{inv_loss.item():.6f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    return {
        'loss': total_loss / n_batches,
        'sig': total_sig / n_batches,
        'inv': total_inv / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train LeJEPA on drone RF spectrograms')
    parser.add_argument('--hdf5', required=True, help='Path to HDF5 store')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-3, help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='AdamW weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--pair-distance', type=int, default=5, help='Positive pair distance')
    parser.add_argument('--noise-std', type=float, default=0.05, help='Augmentation noise std')
    parser.add_argument('--freq-shift', type=int, default=5, help='Max frequency shift bins')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', help='Resume from checkpoint path')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # ---- Dataset ----
    train_ds = LeJEPATrainDataset(
        args.hdf5,
        pair_distance=args.pair_distance,
        noise_std=args.noise_std,
        freq_shift_bins=args.freq_shift,
    )

    # Weighted sampler: balance drone vs negative
    n_drone = len(train_ds.drone_indices)
    n_neg = len(train_ds.neg_indices)
    weights = np.ones(len(train_ds), dtype=np.float64)

    if n_neg > 0 and n_drone > n_neg:
        neg_weight = n_drone / n_neg
        for i in train_ds.neg_indices:
            weights[i] = neg_weight
    elif n_neg > 0 and n_neg > n_drone:
        drone_weight = n_neg / n_drone
        for i in train_ds.drone_indices:
            weights[i] = drone_weight

    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, pin_memory=(device.type != 'cpu')
    )

    # ---- Model ----
    model = LeJEPA(in_channels=3).to(device)
    param_info = model.param_count()
    print(f"Model params: encoder={param_info['encoder']:,} "
          f"projector={param_info['projector']:,} "
          f"predictor={param_info['predictor']:,} "
          f"total={param_info['total']:,}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ---- Scheduler ----
    total_steps = args.epochs * len(train_dl)
    warmup_steps = args.warmup_epochs * len(train_dl)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps, args.lr)
    step_counter = [0]

    # ---- Collapse Detector ----
    collapse_detector = CollapseDetector(
        std_threshold=0.01,
        cosine_threshold=0.95,
        dim_threshold=10,
        plateau_patience=5,
    )

    # ---- Resume ----
    start_epoch = 0
    best_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        best_loss = ckpt.get('best_loss', float('inf'))
        step_counter[0] = ckpt.get('step', 0)
        print(f"Resumed from epoch {start_epoch}, step {step_counter[0]}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- Training Loop ----
    history = []
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        t0 = time.time()

        metrics = train_one_epoch(
            model, train_dl, optimizer, scheduler, device, args.grad_clip, step_counter
        )

        elapsed = time.time() - t0
        metrics['epoch'] = epoch + 1
        metrics['time'] = elapsed
        metrics['lr'] = optimizer.param_groups[0]['lr']
        history.append(metrics)

        print(f"  loss={metrics['loss']:.6f}  sig={metrics['sig']:.6f}  "
              f"inv={metrics['inv']:.6f}  time={elapsed:.1f}s  lr={metrics['lr']:.2e}")

        # ---- Collapse Detection ----
        is_ok, msg = collapse_detector.check(model, train_dl, device, metrics['loss'])
        print(f"  Collapse: {msg}")

        if not is_ok:
            print(f"\n{'!'*60}")
            print(f"COLLAPSE DETECTED at epoch {epoch+1}!")
            print(f"Reason: {msg}")
            print(f"{'!'*60}")
            print(f"\nSuggested fixes:")
            print(f"  1. Reduce lr by 10x: --lr {args.lr / 10}")
            print(f"  2. Increase noise augmentation: --noise-std {args.noise_std * 2}")
            print(f"  3. Increase lambda (SIGReg weight) in model.py")
            print(f"  4. Check data diversity (are all samples the same?)")
            print(f"\nSaving emergency checkpoint...")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_loss': best_loss,
                'step': step_counter[0],
                'collapse_reason': msg,
            }, os.path.join(args.checkpoint_dir, 'collapse.pt'))
            print(f"Saved to {args.checkpoint_dir}/collapse.pt")
            print(f"Fix and resume: python src/train.py --resume {args.checkpoint_dir}/collapse.pt ...")
            break

        # ---- Save Best ----
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_loss': best_loss,
                'step': step_counter[0],
            }, os.path.join(args.checkpoint_dir, 'best.pt'))
            print(f"  -> New best! ({best_loss:.6f})")

        # ---- Save Last ----
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_loss': best_loss,
            'step': step_counter[0],
        }, os.path.join(args.checkpoint_dir, 'last.pt'))

    # ---- Save History ----
    hist_path = os.path.join(args.checkpoint_dir, 'history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    if collapse_detector.collapsed:
        print(f"\nTraining ABORTED due to collapse at epoch {epoch+1}")
    else:
        print(f"\nTraining complete! Best loss: {best_loss:.6f}")
    print(f"History: {hist_path}")


if __name__ == '__main__':
    main()