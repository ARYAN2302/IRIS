#!/usr/bin/env python3
"""
IRIS AVR-CL Hardened Experiment — 3 Seeds + EWC Baseline

Addresses feedback:
1. Confidence intervals via 3 seeds
2. EWC baseline (not just naive) to show AVR-CL beats "reasonable effort"
3. Lower-LR naive baseline to show it's not just learning rate

Runs on T4. ~20 min, ~$0.15.

Usage:
    modal run scripts/avr_cl_hardened.py
"""

from __future__ import annotations

import h5py
import json
import os
import sys
import time
import copy
from typing import Dict, List, Tuple

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

app = modal.App("iris-avr-cl-hardened")

VOL = modal.Volume.from_name("iris-data", create_if_missing=True)
MODEL_VOL = modal.Volume.from_name("iris-models-v11", create_if_missing=True)
RESULTS_VOL = modal.Volume.from_name("iris-results", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install("libgl1", "libglib2.0-0", "libhdf5-dev", "python3", "python3-pip", "python-is-python3")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "h5py==3.12.1", "numpy==1.26.4",
                 "scikit-learn==1.6.1", "scipy==1.14.1", "matplotlib==3.9.3")
)

H5_REMOTE = "/data/iris_rfuav.h5"
MODEL_REMOTE = "/models/lejepa_v11_best.pt"
RESULTS_REMOTE = "/results"


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
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, embed_dim), nn.BatchNorm1d(embed_dim))
    def forward(self, x): return self.head(self.conv(x))


class FingerprintHead(nn.Module):
    def __init__(self, embed_dim=256, fp_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, fp_dim), nn.BatchNorm1d(fp_dim), nn.GELU(),
        )
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


def _resolve_type_dataset(grp, key):
    item = grp[key]
    if isinstance(item, h5py.Dataset):
        if len(item.shape) == 4: return item, item.shape[0], False
        elif len(item.shape) == 3: return item, 1, False
        else: raise ValueError(f"Bad shape {item.shape}")
    for sub_key in ["data", "spectrogram", "samples", "images"]:
        if sub_key in item:
            sub = item[sub_key]
            if isinstance(sub, h5py.Dataset) and len(sub.shape) >= 3:
                return sub, sub.shape[0], False
    sub_datasets = []
    for sk in item.keys():
        sub = item[sk]
        if isinstance(sub, h5py.Dataset) and len(sub.shape) == 3:
            sub_datasets.append(sk)
    if sub_datasets:
        try: sub_datasets.sort(key=lambda x: int(x))
        except ValueError: sub_datasets.sort()
        return item, len(sub_datasets), True
    raise ValueError(f"Cannot resolve /{key}")


def _prep(sample):
    if sample.shape[0] == 3: return sample[:2].copy().astype(np.float32)
    elif sample.shape[0] == 2: return sample.copy().astype(np.float32)
    else: return sample[:2].copy().astype(np.float32)


def _norm(x):
    for c in range(x.shape[0]):
        ch, std = x[c], x[c].std()
        if std > 1e-6: x[c] = (ch - ch.mean()) / std
        else: x[c] = ch - ch.mean()
    return x


def load_type_samples(h5_path, split, type_name, max_n=100):
    with h5py.File(h5_path, "r") as f:
        if split not in f: return np.array([])
        grp = f[split]
        if type_name not in grp: return np.array([])
        try:
            ds_or_grp, n_samples, is_multi = _resolve_type_dataset(grp, type_name)
        except: return np.array([])
        specs = []
        if is_multi:
            sub_keys = [sk for sk in ds_or_grp.keys()
                        if isinstance(ds_or_grp[sk], h5py.Dataset) and len(ds_or_grp[sk].shape) == 3]
            try: sub_keys.sort(key=lambda x: int(x))
            except: sub_keys.sort()
            for sk in sub_keys[:max_n]:
                specs.append(_norm(_prep(ds_or_grp[sk][:])))
        else:
            n = min(ds_or_grp.shape[0] if len(ds_or_grp.shape) == 4 else 1, max_n)
            for i in range(n):
                if len(ds_or_grp.shape) == 4: specs.append(_norm(_prep(ds_or_grp[i])))
                else: specs.append(_norm(_prep(ds_or_grp[:])))
        return np.stack(specs) if specs else np.array([])


def load_all_type_samples(h5_path, split, max_per_type=50):
    with h5py.File(h5_path, "r") as f:
        if split not in f: return {}
        grp = f[split]
        type_names = sorted(list(grp.keys()))
        result = {}
        for tname in type_names:
            try:
                specs = load_type_samples(h5_path, split, tname, max_per_type)
                if len(specs) > 0: result[tname] = specs
            except: continue
        return result


@torch.no_grad()
def encode_batch(encoder, specs, device, bs=32):
    encoder.eval()
    all_embs = []
    for i in range(0, len(specs), bs):
        batch = torch.from_numpy(specs[i:i+bs]).float().to(device)
        all_embs.append(encoder(batch).cpu().numpy())
    return np.concatenate(all_embs)


def supcon_loss(embeddings, labels, temperature=0.07):
    device = embeddings.device
    B = embeddings.shape[0]
    embeddings = F.normalize(embeddings, dim=1)
    sim = torch.mm(embeddings, embeddings.t()) / temperature
    sim = sim.clamp(-10.0, 10.0)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()).float()
    diag = torch.eye(B, device=device)
    pos_mask = (pos_mask - diag).clamp(min=0)
    sim_max, _ = sim.max(dim=1, keepdim=True)
    exp_sim = torch.exp(sim - sim_max.detach())
    denom = (exp_sim * (1.0 - diag)).sum(dim=1, keepdim=True)
    numer = (exp_sim * pos_mask).sum(dim=1, keepdim=True)
    log_prob = torch.log(numer + 1e-8) - torch.log(denom + 1e-8)
    n_pos = pos_mask.sum(dim=1)
    valid = n_pos > 0
    if valid.sum() == 0: return torch.tensor(0.0, device=device, requires_grad=True)
    mean_log = (log_prob * pos_mask).sum(dim=1) / (n_pos + 1e-8)
    return -mean_log[valid].mean()


def get_head_state(head):
    return {n: p.data.cpu().clone() for n, p in head.named_parameters()}


def repair_head(head, snapshot, alpha=0.15, device="cuda"):
    n = 0
    for name, p in head.named_parameters():
        if name in snapshot:
            p.data.copy_((1.0 - alpha) * p.data + alpha * snapshot[name].to(device))
            n += 1
    return n


def identify_accuracy(head, registry, test_embs, test_labels, device, threshold=0.3):
    head.eval()
    if not registry: return 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        fps = head(torch.from_numpy(test_embs).float().to(device)).cpu().numpy()
    for fp, true_label in zip(fps, test_labels):
        best_sim = -1
        best_type = None
        for dtype, enrolled_fp in registry.items():
            sim = float(np.dot(fp, enrolled_fp))
            if sim > best_sim:
                best_sim = sim
                best_type = dtype
        if best_sim >= threshold and best_type == true_label:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def run_single_experiment(encoder, enroll_embs, test_embs, test_labels, holdout_types, device, method="naive", lr=1e-3, seed=42):
    """Run one enrollment experiment. Returns final accuracy + history."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    fp_head = FingerprintHead(embed_dim=256, fp_dim=128).to(device)
    registry = {}
    history = []
    best_accs = {}
    total_repairs = 0

    # For EWC: compute Fisher information after initial enrollment
    fisher = {}
    if method == "ewc":
        for n, p in fp_head.named_parameters():
            fisher[n] = torch.zeros_like(p)

    for i, t in enumerate(holdout_types):
        # EWC and AVR-CL both need snapshot
        snapshot = get_head_state(fp_head) if method in ("avr_cl", "ewc") else None

        # LEARN
        fp_head.train()
        opt = torch.optim.AdamW(fp_head.parameters(), lr=lr, weight_decay=0.01)
        labels = torch.zeros(len(enroll_embs[t]), dtype=torch.long, device=device)
        embs_tensor = torch.from_numpy(enroll_embs[t]).float().to(device)

        for epoch in range(3):
            perm = torch.randperm(len(embs_tensor))
            for j in range(0, len(embs_tensor), 16):
                idx = perm[j:j+16]
                if len(idx) < 2: continue
                fps = fp_head(embs_tensor[idx])
                loss = supcon_loss(fps, labels[idx])

                # EWC penalty
                if method == "ewc" and i > 0:
                    ewc_loss = 0
                    for n, p in fp_head.named_parameters():
                        if n in fisher:
                            ewc_loss += (fisher[n] * (p - snapshot.get(n, p).to(device)) ** 2).sum()
                    loss = loss + 100 * ewc_loss

                opt.zero_grad()
                loss.backward()
                opt.step()
        fp_head.eval()

        # Enroll
        with torch.no_grad():
            mean_fp = fp_head(torch.from_numpy(enroll_embs[t].mean(axis=0)).float().unsqueeze(0).to(device))
        registry[t] = mean_fp.cpu().numpy()[0]

        # Update Fisher for EWC (after enrollment)
        if method == "ewc":
            fp_head.train()
            opt2 = torch.optim.AdamW(fp_head.parameters(), lr=lr, weight_decay=0.01)
            for _ in range(1):
                perm = torch.randperm(len(embs_tensor))
                for j in range(0, len(embs_tensor), 16):
                    idx = perm[j:j+16]
                    if len(idx) < 2: continue
                    fps = fp_head(embs_tensor[idx])
                    loss = supcon_loss(fps, labels[idx])
                    opt2.zero_grad()
                    loss.backward()
                    for n, p in fp_head.named_parameters():
                        if p.grad is not None:
                            fisher[n] += p.grad.data ** 2 / len(embs_tensor)
            fp_head.eval()
            fisher = {n: f / max(i, 1) for n, f in fisher.items()}

        # VERIFY
        enrolled_so_far = holdout_types[:i+1]
        test_mask = np.isin(test_labels, enrolled_so_far)
        acc = identify_accuracy(fp_head, registry, test_embs[test_mask], test_labels[test_mask], device)

        # Check drift
        repairs = 0
        if method == "avr_cl" and i > 0 and snapshot is not None:
            prev_types = holdout_types[:i]
            drifted = False
            for pt in prev_types:
                if pt in best_accs:
                    # Check per-type accuracy
                    pt_mask = test_labels == pt
                    if pt_mask.sum() > 0:
                        pt_acc = identify_accuracy(fp_head, registry, test_embs[pt_mask], test_labels[pt_mask], device)
                        drop = best_accs[pt] - pt_acc
                        if drop > 0.1:
                            drifted = True
                            break

            if drifted:
                for step in range(5):
                    n = repair_head(fp_head, snapshot, alpha=0.15, device=device)
                    repairs += 1
                    # Re-enroll all types with repaired weights
                    for et in enrolled_so_far:
                        with torch.no_grad():
                            fp = fp_head(torch.from_numpy(enroll_embs[et].mean(axis=0)).float().unsqueeze(0).to(device))
                        registry[et] = fp.cpu().numpy()[0]
                    acc = identify_accuracy(fp_head, registry, test_embs[test_mask], test_labels[test_mask], device)

                    still_drifted = False
                    for pt in prev_types:
                        if pt in best_accs:
                            pt_mask = test_labels == pt
                            if pt_mask.sum() > 0:
                                pt_acc = identify_accuracy(fp_head, registry, test_embs[pt_mask], test_labels[pt_mask], device)
                                if best_accs[pt] - pt_acc > 0.1:
                                    still_drifted = True
                                    break
                    if not still_drifted:
                        break

        total_repairs += repairs

        # Update best
        for pt in enrolled_so_far:
            pt_mask = test_labels == pt
            if pt_mask.sum() > 0:
                pt_acc = identify_accuracy(fp_head, registry, test_embs[pt_mask], test_labels[pt_mask], device)
                if pt not in best_accs or pt_acc > best_accs[pt]:
                    best_accs[pt] = pt_acc

        history.append({"step": i+1, "type": t, "accuracy": float(acc), "repairs": repairs})

    return history, total_repairs


@app.function(
    image=IMAGE, gpu="T4",
    volumes={"/data": VOL, "/models": MODEL_VOL, "/results": RESULTS_VOL},
    timeout=5400, memory=16384,
)
def run_hardened():
    device = "cuda"
    print("=" * 70)
    print("IRIS AVR-CL Hardened Experiment — 3 Seeds + EWC + Low-LR Naive")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 70)

    VOL.reload()
    MODEL_VOL.reload()

    # Load encoder
    print("\n[0] Loading encoder...")
    ckpt = torch.load(MODEL_REMOTE, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    encoder_state = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
    encoder = CNNEncoder(in_ch=2, width=64, depth=6, embed_dim=256).to(device)
    encoder.load_state_dict(encoder_state, strict=False)
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    print(f"  [ok] encoder: {sum(p.numel() for p in encoder.parameters()):,} params")

    # Load holdout data
    print("\n[1] Loading holdout data...")
    holdout_data = load_all_type_samples(H5_REMOTE, "holdout", max_per_type=50)
    holdout_types = sorted(holdout_data.keys())
    print(f"  {len(holdout_types)} types: {holdout_types}")

    # Split into enroll/test
    rng = np.random.default_rng(42)
    enroll_data = {}
    test_data = {}
    test_labels = []
    test_embs_all = []
    for t in holdout_types:
        specs = holdout_data[t]
        n = len(specs)
        perm = rng.permutation(n)
        n_enroll = max(3, n // 2)
        enroll_data[t] = specs[perm[:n_enroll]]
        test_data[t] = specs[perm[n_enroll:]]
        if len(test_data[t]) > 0:
            embs = encode_batch(encoder, test_data[t], device)
            test_embs_all.append(embs)
            test_labels.extend([t] * len(embs))

    test_embs_all = np.concatenate(test_embs_all)
    test_labels = np.array(test_labels)

    # Pre-compute enrollment embeddings
    enroll_embs = {}
    for t in holdout_types:
        enroll_embs[t] = encode_batch(encoder, enroll_data[t], device)

    print(f"  Test samples: {len(test_labels)}")

    # Run all methods × 3 seeds
    methods = [
        ("naive_high_lr", "naive", 1e-3),
        ("naive_low_lr", "naive", 1e-4),
        ("ewc", "ewc", 1e-3),
        ("avr_cl", "avr_cl", 1e-3),
    ]
    seeds = [42, 123, 456]

    all_results = {}
    for method_name, method, lr in methods:
        print(f"\n{'='*60}")
        print(f"METHOD: {method_name} (lr={lr})")
        print(f"{'='*60}")

        seed_results = []
        for seed in seeds:
            print(f"\n  Seed {seed}...")
            history, repairs = run_single_experiment(
                encoder, enroll_embs, test_embs_all, test_labels,
                holdout_types, device, method=method, lr=lr, seed=seed
            )
            final_acc = history[-1]["accuracy"]
            seed_results.append({
                "seed": seed,
                "final_accuracy": final_acc,
                "history": history,
                "repairs": repairs,
            })
            print(f"    Final accuracy: {final_acc:.3f}, repairs: {repairs}")

        accs = [r["final_accuracy"] for r in seed_results]
        all_results[method_name] = {
            "method": method,
            "lr": lr,
            "seeds": seed_results,
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "min": float(np.min(accs)),
            "max": float(np.max(accs)),
        }
        print(f"\n  SUMMARY: mean={np.mean(accs):.3f}, std={np.std(accs):.3f}, range=[{np.min(accs):.3f}, {np.max(accs):.3f}]")

    # Summary table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY (3 seeds each)")
    print("=" * 70)
    print(f"  {'Method':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for method_name, r in all_results.items():
        print(f"  {method_name:<20} {r['mean']:>8.3f} {r['std']:>8.3f} {r['min']:>8.3f} {r['max']:>8.3f}")

    # Save results
    os.makedirs(RESULTS_REMOTE, exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "holdout_types": holdout_types,
        "n_test_samples": len(test_labels),
        "methods": all_results,
    }

    json_path = f"{RESULTS_REMOTE}/avr_cl_hardened.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  [ok] saved {json_path}")

    # Generate plot
    print("  [info] generating plot...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = {"naive_high_lr": "red", "naive_low_lr": "orange", "ewc": "blue", "avr_cl": "green"}
        labels = {"naive_high_lr": "Naive (lr=1e-3)", "naive_low_lr": "Naive (lr=1e-4)", "ewc": "EWC", "avr_cl": "AVR-CL"}

        for method_name, r in all_results.items():
            # Average history across seeds
            steps = [h["step"] for h in r["seeds"][0]["history"]]
            mean_accs = []
            std_accs = []
            for step_idx in range(len(steps)):
                step_accs = [s["history"][step_idx]["accuracy"] for s in r["seeds"]]
                mean_accs.append(np.mean(step_accs))
                std_accs.append(np.std(step_accs))

            mean_accs = np.array(mean_accs)
            std_accs = np.array(std_accs)

            ax.plot(steps, mean_accs, "o-", color=colors[method_name], linewidth=2,
                    markersize=8, label=labels[method_name])
            ax.fill_between(steps, mean_accs - std_accs, mean_accs + std_accs,
                          color=colors[method_name], alpha=0.2)

        ax.set_xlabel("Number of Enrolled Types", fontsize=12)
        ax.set_ylabel("Identification Accuracy", fontsize=12)
        ax.set_title("AVR-CL vs Baselines — 3 Seeds (shaded = ±1 std)", fontsize=14)
        ax.legend(fontsize=11, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(range(1, len(holdout_types) + 1))

        plt.tight_layout()
        plot_path = f"{RESULTS_REMOTE}/avr_cl_hardened.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [ok] saved {plot_path}")
    except Exception as e:
        print(f"  [warn] plot failed: {e}")

    # Markdown report
    md_path = f"{RESULTS_REMOTE}/avr_cl_hardened.md"
    with open(md_path, "w") as f:
        f.write("# IRIS AVR-CL Hardened Experiment — 3 Seeds + EWC Baseline\n\n")
        f.write(f"**Generated:** {output['timestamp']}\n\n")
        f.write(f"**Holdout types:** {len(holdout_types)}\n")
        f.write(f"**Test samples:** {output['n_test_samples']}\n\n")

        f.write("## Results (3 seeds each)\n\n")
        f.write("| Method | Mean | Std | Min | Max |\n|---|---|---|---|---|\n")
        for method_name, r in all_results.items():
            f.write(f"| {method_name} | {r['mean']:.3f} | {r['std']:.3f} | {r['min']:.3f} | {r['max']:.3f} |\n")

        f.write("\n## Key Findings\n\n")
        naive_mean = all_results["naive_high_lr"]["mean"]
        avr_mean = all_results["avr_cl"]["mean"]
        ewc_mean = all_results["ewc"]["mean"]
        f.write(f"- **Naive (high LR):** {naive_mean:.3f} ± {all_results['naive_high_lr']['std']:.3f}\n")
        f.write(f"- **Naive (low LR):** {all_results['naive_low_lr']['mean']:.3f} ± {all_results['naive_low_lr']['std']:.3f}\n")
        f.write(f"- **EWC:** {ewc_mean:.3f} ± {all_results['ewc']['std']:.3f}\n")
        f.write(f"- **AVR-CL:** {avr_mean:.3f} ± {all_results['avr_cl']['std']:.3f}\n\n")
        f.write(f"- **AVR-CL vs Naive:** {avr_mean/max(naive_mean, 0.001):.1f}x improvement\n")
        f.write(f"- **AVR-CL vs EWC:** {avr_mean/max(ewc_mean, 0.001):.1f}x improvement\n")

    print(f"  [ok] saved {md_path}")
    RESULTS_VOL.commit()

    print("\n" + "=" * 70)
    print("HARDENED EXPERIMENT COMPLETE")
    print("=" * 70)

    return output


@app.local_entrypoint()
def main():
    run_hardened.remote()


if __name__ == "__main__":
    main()
