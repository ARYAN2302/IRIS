#!/usr/bin/env python3
"""IRIS evaluation — implements the protocol in configs/eval_protocol.json.

Metrics:
  PRIMARY:   k-NN accuracy on held-out types (zero-shot, no training)
  SECONDARY: Linear probe accuracy (single linear layer on frozen embeddings)
  TERTIARY:  Centroid analysis, silhouette score, UMAP visualization

DO NOT modify after training starts. Protocol is committed.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import h5py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
import umap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import LeJEPA
from src.train_dataset import LeJEPAEvalDataset


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ──────────────────────────────────────────────
# Embedding Extraction
# ──────────────────────────────────────────────

def extract_all_embeddings(model, hdf5_path, device, batch_size=64):
    """Extract encoder embeddings for all samples in HDF5.

    Returns:
        embeddings: (N, embed_dim) numpy array
        labels: list of string labels
        splits: list of 'train'/'holdout'/'negatives'
    """
    model.eval()

    all_embeddings = []
    all_labels = []
    all_splits = []

    for split in ['train', 'holdout', 'negatives']:
        ds = LeJEPAEvalDataset(hdf5_path, split=split)
        if len(ds) == 0:
            print(f"  {split}: 0 samples (skipping)")
            continue

        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        with torch.no_grad():
            for tensor, label_idx, label_str in dl:
                tensor = tensor.to(device)
                z = model.encode(tensor)
                all_embeddings.append(z.cpu().numpy())
                all_labels.extend([str(l) for l in label_str])
                all_splits.extend([split] * len(label_str))

        print(f"  {split}: {len(ds)} samples")

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings, all_labels, all_splits


# ──────────────────────────────────────────────
# PRIMARY: k-NN Accuracy (Zero-Shot)
# ──────────────────────────────────────────────

def knn_accuracy(embeddings, labels, splits, k_values=[1, 5]):
    """k-NN accuracy on held-out types.

    Binary task: drone vs background.
    For each holdout sample, find k nearest neighbors among train+negatives.
    If majority of neighbors are drone type (not negative), count as correct.
    """
    train_mask = np.array([s == 'train' for s in splits])
    holdout_mask = np.array([s == 'holdout' for s in splits])

    binary_labels = np.array([0 if l == 'NEGATIVE' else 1 for l in labels])

    train_X = embeddings[train_mask]
    train_y = binary_labels[train_mask]
    holdout_X = embeddings[holdout_mask]
    holdout_y = binary_labels[holdout_mask]

    if len(holdout_X) == 0:
        print("WARNING: No holdout samples found!")
        return {}

    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(train_X, train_y)
        pred = knn.predict(holdout_X)
        acc = accuracy_score(holdout_y, pred)

        drone_mask = holdout_y == 1
        neg_mask = holdout_y == 0

        drone_acc = accuracy_score(holdout_y[drone_mask], pred[drone_mask]) if drone_mask.sum() > 0 else 0.0
        neg_acc = accuracy_score(holdout_y[neg_mask], pred[neg_mask]) if neg_mask.sum() > 0 else 0.0

        results[f'knn_k{k}_overall'] = float(acc)
        results[f'knn_k{k}_drone_recall'] = float(drone_acc)
        results[f'knn_k{k}_negative_recall'] = float(neg_acc)

        print(f"  k-NN (k={k}): overall={acc:.4f}, drone_recall={drone_acc:.4f}, neg_recall={neg_acc:.4f}")

    # Per-type accuracy for holdout drones
    holdout_labels = np.array(labels)[holdout_mask]
    holdout_types = set(holdout_labels) - {'NEGATIVE'}
    print(f"\n  Per-type k-NN (k=1) breakdown:")
    for t in sorted(holdout_types):
        mask = (holdout_labels == t) & (holdout_y == 1)
        if mask.sum() > 0:
            type_acc = accuracy_score(holdout_y[mask], pred[mask])
            results[f'knn_k1_{t}'] = float(type_acc)
            print(f"    {t}: {type_acc:.4f} ({mask.sum()} samples)")

    return results


# ──────────────────────────────────────────────
# SECONDARY: Linear Probe
# ──────────────────────────────────────────────

def linear_probe(embeddings, labels, splits):
    """Linear probe: train LogisticRegression on frozen train embeddings,
    test on holdout. Binary: drone vs background.
    """
    train_mask = np.array([s == 'train' for s in splits])
    holdout_mask = np.array([s == 'holdout' for s in splits])

    binary_labels = np.array([0 if l == 'NEGATIVE' else 1 for l in labels])

    train_X = embeddings[train_mask]
    train_y = binary_labels[train_mask]
    holdout_X = embeddings[holdout_mask]
    holdout_y = binary_labels[holdout_mask]

    if len(holdout_X) == 0:
        return {}

    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    clf.fit(train_X, train_y)
    pred = clf.predict(holdout_X)
    acc = accuracy_score(holdout_y, pred)

    drone_mask = holdout_y == 1
    neg_mask = holdout_y == 0
    drone_acc = accuracy_score(holdout_y[drone_mask], pred[drone_mask]) if drone_mask.sum() > 0 else 0.0
    neg_acc = accuracy_score(holdout_y[neg_mask], pred[neg_mask]) if neg_mask.sum() > 0 else 0.0

    # FPR at 95% TPR
    if drone_mask.sum() > 0 and neg_mask.sum() > 0:
        scores = clf.predict_proba(holdout_X)[:, 1]
        sorted_thresh = np.sort(scores[drone_mask])
        thresh_95 = sorted_thresh[max(0, int(0.05 * len(sorted_thresh)))]
        fpr_at_95tpr = (scores[neg_mask] >= thresh_95).mean()
    else:
        fpr_at_95tpr = -1.0

    results = {
        'linear_probe_overall': float(acc),
        'linear_probe_drone': float(drone_acc),
        'linear_probe_negative': float(neg_acc),
        'fpr_at_95tpr': float(fpr_at_95tpr),
    }

    print(f"  Linear probe: overall={acc:.4f}, drone={drone_acc:.4f}, neg={neg_acc:.4f}")
    print(f"  FPR @ 95% TPR: {fpr_at_95tpr:.4f}")

    return results


# ──────────────────────────────────────────────
# TERTIARY: Centroid Analysis
# ──────────────────────────────────────────────

def centroid_analysis(embeddings, labels, splits):
    """For each holdout type, check: closer to nearest train drone or background?"""
    train_mask = np.array([s == 'train' for s in splits])
    holdout_mask = np.array([s == 'holdout' for s in splits])
    neg_mask = np.array([s == 'negatives' for s in splits])
    labels_arr = np.array(labels)

    # Train drone centroids
    train_types = sorted(set(l for l, m in zip(labels, train_mask) if m and l != 'NEGATIVE'))
    train_centroids = {}
    for t in train_types:
        mask = train_mask & (labels_arr == t)
        train_centroids[t] = embeddings[mask].mean(axis=0)

    # Holdout type centroids
    holdout_types = sorted(set(l for l, m in zip(labels, holdout_mask) if m and l != 'NEGATIVE'))

    # Background centroid
    bg_centroid = embeddings[neg_mask].mean(axis=0) if neg_mask.sum() > 0 else None

    results = {}
    detected = 0
    total = 0

    print(f"  Holdout centroid analysis:")
    for t in holdout_types:
        mask = holdout_mask & (labels_arr == t)
        h_centroid = embeddings[mask].mean(axis=0)

        # Min cosine distance to any train drone centroid
        d_drone = min(
            1 - np.dot(h_centroid, tc) / (np.linalg.norm(h_centroid) * np.linalg.norm(tc) + 1e-8)
            for tc in train_centroids.values()
        )

        # Cosine distance to background
        if bg_centroid is not None:
            d_bg = 1 - np.dot(h_centroid, bg_centroid) / (np.linalg.norm(h_centroid) * np.linalg.norm(bg_centroid) + 1e-8)
        else:
            d_bg = float('inf')

        is_detected = d_bg > d_drone
        detected += int(is_detected)
        total += 1

        results[t] = {
            'd_drone': float(d_drone),
            'd_background': float(d_bg),
            'detected': bool(is_detected),
        }
        status = 'DETECTED' if is_detected else 'MISSED'
        print(f"    {t}: d_drone={d_drone:.4f}, d_bg={d_bg:.4f} -> {status}")

    results['detection_rate'] = float(detected / total) if total > 0 else 0.0
    print(f"  Detection: {detected}/{total} ({results['detection_rate']:.1%})")
    return results


# ──────────────────────────────────────────────
# TERTIARY: Silhouette Score
# ──────────────────────────────────────────────

def compute_silhouette(embeddings, labels, splits):
    """Silhouette score for holdout samples (drone cluster vs background)."""
    binary = np.array([0 if l == 'NEGATIVE' else 1 for l in labels])
    splits_arr = np.array(splits)

    # Only compute on holdout + negatives for clarity
    mask = (splits_arr == 'holdout') | (splits_arr == 'negatives')
    if mask.sum() < 10:
        print("  Silhouette: too few samples")
        return {'silhouette': -2.0}

    X = embeddings[mask]
    y = binary[mask]
    n_classes = len(set(y))

    if n_classes < 2:
        print("  Silhouette: only 1 class present, cannot compute")
        return {'silhouette': -2.0}

    sil = silhouette_score(X, y, metric='cosine')
    print(f"  Silhouette (holdout + negatives): {sil:.4f}")
    return {'silhouette': float(sil)}


# ──────────────────────────────────────────────
# TERTIARY: UMAP Visualization
# ──────────────────────────────────────────────

def umap_plot(embeddings, labels, splits, save_path='umap.png'):
    """2D UMAP: train=blue, holdout=RED, negatives=gray."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("  Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine',
                        n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 9))
    splits_arr = np.array(splits)

    # Negatives (gray, background)
    neg_mask = splits_arr == 'negatives'
    if neg_mask.sum() > 0:
        ax.scatter(coords[neg_mask, 0], coords[neg_mask, 1],
                   c='lightgray', s=10, alpha=0.4, label=f'Background RF ({neg_mask.sum()})',
                   zorder=1)

    # Train (blue)
    train_mask = splits_arr == 'train'
    if train_mask.sum() > 0:
        ax.scatter(coords[train_mask, 0], coords[train_mask, 1],
                   c='steelblue', s=12, alpha=0.5, label=f'Train drones ({train_mask.sum()})',
                   zorder=2)

    # Holdout (RED)
    holdout_mask = splits_arr == 'holdout'
    if holdout_mask.sum() > 0:
        ax.scatter(coords[holdout_mask, 0], coords[holdout_mask, 1],
                   c='red', s=25, alpha=0.85, label=f'HOLDOUT unseen ({holdout_mask.sum()})',
                   edgecolors='darkred', linewidth=0.5, zorder=3)

    ax.legend(loc='best', fontsize=11)
    ax.set_title('IRIS: Zero-Shot Drone Detection via LeJEPA', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1', fontsize=11)
    ax.set_ylabel('UMAP 2', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  UMAP saved to {save_path}")
    plt.close()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def convert_for_json(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(i) for i in obj]
    elif isinstance(obj, (bool,)):
        return bool(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description='IRIS Evaluation Protocol')
    parser.add_argument('--hdf5', required=True, help='Path to HDF5 store')
    parser.add_argument('--checkpoint', required=True, help='Path to best.pt')
    parser.add_argument('--output-dir', default='eval_results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for embedding extraction')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load model
    model = LeJEPA(in_channels=3)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}, best_loss={ckpt.get('best_loss', '?')}")
    print(f"Params: {model.param_count()}")

    # Extract embeddings
    print(f"\nExtracting embeddings from {args.hdf5}...")
    embeddings, labels, splits = extract_all_embeddings(model, args.hdf5, device, args.batch_size)

    unique_labels = set(labels)
    n_drone_types = len(unique_labels - {'NEGATIVE'})
    has_negatives = 'NEGATIVE' in unique_labels
    print(f"\nTotal: {len(embeddings)} embeddings, {n_drone_types} drone types, "
          f"negatives={'yes' if has_negatives else 'NO'}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── PRIMARY: k-NN ──
    print(f"\n{'='*60}")
    print("PRIMARY METRIC: k-NN Accuracy (Zero-Shot)")
    print(f"{'='*60}")
    knn_results = knn_accuracy(embeddings, labels, splits)

    # ── SECONDARY: Linear Probe ──
    print(f"\n{'='*60}")
    print("SECONDARY METRIC: Linear Probe")
    print(f"{'='*60}")
    linear_results = linear_probe(embeddings, labels, splits)

    # ── TERTIARY: Centroid ──
    print(f"\n{'='*60}")
    print("TERTIARY: Centroid Analysis")
    print(f"{'='*60}")
    centroid_results = centroid_analysis(embeddings, labels, splits)

    # ── TERTIARY: Silhouette ──
    print(f"\n{'='*60}")
    print("TERTIARY: Silhouette Score")
    print(f"{'='*60}")
    sil_results = compute_silhouette(embeddings, labels, splits)

    # ── TERTIARY: UMAP ──
    print(f"\n{'='*60}")
    print("TERTIARY: UMAP Visualization")
    print(f"{'='*60}")
    umap_path = os.path.join(args.output_dir, 'umap.png')
    umap_plot(embeddings, labels, splits, save_path=umap_path)

    # ── Save Results ──
    all_results = convert_for_json({
        'knn': knn_results,
        'linear_probe': linear_results,
        'centroid': centroid_results,
        'silhouette': sil_results,
        'meta': {
            'checkpoint_epoch': ckpt.get('epoch'),
            'checkpoint_loss': ckpt.get('best_loss'),
            'n_embeddings': len(embeddings),
            'n_drone_types': n_drone_types,
            'has_negatives': has_negatives,
        }
    })

    results_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    knn_k1 = knn_results.get('knn_k1_overall', 0)
    knn_drone = knn_results.get('knn_k1_drone_recall', 0)
    lp = linear_results.get('linear_probe_overall', 0)
    det = centroid_results.get('detection_rate', 0)
    sil = sil_results.get('silhouette', -2)
    fpr = linear_results.get('fpr_at_95tpr', -1)

    print(f"  k-NN (k=1) overall:       {knn_k1:.1%}")
    print(f"  k-NN (k=1) drone recall:  {knn_drone:.1%}")
    print(f"  Linear probe overall:      {lp:.1%}")
    print(f"  FPR @ 95% TPR:             {fpr:.1%}" if fpr >= 0 else "  FPR @ 95% TPR: N/A")
    print(f"  Holdout detection rate:    {det:.1%}")
    print(f"  Silhouette score:          {sil:.4f}" if sil > -2 else "  Silhouette: N/A")

    print(f"\n  Protocol thresholds:")
    print(f"    k-NN > 70%:  {'PASS' if knn_k1 > 0.70 else 'FAIL'}")
    print(f"    k-NN > 85%:  {'PASS' if knn_k1 > 0.85 else 'NOT YET'}")
    print(f"    Detection > 5/7 types:  {'PASS' if det >= 5/7 else 'FAIL'}")

    print(f"\n  UMAP: {umap_path}")
    print(f"  Results: {results_path}")


if __name__ == '__main__':
    main()