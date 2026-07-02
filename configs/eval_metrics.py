#!/usr/bin/env python3
"""IRIS evaluation metrics — run AFTER training.

Implements the evaluation protocol defined in configs/eval_protocol.json.
DO NOT modify this after training starts.
"""

import numpy as np
import torch
import h5py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
import umap
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.model import LeJEPA
from src.train_dataset import LeJEPAEvalDataset
from torch.utils.data import DataLoader


def extract_all_embeddings(model, hdf5_path, device, batch_size=64):
    """Extract embeddings for all samples in HDF5."""
    model.eval()
    
    embeddings = []
    labels = []
    splits = []  # 'train', 'holdout', or 'negative'
    
    for split in ['train', 'holdout', 'negatives']:
        ds = LeJEPAEvalDataset(hdf5_path, split=split)
        if len(ds) == 0:
            continue
            
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
        with torch.no_grad():
            for tensor, label_idx, label_str in dl:
                tensor = tensor.to(device)
                z = model.encode(tensor)  # (B, embed_dim)
                embeddings.append(z.cpu().numpy())
                labels.extend([str(l) for l in label_str])
                splits.extend([split] * len(label_str))
    
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, labels, splits


def knn_accuracy(embeddings, labels, splits, k_values=[1, 5]):
    """Primary metric: k-NN accuracy on held-out types.
    
    Binary task: drone vs background.
    For each holdout sample, find k nearest neighbors among train+negatives.
    If majority of neighbors are drone type (not negative), count as correct.
    """
    train_mask = np.array([s == 'train' for s in splits])
    holdout_mask = np.array([s == 'holdout' for s in splits])
    
    # Binary labels: drone=1, negative=0
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
        
        # Per-class accuracy
        drone_mask = holdout_y == 1
        neg_mask = holdout_y == 0
        
        drone_acc = accuracy_score(holdout_y[drone_mask], pred[drone_mask]) if drone_mask.sum() > 0 else 0
        neg_acc = accuracy_score(holdout_y[neg_mask], pred[neg_mask]) if neg_mask.sum() > 0 else 0
        
        results[f'knn_k{k}_overall'] = acc
        results[f'knn_k{k}_drone'] = drone_acc
        results[f'knn_k{k}_negative'] = neg_acc
        
        print(f"k-NN (k={k}): overall={acc:.4f}, drone_recall={drone_acc:.4f}, neg_recall={neg_acc:.4f}")
    
    return results


def linear_probe(embeddings, labels, splits):
    """Secondary metric: linear probe on frozen embeddings.
    
    Train LogisticRegression on train embeddings, test on holdout.
    Binary: drone vs background.
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
    
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(train_X, train_y)
    pred = clf.predict(holdout_X)
    acc = accuracy_score(holdout_y, pred)
    
    drone_mask = holdout_y == 1
    neg_mask = holdout_y == 0
    drone_acc = accuracy_score(holdout_y[drone_mask], pred[drone_mask]) if drone_mask.sum() > 0 else 0
    neg_acc = accuracy_score(holdout_y[neg_mask], pred[neg_mask]) if neg_mask.sum() > 0 else 0
    
    print(f"Linear probe: overall={acc:.4f}, drone={drone_acc:.4f}, neg={neg_acc:.4f}")
    
    return {
        'linear_probe_overall': acc,
        'linear_probe_drone': drone_acc,
        'linear_probe_negative': neg_acc,
    }


def centroid_analysis(embeddings, labels, splits):
    """For each holdout type, check: closer to train drones or background?"""
    train_mask = np.array([s == 'train' for s in splits])
    holdout_mask = np.array([s == 'holdout' for s in splits])
    neg_mask = np.array([s == 'negatives' for s in splits])
    
    binary_labels = np.array(labels)
    
    # Train drone centroids
    train_types = set(l for l, m in zip(labels, train_mask) if m and l != 'NEGATIVE')
    train_centroids = {}
    for t in train_types:
        mask = train_mask & (np.array(labels) == t)
        train_centroids[t] = embeddings[mask].mean(axis=0)
    
    # Holdout type centroids
    holdout_types = set(l for l, m in zip(labels, holdout_mask) if m and l != 'NEGATIVE')
    
    # Background centroid
    bg_centroid = embeddings[neg_mask].mean(axis=0) if neg_mask.sum() > 0 else None
    
    results = {}
    detected = 0
    total = 0
    
    for t in sorted(holdout_types):
        mask = holdout_mask & (np.array(labels) == t)
        h_centroid = embeddings[mask].mean(axis=0)
        
        # Min distance to any train drone centroid
        d_drone = min(np.linalg.norm(h_centroid - tc) for tc in train_centroids.values())
        
        # Distance to background
        d_bg = np.linalg.norm(h_centroid - bg_centroid) if bg_centroid is not None else float('inf')
        
        is_detected = d_bg > d_drone
        detected += int(is_detected)
        total += 1
        
        results[t] = {
            'd_drone': float(d_drone),
            'd_background': float(d_bg),
            'detected': is_detected,
        }
        print(f"  {t}: d_drone={d_drone:.3f}, d_bg={d_bg:.3f} -> {'DETECTED' if is_detected else 'MISSED'}")
    
    results['detection_rate'] = detected / total if total > 0 else 0
    print(f"\nHoldout detection: {detected}/{total} ({results['detection_rate']:.1%})")
    return results


def umap_plot(embeddings, labels, splits, save_path='umap.png'):
    """2D UMAP visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    coords = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    splits_arr = np.array(splits)
    
    # Plot negatives first (gray, background)
    neg_mask = splits_arr == 'negatives'
    ax.scatter(coords[neg_mask, 0], coords[neg_mask, 1], 
               c='lightgray', s=10, alpha=0.5, label='Background RF', zorder=1)
    
    # Plot train (blue)
    train_mask = splits_arr == 'train'
    ax.scatter(coords[train_mask, 0], coords[train_mask, 1],
               c='steelblue', s=15, alpha=0.6, label='Train drones (30 types)', zorder=2)
    
    # Plot holdout (RED, prominent)
    holdout_mask = splits_arr == 'holdout'
    ax.scatter(coords[holdout_mask, 0], coords[holdout_mask, 1],
               c='red', s=30, alpha=0.9, label='HOLDOUT drones (7 unseen)', 
               edgecolors='darkred', linewidth=0.5, zorder=3)
    
    ax.legend(loc='best', fontsize=10)
    ax.set_title('IRIS: Drone-ness is Learnable — Zero-Shot Detection of Unseen Drones', fontsize=12)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"UMAP saved to {save_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5', required=True)
    parser.add_argument('--checkpoint', required=True, help='Path to best.pt')
    parser.add_argument('--output-dir', default='eval_results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = LeJEPA(in_channels=3)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, loss={ckpt.get('best_loss', '?')}")
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings, labels, splits = extract_all_embeddings(model, args.hdf5, device)
    print(f"Total: {len(embeddings)} embeddings, {len(set(labels))} unique labels")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run full evaluation protocol
    print("\n" + "="*60)
    print("PRIMARY METRIC: k-NN Accuracy (Zero-Shot)")
    print("="*60)
    knn_results = knn_accuracy(embeddings, labels, splits)
    
    print("\n" + "="*60)
    print("SECONDARY METRIC: Linear Probe")
    print("="*60)
    linear_results = linear_probe(embeddings, labels, splits)
    
    print("\n" + "="*60)
    print("TERTIARY METRICS: Centroid Analysis")
    print("="*60)
    centroid_results = centroid_analysis(embeddings, labels, splits)
    
    print("\n" + "="*60)
    print("UMAP Visualization")
    print("="*60)
    umap_plot(embeddings, labels, splits, 
              save_path=os.path.join(args.output_dir, 'umap.png'))
    
    # Save all results
    all_results = {
        'knn': knn_results,
        'linear_probe': linear_results,
        'centroid': centroid_results,
        'checkpoint_epoch': ckpt.get('epoch'),
        'checkpoint_loss': ckpt.get('best_loss'),
        'n_embeddings': len(embeddings),
        'n_labels': len(set(labels)),
    }
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        return obj
    
    results_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    knn_k1 = knn_results.get('knn_k1_overall', 0)
    print(f"k-NN (k=1) accuracy:      {knn_k1:.1%}")
    print(f"  Threshold (>70%):        {'PASS' if knn_k1 > 0.70 else 'FAIL'}")
    print(f"  Stretch goal (>85%):     {'PASS' if knn_k1 > 0.85 else 'NOT YET'}")
    
    lp_acc = linear_results.get('linear_probe_overall', 0)
    print(f"Linear probe accuracy:     {lp_acc:.1%}")
    
    det_rate = centroid_results.get('detection_rate', 0)
    print(f"Holdout detection rate:    {det_rate:.1%}")
    
    print(f"\nProtocol: configs/eval_protocol.json")
    print(f"UMAP plot: {os.path.join(args.output_dir, 'umap.png')}")


if __name__ == '__main__':
    main()