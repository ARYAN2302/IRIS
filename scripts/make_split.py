#!/usr/bin/env python3
"""Decide train/holdout split for IRIS zero-shot experiment.

CRITICAL: This script is run ONCE before training. The split is committed
to JSON and never modified. This ensures evaluation integrity.

Strategy:
  - RFUAV has 37 drone types across train+valid
  - We select 7 types as HOLDOUT (never seen during training)
  - Selection is STRATIFIED by drone category (multirotor/fixed-wing/hybrid)
  - All images of those 7 types → /holdout/ in HDF5
  - All images of remaining 30 types → /train/ in HDF5
  - Negatives stay in /negatives/

Usage:
    # Dry run (see the split without modifying HDF5):
    python scripts/make_split.py --hdf5 data/processed/iris.h5 --dry-run

    # Commit the split:
    python scripts/make_split.py --hdf5 data/processed/iris.h5

    # Force redo (DANGEROUS after training has started):
    python scripts/make_split.py --hdf5 data/processed/iris.h5 --force
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import h5py
import numpy as np

SPLIT_JSON = 'configs/split.json'


def categorize_drone_type(type_name):
    """Categorize drone type based on name heuristics.
    
    RFUAV has 37 types. We need broad categories to ensure
    the holdout set isn't all one physical class.
    """
    name_lower = type_name.lower()
    
    # Fixed-wing indicators
    fixed_wing = ['plane', 'wing', 'glider', 'disco', 'surveillance']
    # Multirotor indicators  
    multirotor = ['quad', 'hex', 'octo', 'copter', 'phantom', 'mavic', 
                  'bebop', 'parrot', 'dji', 'yuneec', 'syma', 'hubsan',
                  'walkera', 'blade', 'arris', 'holy']
    # Hybrid / VTOL
    hybrid = ['vtol', 'hybrid', 'convertible', 'tilt']
    
    if any(kw in name_lower for kw in fixed_wing):
        return 'fixed_wing'
    elif any(kw in name_lower for kw in hybrid):
        return 'hybrid'
    elif any(kw in name_lower for kw in multirotor):
        return 'multirotor'
    else:
        return 'unknown'


def scan_hdf5_types(hdf5_path):
    """Scan HDF5 and return {drone_type: count} for all types in /train/."""
    types = Counter()
    with h5py.File(hdf5_path, 'r') as f:
        if 'train' not in f:
            print("ERROR: No /train/ group in HDF5. Run ingest first.")
            sys.exit(1)
        for dtype in f['train']:
            types[dtype] = len(f['train'][dtype])
    return types


def count_negatives(hdf5_path):
    """Count negative samples in HDF5."""
    with h5py.File(hdf5_path, 'r') as f:
        if 'negatives' in f:
            return len(f['negatives'])
    return 0


def select_holdout_types(type_counts, holdout_count=7, min_samples=100, seed=42):
    """Select holdout types with CATEGORY STRATIFICATION.
    
    Ensures holdout set represents all drone categories present in data.
    This prevents the zero-shot test from being trivially impossible
    (e.g., all training = multirotors, all holdout = fixed-wing).
    """
    rng = np.random.RandomState(seed)

    # Filter eligible types
    eligible = [(t, c) for t, c in type_counts.items() if c >= min_samples]
    eligible.sort()

    print(f"Eligible types (>={min_samples} samples): {len(eligible)}/{len(type_counts)}")

    # Categorize eligible types
    categories = {}
    for t, c in eligible:
        cat = categorize_drone_type(t)
        categories.setdefault(cat, []).append((t, c))
        print(f"  {t}: {c} images -> {cat}")

    print(f"\nCategory distribution:")
    for cat, types in sorted(categories.items()):
        print(f"  {cat}: {len(types)} types ({sum(c for _, c in types)} images)")

    if len(eligible) < holdout_count:
        print(f"WARNING: Only {len(eligible)} eligible types, reducing holdout")
        holdout_count = len(eligible)

    # Stratified selection: pick from each category proportionally
    holdout_per_cat = {}
    remaining = holdout_count

    total_types = len(eligible)
    for cat, types in sorted(categories.items()):
        proportion = len(types) / total_types
        n = max(1, round(proportion * holdout_count))
        n = min(n, len(types))
        holdout_per_cat[cat] = n
        remaining -= n

    # Adjust if we over/under-allocated
    if remaining > 0:
        cats_by_size = sorted(categories.keys(), key=lambda c: len(categories[c]), reverse=True)
        for cat in cats_by_size:
            if remaining <= 0:
                break
            if holdout_per_cat[cat] < len(categories[cat]):
                holdout_per_cat[cat] += 1
                remaining -= 1
    elif remaining < 0:
        cats_by_size = sorted(categories.keys(), key=lambda c: holdout_per_cat.get(c, 0), reverse=True)
        for cat in cats_by_size:
            if remaining >= 0:
                break
            if holdout_per_cat[cat] > 1:
                holdout_per_cat[cat] -= 1
                remaining += 1

    # Select random types from each category
    holdout = []
    print(f"\nHoldout selection (total={holdout_count}):")
    for cat, n in sorted(holdout_per_cat.items()):
        cat_types = categories[cat]
        indices = rng.choice(len(cat_types), size=n, replace=False)
        selected = [cat_types[i][0] for i in indices]
        holdout.extend(selected)
        print(f"  {cat}: holding out {n}/{len(cat_types)} -> {selected}")

    holdout.sort()
    train = sorted([t for t, _ in eligible if t not in holdout])

    # Types with too few samples -> train (can't be holdout)
    too_few = [(t, c) for t, c in type_counts.items() if c < min_samples]
    for t, c in too_few:
        train.append(t)
        print(f"  NOTE: {t} has only {c} samples -- kept in train (too few for holdout)")

    # Print stratification verification
    print(f"\nStratification verification:")
    for cat in sorted(set(categories.keys())):
        n_train = sum(1 for t in train if categorize_drone_type(t) == cat)
        n_hold = sum(1 for t in holdout if categorize_drone_type(t) == cat)
        n_total = sum(1 for t in type_counts if categorize_drone_type(t) == cat)
        print(f"  {cat}: total={n_total}, train={n_train}, holdout={n_hold}")

    return train, holdout


def apply_split(hdf5_path, train_types, holdout_types):
    """Reorganize HDF5: move holdout types from /train/ to /holdout/."""
    with h5py.File(hdf5_path, 'a') as f:
        # Create holdout group
        if 'holdout' in f:
            del f['holdout']
        holdout_grp = f.create_group('holdout')

        moved = 0
        for dtype in holdout_types:
            if dtype not in f['train']:
                print(f"  WARNING: {dtype} not found in /train/, skipping")
                continue

            # Copy all samples from /train/{dtype} to /holdout/{dtype}
            src_grp = f['train'][dtype]
            dst_grp = holdout_grp.create_group(dtype)
            for key in list(src_grp.keys()):
                src_grp.copy(key, dst_grp)
                moved += 1

            # Delete from /train/
            del f['train'][dtype]
            print(f"  Moved {dtype}: {len(list(dst_grp.keys()))} samples -> /holdout/")

        print(f"\nTotal moved: {moved} samples")


def main():
    parser = argparse.ArgumentParser(description='Create train/holdout split for IRIS')
    parser.add_argument('--hdf5', required=True, help='Path to HDF5 store')
    parser.add_argument('--holdout-count', type=int, default=7,
                        help='Number of types to hold out (default: 7)')
    parser.add_argument('--min-samples', type=int, default=100,
                        help='Min samples required for a type to be eligible for holdout')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show split without modifying HDF5')
    parser.add_argument('--force', action='store_true',
                        help='Force re-split even if split.json exists (DANGEROUS after training)')
    args = parser.parse_args()

    # Check if split already exists
    if os.path.exists(SPLIT_JSON) and not args.force:
        with open(SPLIT_JSON) as f:
            existing = json.load(f)
        print(f"Split already exists in {SPLIT_JSON}!")
        print(f"  Train types: {len(existing['train_types'])}")
        print(f"  Holdout types: {existing['holdout_types']}")
        print(f"  Created: {existing.get('created_at', 'unknown')}")
        if not args.dry_run:
            print("\nUse --force to overwrite (NOT recommended after training has started)")
        return

    # Scan current data
    type_counts = scan_hdf5_types(args.hdf5)
    n_neg = count_negatives(args.hdf5)

    print(f"\n{'='*60}")
    print(f"IRIS Split Decision")
    print(f"{'='*60}")
    print(f"Total drone types: {len(type_counts)}")
    print(f"Total drone images: {sum(type_counts.values())}")
    print(f"Negative samples: {n_neg}")
    print(f"Target: {len(type_counts) - args.holdout_count} train / {args.holdout_count} holdout")
    print()

    # Select split
    train_types, holdout_types = select_holdout_types(
        type_counts, args.holdout_count, args.min_samples, args.seed
    )

    # Count samples in each split
    train_count = sum(type_counts[t] for t in train_types)
    holdout_count = sum(type_counts[t] for t in holdout_types)

    print(f"\n{'='*60}")
    print(f"PROPOSED SPLIT:")
    print(f"{'='*60}")
    print(f"\nTRAIN ({len(train_types)} types, {train_count} images):")
    for t in train_types:
        print(f"  {t}: {type_counts[t]} [{categorize_drone_type(t)}]")
    print(f"\nHOLDOUT ({len(holdout_types)} types, {holdout_count} images) -- UNSEEN:")
    for t in holdout_types:
        print(f"  {t}: {type_counts[t]} [{categorize_drone_type(t)}]")
    print(f"\nNEGATIVES: {n_neg} background RF samples")

    if args.dry_run:
        print(f"\n[DRY RUN] No changes made. Remove --dry-run to commit this split.")
        return

    # Confirm
    print(f"\n{'!'*60}")
    print(f"WARNING: This will PERMANENTLY reorganize the HDF5 file.")
    print(f"Holdout types will be moved from /train/ to /holdout/.")
    print(f"This decision should NOT be changed after training begins.")
    print(f"{'!'*60}")

    response = input("\nType 'COMMIT' to proceed: ")
    if response != 'COMMIT':
        print("Aborted. No changes made.")
        return

    # Apply split
    print("\nApplying split...")
    apply_split(args.hdf5, train_types, holdout_types)

    # Save split.json
    split_info = {
        'train_types': train_types,
        'holdout_types': holdout_types,
        'train_count': train_count,
        'holdout_count': holdout_count,
        'negative_count': n_neg,
        'seed': args.seed,
        'min_samples': args.min_samples,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hdf5_path': os.path.abspath(args.hdf5),
        'train_categories': {t: categorize_drone_type(t) for t in train_types},
        'holdout_categories': {t: categorize_drone_type(t) for t in holdout_types},
    }

    os.makedirs(os.path.dirname(SPLIT_JSON) or '.', exist_ok=True)
    with open(SPLIT_JSON, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"\nSplit committed to {SPLIT_JSON}")

    # Print final stats
    print(f"\n{'='*60}")
    print(f"FINAL HDF5 STATS:")
    print(f"{'='*60}")
    with h5py.File(args.hdf5, 'r') as f:
        for group in ['train', 'holdout', 'negatives']:
            if group in f:
                if group == 'negatives':
                    n = len(f['negatives'])
                else:
                    n = sum(len(f[group][t]) for t in f[group])
                print(f"  /{group}/: {n} samples")
    print("\nSplit complete. You may now begin training.")


if __name__ == '__main__':
    main()