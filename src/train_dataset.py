"""LeJEPA training and evaluation datasets.

Handles mixed-channel inputs:
  - RFUAV RGB spectrograms: (3, H, W) from HDF5
  - DroneRF background STFT: (2, H, W) → zero-padded to (3, H, W)

Training dataset returns (x1, x2, is_positive):
  - Drone pairs: two augmented views from same type, is_positive=1
  - Negatives: single augmented view duplicated, is_positive=0
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class LeJEPATrainDataset(Dataset):
    """Generates positive pairs from drone spectrograms + loads negatives.

    Positive pairs: two augmented views from same drone type.
    Negatives: single views from background RF, is_positive=0.
    """

    def __init__(self, hdf5_path, pair_distance=5, noise_std=0.05, freq_shift_bins=5):
        """
        Args:
            hdf5_path: path to HDF5 store
            pair_distance: unused (kept for CLI compat), pairing is by type
            noise_std: std of Gaussian noise augmentation
            freq_shift_bins: max frequency shift in bins
        """
        self.hdf5_path = hdf5_path
        self.pair_distance = pair_distance
        self.noise_std = noise_std
        self.freq_shift_bins = freq_shift_bins

        # Scan HDF5 to build index
        self.samples = []  # list of (hdf5_path_str, drone_type, is_negative)
        with h5py.File(hdf5_path, 'r') as f:
            # Drone samples from /train/
            if 'train' in f:
                for dtype in f['train']:
                    group = f['train'][dtype]
                    for key in group:
                        self.samples.append((f'train/{dtype}/{key}', dtype, False))

            # Negatives from /negatives/
            if 'negatives' in f:
                for key in f['negatives']:
                    self.samples.append((f'negatives/{key}', 'NEGATIVE', True))

        # Separate indices
        self.drone_indices = [i for i, s in enumerate(self.samples) if not s[2]]
        self.neg_indices = [i for i, s in enumerate(self.samples) if s[2]]

        # Group drone samples by type for positive pair generation
        self.type_indices = {}
        for i, (path, dtype, is_neg) in enumerate(self.samples):
            if not is_neg:
                self.type_indices.setdefault(dtype, []).append(i)

        print(f"TrainDataset: {len(self.drone_indices)} drone samples, "
              f"{len(self.neg_indices)} negatives, "
              f"{len(self.type_indices)} drone types")

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, idx):
        """Load a sample from HDF5, normalize, ensure 3 channels."""
        path, dtype, is_neg = self.samples[idx]

        with h5py.File(self.hdf5_path, 'r') as f:
            data = f[path][()]

        # data shape: (C, H, W) where C=2 (STFT) or C=3 (RGB spectrogram)
        tensor = torch.from_numpy(data).float()

        # Normalize to [0, 1] if needed (RGB spectrograms might be 0-255)
        if tensor.max() > 1.5:
            tensor = tensor / 255.0

        # Per-sample normalization (zero mean, unit variance)
        for ch in range(tensor.shape[0]):
            mu = tensor[ch].mean()
            std = tensor[ch].std()
            if std > 1e-8:
                tensor[ch] = (tensor[ch] - mu) / std
            else:
                tensor[ch] = tensor[ch] - mu

        # Zero-pad 2-channel to 3-channel
        if tensor.shape[0] == 2:
            pad = torch.zeros(1, tensor.shape[1], tensor.shape[2])
            tensor = torch.cat([tensor, pad], dim=0)

        return tensor

    def _augment(self, x):
        """Apply augmentations: Gaussian noise + frequency shift."""
        # Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # Random frequency shift (circular shift along freq axis)
        if self.freq_shift_bins > 0:
            shift = torch.randint(-self.freq_shift_bins, self.freq_shift_bins + 1, (1,)).item()
            x = torch.roll(x, shifts=shift, dims=1)

        return x

    def __getitem__(self, idx):
        path, dtype, is_neg = self.samples[idx]

        if is_neg:
            # Negatives: single view, duplicated for compatible batch shape
            x = self._load_sample(idx)
            x = self._augment(x)
            return x, x, torch.tensor(0.0)

        # Drone: generate positive pair from same type
        x1 = self._load_sample(idx)
        x1 = self._augment(x1)

        # Find a neighbor from same drone type
        same_type = self.type_indices[dtype]
        if len(same_type) > 1:
            choices = [i for i in same_type if i != idx]
            if choices:
                neighbor_idx = int(np.random.choice(choices))
            else:
                neighbor_idx = idx
        else:
            neighbor_idx = idx

        x2 = self._load_sample(neighbor_idx)
        x2 = self._augment(x2)

        return x1, x2, torch.tensor(1.0)


class LeJEPAEvalDataset(Dataset):
    """Evaluation dataset: single spectrograms with labels, no pairing/augmentation."""

    def __init__(self, hdf5_path, split='all'):
        """
        Args:
            hdf5_path: path to HDF5 store
            split: 'train', 'holdout', 'negatives', or 'all'
        """
        self.hdf5_path = hdf5_path
        self.samples = []  # list of (path, label)

        with h5py.File(hdf5_path, 'r') as f:
            groups_to_load = []
            if split in ('train', 'all') and 'train' in f:
                groups_to_load.append(('train', f['train']))
            if split in ('holdout', 'all') and 'holdout' in f:
                groups_to_load.append(('holdout', f['holdout']))
            if split in ('negatives', 'all') and 'negatives' in f:
                groups_to_load.append(('negatives', f['negatives']))

            for group_name, group in groups_to_load:
                if group_name == 'negatives':
                    for key in group:
                        self.samples.append((f'negatives/{key}', 'NEGATIVE'))
                else:
                    for dtype in group:
                        for key in group[dtype]:
                            self.samples.append((f'{group_name}/{dtype}/{key}', dtype))

        # Build label index
        self.unique_labels = sorted(set(label for _, label in self.samples))
        self.label_to_idx = {l: i for i, l in enumerate(self.unique_labels)}

        print(f"EvalDataset ({split}): {len(self.samples)} samples, "
              f"{len(self.unique_labels)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with h5py.File(self.hdf5_path, 'r') as f:
            data = f[path][()]

        tensor = torch.from_numpy(data).float()

        # Normalize to [0, 1] if needed
        if tensor.max() > 1.5:
            tensor = tensor / 255.0

        # Per-sample normalization
        for ch in range(tensor.shape[0]):
            mu = tensor[ch].mean()
            std = tensor[ch].std()
            if std > 1e-8:
                tensor[ch] = (tensor[ch] - mu) / std
            else:
                tensor[ch] = tensor[ch] - mu

        # Zero-pad 2-channel to 3-channel
        if tensor.shape[0] == 2:
            pad = torch.zeros(1, tensor.shape[1], tensor.shape[2])
            tensor = torch.cat([tensor, pad], dim=0)

        label_idx = self.label_to_idx[label]
        return tensor, label_idx, label