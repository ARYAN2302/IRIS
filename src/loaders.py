"""
Dataset loaders for IRIS.
Each loader takes a file path and returns a 1D complex64 numpy array.
That's it. The STFT engine handles everything after.

Three formats cover all 5 datasets:
  1. InterleavedBinary  → RFUAV, CDRF
  2. MatSeparateIQ      → DRFF-R2
  3. MatComplex         → DroneDetect, DroneRF
"""

import numpy as np
import h5py
from pathlib import Path
import xml.etree.ElementTree as ET


class InterleavedBinaryLoader:
    """
    Loads interleaved fp32 I/Q data from flat binary files.
    
    On-disk: I₀(f32), Q₀(f32), I₁(f32), Q₁(f32), ...
    Output:  1D complex64 numpy array
    
    Used by: RFUAV (.dat files), CDRF (.dat files)
    """
    
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
    
    def load(self, max_samples: int | None = None) -> np.ndarray:
        raw = np.fromfile(self.file_path, dtype=np.float32)
        
        if max_samples is not None:
            raw = raw[:max_samples * 2]
        
        # De-interleave: [I0,Q0,I1,Q1,...] → [[I0,Q0],[I1,Q1],...]
        iq = raw.reshape(-1, 2)
        
        # Form complex
        return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
    
    def load_xml_metadata(self) -> dict:
        """Parse companion XML metadata (RFUAV specific)."""
        xml_path = self.file_path.with_suffix('.xml')
        if not xml_path.exists():
            return {}
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        meta = {}
        for child in root:
            try:
                meta[child.tag] = float(child.text)
            except (ValueError, TypeError):
                meta[child.tag] = child.text
        return meta


class MatSeparateIQLoader:
    """
    Loads I/Q from .mat files where I and Q are stored as separate arrays.
    
    On-disk: HDF5/MATLAB .mat v7.3 with keys RF0_I, RF0_Q
    Output:  1D complex64 numpy array
    
    Used by: DRFF-R2
    """
    
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
    
    def load(self, max_samples: int | None = None) -> np.ndarray:
        with h5py.File(self.file_path, 'r') as f:
            # DRFF-R2 style: RF0_I + RF0_Q
            if 'RF0_I' in f and 'RF0_Q' in f:
                I = np.array(f['RF0_I']).flatten()
                Q = np.array(f['RF0_Q']).flatten()
            
            # Fallback: scan for pair of equal-length 1D real arrays
            else:
                I, Q = self._find_iq_pair(f)
        
        if max_samples is not None:
            I = I[:max_samples]
            Q = Q[:max_samples]
        
        return (I + 1j * Q).astype(np.complex64)
    
    def load_metadata(self) -> dict:
        """Load scalar metadata from .mat file."""
        meta = {}
        scalar_keys = ['Fs', 'CenterFrequence', 'Gain',
                       'State', 'Distance', 'Height', 'FlightMode']
        with h5py.File(self.file_path, 'r') as f:
            for key in scalar_keys:
                if key in f:
                    val = np.array(f[key])
                    if val.ndim == 0:
                        meta[key] = float(val)
                    else:
                        meta[key] = val.flatten().tolist()
        return meta
    
    def _find_iq_pair(self, f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
        """Fallback: find two 1D real arrays of equal length."""
        arrays = {}
        for key in f.keys():
            arr = np.array(f[key])
            if arr.ndim == 1 and arr.dtype in (np.float32, np.float64):
                arrays[key] = arr
        
        keys = list(arrays.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if arrays[keys[i]].shape == arrays[keys[j]].shape:
                    return arrays[keys[i]], arrays[keys[j]]
        
        raise ValueError(
            f"No I/Q pair found in {self.file_path}. "
            f"Available keys: {list(f.keys())}"
        )


class MatComplexLoader:
    """
    Loads complex I/Q from .mat files where data is already stored
    as complex MATLAB arrays.
    
    On-disk: .mat v7.3 with complex-valued variable(s)
    Output:  1D complex64 numpy array
    
    Used by: DroneRF, DroneDetect
    """
    
    def __init__(self, file_path: str | Path, var_name: str | None = None):
        self.file_path = Path(file_path)
        self.var_name = var_name  # None = auto-detect
    
    def load(self, max_samples: int | None = None) -> np.ndarray:
        with h5py.File(self.file_path, 'r') as f:
            # If variable name specified, use it
            if self.var_name and self.var_name in f:
                data = np.array(f[self.var_name]).flatten()
            else:
                # Auto-detect: find first 1D complex or (2, N) real array
                data = self._auto_detect(f)
        
        if max_samples is not None:
            data = data[:max_samples]
        
        # Ensure complex64
        if data.dtype in (np.complex64, np.complex128):
            return data.astype(np.complex64)
        elif data.ndim == 1 and data.dtype in (np.float32, np.float64):
            # Might be real-valued, return as-is (shouldn't happen for I/Q)
            return data.astype(np.float32)
        
        return data.astype(np.complex64)
    
    def _auto_detect(self, f: h5py.File) -> np.ndarray:
        """Find the I/Q data array in the .mat file."""
        for key in f.keys():
            arr = np.array(f[key])
            
            # Complex 1D array — perfect
            if arr.ndim == 1 and np.issubdtype(arr.dtype, np.complexfloating):
                return arr
            
            # (2, N) real array — row 0 = I, row 1 = Q
            if arr.ndim == 2 and arr.shape[0] == 2:
                return arr[0] + 1j * arr[1]
        
        raise ValueError(
            f"No I/Q data found in {self.file_path}. "
            f"Keys: {list(f.keys())}, "
            f"Shapes/dtypes: {[(k, f[k].shape, f[k].dtype) for k in f.keys()]}"
        )


# ──────────────────────────────────────────────
# Auto-detect loader from file extension
# ──────────────────────────────────────────────

def load_iq(file_path: str | Path, max_samples: int | None = None) -> np.ndarray:
    """
    Automatically pick the right loader and return complex64 I/Q array.
    
    Heuristics:
      .mat + has RF0_I key  → MatSeparateIQ
      .mat + otherwise      → MatComplex
      .bin/.dat             → InterleavedBinary
      .npy                  → numpy load
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    
    if ext == '.mat':
        # Peek inside to decide which loader
        with h5py.File(path, 'r') as f:
            keys = set(f.keys())
        
        if 'RF0_I' in keys and 'RF0_Q' in keys:
            return MatSeparateIQLoader(path).load(max_samples)
        else:
            return MatComplexLoader(path).load(max_samples)
    
    elif ext in ('.bin', '.dat'):
        return InterleavedBinaryLoader(path).load(max_samples)
    
    elif ext == '.npy':
        arr = np.load(path)
        if arr.dtype in (np.complex64, np.complex128):
            return arr.astype(np.complex64)[:max_samples]
        raise ValueError(f"Unexpected dtype in .npy: {arr.dtype}")
    
    else:
        # Try binary as last resort
        return InterleavedBinaryLoader(path).load(max_samples)