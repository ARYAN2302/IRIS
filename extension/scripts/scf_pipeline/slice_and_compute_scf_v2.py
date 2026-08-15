#!/usr/bin/env python3
"""Memory-efficient Zenodo drone RF → SCF image generator.

Memory optimizations vs v1:
  1. Memory-map the .bin file (no full load into RAM)
  2. Process traces one-at-a-time (no big batch in memory)
  3. Write SCF images incrementally to HDF5 (no accumulation in RAM)
  4. Use float32 throughout
  5. Use HDF5 chunking for efficient incremental writes

Output:
  /home/z/my-project/data/processed/zenodo_scf/scf_samples.h5
  /home/z/my-project/data/processed/zenodo_scf/manifest.json
"""
import os, sys, json, time
import numpy as np
import h5py
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/z/my-project/IRIS_repo/extension/scripts")
from train_rf_scf_core import iq_to_scf_image

# ---------- config ----------
SRC_DIR = "/home/z/my-project/data/sources/zenodo_4264467"
OUT_DIR = "/home/z/my-project/data/processed/zenodo_scf"
H5_PATH = os.path.join(OUT_DIR, "scf_samples.h5")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")

TRACES_PER_FILE = 500
TRACE_LEN = 4096
IMG_SHAPE = (2, 256, 256)
DTYPE = np.float32

DRONE_MAP = {
    "DJI_matrice_100_2G.bin":         ("DJI Matrice 100",       "2.4GHz"),
    "DJI_matrice_210_2G.bin":         ("DJI Matrice 210",       "2.4GHz"),
    "DJI_inspire_2_2G.bin":           ("DJI Inspire 2",         "2.4GHz"),
    "DJI_mavic_pro_2G.bin":           ("DJI Mavic Pro",         "2.4GHz"),
    "DJI_mavic_mini_2G.bin":          ("DJI Mavic Mini",        "2.4GHz"),
    "DJI_phantom_4_2G.bin":           ("DJI Phantom 4",         "2.4GHz"),
    "DJI_phantom_4_pro_plus_2G.bin":  ("DJI Phantom 4 Pro+",    "2.4GHz"),
    "Parrot_disco_2G.bin":            ("Parrot Disco",          "2.4GHz"),
    "Parrot_mambo_control_2G.bin":    ("Parrot Mambo (ctrl)",   "2.4GHz"),
    "Parrot_mambo_video_2G.bin":      ("Parrot Mambo (video)",  "2.4GHz"),
    "Yuneec_typhoon_h_2G_1of2.bin":   ("Yuneec Typhoon H",      "2.4GHz"),
    "Yuneec_typhoon_h_5G.bin":        ("Yuneec Typhoon H",      "5.8GHz"),
}

SOURCE_INFO = {
    "name": "Zenodo 4264467 — Radio-Frequency Control and Video Signal Recordings of Drones",
    "url": "https://zenodo.org/records/4264467",
    "doi": "10.5281/zenodo.4264467",
    "author": "Pärlin, Karel",
    "year": 2020,
    "license": "CC-BY 4.0",
    "format": "interleaved int16 LE IQ, 4 bytes/complex",
    "sample_rate": "120 MSps (2.4GHz) / 200 MSps (5.8GHz)",
    "citation": "Pärlin, K. (2020). Radio-Frequency Control and Video Signal "
                "Recordings of Drones [Data set]. Zenodo. "
                "https://doi.org/10.5281/zenodo.4264467",
}

def memmap_iq(path):
    """Memory-map a Zenodo .bin file as complex128 IQ."""
    sz = os.path.getsize(path)
    n_complex = sz // 4
    # memmap as int16 with shape (n_complex*2,)
    arr = np.memmap(path, dtype="<i2", mode="r", shape=(n_complex * 2,))
    # Build a view as complex128 — but we can't memmap directly as complex128
    # because int16 → complex128 needs a type cast (4 bytes → 16 bytes).
    # So we read chunks of int16 and convert on-the-fly.
    return arr, n_complex

def get_trace(iq_memmap, n_complex, trace_idx, trace_len=TRACE_LEN):
    """Read one 4096-sample trace from the memmapped file as complex128."""
    s = trace_idx * trace_len
    e = s + trace_len
    # Read 2*trace_len int16 values, convert to complex128
    raw = np.asarray(iq_memmap[2*s:2*e], dtype=np.float64)  # copy out of memmap
    return raw.view(np.complex128)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Remove old H5 if exists
    if os.path.exists(H5_PATH):
        os.remove(H5_PATH)

    print("="*70)
    print("Zenodo Drone RF → SCF Image Generator (memory-efficient v2)")
    print("="*70)
    print(f"Source: {SRC_DIR}")
    print(f"Output: {H5_PATH}")
    print(f"Traces per file: {TRACES_PER_FILE}  (trace length: {TRACE_LEN})")
    print(f"Total target samples: {TRACES_PER_FILE * len(DRONE_MAP)}")
    print(f"Image shape: {IMG_SHAPE}  dtype: {DTYPE}")
    print()

    # Build label mapping
    drone_types = sorted(set(v[0] for v in DRONE_MAP.values()))
    type_to_idx = {t: i for i, t in enumerate(drone_types)}
    print(f"Drone types ({len(drone_types)}):")
    for t, i in type_to_idx.items():
        print(f"  [{i}] {t}")
    print()

    # Create HDF5 with chunked datasets for incremental writes
    h5 = h5py.File(H5_PATH, "w")
    total_expected = TRACES_PER_FILE * len(DRONE_MAP)

    # Use chunking: 500 images per chunk = full file per chunk
    # Disable gzip during writes (apply at end if needed) for speed
    chunk_shape = (TRACES_PER_FILE,) + IMG_SHAPE
    imgs_dset = h5.create_dataset(
        "images",
        shape=(total_expected,) + IMG_SHAPE,
        dtype=DTYPE,
        chunks=chunk_shape,
        compression=None,  # no compression during write for speed
    )
    labels_dset = h5.create_dataset("labels", shape=(total_expected,), dtype=np.int32)
    types_dset = h5.create_dataset(
        "types", shape=(total_expected,), dtype="S32",
    )
    sources_dset = h5.create_dataset(
        "sources", shape=(total_expected,), dtype="S64",
    )
    bands_dset = h5.create_dataset(
        "bands", shape=(total_expected,), dtype="S8",
    )

    # Write metadata attributes
    h5.attrs["source"] = SOURCE_INFO["name"]
    h5.attrs["source_url"] = SOURCE_INFO["url"]
    h5.attrs["source_doi"] = SOURCE_INFO["doi"]
    h5.attrs["license"] = SOURCE_INFO["license"]
    h5.attrs["citation"] = SOURCE_INFO["citation"]
    h5.attrs["n_samples_target"] = total_expected
    h5.attrs["trace_length"] = TRACE_LEN
    h5.attrs["image_shape"] = list(IMG_SHAPE)
    h5.attrs["drone_types"] = drone_types
    h5.attrs["created_utc"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    cursor = 0
    file_stats = []
    type_counts = {t: 0 for t in drone_types}
    t_start = time.time()

    for fname, (drone_type, band) in DRONE_MAP.items():
        path = os.path.join(SRC_DIR, fname)
        if not os.path.exists(path):
            print(f"MISSING: {fname}")
            continue

        print(f"\n--- [{cursor//TRACES_PER_FILE + 1}/{len(DRONE_MAP)}] {fname} ---")
        print(f"  Drone: {drone_type}   Band: {band}")
        t0 = time.time()
        iq_mm, n_total = memmap_iq(path)
        print(f"  Memmap'd {n_total:,} complex samples in {time.time()-t0:.2f}s")

        # Pick random non-overlapping trace indices
        n_possible = n_total // TRACE_LEN
        rng = np.random.RandomState(hash(fname) & 0xFFFF)
        trace_ids = sorted(rng.choice(n_possible, size=TRACES_PER_FILE, replace=False))

        # Pre-allocate batch buffer
        batch_imgs = np.empty((TRACES_PER_FILE,) + IMG_SHAPE, dtype=DTYPE)
        batch_labels = np.empty(TRACES_PER_FILE, dtype=np.int32)
        batch_types = np.empty(TRACES_PER_FILE, dtype="S32")
        batch_sources = np.empty(TRACES_PER_FILE, dtype="S64")
        batch_bands = np.empty(TRACES_PER_FILE, dtype="S8")

        t0 = time.time()
        for i, tid in enumerate(trace_ids):
            iq_trace = get_trace(iq_mm, n_total, tid, TRACE_LEN)
            img = iq_to_scf_image(iq_trace)  # (2, 256, 256) float32
            batch_imgs[i] = img.astype(DTYPE)
            batch_labels[i] = type_to_idx[drone_type]
            batch_types[i] = drone_type.encode("utf-8")
            batch_sources[i] = fname.encode("utf-8")
            batch_bands[i] = band.encode("utf-8")

            if (i+1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i+1) / elapsed
                eta = (TRACES_PER_FILE - i - 1) / rate
                mem = os.popen("free -m | awk '/^Mem:/ {print $3\"MB used / \"$2\"MB total\"}'").read().strip()
                print(f"    SCF {i+1}/{TRACES_PER_FILE}  elapsed={elapsed:.1f}s  "
                      f"rate={rate:.1f}/s  eta={eta:.1f}s  mem={mem}", flush=True)

        # Write entire batch to HDF5 at once (much faster than per-image)
        imgs_dset[cursor:cursor+TRACES_PER_FILE] = batch_imgs
        labels_dset[cursor:cursor+TRACES_PER_FILE] = batch_labels
        types_dset[cursor:cursor+TRACES_PER_FILE] = batch_types
        sources_dset[cursor:cursor+TRACES_PER_FILE] = batch_sources
        bands_dset[cursor:cursor+TRACES_PER_FILE] = batch_bands
        cursor += TRACES_PER_FILE

        # Free memmap and batch buffers
        del iq_mm, batch_imgs, batch_labels, batch_types, batch_sources, batch_bands
        dt = time.time() - t0
        print(f"  Computed {TRACES_PER_FILE} SCF images in {dt:.1f}s "
              f"({TRACES_PER_FILE/dt:.1f} img/s)")

        type_counts[drone_type] += TRACES_PER_FILE
        file_stats.append({
            "file": fname,
            "drone_type": drone_type,
            "band": band,
            "n_traces": TRACES_PER_FILE,
            "n_samples_in_file": int(n_total),
            "compute_time_s": round(dt, 1),
        })

        # Flush HDF5 to disk
        h5.flush()

        # Show running disk size
        sz = os.path.getsize(H5_PATH) / 1e6
        print(f"  HDF5 size so far: {sz:.1f} MB")

    # Truncate to actual cursor (in case some files were missing)
    if cursor < total_expected:
        h5["images"].resize((cursor,) + IMG_SHAPE)
        h5["labels"].resize((cursor,))
        h5["types"].resize((cursor,))
        h5["sources"].resize((cursor,))
        h5["bands"].resize((cursor,))

    h5.attrs["n_samples"] = cursor
    h5.close()

    total_dt = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"DONE — {cursor} SCF samples in {total_dt:.1f}s ({cursor/total_dt:.1f} img/s)")
    print(f"HDF5: {H5_PATH}  ({os.path.getsize(H5_PATH)/1e6:.1f} MB)")

    # Write manifest
    manifest = {
        "source": SOURCE_INFO,
        "total_samples": cursor,
        "trace_length": TRACE_LEN,
        "image_shape": list(IMG_SHAPE),
        "image_dtype": str(DTYPE),
        "channels": {
            "0": "log10(|SCF| + eps), normalized per-channel (zero mean, unit std)",
            "1": "|COH| (spectral coherence) in [0,1], normalized per-channel",
        },
        "drone_types": drone_types,
        "type_to_label": type_to_idx,
        "per_file": file_stats,
        "per_type_counts": type_counts,
        "compute_time_s": round(total_dt, 1),
        "created_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {MANIFEST_PATH}")

    print(f"\nPer drone type counts:")
    for t, c in type_counts.items():
        print(f"  {t:30s} {c:>5} samples")
    print(f"\nTotal: {cursor} samples (target was {total_expected})")

if __name__ == "__main__":
    main()
