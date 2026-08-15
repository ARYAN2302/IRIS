"""Modal launcher: Zenodo drone RF → SCF image HDF5.

Reads all 12 .bin files from /raw_iq/ on the iris-cuas-data volume,
computes SCF images (2,256,256) per 4096-sample IQ trace, and writes
the combined HDF5 + manifest to /data/ on the same volume.

Usage:
    python /home/z/my-project/scripts/spawn_zenodo_scf.py
"""
import modal, os

app = modal.App("iris-cuas-zenodo-scf")

DATA_VOL = modal.Volume.from_name("iris-cuas-data", create_if_missing=True)
RESULTS_VOL = modal.Volume.from_name("iris-cuas-results", create_if_missing=True)

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install("libgl1", "libglib2.0-0", "libhdf5-dev", "python3", "python3-pip", "python-is-python3")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "h5py==3.12.1",
                 "numpy==1.26.4", "scikit-learn==1.6.1", "scipy==1.14.1")
)

# The core SCF logic as a string we mount into the container
SCF_CORE = '''
"""SCF core: read .bin files from volume, compute SCF images, write HDF5."""
import os, json, time
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve

# ---------- config ----------
RAW_DIR = "/raw_iq"
OUT_DIR = "/data"
H5_PATH = os.path.join(OUT_DIR, "zenodo_scf_samples_v2.h5")
MANIFEST_PATH = os.path.join(OUT_DIR, "zenodo_scf_manifest_v2.json")

TRACES_PER_FILE = 1250
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


def iq_to_scf_image(iq, out_size=256, n_fft=1<<12, alpha_max=0.5, window_len=128, n_alpha=128):
    """SCF + COH image (matches train_rf_scf_core.iq_to_scf_image exactly)."""
    z = np.asarray(iq, dtype=np.complex128)
    N = len(z)
    if N < n_fft:
        z = np.concatenate([z, np.zeros(n_fft - N, dtype=z.dtype)])
    else:
        z = z[:n_fft]
    N = n_fft
    z = z * np.hanning(N)
    X = np.fft.fftshift(np.fft.fft(z))
    n_alpha = n_alpha
    alphas = np.linspace(0.0, alpha_max, n_alpha)
    n_freq = max(N // window_len, 1)
    win = np.hanning(window_len)
    SCF = np.zeros((n_alpha, n_freq), dtype=np.complex128)
    Sx = np.abs(X) ** 2
    for i, a in enumerate(alphas):
        shift = int(round(a * N / 2.0))
        scf_slice = np.roll(X, -shift) * np.conj(np.roll(X, shift))
        SCF[i, :] = np.convolve(scf_slice, win, mode="same")[::window_len][:n_freq]
    SCF[0, :] = 0
    eps = 1e-12 * (Sx.max() + 1e-30)
    COH = np.zeros((n_alpha, n_freq), dtype=np.float64)
    for i, a in enumerate(alphas):
        shift = int(round(a * len(X) / 2.0))
        Splus = np.convolve(np.roll(Sx, -shift), win, mode="same")[::window_len][:n_freq]
        Sminus = np.convolve(np.roll(Sx, shift), win, mode="same")[::window_len][:n_freq]
        denom = np.sqrt(Splus * Sminus) + eps
        COH[i, :] = np.abs(SCF[i, :]) / denom
    COH = np.clip(COH, 0.0, 1.0)
    ch0 = np.log10(np.abs(SCF) + 1e-12).astype(np.float64)
    ch1 = COH.astype(np.float64)
    img = np.stack([ch0, ch1], axis=0)
    C, H, W = img.shape
    if H != out_size or W != out_size:
        t = torch.from_numpy(img).float().unsqueeze(0)
        t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
        img = t.squeeze(0).numpy()
    for c in range(img.shape[0]):
        mu, sd = img[c].mean(), img[c].std() + 1e-8
        img[c] = (img[c] - mu) / sd
    return img.astype(np.float32)


def memmap_iq(path):
    sz = os.path.getsize(path)
    n_complex = sz // 4
    arr = np.memmap(path, dtype="<i2", mode="r", shape=(n_complex * 2,))
    return arr, n_complex


def get_trace(iq_mm, trace_idx, trace_len=TRACE_LEN):
    s = trace_idx * trace_len
    e = s + trace_len
    raw = np.asarray(iq_mm[2*s:2*e], dtype=np.float64)
    return raw.view(np.complex128)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(H5_PATH):
        os.remove(H5_PATH)

    print("="*70, flush=True)
    print("Zenodo Drone RF → SCF (running in Modal)", flush=True)
    print("="*70, flush=True)
    print(f"Input:  {RAW_DIR}", flush=True)
    print(f"Output: {H5_PATH}", flush=True)
    print(f"Traces per file: {TRACES_PER_FILE}  (trace length: {TRACE_LEN})", flush=True)
    print(f"Total target samples: {TRACES_PER_FILE * len(DRONE_MAP)}", flush=True)
    print(f"Image shape: {IMG_SHAPE}  dtype: {DTYPE}", flush=True)
    print(flush=True)

    drone_types = sorted(set(v[0] for v in DRONE_MAP.values()))
    type_to_idx = {t: i for i, t in enumerate(drone_types)}
    print(f"Drone types ({len(drone_types)}):", flush=True)
    for t, i in type_to_idx.items():
        print(f"  [{i}] {t}", flush=True)
    print(flush=True)

    h5 = h5py.File(H5_PATH, "w")
    total_expected = TRACES_PER_FILE * len(DRONE_MAP)
    chunk_shape = (TRACES_PER_FILE,) + IMG_SHAPE
    imgs_dset = h5.create_dataset(
        "images",
        shape=(total_expected,) + IMG_SHAPE,
        dtype=DTYPE,
        chunks=chunk_shape,
        compression=None,
    )
    labels_dset = h5.create_dataset("labels", shape=(total_expected,), dtype=np.int32)
    types_dset = h5.create_dataset("types", shape=(total_expected,), dtype="S32")
    sources_dset = h5.create_dataset("sources", shape=(total_expected,), dtype="S64")
    bands_dset = h5.create_dataset("bands", shape=(total_expected,), dtype="S8")

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
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            print(f"MISSING: {fname}", flush=True)
            continue

        print(f"\\n--- [{cursor//TRACES_PER_FILE + 1}/{len(DRONE_MAP)}] {fname} ---", flush=True)
        print(f"  Drone: {drone_type}   Band: {band}", flush=True)
        t0 = time.time()
        iq_mm, n_total = memmap_iq(path)
        print(f"  Memmap'd {n_total:,} complex samples in {time.time()-t0:.2f}s", flush=True)

        n_possible = n_total // TRACE_LEN
        rng = np.random.RandomState(hash(fname) & 0xFFFF)
        trace_ids = sorted(rng.choice(n_possible, size=TRACES_PER_FILE, replace=False))

        batch_imgs = np.empty((TRACES_PER_FILE,) + IMG_SHAPE, dtype=DTYPE)
        batch_labels = np.empty(TRACES_PER_FILE, dtype=np.int32)
        batch_types = np.empty(TRACES_PER_FILE, dtype="S32")
        batch_sources = np.empty(TRACES_PER_FILE, dtype="S64")
        batch_bands = np.empty(TRACES_PER_FILE, dtype="S8")

        t0 = time.time()
        for i, tid in enumerate(trace_ids):
            iq_trace = get_trace(iq_mm, tid, TRACE_LEN)
            img = iq_to_scf_image(iq_trace)
            batch_imgs[i] = img.astype(DTYPE)
            batch_labels[i] = type_to_idx[drone_type]
            batch_types[i] = drone_type.encode("utf-8")
            batch_sources[i] = fname.encode("utf-8")
            batch_bands[i] = band.encode("utf-8")

            if (i+1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i+1) / elapsed
                eta = (TRACES_PER_FILE - i - 1) / rate
                print(f"    SCF {i+1}/{TRACES_PER_FILE}  elapsed={elapsed:.1f}s  "
                      f"rate={rate:.1f}/s  eta={eta:.1f}s", flush=True)

        imgs_dset[cursor:cursor+TRACES_PER_FILE] = batch_imgs
        labels_dset[cursor:cursor+TRACES_PER_FILE] = batch_labels
        types_dset[cursor:cursor+TRACES_PER_FILE] = batch_types
        sources_dset[cursor:cursor+TRACES_PER_FILE] = batch_sources
        bands_dset[cursor:cursor+TRACES_PER_FILE] = batch_bands
        cursor += TRACES_PER_FILE

        del iq_mm, batch_imgs, batch_labels, batch_types, batch_sources, batch_bands
        dt = time.time() - t0
        print(f"  Computed {TRACES_PER_FILE} SCF images in {dt:.1f}s "
              f"({TRACES_PER_FILE/dt:.1f} img/s)", flush=True)

        type_counts[drone_type] += TRACES_PER_FILE
        file_stats.append({
            "file": fname,
            "drone_type": drone_type,
            "band": band,
            "n_traces": TRACES_PER_FILE,
            "n_samples_in_file": int(n_total),
            "compute_time_s": round(dt, 1),
        })

        h5.flush()
        # Commit volume so progress is persisted
        try:
            DATA_VOL.commit()
        except Exception:
            pass
        print(f"  HDF5 size so far: {os.path.getsize(H5_PATH)/1e6:.1f} MB", flush=True)

    if cursor < total_expected:
        h5["images"].resize((cursor,) + IMG_SHAPE)
        h5["labels"].resize((cursor,))
        h5["types"].resize((cursor,))
        h5["sources"].resize((cursor,))
        h5["bands"].resize((cursor,))

    h5.attrs["n_samples"] = cursor
    h5.close()

    total_dt = time.time() - t_start
    print(f"\\n{'='*70}", flush=True)
    print(f"DONE — {cursor} SCF samples in {total_dt:.1f}s ({cursor/total_dt:.1f} img/s)", flush=True)
    print(f"HDF5: {H5_PATH}  ({os.path.getsize(H5_PATH)/1e6:.1f} MB)", flush=True)

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
    print(f"Manifest: {MANIFEST_PATH}", flush=True)

    print(f"\\nPer drone type counts:", flush=True)
    for t, c in type_counts.items():
        print(f"  {t:30s} {c:>5} samples", flush=True)
    print(f"\\nTotal: {cursor} samples (target was {total_expected})", flush=True)

    return {
        "n_samples": cursor,
        "h5_path": H5_PATH,
        "h5_size_mb": os.path.getsize(H5_PATH) / 1e6,
        "manifest_path": MANIFEST_PATH,
        "compute_time_s": round(total_dt, 1),
        "per_type_counts": type_counts,
    }
'''

# Write the core module to a temp file and add to image
CORE_PATH = "/tmp/zenodo_scf_core.py"
with open(CORE_PATH, "w") as f:
    f.write(SCF_CORE)

IMAGE = IMAGE.add_local_file(CORE_PATH, "/root/zenodo_scf_core.py")


@app.function(
    image=IMAGE,
    gpu="T4",
    volumes={"/data": DATA_VOL},
    timeout=3600,
    memory=16384,
)
def run_zenodo_scf():
    import sys, os
    sys.path.insert(0, "/root")
    # Both /raw_iq and output live under the same volume mount /data
    # Patch the module paths accordingly before importing
    from zenodo_scf_core import main
    # Monkey-patch the paths
    import zenodo_scf_core as m
    m.RAW_DIR = "/data/raw_iq"
    m.OUT_DIR = "/data"
    m.H5_PATH = "/data/zenodo_scf_samples_v2.h5"
    m.MANIFEST_PATH = "/data/zenodo_scf_manifest_v2.json"
    return main()


if __name__ == "__main__":
    with app.run(detach=True):
        fc = run_zenodo_scf.spawn()
        print(f"SPAWNED: {fc.object_id}")
        print(f"App: {app.name}")
        print(f"Track: https://modal.com/logs/call/{fc.object_id}")
        print("="*60)
        print("To check status:")
        print(f"  python -m modal call {fc.object_id}")
