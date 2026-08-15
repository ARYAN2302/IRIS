#!/usr/bin/env python3
"""Download MISSING Zenodo 4264467 drone RF recordings.

Already present in Modal volume (6 files): DJI Mavic Pro/Mini, Phantom 4,
Parrot Disco/Mambo control/Mambo video — all at 2.4 GHz.

This script downloads the 6 MISSING files representing 5 NEW drone classes
plus a 5.8 GHz variant for frequency diversity.

Source: https://zenodo.org/records/4264467
Format: interleaved int16 LE IQ, ~120M complex samples per file (~480MB)
License: CC-BY 4.0
Citation: Pärlin, K. (2020). Radio-Frequency Control and Video Signal
          Recordings of Drones. Zenodo. https://doi.org/10.5281/zenodo.4264467
"""
import os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

DEST = "/home/z/my-project/data/sources/zenodo_4264467"
BASE = "https://zenodo.org/api/records/4264467/files"

# Missing files: 5 new drone types + 1 5.8GHz variant
TARGETS = [
    "DJI_matrice_100_2G.bin",          # NEW drone type: DJI Matrice 100
    "DJI_matrice_210_2G.bin",          # NEW drone type: DJI Matrice 210
    "DJI_inspire_2_2G.bin",            # NEW drone type: DJI Inspire 2
    "DJI_phantom_4_pro_plus_2G.bin",   # NEW drone type: Phantom 4 Pro+ (vs Phantom 4)
    "Yuneec_typhoon_h_2G_1of2.bin",    # NEW drone type: Yuneec Typhoon H (2.4GHz)
    "Yuneec_typhoon_h_5G.bin",         # 5.8 GHz variant of Yuneec Typhoon H
]

def download_one(fname, max_retries=5, chunk=1<<20):
    url = f"{BASE}/{fname}/content"
    dst = os.path.join(DEST, fname)
    expected_min = 400_000_000  # 5.8GHz files are 400MB, 2.4GHz are 480MB

    if os.path.exists(dst) and os.path.getsize(dst) >= expected_min:
        return fname, os.path.getsize(dst), "SKIP-CACHED"

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            with urllib.request.urlopen(req, timeout=180) as r:
                tmp = dst + ".part"
                got = 0
                t0 = time.time()
                last_print = t0
                with open(tmp, "wb") as f:
                    while True:
                        buf = r.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        got += len(buf)
                        now = time.time()
                        if now - last_print > 15:
                            rate = got / max(now - t0, 1)
                            print(f"  [{fname}] {got/1e6:.0f}/{((int(r.headers.get('Content-Length','0')) or got)/1e6):.0f} MB @ {rate/1e6:.1f} MB/s", flush=True)
                            last_print = now
                os.replace(tmp, dst)
                rate = got / max(time.time() - t0, 1)
                return fname, got, f"OK {rate/1e6:.1f} MB/s"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = str(e)
            print(f"  [{fname}] attempt {attempt} failed: {last_err}", flush=True)
            time.sleep(3 * attempt)
    return fname, 0, f"FAIL: {last_err}"

def main():
    os.makedirs(DEST, exist_ok=True)
    print(f"Downloading {len(TARGETS)} MISSING files to {DEST}")
    print(f"Expected total: ~2.8 GB")
    print()

    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(download_one, f): f for f in TARGETS}
        for fut in as_completed(futs):
            fname, size, status = fut.result()
            print(f"  [{size/1e6:6.1f} MB] {fname:42s} -> {status}", flush=True)

    # Final inventory
    print("\n=== Final Inventory ===")
    total = 0
    for f in sorted(os.listdir(DEST)):
        if not f.endswith(".bin"):
            continue
        sz = os.path.getsize(os.path.join(DEST, f))
        total += sz
        print(f"  {f:42s} {sz/1e6:7.1f} MB")
    print(f"\nTotal: {total/1e9:.2f} GB")

if __name__ == "__main__":
    main()
