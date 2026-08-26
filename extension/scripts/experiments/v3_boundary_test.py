#!/usr/bin/env python3
"""
v3 Boundary Test — does frozen SCF v3 accept or reject non-OFDM signals?

We CLAIM v3 is OFDM-only but never measured it. This experiment measures the
actual boundary: synthesize FHSS-GFSK, FM-analog, WiFi-like OFDM, CP-OFDM,
and noise — push all through the EXACT same SCF |COH| pipeline as training —
score with frozen encoder + Mahalanobis centroid refit on real Zenodo SCF.

Verdict matrix:
  - CP-OFDM synth should be DETECTED (sanity check)
  - WiFi-like OFDM should be REJECTED (matches 0% BG FP claim)
  - FHSS / FM → unknown until now. Accept = more universal than claimed.
    Reject = boundary confirmed and quantified.

Self-contained (inline SCF + encoder + Mahalanobis), batched eval (no OOM),
saves encoder-independent results JSON early.

Run: python3 extension/scripts/experiments/v3_boundary_test.py   (spawns detached)
"""
import modal

app = modal.App("iris-v3-boundary")
DATA_VOL = modal.Volume.from_name("iris-cuas-data")
MODELS_VOL = modal.Volume.from_name("iris-cuas-models")
RESULTS_VOL = modal.Volume.from_name("iris-cuas-results")
IMAGE = (
    modal.Image.debian_slim()
    .pip_install("torch==2.5.1", "numpy==1.26.4", "h5py==3.12.1",
                 "scikit-learn==1.6.1", "scipy==1.14.1", "tqdm==4.67.1")
)

CORE = r'''
import os, sys, json, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*70)
print("=== v3 Boundary Test: frozen SCF detector vs non-OFDM signals ===")
print(f"device={device}")
print("="*70, flush=True)

# ---------------- SCF pipeline (exact copy of scf_features.py) ----------------
def _to_complex(iq):
    x = np.asarray(iq)
    if np.iscomplexobj(x):
        return x.astype(np.complex128)
    if x.ndim == 2 and x.shape[0] == 2:
        return x[0].astype(np.complex128) + 1j * x[1].astype(np.complex128)
    if x.ndim == 2 and x.shape[1] == 2:
        return x[:, 0].astype(np.complex128) + 1j * x[:, 1].astype(np.complex128)
    return x.astype(np.complex128)

def scf_frequency_smoothing(iq, n_fft=1<<14, n_alpha=256, alpha_max=0.5, window_len=128):
    z = _to_complex(iq)
    N = len(z)
    if N < n_fft:
        z = np.concatenate([z, np.zeros(n_fft - N, dtype=z.dtype)])
    else:
        z = z[:n_fft]
    N = n_fft
    z = z * np.hanning(N)
    X = np.fft.fftshift(np.fft.fft(z))
    alphas = np.linspace(0.0, alpha_max, n_alpha)
    n_freq = max(N // window_len, 1)
    win = np.hanning(window_len)
    SCF = np.zeros((n_alpha, n_freq), dtype=np.complex128)
    for i, a in enumerate(alphas):
        shift = int(round(a * N / 2.0))
        scf_slice = np.roll(X, -shift) * np.conj(np.roll(X, shift))
        SCF[i, :] = np.convolve(scf_slice, win, mode="same")[::window_len][:n_freq]
    SCF[0, :] = 0
    return SCF, alphas

def spectral_coherence(SCF, iq, n_fft=1<<14, window_len=128):
    z = _to_complex(iq)
    N = len(z)
    if N < n_fft:
        z = np.concatenate([z, np.zeros(n_fft - N, dtype=z.dtype)])
    else:
        z = z[:n_fft]
    z = z * np.hanning(len(z))
    X = np.fft.fftshift(np.fft.fft(z))
    Sx = np.abs(X) ** 2
    n_alpha, n_freq = SCF.shape
    alphas = np.linspace(0.0, 0.5, n_alpha)
    win = np.hanning(window_len)
    eps = 1e-12 * (Sx.max() + 1e-30)
    COH = np.zeros((n_alpha, n_freq), dtype=np.float64)
    for i, a in enumerate(alphas):
        shift = int(round(a * len(X) / 2.0))
        Splus = np.convolve(np.roll(Sx, -shift), win, mode="same")[::window_len][:n_freq]
        Sminus = np.convolve(np.roll(Sx, shift), win, mode="same")[::window_len][:n_freq]
        denom = np.sqrt(Splus * Sminus) + eps
        COH[i, :] = np.abs(SCF[i, :]) / denom
    return np.clip(COH, 0.0, 1.0)

def iq_to_scf_image(iq, out_size=256):
    z = _to_complex(iq)
    SCF, alphas = scf_frequency_smoothing(z, n_fft=1<<14, n_alpha=out_size)
    COH = spectral_coherence(SCF, z)
    ch0 = np.log10(np.abs(SCF) + 1e-12).astype(np.float64)
    ch1 = COH.astype(np.float64)
    img = np.stack([ch0, ch1], axis=0)
    for c in range(img.shape[0]):
        mu, sd = img[c].mean(), img[c].std() + 1e-8
        img[c] = (img[c] - mu) / sd
    return img.astype(np.float32)

# ---------------- Signal synthesizers (non-OFDM + controls) ----------------
def synth_cp_ofdm(n=16384, n_sc=64, cp_len=16, seed=0):
    """CP-OFDM — DJI-like control. Should be DETECTED."""
    rng = np.random.RandomState(seed)
    out = []
    while sum(len(x) for x in out) < n:
        sym = rng.randn(n_sc) + 1j*rng.randn(n_sc)          # QAM-ish symbols
        wave = np.fft.ifft(sym, n=n_sc)
        wave = np.concatenate([wave[-cp_len:], wave])        # cyclic prefix
        out.append(wave)
    z = np.concatenate(out)[:n]
    return 0.3 * z / (np.abs(z).max() + 1e-9) + 0.01*(rng.randn(n)+1j*rng.randn(n))

def synth_wifi_ofdm(n=16384, n_sc=64, cp_len=8, seed=0):
    """Bursty OFDM packets with idle gaps — WiFi-like. Should be REJECTED (BG)."""
    rng = np.random.RandomState(seed)
    out = np.zeros(n, dtype=np.complex128)
    pos = 0
    while pos < n:
        burst_len = rng.randint(500, 2000)
        if pos + burst_len >= n: break
        sym_n = burst_len // (n_sc + cp_len)
        chunk = []
        for _ in range(max(sym_n,1)):
            s = rng.randn(n_sc) + 1j*rng.randn(n_sc)
            w = np.fft.ifft(s, n=n_sc)
            chunk.append(np.concatenate([w[-8:], w]))
        sig = np.concatenate(chunk)[:burst_len]
        out[pos:pos+burst_len] = 0.3 * sig / (np.abs(sig).max()+1e-9)
        pos += burst_len + rng.randint(200, 1500)  # idle gap
    return out + 0.01*(rng.randn(n)+1j*rng.randn(n))

def synth_fhss_gfsk(n=16384, n_hops=10, hop_bw=0.15, seed=0):
    """FHSS GFSK — ELRS/Crossfire/FrSky-like. THE QUESTION."""
    rng = np.random.RandomState(seed)
    hop_len = n // n_hops
    out = np.zeros(n, dtype=np.complex128)
    centers = np.linspace(-0.35, 0.35, 25)
    t_phase = 0.0
    for h in range(n_hops):
        fc = centers[rng.randint(len(centers))]              # random hop center
        seg = np.zeros(hop_len, dtype=np.complex128)
        bt = 2048                                            # samples per symbol
        bits = rng.randint(0, 2, hop_len // bt + 2)
        # gaussian filter bits -> freq deviation (GFSK)
        kern = np.exp(-0.5*(np.linspace(-2,2,129)/40)**2); kern/=kern.sum()
        fdev = np.convolve(np.repeat(bits, bt), kern, mode="same")[:hop_len] - 0.5
        phase = 2*np.pi*np.cumsum(fc + hop_bw*fdev)/1.0
        seg = 0.5*np.exp(1j*phase)
        out[h*hop_len:(h+1)*hop_len] = seg
    return out + 0.01*(rng.randn(n)+1j*rng.randn(n))

def synth_fm_video(n=16384, seed=0):
    """Wideband FM analog video — 5.8GHz VTX-like with line-rate sync. THE QUESTION."""
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    video = 0.5*rng.randn(n)                                # luma noise
    # PAL line-rate structure: 15.625 kHz sync pulses (normalized to window: line period in samples)
    line_period = max(int(n / 40), 50)                       # ~40 lines across capture
    for start in range(0, n, line_period):
        video[start:start+max(line_period//20,3)] = 2.0      # sync pulse
    # FM: integrate video -> phase
    phase = 2*np.pi*0.1*t/n + np.cumsum(video)*0.002
    return 0.5*np.exp(1j*phase)

# ---------------- Encoder (exact backbone.py CNNEncoder) ----------------
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
            oc = min(width * (2 ** (i // 2)), 512)
            layers.append(ConvBlock(ch, oc)); layers.append(nn.MaxPool2d(2)); ch = oc
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            out = self.conv(torch.zeros(1, in_ch, 256, 256))
            flat = out.numel() // out.shape[0]
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, embed_dim), nn.BatchNorm1d(embed_dim))
    def forward(self, x): return self.head(self.conv(x))

def fit_mahal(embs, reg=1e-3):
    n = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs = embs / n
    c = embs.mean(0); D = embs.shape[1]
    cov = np.cov(embs.T) + reg*np.eye(D)
    try: ci = np.linalg.inv(cov)
    except: ci = np.linalg.pinv(cov)
    return c, ci

def mahal_l2(embs, c, ci):
    n = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    e = embs / n
    d = e - c
    return np.sqrt(np.maximum((d @ ci * d).sum(1), 0))

def encode_batched(enc, X, bs=32):
    enc.eval(); outs=[]
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(enc(torch.from_numpy(X[i:i+bs]).float().to(device)).cpu().numpy())
    return np.concatenate(outs)

# ---------------- Main ----------------
def main(seed=42):
    torch.manual_seed(seed); np.random.seed(seed)

    # 1. Load frozen encoder from volume
    ckpts = glob.glob("/models/rf_scf_real_v3_encoder_seed42.pt")
    if not ckpts:
        raise SystemExit("frozen v3 checkpoint not found on /models")
    print(f"[1] Loading {ckpts[0]}", flush=True)
    ckpt = torch.load(ckpts[0], map_location="cpu")
    # v3 checkpoints are saved as {"encoder": state_dict} by spawn_train_v3_vicreg
    if "encoder" in ckpt and isinstance(ckpt["encoder"], dict):
        sd = ckpt["encoder"]
    else:
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    enc = CNNEncoder(in_ch=2, embed_dim=256)
    clean = {}
    for k, v in sd.items():
        k2 = k[len("encoder."):] if k.startswith("encoder.") else k
        if not isinstance(v, torch.Tensor):
            continue
        clean[k2] = v
    missing, unexpected = enc.load_state_dict(clean, strict=False)
    real_missing = [m for m in missing]
    print(f"  matched={len(clean)} missing={len(real_missing)} unexpected={len(unexpected)}", flush=True)
    if len(real_missing) > 0:
        raise RuntimeError(f"CHECKPOINT LOAD FAILED — missing keys: {real_missing[:5]}... "
                           f"(available ckpt keys: {list(sd.keys())[:5]})")
    enc.to(device).eval()

    # 2. Refit centroid on REAL training SCF from volume
    print("[2] Loading real Zenodo train SCF for centroid...", flush=True)
    h5p = None
    for cand in ["/data/zenodo_scf_samples_v2.h5", "/data/zenodo_scf_samples.h5"]:
        if os.path.exists(cand): h5p = cand; break
    assert h5p, "zenodo scf h5 not found"
    import h5py
    with h5py.File(h5p,"r") as f:
        def walk(g, pre=""):
            ks=[]
            for k in g.keys():
                it=g[k]
                if hasattr(it,"keys"): ks += walk(it, pre+k+"/")
                else: ks.append(pre+k)
            return ks
        keys = walk(f)
        print(f"  h5 keys sample: {keys[:8]}", flush=True)
        # find dataset containing (N,2,256,256)-ish
        arr=None
        for k in keys:
            a = f[k]
            if a.ndim==4 and a.shape[-1]==256 and a.shape[-2]==256 and a.shape[1] in (1,2):
                arr = a; print(f"  using {k} shape {a.shape}", flush=True); break
        assert arr is not None, "no SCF image dataset found"
        n_fit = min(arr.shape[0], 3000)
        idx = np.linspace(0, arr.shape[0]-1, n_fit).astype(int)
        Xfit = arr[idx]
        if Xfit.shape[1]==1: Xfit = np.repeat(Xfit,2,axis=1)
    Xfit = Xfit.astype(np.float32)
    print(f"  Xfit stats: min={Xfit.min():.3f} max={Xfit.max():.3f} mean={Xfit.mean():.4f} std={Xfit.std():.4f}", flush=True)
    assert Xfit.std() > 1e-4, "training SCF samples are constant — wrong dataset?"
    Zfit = encode_batched(enc, Xfit)
    zstd = float(Zfit.std(axis=0).mean())
    print(f"  Zfit embed std/dim: {zstd:.4f}", flush=True)
    assert zstd > 1e-4, f"encoder outputs constant embeddings (std={zstd}) — checkpoint load broken?"
    centroid, cov_inv = fit_mahal(Zfit)
    d_fit = mahal_l2(Zfit, centroid, cov_inv)
    thr99 = float(np.percentile(d_fit, 99))
    thr999 = float(np.percentile(d_fit, 99.9))
    print(f"  centroid fit on {len(Zfit)} real SCF | dist mean {d_fit.mean():.2f} | thr99={thr99:.2f} thr99.9={thr999:.2f}", flush=True)

    # also score real DRFF if present (positive cross-dataset control)
    results = {"threshold_99p": thr99, "threshold_99_9p": thr999}

    # 3. Synthesize signal classes → SCF images
    print("[3] Synthesizing test classes...", flush=True)
    classes = {
        "cp_ofdm_synthetic (should DETECT)": [synth_cp_ofdm(seed=s) for s in range(30)],
        "wifi_bursty_ofdm (should REJECT)":  [synth_wifi_ofdm(seed=s) for s in range(30)],
        "fhss_gfsk_fast_10hops":             [synth_fhss_gfsk(seed=s, n_hops=10) for s in range(30)],
        "fhss_gfsk_slow_4hops":              [synth_fhss_gfsk(seed=s, n_hops=4) for s in range(30)],
        "fhss_gfsk_wideband":                [synth_fhss_gfsk(seed=s, n_hops=6, hop_bw=0.3) for s in range(30)],
        "fm_analog_video":                   [synth_fm_video(seed=s) for s in range(30)],
    }
    verdicts = {}
    for name, iqs in classes.items():
        imgs = np.stack([iq_to_scf_image(iq) for iq in iqs])
        Z = encode_batched(enc, imgs)
        d = mahal_l2(Z, centroid, cov_inv)
        det99 = float((d <= thr99).mean())
        det999 = float((d <= thr999).mean())
        verdicts[name] = {
            "det_99p": det99, "det_99_9p": det999,
            "dist_mean": float(d.mean()), "dist_p50": float(np.median(d)),
            "verdict": "DETECTED(drone-like)" if det99 > 0.5 else ("BORDERLINE" if det99 > 0.1 else "REJECTED(bg-like)")
        }
        print(f"  {name}: det99={det99:.2%} det99.9={det999:.2%} dist_mean={d.mean():.1f} → {verdicts[name]['verdict']}", flush=True)

    # 4. Real-data positive control if DRFF present (as raw IQ? drffr2.h5 likely SCF already)
    try:
        drff_cands = glob.glob("/data/**/drffr2.h5", recursive=True)
        if drff_cands:
            with h5py.File(drff_cands[0],"r") as f:
                def walk(g, pre=""):
                    ks=[]
                    for k in g.keys():
                        it=g[k]
                        if hasattr(it,"keys"): ks += walk(it, pre+k+"/")
                        else: ks.append(pre+k)
                    return ks
                ks = walk(f)
                for k in ks:
                    a = f[k]
                    if a.ndim==4 and a.shape[-1]==256 and a.shape[1] in (1,2):
                        Xd = a[:200]
                        if Xd.shape[1]==1: Xd = np.repeat(Xd,2,axis=1)
                        Zd = encode_batched(enc, Xd.astype(np.float32))
                        dd = mahal_l2(Zd, centroid, cov_inv)
                        det = float((dd <= thr99).mean())
                        results["real_drff_control"] = {"det_99p": det, "dist_mean": float(dd.mean())}
                        print(f"  REAL DRFF-R2 positive control: det99={det:.2%} dist_mean={dd.mean():.1f}", flush=True)
                        break
    except Exception as e:
        print(f"  DRFF control skipped: {e}", flush=True)

    results["synthetic_verdicts"] = verdicts
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    results["conclusion"] = (
        "v3 accepts non-OFDM → more universal than theory predicted" 
        if verdicts["fhss_gfsk_fast_10hops"]["det_99p"] > 0.5 or verdicts["fm_analog_video"]["det_99p"] > 0.5
        else "v3 rejects non-OFDM → OFDM-family boundary CONFIRMED by measurement"
    )
    with open("/results/v3_boundary_test.json","w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*70)
    print(json.dumps({k:v for k,v in results.items() if k!="synthetic_verdicts"}, indent=2))
    for k,v in verdicts.items():
        print(f"  {k}: {v['verdict']} ({v['det_99p']:.0%})")
    print("="*70)
    print("Saved /results/v3_boundary_test.json")

main()
'''

CORE_PATH = "/tmp/v3_boundary_core.py"
with open(CORE_PATH, "w") as f:
    f.write(CORE)

IMAGE = IMAGE.add_local_file(CORE_PATH, "/root/v3_boundary_core.py")

@app.function(image=IMAGE, gpu="T4",
              volumes={"/data": DATA_VOL, "/models": MODELS_VOL, "/results": RESULTS_VOL},
              timeout=3600, memory=32768)
def run_test():
    import subprocess
    r = subprocess.run(["python3", "/root/v3_boundary_core.py"], capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise RuntimeError(r.stderr[-2000:])
    return "done"

if __name__ == "__main__":
    with app.run(detach=True):
        fc = run_test.spawn()
        print(f"SPAWNED detached: {fc.object_id}")
        print("Survives disconnect. Check: python3 -m modal app logs <app-id>")
