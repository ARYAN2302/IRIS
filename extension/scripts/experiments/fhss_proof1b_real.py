#!/usr/bin/env python3
"""
Proof 1b — Real FHSS capture vs RFUAV paper-reported hop timing.

Downloads one RFUAV FHSS-type raw-IQ archive (RadioMaster BOXER — paper reports
FHSDT=6.84ms dwell, FHSPP=422.8ms sequence period), extracts, runs the validated
envelope-coherence pipeline on real captures, and checks dominant comb candidates
against {1/FHSDT ~146Hz transition rate, 1/FHSPP ~2.37Hz cycle}.

Run: python3 extension/scripts/experiments/fhss_proof1b_real.py   (spawns detached)
"""
import modal

app = modal.App("fhss-proof1b-real")
DATA_VOL = modal.Volume.from_name("iris-data")
RESULTS_VOL = modal.Volume.from_name("iris-cuas-results")
IMAGE = (
    modal.Image.debian_slim()
    .run_commands(
        r"printf 'Types: deb\nURIs: http://deb.debian.org/debian\nSuites: bookworm\nComponents: main non-free non-free-firmware\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\n' > /etc/apt/sources.list.d/nonfree.sources",
        "apt-get update && apt-get install -y unrar p7zip-full wget",
    )
    .pip_install("numpy==1.26.4", "scipy==1.14.1", "huggingface_hub==0.24.7",
                 "tqdm==4.67.1")
)

@app.function(image=IMAGE, volumes={"/data": DATA_VOL, "/results": RESULTS_VOL},
              timeout=7200, memory=16384)
def run():
    import os, json, time, glob, subprocess
    import numpy as np
    from scipy.signal import butter, filtfilt

    print("="*70)
    print("=== Proof 1b: real RFUAV FHSS capture (BOXER) envelope comb ===")
    print("="*70, flush=True)

    # ---------- 1. find + download BOXER rar ----------
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = api.list_repo_files("kitofrank/RFUAV", repo_type="dataset")
    cands = [f for f in files if f.lower().endswith((".rar",".zip")) and "boxer" in f.lower()]
    print(f"repo has {len(files)} files; BOXER archives: {cands}", flush=True)
    if not cands:
        # fallback: any FHSS-controller candidate
        for name in ["tx16s","jumper","t14"]:
            cands = [f for f in files if name in f.lower() and f.lower().endswith((".rar",".zip"))]
            if cands: break
    assert cands, "no candidate archive found in repo"
    arch = cands[0]
    local = hf_hub_download("kitofrank/RFUAV", arch, repo_type="dataset", local_dir="/tmp/rfuav")
    print(f"downloaded {arch} -> {os.path.getsize(local)/1e9:.2f} GB", flush=True)
    results = {"archive": arch, "size_gb": os.path.getsize(local)/1e9}

    # ---------- 2. extract ----------
    exdir = "/tmp/extract"
    os.makedirs(exdir, exist_ok=True)
    for cmd in (["unrar","x","-o+",local,exdir+"/"],
                ["7z","x",local,f"-o{exdir}","-y"],
                ["bsdtar","-xf",local,"-C",exdir]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            got = glob.glob(exdir+"/**/*", recursive=True)
            big = [g for g in got if os.path.isfile(g) and os.path.getsize(g)>1e6]
            print(f"  {cmd[0]}: rc={r.returncode} big_files={len(big)} "
                  f"stderr_tail={r.stderr[-200:] if r.stderr else ''}", flush=True)
            if r.returncode==0 and big:
                print(f"extract OK via {cmd[0]}", flush=True); break
        except Exception as e:
            print(f"  {cmd[0]} failed: {e}", flush=True)
    raws = [g for g in glob.glob(exdir+"/**/*", recursive=True)
            if os.path.isfile(g) and os.path.getsize(g) > 50e6]
    print("big extracted files:", [(os.path.basename(g), round(os.path.getsize(g)/1e6)) for g in raws][:10], flush=True)
    assert raws, "extraction produced no large files"

    # persist extracted IQ to volume so hybrid training reuses without re-download
    os.makedirs("/data/fhss_elrs", exist_ok=True)
    for g in raws[:4]:
        dst = f"/data/fhss_elrs/boxer_{os.path.basename(g)}"
        if not os.path.exists(dst):
            subprocess.run(["cp", g, dst], check=False)
    subprocess.run(["cp", local, "/data/fhss_elrs/_archive.rar"], check=False)
    print("persisted to /data/fhss_elrs:", os.listdir("/data/fhss_elrs"), flush=True)
    results["best_comb_search_archive"] = {"rar": os.path.basename(local)}

    # ---------- 3. load IQ windows ----------
    def load_iq(path, n_bytes=32_000_000, offset_frac=0.25):
        sz = os.path.getsize(path)
        with open(path,"rb") as f:
            f.seek(int(sz*offset_frac))
            buf = f.read(n_bytes)
        for itemp in (np.float32, np.int16):
            try:
                a = np.frombuffer(buf[:len(buf)//(4 if itemp==np.float32 else 2)*(4 if itemp==np.float32 else 2)], dtype=itemp)
                if itemp==np.int16: a=a.astype(np.float32)/32768.0
                if len(a)%2: a=a[:-1]
                iq = a[0::2] + 1j*a[1::2]
                if np.abs(iq).std() > 1e-6:
                    return iq
            except Exception: continue
        raise RuntimeError("could not parse IQ")

    # ---------- 4. validated envelope pipeline (from Proof 1a) ----------
    def power_envelope(iq, decim=100):
        n=len(iq)//decim*decim
        p=(np.abs(iq[:n].reshape(-1,decim))**2).mean(axis=1)
        b,a=butter(4,0.8,btype="low"); return filtfilt(b,a,p)

    def envelope_spectrum(e):
        e=e-e.mean(); N=len(e)
        nfft=1<<int(np.ceil(np.log2(2*N)))
        E=np.fft.rfft(e,nfft); acf=np.fft.irfft(E*np.conj(E),nfft)[:N]
        seg=acf[1:N//2]; S=np.abs(np.fft.rfft(seg*np.hanning(len(seg))))
        alphas=np.fft.rfftfreq(len(seg))
        return alphas, S/(acf[0]+1e-30)/len(seg)

    def top_comb_candidates(alphas_hz, C, n_top=6, fmin=5.0, fmax=None):
        """peak-pick with local contrast, returns sorted candidates."""
        if fmax is None: fmax = alphas_hz.max()*0.45
        m=(alphas_hz>=fmin)&(alphas_hz<=fmax)
        ah,Cc = alphas_hz[m], C[m]
        order=np.argsort(Cc)[::-1]
        picked=[]
        for i in order:
            f=ah[i]; v=Cc[i]
            if any(abs(f-p)/max(p,1e-12)<0.02 for p,_ in picked): continue
            loc=(ah>=f*0.8)&(ah<=f*1.2)&(np.abs(ah-f)>f*0.01)
            # perfect combs zero-out inter-line bins -> median can be exactly 0
            if not loc.any() or not np.isfinite(v) or v<=0:
                continue
            floor=float(np.percentile(Cc[loc],25))+1e-12
            picked.append((float(f), float(v/floor)))
            if len(picked)>=n_top: break
        return picked

    def best_comb_search(alphas_hz, C, f_lo=20.0, f_hi=600.0, n_harm=8, step=0.25):
        """Scan candidate fundamentals; score = MIN over k=1..n_harm of
        C(k*f0) / p75(C). A true TDMA comb maximizes the minimum (all harmonics present).
        Returns top 3 (f0, score, harmonics_found)."""
        out=[]
        hi = min(f_hi, alphas_hz.max()/n_harm)
        for f0 in np.arange(f_lo, hi, step):
            ratios=[]
            for k in range(1, n_harm+1):
                fk=f0*k
                m=(alphas_hz>=fk*0.98)&(alphas_hz<=fk*1.02)
                if not m.any(): ratios=[0]; break
                ratios.append(float(C[m].max()))
            if len(ratios)<n_harm: continue
            floor=float(np.percentile(C,75))+1e-30
            score=min(r/floor for r in ratios)
            out.append((float(f0), float(score), n_harm))
        out.sort(key=lambda t:-t[1])
        return out[:3]

    # sample-rate: RFUAV metadata says USRP 100 MS/s; verify by file heuristics later
    FS = 100e6; DECIM=4000   # env fs = 25kHz, resolves up to 12.5kHz transitions
    all_peaks={}; comb_scores={}
    for path in raws[:2]:
        for off in (0.25, 0.55):
            try:
                iq = load_iq(path, offset_frac=off)
            except Exception as e:
                print(f"  load fail {path}@{off}: {e}", flush=True); continue
            e = power_envelope(iq, decim=DECIM)
            alphas, C = envelope_spectrum(e)
            ah = alphas * (FS/DECIM)
            peaks = top_comb_candidates(ah, C, n_top=6, fmin=5.0)
            tag=f"{os.path.basename(path)}@{off}"
            all_peaks[tag]=peaks
            combs = best_comb_search(ah, C)
            comb_scores[tag]=combs
            print(f"  {tag}: top lines: " +
                  ", ".join(f"{f:.1f}({c:.0f}x)" for f,c in peaks[:4]), flush=True)
            print(f"      best comb: " +
                  " | ".join(f"f0={f:.1f}Hz score={s:.0f}" for f,s,_ in combs), flush=True)

    # expected from paper Table: BOXER FHSDT=6.84ms -> transitions ~146.2Hz; FHSPP=422.8ms -> cycle 2.365Hz
    expected = {"hop_transition_hz": 1/0.00684, "sequence_cycle_hz": 1/0.4228}
    match={}
    for tag,peaks in all_peaks.items():
        for lbl,exp in expected.items():
            best=min(peaks, key=lambda pc: abs(pc[0]-exp)) if peaks else (0,0)
            match[f"{tag}:{lbl}"]={"expected":round(exp,2),"nearest_found":round(best[0],2),
                                   "rel_err":round(abs(best[0]-exp)/exp,3),"contrast":round(best[1],1)}
    results["expected_from_paper"]=expected
    results["peaks"]=all_peaks
    results["best_comb_search"]=comb_scores
    results["match_analysis"]=match
    results["timestamp"]=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    with open("/results/fhss_proof1b_real.json","w") as f:
        json.dump(results,f,indent=2)
    print("\nExpected (RFUAV paper):", expected, flush=True)
    print(json.dumps(match, indent=2), flush=True)
    print("Saved /results/fhss_proof1b_real.json")
    return "done"

@app.local_entrypoint()
def main():
    print(run.remote())
