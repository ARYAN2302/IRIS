#!/usr/bin/env python3
"""
Proof 1a — Envelope cyclostationarity pipeline verification (DSP only, no CNN).

Falsifies/validates Stage 1-2 of the FHSS coherence framework:
  1. Synthesize GFSK-FHSS at EXACTLY known packet rates (500Hz, 250Hz, 150Hz)
  2. Run: IQ -> power envelope -> cyclic autocorrelation -> coherence-normalized C(alpha)
  3. PASS if comb peak appears at the configured alpha (= packet rate, normalized)
  4. Gain invariance: scale IQ by random gains (+/-30dB) -> C(alpha) unchanged
  5. Controls: pure noise (no comb), CP-OFDM (packet-free, no burst comb)

Run: python3 extension/scripts/experiments/fhss_proof1_dsp.py   (spawns detached)
"""
import modal

app = modal.App("fhss-proof1-dsp")
RESULTS_VOL = modal.Volume.from_name("iris-cuas-results")
IMAGE = modal.Image.debian_slim().pip_install("numpy==1.26.4", "scipy==1.14.1", "tqdm==4.67.1")

@app.function(image=IMAGE, volumes={"/results": RESULTS_VOL}, timeout=1800)
def run():
    import json, time
    import numpy as np
    from scipy.signal import butter, filtfilt

    rng_master = np.random.RandomState(42)
    print("="*70)
    print("=== Proof 1a: envelope cyclic pipeline vs known FHSS packet rates ===")
    print("="*70, flush=True)

    # ---------------- Signal synthesizers ----------------
    def synth_gfsk_fhss(fs=1_000_000.0, packet_rate_hz=500.0, n_packets=None,
                        duration_s=None, seed=0, dev_frac=0.05):
        """ELRS-like: fixed-interval packets, each at a random carrier offset."""
        rng = np.random.RandomState(seed)
        if duration_s is None:
            n_packets = n_packets or 20
            duration_s = n_packets / packet_rate_hz
        n = int(duration_s * fs)
        out = np.zeros(n, dtype=np.complex128)
        packet_samples = int(fs / packet_rate_hz * 0.7)   # 70% duty cycle
        centers = np.linspace(-0.30, 0.30, 25)             # hop channels (normalized)
        pos = 0
        t_packet = 0
        while True:
            start = int(t_packet * fs / packet_rate_hz)
            if start + packet_samples >= n: break
            fc = centers[rng.randint(len(centers))]
            bits = rng.randint(0, 2, packet_samples // 64 + 2)
            # gaussian pulse -> GFSK freq deviation
            kern = np.exp(-0.5*(np.linspace(-3,3,201)/30)**2); kern /= kern.sum()
            fdev = np.convolve(np.repeat(bits, 64), kern, mode="same")[:packet_samples] - 0.5
            phase = 2*np.pi*np.cumsum(fc + dev_frac*fdev)/fs*fs  # phase in rad (normalized units)
            out[start:start+packet_samples] = 0.7*np.exp(1j*phase)
            t_packet += 1
        out += 0.02*(rng.randn(n)+1j*rng.randn(n))
        return out

    def synth_cp_ofdm(fs=1_000_000.0, duration_s=0.04, seed=0):
        """Continuous OFDM — no packet bursts. Should show NO burst comb."""
        rng = np.random.RandomState(seed)
        n = int(duration_s*fs); out=[]
        n_sc=64; cp=16
        while sum(len(x) for x in out) < n:
            s = rng.randn(n_sc)+1j*rng.randn(n_sc)
            w = np.fft.ifft(s, n=n_sc)
            out.append(np.concatenate([w[-cp:], w]))
        z = np.concatenate(out)[:n]
        return 0.3*z/(np.abs(z).max()+1e-9) + 0.01*(rng.randn(n)+1j*rng.randn(n))

    # ---------------- Envelope cyclic pipeline ----------------
    def power_envelope(iq, decim=100):
        """Wideband power envelope, lowpass+decimate (20MS/s-class -> kS/s)."""
        n = len(iq)//decim*decim
        p = (np.abs(iq[:n].reshape(-1, decim))**2).mean(axis=1)
        b,a = butter(4, 0.8, btype="low")
        return filtfilt(b, a, p)

    def envelope_coherence(e, n_alpha=512, alpha_max_norm=0.5):
        """
        C(alpha) = |sum_tau R_e(alpha,tau)|^2 / (P_bar * R_e(0))
        alpha normalized to fs_env. Regularity/power — gain invariant by construction.
        Implemented via FFT autocorrelation of mean-removed envelope (fast, exact).
        """
        e = e - e.mean()
        N = len(e)
        nfft = 1 << int(np.ceil(np.log2(2*N)))
        E = np.fft.rfft(e, nfft)
        acf_full = np.fft.irfft(E*np.conj(E), nfft)[:N]
        R0 = acf_full[0] + 1e-30
        # spectrum of the autocorrelation-as-time-series over lag tau:
        # For packet-train, R_e(tau) is itself periodic with the packet period.
        # Take FFT over tau window [0, N//2] to find periodicity of the ACF.
        seg = acf_full[1:N//2]
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        alphas = np.fft.rfftfreq(len(seg), d=1.0)   # cycles per env-sample
        C = S / (R0 * len(seg))
        return alphas, C

    def harmonic_score(alphas, C, f0, fs_env, n_harm=3):
        """True fixed-rate packet trains have SHARP lines at f0 AND 2f0 AND 3f0.
        Score = min over harmonics of LOCAL CONTRAST:
            peak(C in +-1% of fk) / median(C in +-20% window, excl +-1%)
        Local contrast kills the 'decaying-spectrum floor' false positive."""
        elevs = []
        alphas_hz = alphas * fs_env
        for h in range(1, n_harm+1):
            fk = f0*h
            pk_mask = (alphas_hz >= fk*0.99) & (alphas_hz <= fk*1.01)
            if not pk_mask.any(): return 0.0
            pk = float(C[pk_mask].max())
            loc = (alphas_hz >= fk*0.8) & (alphas_hz <= fk*1.2) & (~pk_mask)
            if not loc.any(): return 0.0
            floor = float(np.median(C[loc])) + 1e-30
            elevs.append(pk/floor)
        return float(min(elevs))

    def detect_peak(alphas, C, f_true, fs_env, tol_rel=0.03):
        """Find max peak within tol of true packet frequency; returns (peak_freq, peak_val, snr).
        alphas are cycles/env-sample; convert to Hz via fs_env."""
        alphas_hz = alphas * fs_env
        mask = (alphas_hz >= f_true*(1-tol_rel)) & (alphas_hz <= f_true*(1+tol_rel))
        if not mask.any(): return 0.0, 0.0, 0.0
        seg = C[mask]
        pk = float(seg.max())
        far = (alphas_hz < f_true*0.5) | (alphas_hz > f_true*1.5)
        floor = float(np.median(C[far])) if far.any() else 1e-30
        return float(alphas_hz[mask][np.argmax(seg)]), pk, pk/(floor+1e-30)

    # ---------------- Tests ----------------
    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    verdict_all = True

    print("\n[Test 1] Comb at configured packet rate + gain invariance", flush=True)
    for rate in [500.0, 250.0, 150.0]:
        iq = synth_gfsk_fhss(packet_rate_hz=rate, duration_s=max(40/rate, 0.08), seed=int(rate))
        e = power_envelope(iq)
        fs_env = 1e6 / 100.0
        alphas, C = envelope_coherence(e)
        pf, pv, snr = detect_peak(alphas, C, rate, fs_env)
        hs = harmonic_score(alphas, C, rate, fs_env)
        ok_rate = abs(pf - rate)/rate < 0.03 and snr > 10.0 and hs > 5.0

        # gain invariance: x10 and x0.001 amplitude
        inv_ok = True; shifts=[]
        for g in [10.0, 0.001, 316.0]:
            iq_g = iq * g
            e_g = power_envelope(iq_g)
            _, C_g = envelope_coherence(e_g)
            _, pv_g, _ = detect_peak(alphas, C_g, rate, fs_env)
            rel = abs(pv_g - pv)/max(pv,1e-30)
            shifts.append(rel)
            if rel > 0.15: inv_ok=False
        results[f"fhss_{int(rate)}hz"] = {
            "peak_hz": pf, "true_hz": rate, "comb_snr": snr,
            "harmonic_score_1to3": hs,
            "gain_invariance_maxrel": float(max(shifts)),
            "pass": bool(ok_rate and inv_ok),
        }
        verdict_all &= bool(ok_rate and inv_ok)
        print(f"  {rate:6.0f} Hz: peak={pf:7.1f} Hz  SNR={snr:6.1f}  harm3={hs:8.1f}  "
              f"gain-inv={max(shifts):.3f}  {'PASS' if ok_rate and inv_ok else 'FAIL'}", flush=True)

    print("\n[Test 2] Controls", flush=True)
    # noise control
    iq_noise = 0.05*(rng_master.randn(int(0.08e6))+1j*rng_master.randn(int(0.08e6)))
    e = power_envelope(iq_noise); alphas, C = envelope_coherence(e)
    pf,pv,snr = detect_peak(alphas, C, 500.0, 1e4)
    results["control_noise"] = {"comb_snr": snr, "pass": bool(snr < 5.0)}
    verdict_all &= snr < 5.0
    print(f"  pure noise @500Hz: comb_snr={snr:.1f}  {'PASS(no comb)' if snr<5 else 'FAIL'}", flush=True)

    # continuous OFDM control (no bursts)
    iq_ofdm = synth_cp_ofdm()
    e = power_envelope(iq_ofdm); alphas, C = envelope_coherence(e)
    pf,pv,snr = detect_peak(alphas, C, 500.0, 1e4)
    results["control_cp_ofdm"] = {"comb_snr": snr, "pass": bool(snr < 5.0)}
    verdict_all &= snr < 5.0
    print(f"  continuous OFDM @500Hz: comb_snr={snr:.1f}  {'PASS(no burst comb)' if snr<5 else 'FAIL'}", flush=True)

    # WiFi-bursty control: irregular gaps should NOT give clean comb at any fixed rate
    def synth_bursty_irregular(seed=0):
        """WiFi-like: OFDM-noise-filled bursts, random widths/gaps (CSMA)."""
        rng = np.random.RandomState(seed)
        n = int(0.08e6); out=np.zeros(n, dtype=np.complex128); pos=0
        while pos < n:
            bl = rng.randint(300, 2500)
            if pos+bl>=n: break
            noise = rng.randn(bl)+1j*rng.randn(bl)          # incoherent, not tonal
            out[pos:pos+bl] = 0.5*noise/np.abs(noise).max()
            pos += bl + rng.randint(100, 4000)
        return out + 0.02*(rng.randn(n)+1j*rng.randn(n))
    iq_irr = synth_bursty_irregular()
    e = power_envelope(iq_irr); alphas, C = envelope_coherence(e)
    best_snr = 0.0; best_harm = 0.0
    for probe in [200.,333.,500.,800.,1600.]:
        _,_,snr_p = detect_peak(alphas, C, probe, 1e4)
        hs_p = harmonic_score(alphas, C, probe, 1e4)
        best_snr = max(best_snr, snr_p); best_harm = max(best_harm, hs_p)
    # irregular bursts may show single-line SNR but MUST fail the 3-harmonic comb test
    results["control_irregular_bursts"] = {
        "max_comb_snr": best_snr, "max_harmonic_score": best_harm,
        "pass": bool(best_harm < 5.0)}
    verdict_all &= best_harm < 5.0
    print(f"  irregular bursts (WiFi-like): max_snr={best_snr:.1f} max_harm3={best_harm:.1f}  "
          f"{'PASS(no fixed-rate comb)' if best_harm<5 else 'FAIL'}", flush=True)

    results["ALL_PASS"] = bool(verdict_all)
    with open("/results/fhss_proof1_dsp.json","w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*70)
    print(f"VERDICT: {'ALL PASS — envelope coherence framework validated' if verdict_all else 'FAILURES — see above'}")
    print("Saved /results/fhss_proof1_dsp.json")
    return results

@app.local_entrypoint()
def main():
    print(run.remote())
