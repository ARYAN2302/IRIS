#!/usr/bin/env python3
"""
Proof 2 — ONE encoder detects BOTH OFDM and FHSS.

4-channel input per sample (all computed from same raw IQ, no modulation knowledge):
  ch0: log|SCF|          ch1: |COH|          (OFDM energy lives here)
  ch2: log envelope-comb ch3: envelope ACF   (FHSS burst-regularity lives here)

Corpus:
  OFDM+: Zenodo raw bins (/data2/raw_iq, 12 files) -> full 4ch
  FHSS+: BOXER ELRS capture (/data/fhss_elrs/boxer_*.iq) -> full 4ch
  BG   : synth WiFi-bursty OFDM + irregular bursts + noise -> full 4ch
Eval: LOTO Zenodo type, held-out BOXER file, BG FP@99p. Save encoder PRE-eval.

Run: python3 extension/scripts/experiments/hybrid_universal_train.py  (spawns detached)
"""
import modal

app = modal.App("iris-hybrid-universal")
DATA_VOL = modal.Volume.from_name("iris-cuas-data")
DATA_VOL2 = modal.Volume.from_name("iris-data")
MODELS_VOL = modal.Volume.from_name("iris-cuas-models")
RESULTS_VOL = modal.Volume.from_name("iris-cuas-results")
IMAGE = (
    modal.Image.debian_slim()
    .apt_install("libhdf5-dev")
    .pip_install("torch==2.5.1", "numpy==1.26.4", "h5py==3.12.1",
                 "scipy==1.14.1", "scikit-learn==1.6.1", "tqdm==4.67.1")
)

CORE = r'''
import os, sys, json, time, glob, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

device="cuda" if torch.cuda.is_available() else "cpu"
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("="*70); print("=== Proof 2: ONE encoder, OFDM+FHSS (4-ch hybrid) ==="); print("="*70, flush=True)

# ---------- SCF (from scf_features.py, trimmed) ----------
def _cx(iq):
    x=np.asarray(iq)
    return x.astype(np.complex128) if np.iscomplexobj(x) else (x[0]+1j*x[1] if x.ndim==2 and x.shape[0]==2 else x[:,0]+1j*x[:,1])

def scf_image(iq, n_fft=1<<14, out_size=256):
    z=_cx(iq); N=min(len(z),n_fft); z=z[:N]*np.hanning(N)
    if N<n_fft: pass
    X=np.fft.fftshift(np.fft.fft(z,n=n_fft))
    alphas=np.linspace(0,.5,out_size); win=np.hanning(128); nf=max(n_fft//128,1)
    SCF=np.zeros((out_size,nf),complex)
    for i,a in enumerate(alphas):
        sh=int(round(a*n_fft/2))
        SCF[i]=np.convolve(np.roll(X,-sh)*np.conj(np.roll(X,sh)),win,"same")[::128][:nf]
    SCF[0]=0
    Sx=np.abs(X)**2; eps=1e-12*(Sx.max()+1e-30)
    COH=np.zeros_like(SCF,dtype=float)
    for i,a in enumerate(alphas):
        sh=int(round(a*n_fft/2))
        Sp=np.convolve(np.roll(Sx,-sh),win,"same")[::128][:nf]
        Sm=np.convolve(np.roll(Sx, sh),win,"same")[::128][:nf]
        COH[i]=np.abs(SCF[i])/(np.sqrt(Sp*Sm)+eps)
    COH=np.clip(COH,0,1)
    img=np.stack([np.log10(np.abs(SCF)+1e-12),COH]).astype(np.float32)
    t=F.interpolate(torch.from_numpy(img)[None],size=(out_size,out_size),mode="bilinear",align_corners=False)[0].numpy()
    for c in range(2):
        m,s=t[c].mean(),t[c].std()+1e-8; t[c]=(t[c]-m)/s
    return t

# ---------- Envelope comb image (from validated Proof 1a/1b chain) ----------
def env_image(iq, decim=100, n_win_max=48):
    """Time-resolved envelope comb: rows=subwindows, cols=alpha bins.
    Adaptive decim/subwindows so ANY capture length works."""
    from scipy.signal import butter, filtfilt
    decim = int(np.clip(decim, 1, max(1, len(iq)//512)))   # keep >=512 env pts
    n=len(iq)//decim*decim
    p=(np.abs(iq[:n].reshape(-1,decim))**2).mean(1)
    b,a=butter(4,0.8,btype="low"); e=filtfilt(b,a,p)
    n_win=int(np.clip(len(e)//24, 4, n_win_max))
    W=len(e)//n_win
    if W < 8: raise ValueError(f"envelope too short: len(e)={len(e)}")
    e=e[:W*n_win]
    rows=[]
    for w in range(n_win):
        seg=e[w*W:(w+1)*W]; seg=seg-seg.mean()
        nfft=1<<int(np.ceil(np.log2(max(2*len(seg),4))))
        E=np.fft.rfft(seg,nfft); acf=np.fft.irfft(E*np.conj(E))[:len(seg)]
        half=len(seg)//2
        if half < 8: raise ValueError("acf segment too short")
        segA=acf[1:half]
        S=np.abs(np.fft.rfft(segA*np.hanning(len(segA))))
        S=S/(acf[0]+1e-30)/max(len(segA),1)
        idx=np.linspace(0,len(S)-1,256).astype(int)
        rows.append(np.log10(S[idx]+1e-12))
    spec=np.stack(rows)                      # (n_win,256)
    acf_img=np.tile(np.abs(acf[:256])[None]/(acf[0]+1e-30),(256,1))
    ch2=spec.T.astype(np.float32)            # (256, n_win) — resize BEFORE stack
    ch3=acf_img.astype(np.float32)
    ch2r=F.interpolate(torch.from_numpy(ch2)[None,None],size=(256,256),mode="bilinear",align_corners=False)[0,0].numpy()
    img=np.stack([ch2r,ch3])
    t=F.interpolate(torch.from_numpy(img)[None],size=(256,256),mode="bilinear",align_corners=False)[0].numpy()
    for c in range(2):
        m,s=t[c].mean(),t[c].std()+1e-8; t[c]=(t[c]-m)/s
    return t

def hybrid_4ch(iq):
    a=scf_image(iq); b=env_image(iq)
    img=np.concatenate([a,b],axis=0)
    assert img.shape==(4,256,256)
    return img.astype(np.float32)

# ---------- Synth BG generators (validated in Proof 1a) ----------
def synth_wifi_ofdm(n=16384, n_sc=64, cp_len=8, seed=0):
    rng=np.random.RandomState(seed); out=np.zeros(n,complex); pos=0
    while pos<n:
        bl=rng.randint(500,2000)
        if pos+bl>=n: break
        sym=max(bl//(n_sc+cp_len),1); chunk=[]
        for _ in range(sym):
            s=rng.randn(n_sc)+1j*rng.randn(n_sc); w=np.fft.ifft(s,n=n_sc)
            chunk.append(np.concatenate([w[-cp_len:],w]))
        sig=np.concatenate(chunk); end=min(pos+len(sig),n); seg=sig[:end-pos]
        out[pos:end]=0.3*seg/(np.abs(seg).max()+1e-9); pos=end+rng.randint(200,1500)
    return out+0.01*(rng.randn(n)+1j*rng.randn(n))

def synth_irr_bursts(n=16384, seed=0):
    rng=np.random.RandomState(seed); out=np.zeros(n,complex); pos=0
    while pos<n:
        bl=rng.randint(300,2500)
        if pos+bl>=n: break
        nz=rng.randn(bl)+1j*rng.randn(bl)
        out[pos:pos+bl]=0.5*nz/np.abs(nz).max(); pos+=bl+rng.randint(100,4000)
    return out+0.02*(rng.randn(n)+1j*rng.randn(n))

# ---------- Encoder ----------
class ConvBlock(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.b=nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.GELU(),nn.Conv2d(o,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.GELU())
    def forward(s,x): return s.b(x)
class CNNEncoder(nn.Module):
    def __init__(s,in_ch=4,width=64,depth=6,embed_dim=256):
        super().__init__(); L=[];ch=in_ch
        for i in range(depth):
            oc=min(width*(2**(i//2)),512); L += [ConvBlock(ch,oc), nn.MaxPool2d(2)]; ch=oc
        s.conv=nn.Sequential(*L)
        with torch.no_grad(): flat=s.conv(torch.zeros(1,in_ch,256,256)).numel()
        s.head=nn.Sequential(nn.Flatten(),nn.Linear(flat,embed_dim),nn.BatchNorm1d(embed_dim))
    def forward(s,x): return s.head(s.conv(x))
class SIGReg(nn.Module):
    def __init__(s,d=256,k=256):
        super().__init__()
        g=torch.Generator().manual_seed(42); W=torch.randn(k,d,generator=g); W=W/W.norm(dim=1,keepdim=True); s.register_buffer("W",W)
    def forward(s,z): return ((F.linear(z,s.W).var(0)-1)**2).mean()
class VICReg(nn.Module):
    def forward(s,z):
        var=torch.relu(1-torch.sqrt(z.var(0)+1e-4)).mean()*25
        N,D=z.shape; zc=z-z.mean(0); cov=(zc.T@zc)/(N-1)
        return var+((cov-torch.diag(torch.diag(cov)))**2).sum()/D

def enc_batch(enc,X,bs=32):
    enc.eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(X),bs):
            o.append(enc(torch.from_numpy(X[i:i+bs]).float().to(device)).cpu().numpy())
    return np.concatenate(o)

def main():
    X=[];Y=[];META=[]  # META: ('ofdm',type) ('fhss','boxer') ('bg',kind)
    print("[1] OFDM positives: Zenodo bins", flush=True)
    bins=sorted(glob.glob("/data2/raw_iq/*.bin"))
    for bi,path in enumerate(bins):
        dtype=np.float32; raw=np.fromfile(path,dtype=dtype,count=8_000_000)
        iq=(raw[0::2]+1j*raw[1::2])
        std=iq.std()
        if std<1e-7: continue
        iq=iq/std
        w=len(iq)//16384
        tname=os.path.basename(path).replace(".bin","")
        take=min(w//3+1, 12)   # ~12 windows/file
        for k in range(take):
            seg=iq[k*16384:(k+1)*16384]
            if len(seg)<16384: break
            try: X.append(hybrid_4ch(seg)); Y.append(1); META.append(("ofdm",tname))
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  seg_len={len(seg)}", flush=True)
                raise SystemExit(1)
        print(f"  {tname}: total windows so far {len(X)}", flush=True)
    n_ofdm_types=len(set(m[1] for m in META if m[0]=='ofdm'))
    print(f"OFDM+ done: {sum(1 for m in META if m[0]=='ofdm')} windows / {n_ofdm_types} types", flush=True)

    print("[2] FHSS positives: BOXER ELRS", flush=True)
    fhs=sorted(glob.glob("/data/fhss_elrs/boxer_*")) or glob.glob("/data/fhss_elrs/*")
    fh_count=0
    for path in [p for p in fhs if not p.endswith('.rar')]:
        sz=os.path.getsize(path)
        raw=np.fromfile(path,dtype=np.float32,count=16_000_000)
        if len(raw)<1000:
            continue
        iq=raw[0::2]+1j*raw[1::2]
        # normalize RFUAV int-interleave scale heuristics: try /32768 if huge
        if np.abs(iq).std()>1e3: iq=iq/32768.0
        std=iq.std();
        if std<1e-9: continue
        iq=iq/std
        w=len(iq)//16384
        for k in range(min(w,40)):
            seg=iq[k*16384:(k+1)*16384]
            if len(seg)<16384: break
            try: X.append(hybrid_4ch(seg)); Y.append(1); META.append(("fhss","boxer")); fh_count+=1
            except Exception as e:
                import traceback, sys as _s; traceback.print_exc(); _s.stdout.flush(); raise SystemExit(1)
    print(f"FHSS+ added {fh_count}", flush=True)

    print("[3] BG: synth wifi/ofdm-continuous/irregular/noise", flush=True)
    gens=[synth_wifi_ofdm, synth_irr_bursts]
    for gi,g in enumerate(gens):
        for s in range(60):
            iq=g(seed=100+s+gi*100)
            try: X.append(hybrid_4ch(iq)); Y.append(0); META.append(("bg",g.__name__))
            except Exception as e:
                import traceback, sys as _s; traceback.print_exc(); _s.stdout.flush(); raise SystemExit(1)
    for s in range(60):
        iq=0.05*(np.random.RandomState(s).randn(16384)+1j*np.random.RandomState(s+999).randn(16384))
        try:
            X.append(hybrid_4ch(iq)); Y.append(0); META.append(("bg","noise"))
        except Exception as e:
                import traceback, sys as _s; traceback.print_exc(); _s.stdout.flush(); raise SystemExit(1)
    X=np.stack(X).astype(np.float32); Y=np.array(Y,dtype=np.float32)
    print(f"CORPUS: X{X.shape} drones={int(Y.sum())} bg={int((1-Y).sum())}", flush=True)

    # splits: hold out 2 ofdm types entirely + all boxer (fhss unseen-type test is mode-level; boxer IS our only fhss source so we do window-level split for fhss and type-level for ofdm)
    ofdm_types=sorted(set(m[1] for m in META if m[0]=='ofdm'))
    holdout_types=set(ofdm_types[-2:])   # LOTO-style: last two types fully unseen
    tr_i=[];ev_seen=[];ev_hold_type=[];ev_fhss=[];ev_bg=[]
    for i,(y,m) in enumerate(zip(Y,META)):
        if y==1 and m[0]=='ofdm':
            (ev_hold_type if m[1] in holdout_types else (tr_i if random.random()<0.85 else ev_seen)).append(i)
        elif y==1 and m[0]=='fhss':
            ev_fhss.append(i)   # ALL fhss held out of training (universal test)
        else:
            (tr_i if random.random()<0.85 else ev_bg).append(i)
    print(f"splits: train={len(tr_i)} ev_seen={len(ev_seen)} ev_holdtype={len(ev_hold_type)} ev_fhss={len(ev_fhss)} ev_bg={len(ev_bg)}", flush=True)

    enc=CNNEncoder(4).to(device); sig=SIGReg().to(device); vic=VICReg().to(device)
    head=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,1)).to(device)
    opt=torch.optim.AdamW(list(enc.parameters())+list(head.parameters()),lr=1e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20)
    Xt=torch.from_numpy(X[tr_i]); yt=torch.from_numpy(Y[tr_i])
    dl=DataLoader(TensorDataset(Xt,yt),batch_size=32,shuffle=True,drop_last=True)
    print(f"[4] training 20 epochs, {len(dl)} batches", flush=True)
    for ep in range(20):
        enc.train();head.train();tot=0
        for xb,yb in dl:
            xb,yb=xb.to(device),yb.to(device)
            z=enc(xb)
            loss=sig(z)+vic(z)+F.binary_cross_entropy_with_logits(head(z).squeeze(-1),yb)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(enc.parameters(),1.0);opt.step()
            tot+=loss.item()
        sch.step()
        print(f"  ep{ep+1}: loss {tot/max(len(dl),1):.4f}", flush=True)

    torch.save({"encoder":enc.state_dict(),"head":head.state_dict()},"/models/hybrid_universal_encoder.pt")
    print("SAVED /models/hybrid_universal_encoder.pt (pre-eval)", flush=True)

    centroid,covinv=None,None
    Ztr=enc_batch(enc,X[tr_i]); c=Ztr.mean(0); cov=np.cov(Ztr.T)+1e-3*np.eye(256)
    try: ci=np.linalg.inv(cov)
    except: ci=np.linalg.pinv(cov)
    def mh(E): 
        n=np.linalg.norm(E,axis=1,keepdims=True)+1e-8; e=E/n; d=e-c
        return np.sqrt(np.maximum((d@ci*d).sum(1),0))
    d_fit=mh(Ztr); thr=float(np.percentile(d_fit,99)); thr9=float(np.percentile(d_fit,99.9))
    res={"threshold_99p":thr}
    for name,idx in [("seen_ofdm",ev_seen),("holdout_ofdm_types",ev_hold_type),("fhss_boxer_heldout",ev_fhss),("bg",ev_bg)]:
        if not idx: continue
        d=mh(enc_batch(enc,X[idx])); det=(d<=thr).mean(); det9=(d<=thr9).mean()
        res[name]={"det_99p":round(float(det),4),"det_99_9p":round(float(det9),4),"dist_mean":round(float(d.mean()),2)}
        print(f"  {name}: det={det:.2%}/{det9:.2%} dist={d.mean():.1f}", flush=True)
    if ev_bg and ev_fhss:
        y=np.r_[np.zeros(len(ev_bg)),np.ones(len(ev_fhss))]
        s=np.r_[mh(enc_batch(enc,X[ev_bg])), -mh(enc_batch(enc,X[ev_fhss]))]
        res["auc_fhss_vs_bg"]=round(float(roc_auc_score(y,s)),4)
        print(f"  AUC FHSS-vs-BG: {res['auc_fhss_vs_bg']}", flush=True)
    res["timestamp"]=time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime())
    json.dump(res,open("/results/hybrid_universal_result.json","w"),indent=2)
    print(json.dumps(res,indent=2))
    print("DONE")

main()
'''

CORE_PATH="/tmp/hybrid_core.py"
open(CORE_PATH,"w").write(CORE)
IMAGE=IMAGE.add_local_file(CORE_PATH,"/root/hybrid_core.py")

@app.function(image=IMAGE,gpu="T4",
              volumes={"/data":DATA_VOL,"/data2":DATA_VOL2,"/models":MODELS_VOL,"/results":RESULTS_VOL},
              timeout=7200,memory=32768)
def run():
    import subprocess
    r=subprocess.run(["python3","/root/hybrid_core.py"],capture_output=True,text=True)
    print(r.stdout[-8000:])
    if r.returncode!=0:
        print(r.stderr[-3000:]); raise RuntimeError("hybrid train failed")

if __name__=="__main__":
    with app.run(detach=True):
        fc=run.spawn()
        print(f"SPAWNED detached: {fc.object_id}")
