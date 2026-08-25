#!/usr/bin/env python3
"""
Verify P0 fixes on Modal (T4) — lightweight contract test.

Checks:
  1. LeJEPALoss returns {total,sigreg,invariance} and train.py keys match
  2. DisentangledEncoder forward + loss with 128+128 split (was 256 crash)
  3. Backbone CNNEncoder + FusionHead shapes
  4. Mahalanobis round-trip

Run: python3 -m modal run --detach extension/scripts/experiments/verify_p0_fixes.py
"""
import modal

app = modal.App("iris-verify-p0")
IMAGE = modal.Image.debian_slim().pip_install(
    "torch==2.5.1", "numpy==1.26.4", "scikit-learn==1.6.1", "scipy==1.14.1"
)

@app.function(image=IMAGE, gpu="T4", timeout=600)
def verify():
    import torch, numpy as np, json, sys
    sys.path.insert(0, "/")
    # Mount repo via modal mount? Instead copy needed files via inline imports
    # We will pip install the repo? Simpler: just test the logic directly
    # Import from the mounted repo — modal run mounts local files automatically
    results = {}

    # 1. LeJEPALoss keys
    try:
        from src.sigreg import LeJEPALoss
        m = LeJEPALoss(embed_dim=768, K=16)
        out = m(torch.randn(4,768), torch.randn(4,768), torch.randn(4,768))
        assert set(out)=={"total","sigreg","invariance"}, f"keys {set(out)}"
        # Check train.py reads correct keys
        import pathlib
        train_src = pathlib.Path("src/train.py").read_text()
        assert "losses['sigreg']" in train_src
        assert "losses['invariance']" in train_src
        assert "losses['sig']" not in train_src.replace("losses['sigreg']","")
        # Backward
        (out["total"]).backward()
        results["lejepa_keys"] = "PASS"
    except Exception as e:
        results["lejepa_keys"] = f"FAIL: {e}"
        import traceback; traceback.print_exc()

    # 2. DisentangledEncoder
    try:
        from extension.src.intelligence.drone_id import DisentangledEncoder
        model = DisentangledEncoder(in_ch=2, embed_dim=256, n_drone_types=4)
        assert model.bg_head.net[0].in_features == 128, f"bg_head {model.bg_head.net[0].in_features}"
        assert model.id_head.net[0].in_features == 128
        x = torch.randn(4,2,256,256)
        bg = torch.tensor([1.,0.,1.,0.])
        dl = torch.tensor([0,0,1,0])
        rl = torch.tensor([0,0,1,1])
        model.train()
        z, z_det, z_drone = model(x)
        assert z.shape==(4,256) and z_det.shape==(4,128) and z_drone.shape==(4,128)
        loss = model.compute_loss(z, z_det, z_drone, bg, dl, rl)
        assert loss.dim()==0 and torch.isfinite(loss)
        loss.backward()
        results["disentangled"] = "PASS"
    except Exception as e:
        results["disentangled"] = f"FAIL: {e}"
        import traceback; traceback.print_exc()

    # 3. Backbone + Fusion
    try:
        from extension.src.encoders.backbone import CNNEncoder
        from extension.src.fusion import FusionHead
        enc = CNNEncoder(in_ch=2, embed_dim=256)
        enc.eval()
        with torch.no_grad():
            z = enc(torch.randn(2,2,256,256))
            assert z.shape==(2,256)
        head = FusionHead(embed_dim=256, n_modalities=3, use_modality_dropout=True)
        head.eval()
        with torch.no_grad():
            z2 = head([torch.randn(2,256) for _ in range(3)])
            assert z2.shape==(2,256)
            zs = head.forward_silent([torch.randn(2,256) for _ in range(3)], silent_modality=0)
            assert zs.shape==(2,256)
        results["backbone_fusion"] = "PASS"
    except Exception as e:
        results["backbone_fusion"] = f"FAIL: {e}"
        import traceback; traceback.print_exc()

    # 4. Mahalanobis
    try:
        from src.iris_inference import fit_mahalanobis, compute_mahalanobis
        rng = np.random.RandomState(0)
        train = rng.randn(200,32).astype(np.float32)
        c, Ci = fit_mahalanobis(train)
        sc = compute_mahalanobis(train, c, Ci)
        assert sc.shape==(200,) and np.all(np.isfinite(sc)) and np.all(sc>=0)
        assert sc.mean() < compute_mahalanobis(rng.randn(20,32).astype(np.float32)*5+10, c, Ci).mean()
        results["mahalanobis"] = "PASS"
    except Exception as e:
        results["mahalanobis"] = f"FAIL: {e}"
        import traceback; traceback.print_exc()

    # 5. Hardcoded paths
    try:
        import pathlib, re
        banned = re.compile(r"/Users/|/home/adarshthakur")
        offenders=[]
        for p in pathlib.Path(".").rglob("*.py"):
            if ".git" in str(p) or "__pycache__" in str(p): continue
            if banned.search(p.read_text(errors="ignore")):
                offenders.append(str(p))
        assert not offenders, f"hardcoded paths in {offenders}"
        import json as js
        assert "/Users/" not in pathlib.Path("configs/split.json").read_text()
        results["portability"] = "PASS"
    except Exception as e:
        results["portability"] = f"FAIL: {e}"

    print(json.dumps(results, indent=2))
    failed = [k for k,v in results.items() if v!="PASS"]
    if failed:
        print(f"\nFAILED: {failed}")
        raise SystemExit(1)
    print("\nAll P0 contracts PASS")

@app.local_entrypoint()
def main():
    verify.remote()
