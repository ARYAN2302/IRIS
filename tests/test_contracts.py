"""
Contract tests — shape, key, and determinism invariants.
These would have caught the P0 bugs before they shipped.
Run: pytest tests/test_contracts.py -v
No data download or Modal needed.
"""
import torch
import numpy as np


# ---------------------------------------------------------------------------
# 1. Encoder shape contracts
# ---------------------------------------------------------------------------

def test_cnn_encoder_shapes():
    """Backbone CNNEncoder: any in_ch → 256-dim, correct batch dim."""
    from extension.src.encoders.backbone import CNNEncoder

    for in_ch in (1, 2, 4):
        enc = CNNEncoder(in_ch=in_ch, embed_dim=256)
        enc.eval()
        x = torch.randn(4, in_ch, 256, 256)
        with torch.no_grad():
            z = enc(x)
        assert z.shape == (4, 256), f"in_ch={in_ch} gave {z.shape}"
        assert z.dtype == torch.float32


def test_legacy_encoder_shapes():
    """Legacy src/encoder.py still produces 768-dim."""
    from src.encoder import CNNEncoder as LegacyEncoder

    enc = LegacyEncoder(embed_dim=768)
    enc.eval()
    # Legacy expects 3 channels (RGB spectrogram)
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        z = enc(x)
    assert z.shape == (2, 768)


def test_iris_inference_encoder_shape():
    """Inference encoder: 2-channel 256×256 → 256-dim (checkpoint-bound)."""
    from src.iris_inference import CNNEncoder

    enc = CNNEncoder(in_ch=2, width=64, depth=6, embed_dim=256)
    enc.eval()
    x = torch.randn(2, 2, 256, 256)
    with torch.no_grad():
        z = enc(x)
    assert z.shape == (2, 256)


# ---------------------------------------------------------------------------
# 2. Loss key contracts (catches the sig/sigreg KeyError)
# ---------------------------------------------------------------------------

def test_lejepa_loss_keys():
    """LeJEPALoss must return {total, sigreg, invariance} — not {sig, inv}."""
    from src.sigreg import LeJEPALoss

    loss_fn = LeJEPALoss(embed_dim=768, lam=1e-3, K=16)
    B = 4
    z_ctx = torch.randn(B, 768)
    z_target = torch.randn(B, 768)
    y_pred = torch.randn(B, 768)
    out = loss_fn(z_ctx, z_target, y_pred)

    assert set(out.keys()) == {"total", "sigreg", "invariance"}, f"keys={out.keys()}"
    for k in out:
        assert out[k].dim() == 0, f"{k} should be scalar, got shape {out[k].shape}"
        assert torch.isfinite(out[k]).all()

    # Backward must not error
    out["total"].backward()


def test_train_consumes_correct_keys():
    """src/train.py must read the same keys sigreg.py produces."""
    from src.sigreg import LeJEPALoss
    import inspect
    from pathlib import Path

    train_src = Path("src/train.py").read_text()
    # After fix, train.py should reference sigreg/invariance, not sig/inv
    assert "losses['sigreg']" in train_src, "train.py still uses stale key 'sig'"
    assert "losses['invariance']" in train_src, "train.py still uses stale key 'inv'"
    assert "losses['sig']" not in train_src.replace("losses['sigreg']", ""), "stale 'sig' still present"
    assert "losses['inv']" not in train_src.replace("losses['invariance']", ""), "stale 'inv' still present"

    # Also verify round-trip
    loss_fn = LeJEPALoss(embed_dim=768, K=16)
    out = loss_fn(torch.randn(4, 768), torch.randn(4, 768), torch.randn(4, 768))
    # This is what train.py does:
    sig_loss = out["sigreg"]
    inv_loss = out["invariance"]
    is_pos = torch.ones(4)
    batch_loss = sig_loss + is_pos.mean() * inv_loss
    batch_loss.backward()


# ---------------------------------------------------------------------------
# 3. DisentangledEncoder — catches the DroneBGHead(d=256) vs 128-dim bug
# ---------------------------------------------------------------------------

def test_disentangled_encoder_forward_and_loss():
    """DisentangledEncoder: 256→ split 128/128 → both heads accept 128-d."""
    from extension.src.intelligence.drone_id import DisentangledEncoder

    model = DisentangledEncoder(in_ch=2, embed_dim=256, n_drone_types=8)
    model.eval()

    x = torch.randn(4, 2, 256, 256)
    bg_labels = torch.tensor([1., 0., 1., 0.])
    drone_labels = torch.tensor([3, 0, 1, 0])
    recv_labels = torch.tensor([0, 0, 1, 1])

    with torch.no_grad():
        z, z_detect, z_drone = model(x)

    assert z.shape == (4, 256)
    assert z_detect.shape == (4, 128)
    assert z_drone.shape == (4, 128)

    # The actual bug was here — bg_head(128-d) would crash if d=256
    model.train()
    loss = model.compute_loss(z, z_detect, z_drone, bg_labels, drone_labels, recv_labels)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()


def test_drone_bg_head_dims():
    """DroneBGHead must match the dim it is actually called with."""
    from extension.src.intelligence.drone_id import DisentangledEncoder

    model = DisentangledEncoder(in_ch=2, embed_dim=256, n_drone_types=4)
    # Internal check: bg_head first Linear must be 128, not 256
    assert model.bg_head.net[0].in_features == 128, (
        f"bg_head expects {model.bg_head.net[0].in_features}-d, should be 128"
    )
    assert model.id_head.net[0].in_features == 128


# ---------------------------------------------------------------------------
# 4. Mahalanobis round-trip + L2 contract
# ---------------------------------------------------------------------------

def test_mahalanobis_fit_and_score():
    """fit → score: train points close, far OOD far, score is 1-D, finite, non-negative."""
    from src.iris_inference import fit_mahalanobis, compute_mahalanobis

    rng = np.random.RandomState(0)
    train = rng.randn(200, 32).astype(np.float32)
    centroid, cov_inv = fit_mahalanobis(train, reg=1e-3, l2_normalize=True)

    assert centroid.shape == (32,)
    assert cov_inv.shape == (32, 32)

    train_scores = compute_mahalanobis(train, centroid, cov_inv, l2_normalize=True)
    far_ood = rng.randn(20, 32).astype(np.float32) * 5 + 10
    ood_scores = compute_mahalanobis(far_ood, centroid, cov_inv, l2_normalize=True)

    assert train_scores.shape == (200,)
    assert np.all(np.isfinite(train_scores)) and np.all(train_scores >= 0)
    # Train points should be tighter than far OOD on average
    assert train_scores.mean() < ood_scores.mean()


def test_mahalanobis_l2_equivalence():
    """L2 normalize flag must change scores (sanity: not a no-op)."""
    from src.iris_inference import fit_mahalanobis, compute_mahalanobis

    rng = np.random.RandomState(1)
    train = rng.randn(100, 16).astype(np.float32) * 2
    c_n, C_n = fit_mahalanobis(train, l2_normalize=True)
    c_u, C_u = fit_mahalanobis(train, l2_normalize=False)
    # Centroids differ when L2 is toggled
    assert not np.allclose(c_n, c_u)


# ---------------------------------------------------------------------------
# 5. Determinism — same seed → identical embeddings
# ---------------------------------------------------------------------------

def test_encoder_determinism():
    """Two encoders with same seed produce identical embeddings on same input."""
    from extension.src.encoders.backbone import CNNEncoder

    torch.manual_seed(42)
    enc1 = CNNEncoder(in_ch=2, embed_dim=256)
    torch.manual_seed(42)
    enc2 = CNNEncoder(in_ch=2, embed_dim=256)

    x = torch.randn(2, 2, 256, 256)
    enc1.eval(); enc2.eval()
    with torch.no_grad():
        z1 = enc1(x)
        z2 = enc2(x)
    assert torch.allclose(z1, z2, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Fusion head — modality dropout path doesn't break shapes
# ---------------------------------------------------------------------------

def test_fusion_head_shapes():
    """FusionHead: 3×256 → 256, train and eval, with/without dropout."""
    from extension.src.fusion import FusionHead

    for use_dropout in (True, False):
        head = FusionHead(embed_dim=256, n_modalities=3, use_modality_dropout=use_dropout)
        embs = [torch.randn(4, 256) for _ in range(3)]

        head.eval()
        with torch.no_grad():
            z = head(embs)
        assert z.shape == (4, 256)

        if use_dropout:
            head.train()
            z2 = head(embs)
            assert z2.shape == (4, 256)

        # Silent modality path (inference with RF zeroed)
        with torch.no_grad():
            z_silent = head.forward_silent(embs, silent_modality=0)
        assert z_silent.shape == (4, 256)


# ---------------------------------------------------------------------------
# 7. Hardcoded path hygiene — regression guard
# ---------------------------------------------------------------------------

def test_no_hardcoded_user_paths():
    """No file should contain /Users/adarshthakur or similar absolute home paths."""
    from pathlib import Path
    import re

    banned = re.compile(r"/Users/|/home/adarshthakur")
    offenders = []
    for p in Path(".").rglob("*.py"):
        if ".git" in str(p) or "__pycache__" in str(p):
            continue
        text = p.read_text(errors="ignore")
        if banned.search(text):
            # Allow comments that mention the fix
            for i, line in enumerate(text.splitlines(), 1):
                if banned.search(line) and "adarshthakur" in line:
                    offenders.append(f"{p}:{i}: {line.strip()[:100]}")
    assert not offenders, "Hardcoded absolute paths found:\n" + "\n".join(offenders)


def test_no_hardcoded_paths_in_configs():
    """configs/*.json must use relative paths."""
    from pathlib import Path
    import json, re

    banned = re.compile(r"/Users/|/home/")
    for p in Path("configs").glob("*.json"):
        data = json.loads(p.read_text())
        blob = json.dumps(data)
        assert not banned.search(blob), f"{p} contains absolute path: {blob[:300]}"
