#!/usr/bin/env python3
"""
Export IRIS v11 encoder to ONNX format for fast edge inference.

On M1 Mac, ONNX Runtime with the CoreML Execution Provider runs
the encoder ~2-5x faster than PyTorch MPS for small models.

Usage:
    python scripts/export_onnx.py

Outputs:
    models/iris_encoder.onnx    — ONNX model file
    models/iris_encoder.onnx.with_optimizations.onnx  — optimized

After export, test with:
    python scripts/edge_benchmark.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.iris_inference import IRISDetector, CNNEncoder

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "models/lejepa_v11_best.pt"
ONNX_PATH = "models/iris_encoder.onnx"
ONNX_OPTIMIZED_PATH = "models/iris_encoder_optimized.onnx"

INPUT_SHAPE = (1, 2, 256, 256)  # batch=1, channels=2, H=256, W=256
OUTPUT_DIM = 256                 # embedding dimension


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────


def export_to_onnx():
    """Export the IRIS encoder to ONNX format."""
    print("=" * 60)
    print("IRIS v11 — ONNX Export")
    print("=" * 60)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"  [error] checkpoint not found: {CHECKPOINT_PATH}")
        print(f"          run scripts/pull_from_modal.py first")
        sys.exit(1)

    # Load encoder
    print(f"\n  [info] loading encoder from {CHECKPOINT_PATH}...")
    detector = IRISDetector(checkpoint_path=CHECKPOINT_PATH)
    encoder = detector.encoder
    encoder.eval()

    param_count = sum(p.numel() for p in encoder.parameters())
    print(f"  [info] encoder: {param_count:,} params ({param_count * 4 / 1024 / 1024:.1f} MB fp32)")

    # Create dummy input
    dummy = torch.randn(*INPUT_SHAPE, dtype=torch.float32)

    # Verify PyTorch output first
    with torch.no_grad():
        pytorch_out = encoder(dummy)
    print(f"  [info] pytorch output shape: {pytorch_out.shape}")
    print(f"  [info] pytorch output norm:  {pytorch_out.norm().item():.4f}")

    # Export
    print(f"\n  [info] exporting to {ONNX_PATH}...")
    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)

    torch.onnx.export(
        encoder,
        dummy,
        ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["spectrogram"],
        output_names=["embedding"],
        dynamic_axes={
            "spectrogram": {0: "batch"},
            "embedding": {0: "batch"},
        },
        verbose=False,
    )

    file_size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"  [ok] exported {file_size_mb:.1f} MB → {ONNX_PATH}")

    # Verify ONNX model loads and produces same output
    print(f"\n  [info] verifying ONNX model...")
    try:
        import onnxruntime as ort

        # Try CoreML EP first (M1 optimization), fall back to CPU
        available_providers = ort.get_available_providers()
        print(f"  [info] available ORT providers: {available_providers}")

        # Prefer CoreML for M1
        preferred = []
        if "CoreMLExecutionProvider" in available_providers:
            preferred.append("CoreMLExecutionProvider")
        preferred.append("CPUExecutionProvider")

        sess = ort.InferenceSession(ONNX_PATH, providers=preferred)
        input_name = sess.get_inputs()[0].name

        # Run inference
        dummy_np = dummy.numpy()
        onnx_out = sess.run(None, {input_name: dummy_np})[0]

        # Compare
        pytorch_np = pytorch_out.numpy()
        max_diff = np.abs(pytorch_np - onnx_out).max()
        mean_diff = np.abs(pytorch_np - onnx_out).mean()
        print(f"  [ok] ONNX output shape: {onnx_out.shape}")
        print(f"  [ok] max diff vs pytorch:  {max_diff:.6f}")
        print(f"  [ok] mean diff vs pytorch: {mean_diff:.6f}")

        if max_diff < 1e-4:
            print(f"  [ok] ✓ ONNX export verified — outputs match pytorch")
        else:
            print(f"  [warn] ⚠ outputs differ by {max_diff:.6f} — may need re-export")

        # Benchmark
        print(f"\n  [info] benchmarking {len(preferred)} providers...")
        for provider in preferred:
            try:
                sess = ort.InferenceSession(ONNX_PATH, providers=[provider])
                # Warmup
                for _ in range(5):
                    sess.run(None, {input_name: dummy_np})

                # Time
                n_iters = 100
                t0 = time.time()
                for _ in range(n_iters):
                    sess.run(None, {input_name: dummy_np})
                t1 = time.time()
                latency_ms = (t1 - t0) / n_iters * 1000
                print(f"    {provider}: {latency_ms:.2f} ms/inference ({1000/latency_ms:.1f} fps)")
            except Exception as e:
                print(f"    {provider}: failed — {e}")

    except ImportError:
        print(f"  [warn] onnxruntime not installed — pip install onnxruntime")
        print(f"         for M1: pip install onnxruntime-coreml")

    # Also benchmark pytorch for comparison
    print(f"\n  [info] pytorch benchmark (device={detector.device})...")
    dummy_dev = dummy.to(detector.device)
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            encoder(dummy_dev)
    # Time
    n_iters = 100
    if detector.device.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            encoder(dummy_dev)
    if detector.device.type == "mps":
        torch.mps.synchronize()
    t1 = time.time()
    pytorch_latency_ms = (t1 - t0) / n_iters * 1000
    print(f"    pytorch ({detector.device}): {pytorch_latency_ms:.2f} ms/inference ({1000/pytorch_latency_ms:.1f} fps)")

    # Optimize ONNX (optional)
    try:
        from onnxruntime.transformers import optimizer as ort_optimizer
        print(f"\n  [info] optimizing ONNX model...")
        # Use ORT optimizer
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Save optimized model
        optimized_path = ONNX_OPTIMIZED_PATH
        sess = ort.InferenceSession(ONNX_PATH, sess_options, providers=preferred)
        # ORT doesn't directly save optimized model, but the optimization happens on load
        print(f"  [ok] optimizations applied at load time")
    except Exception as e:
        print(f"  [warn] optimization skipped: {e}")

    print("\n" + "=" * 60)
    print("Export complete.")
    print(f"  ONNX model:        {ONNX_PATH} ({file_size_mb:.1f} MB)")
    print(f"  PyTorch fallback:  {CHECKPOINT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    export_to_onnx()
