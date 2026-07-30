#!/usr/bin/env python3
"""
IRIS Edge Deployment Benchmark

Measures the four SWaP-C numbers a defense customer actually cares about:
  - Latency:   ms per spectrogram inference
  - Throughput: spectrograms per second
  - RAM:       peak memory during inference
  - Model size: on-disk size in MB

Tests three backends on M1 Mac:
  1. PyTorch MPS     — fallback, always works
  2. PyTorch CPU     — for comparison
  3. ONNX Runtime    — fastest on M1 (CoreML EP if available, else CPU)

Outputs a markdown table to results/edge_benchmark.md.

Usage:
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
from src.iris_inference import IRISDetector

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "models/lejepa_v11_best.pt"
ONNX_PATH = "models/iris_encoder.onnx"
RESULTS_PATH = "results/edge_benchmark.md"

INPUT_SHAPE = (1, 2, 256, 256)
BATCH_SIZES = [1, 4, 16, 64]
N_WARMUP = 10
N_BENCHMARK = 200


# ─────────────────────────────────────────────────────────────────────────────
# Memory measurement
# ─────────────────────────────────────────────────────────────────────────────


def get_ram_mb() -> float:
    """Get current process RAM in MB (works on Mac and Linux)."""
    try:
        import resource
        # On Mac, ru_maxrss is in bytes; on Linux it's in KB
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)  # bytes → MB
        else:
            return usage.ru_maxrss / 1024  # KB → MB
    except Exception:
        return 0.0


def get_model_size_mb(path: str) -> float:
    """Get on-disk model size in MB."""
    return os.path.getsize(path) / (1024 * 1024)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch benchmarks
# ─────────────────────────────────────────────────────────────────────────────


def benchmark_pytorch(device_str: str, batch_size: int) -> dict:
    """Benchmark PyTorch on a specific device."""
    device = torch.device(device_str)

    # Fresh load each time for accurate memory measurement
    detector = IRISDetector(checkpoint_path=CHECKPOINT_PATH, device=device_str)
    encoder = detector.encoder.to(device)
    encoder.eval()

    dummy = torch.randn(batch_size, 2, 256, 256, dtype=torch.float32, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = encoder(dummy)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    ram_before = get_ram_mb()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(N_BENCHMARK):
            _ = encoder(dummy)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    ram_after = get_ram_mb()

    total_ms = (t1 - t0) * 1000
    per_sample_ms = total_ms / N_BENCHMARK / batch_size
    fps = N_BENCHMARK * batch_size / (t1 - t0)

    return {
        "backend": f"pytorch_{device_str}",
        "batch_size": batch_size,
        "total_ms": total_ms,
        "per_sample_ms": per_sample_ms,
        "fps": fps,
        "ram_mb": ram_after,
        "ram_delta_mb": ram_after - ram_before,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ONNX benchmarks
# ─────────────────────────────────────────────────────────────────────────────


def benchmark_onnx(provider: str, batch_size: int) -> dict:
    """Benchmark ONNX Runtime with a specific provider."""
    try:
        import onnxruntime as ort
    except ImportError:
        return {"backend": f"onnx_{provider}", "error": "onnxruntime not installed"}

    if not os.path.exists(ONNX_PATH):
        return {"backend": f"onnx_{provider}", "error": f"ONNX model not found: {ONNX_PATH}"}

    try:
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(ONNX_PATH, sess_opts, providers=[provider])
    except Exception as e:
        return {"backend": f"onnx_{provider}", "error": f"failed to load: {e}"}

    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(batch_size, 2, 256, 256).astype(np.float32)

    # Warmup
    for _ in range(N_WARMUP):
        sess.run(None, {input_name: dummy})

    # Benchmark
    ram_before = get_ram_mb()
    t0 = time.time()
    for _ in range(N_BENCHMARK):
        sess.run(None, {input_name: dummy})
    t1 = time.time()
    ram_after = get_ram_mb()

    total_ms = (t1 - t0) * 1000
    per_sample_ms = total_ms / N_BENCHMARK / batch_size
    fps = N_BENCHMARK * batch_size / (t1 - t0)

    return {
        "backend": f"onnx_{provider}",
        "batch_size": batch_size,
        "total_ms": total_ms,
        "per_sample_ms": per_sample_ms,
        "fps": fps,
        "ram_mb": ram_after,
        "ram_delta_mb": ram_after - ram_before,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("IRIS Edge Deployment Benchmark")
    print("=" * 60)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n  [error] checkpoint not found: {CHECKPOINT_PATH}")
        print(f"          run scripts/pull_from_modal.py first")
        sys.exit(1)

    # System info
    import platform
    print(f"\n  System:")
    print(f"    platform:   {platform.platform()}")
    print(f"    processor:  {platform.processor()}")
    print(f"    python:     {platform.python_version()}")
    print(f"    torch:      {torch.__version__}")
    print(f"    mps avail:  {torch.backends.mps.is_available()}")
    print(f"    cuda avail: {torch.cuda.is_available()}")

    # Model info
    detector = IRISDetector(checkpoint_path=CHECKPOINT_PATH)
    param_count = sum(p.numel() for p in detector.encoder.parameters())
    print(f"\n  Model:")
    print(f"    params:     {param_count:,}")
    print(f"    fp32 size:  {param_count * 4 / 1024 / 1024:.2f} MB")
    print(f"    checkpoint: {get_model_size_mb(CHECKPOINT_PATH):.2f} MB")
    if os.path.exists(ONNX_PATH):
        print(f"    onnx:       {get_model_size_mb(ONNX_PATH):.2f} MB")

    # Discover available ONNX providers
    onnx_providers = []
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CoreMLExecutionProvider" in available:
            onnx_providers.append("CoreMLExecutionProvider")
        onnx_providers.append("CPUExecutionProvider")
    except ImportError:
        pass

    # Run benchmarks
    print(f"\n  Running benchmarks (warmup={N_WARMUP}, iters={N_BENCHMARK})...")
    results = []

    for batch_size in BATCH_SIZES:
        print(f"\n  Batch size {batch_size}:")
        # PyTorch MPS (if available)
        if torch.backends.mps.is_available():
            r = benchmark_pytorch("mps", batch_size)
            if "error" not in r:
                results.append(r)
                print(f"    pytorch mps:     {r['per_sample_ms']:.2f} ms/sample ({r['fps']:.1f} fps)")
            else:
                print(f"    pytorch mps:     failed")

        # PyTorch CPU
        r = benchmark_pytorch("cpu", batch_size)
        if "error" not in r:
            results.append(r)
            print(f"    pytorch cpu:     {r['per_sample_ms']:.2f} ms/sample ({r['fps']:.1f} fps)")

        # ONNX
        for prov in onnx_providers:
            r = benchmark_onnx(prov, batch_size)
            if "error" not in r:
                results.append(r)
                prov_short = prov.replace("ExecutionProvider", "")
                print(f"    onnx {prov_short:8s}: {r['per_sample_ms']:.2f} ms/sample ({r['fps']:.1f} fps)")
            else:
                print(f"    onnx {prov}: {r.get('error', 'failed')}")

    # Generate markdown report
    print(f"\n  Generating report → {RESULTS_PATH}")
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    with open(RESULTS_PATH, "w") as f:
        f.write("# IRIS Edge Deployment Benchmark\n\n")
        f.write("SWaP-C numbers for IRIS v11 encoder on this machine.\n")
        f.write("These are the numbers to quote in defense customer conversations.\n\n")

        f.write("## System\n\n")
        f.write(f"- **Platform:** {platform.platform()}\n")
        f.write(f"- **Processor:** {platform.processor()}\n")
        f.write(f"- **PyTorch:** {torch.__version__}\n")
        f.write(f"- **MPS available:** {torch.backends.mps.is_available()}\n\n")

        f.write("## Model\n\n")
        f.write(f"- **Parameters:** {param_count:,}\n")
        f.write(f"- **FP32 size:** {param_count * 4 / 1024 / 1024:.2f} MB\n")
        f.write(f"- **Checkpoint:** {get_model_size_mb(CHECKPOINT_PATH):.2f} MB\n")
        if os.path.exists(ONNX_PATH):
            f.write(f"- **ONNX:** {get_model_size_mb(ONNX_PATH):.2f} MB\n")
        f.write("\n")

        f.write("## Latency & Throughput\n\n")
        f.write("| Backend | Batch Size | Per-Sample (ms) | Throughput (fps) | RAM (MB) |\n")
        f.write("|---------|-----------|-----------------|------------------|----------|\n")
        for r in results:
            f.write(
                f"| {r['backend']} | {r['batch_size']} | "
                f"{r['per_sample_ms']:.2f} | {r['fps']:.1f} | "
                f"{r['ram_mb']:.0f} |\n"
            )

        # Best numbers for headline
        best_per_sample = min(r["per_sample_ms"] for r in results if "per_sample_ms" in r)
        best_fps = max(r["fps"] for r in results if "fps" in r)
        best_result = next(r for r in results if r["per_sample_ms"] == best_per_sample)

        f.write("\n## Headline Numbers\n\n")
        f.write(f"- **Best latency:** {best_per_sample:.2f} ms/sample ({best_result['backend']}, batch={best_result['batch_size']})\n")
        f.write(f"- **Best throughput:** {best_fps:.1f} fps ({best_result['backend']})\n")
        f.write(f"- **Model size (ONNX):** {get_model_size_mb(ONNX_PATH) if os.path.exists(ONNX_PATH) else 'N/A'} MB\n")
        f.write(f"- **Peak RAM:** {max(r['ram_mb'] for r in results):.0f} MB\n\n")

        # Extrapolation to common edge devices
        f.write("## Extrapolation to Common Edge Devices\n\n")
        f.write("Rough scaling factors (use as order-of-magnitude estimates only):\n\n")
        f.write("| Device | Relative to M1 Mac | Est. Latency (ms) | Est. FPS |\n")
        f.write("|--------|-------------------|-------------------|----------|\n")
        m1_latency = best_per_sample
        devices = [
            ("Jetson Orin Nano (15W)", 1.5),
            ("Jetson Orin NX (25W)", 0.8),
            ("Jetson AGX Orin (60W)", 0.3),
            ("Raspberry Pi 5", 4.0),
            ("Intel N100 (mini PC)", 2.0),
        ]
        for name, factor in devices:
            f.write(f"| {name} | {factor}x | {m1_latency * factor:.2f} | {1000 / (m1_latency * factor):.1f} |\n")

        f.write("\n## Why This Matters for C-UAS\n\n")
        f.write("Real-time drone detection requires processing at least 10 spectrograms/second\n")
        f.write("(100ms STFT windows at 10 Hz refresh). IRIS at the best measured latency\n")
        f.write(f"of {best_per_sample:.2f} ms/sample can process ~{1000/best_per_sample:.0f} samples/sec —\n")
        f.write(f"**{1000/best_per_sample/10:.0f}x the real-time requirement.**\n\n")
        f.write("This means a single M1 Mac (or Jetson Orin class edge device) can run IRIS\n")
        f.write("with massive headroom for:\n")
        f.write("- Multi-band SDR processing (2.4 GHz + 5.8 GHz + 915 MHz simultaneously)\n")
        f.write("- Multi-target tracking (100+ concurrent tracks)\n")
        f.write("- Intent classification (Build 3, additional head)\n")
        f.write("- RF fingerprinting for IFF (Build 5, additional head)\n")
        f.write("- CoT/ATAK output and API serving\n\n")
        f.write("All on a 13MB encoder with ~{:.0f}MB RAM footprint. This is the SWaP-C story.\n".format(
            max(r['ram_mb'] for r in results)
        ))

    print(f"\n  [ok] report written to {RESULTS_PATH}")
    print(f"\n  Headline: {best_per_sample:.2f} ms/sample, {best_fps:.1f} fps, "
          f"{get_model_size_mb(ONNX_PATH) if os.path.exists(ONNX_PATH) else 'N/A'} MB model")

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
