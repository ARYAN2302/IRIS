# IRIS Phase 1 — Foundation: Local Inference + Live Demo

This phase turns IRIS from a batch script that prints AUC numbers into a **real-time detector running on your M1 Mac**. It's the foundation for every demo moment in the meeting.

## Quickstart (3 commands)

```bash
# 1. Install dependencies (one-time, ~5 min)
pip install -r requirements_demo.txt

# 2. Pull v11 checkpoint from Modal + compute Mahalanobis centroid (~10 min)
python scripts/pull_from_modal.py

# 3. Run the live demo
python scripts/live_demo.py
```

## What's in Phase 1

### Files added

| File | Purpose |
|---|---|
| `src/iris_inference.py` | Clean inference module. Loads v11 encoder, runs Mahalanobis detection. Reproduces exact CNNEncoder architecture from `train_modal_v11.py`. |
| `scripts/pull_from_modal.py` | Downloads v11 checkpoint from Modal storage. Optionally pulls HDF5 data + computes Mahalanobis centroid from training drones. |
| `scripts/export_onnx.py` | Exports encoder to ONNX format. Tests CoreML Execution Provider on M1 for ~2-5x speedup. |
| `scripts/edge_benchmark.py` | Benchmarks latency, throughput, RAM, model size. Generates `results/edge_benchmark.md` with SWaP-C numbers + extrapolation to Jetson/Pi/etc. |
| `scripts/live_demo.py` | Real-time waterfall + detection display. Three modes: synthetic (no files), HDF5 replay, I/Q file playback. |
| `requirements_demo.txt` | Pinned dependencies for M1 Mac. |

### Files generated (when you run the scripts)

| File | What it is |
|---|---|
| `models/lejepa_v11_best.pt` | Trained v11 encoder (downloaded from Modal) |
| `models/drone_centroid.npz` | Precomputed Mahalanobis centroid + covariance + threshold |
| `models/iris_encoder.onnx` | ONNX export of encoder (for fast inference) |
| `~/.iris_samples/drones.npz` | 200 holdout drone spectrograms (for offline demos) |
| `~/.iris_samples/matched_bg.npz` | 200 matched BG spectrograms (for offline demos) |
| `results/edge_benchmark.md` | SWaP-C benchmark report |

## The Demo Moment

Open laptop. Run:

```bash
python scripts/live_demo.py
```

A dark window opens with:
- **Top:** rolling RF spectrogram waterfall (most recent on the right)
- **Middle:** Mahalanobis distance line over time + red threshold line
- **Bottom:** alert banner: "⚠ DRONE DETECTED" when above threshold

Synthetic mode generates drone-like RF bursts every ~5 seconds. You'll see:
- Background noise → distance stays above threshold → "Monitoring..."
- Drone burst → distance drops below threshold → "⚠ DRONE DETECTED" in red
- Burst ends → distance rises → back to monitoring

## The Numbers You'll Quote

After running `python scripts/edge_benchmark.py`, you'll have a markdown report at `results/edge_benchmark.md` with the actual numbers. Expected on M1 Mac 8GB:

- **Latency:** ~5-15 ms per spectrogram (batch=1, ONNX CoreML)
- **Throughput:** ~70-200 fps
- **Model size:** ~13 MB ONNX
- **RAM:** ~500-800 MB peak
- **Encoder params:** 3.4M

Quote these in the meeting:
> *"IRIS runs at X ms per spectrogram on my M1 Mac — extrapolates to ~Y on Jetson Orin Nano. The encoder is 3.4M params, 13MB. This is the SWaP-C story for Samaritan OS."*

## Troubleshooting

### "checkpoint not found"
Run `python scripts/pull_from_modal.py` first. If Modal token isn't set, run:
```bash
modal token set --token-id YOUR_ID --token-secret YOUR_SECRET
```

### "no module named 'modal'"
```bash
pip install modal==0.67.44
```

### "MPS not available"
You need PyTorch 2.5+ on M1 Mac. Verify:
```python
import torch
print(torch.backends.mps.is_available())  # should be True
```

### "onnxruntime has no CoreMLExecutionProvider"
Install onnxruntime properly:
```bash
pip uninstall onnxruntime onnxruntime-coreml -y
pip install onnxruntime==1.20.1
```
CoreML EP is bundled in the standard `onnxruntime` package on M1 Macs.

### Live demo is slow / choppy
- Reduce FPS: `python scripts/live_demo.py --fps 4`
- Use ONNX export first: `python scripts/export_onnx.py` (live_demo uses PyTorch by default; we'll switch to ONNX in Phase 2)
- Close other apps (M1 8GB is tight)

### Want to test without GUI?
```bash
python scripts/live_demo.py --no-display
```
Prints detection results to console for 50 frames, then exits with stats.

## Next Steps

Once Phase 1 is working:

- **Phase 2:** Honest evaluation — recording-grouped CV, SNR curve, cross-dataset transfer. Run `python scripts/honest_eval.py` to generate the only honest drone RF detection numbers in the Indian market.
- **Phase 3:** RF-only intent classifier — the first killer demo.
- **Phase 4:** Remote ID spoof detector — the second killer demo.

See `/home/z/my-project/IRIS/build.md` for the full architecture and roadmap.
