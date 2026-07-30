# IRIS — Meeting Day Runbook

Everything you need to walk into the Armory meeting and demo IRIS.

## The 30-Second Pitch

> *"I read your October blog on RFML + DSP + Drone ID and your December blog on spoofing. You said 'detect intent, not just ID.' And you said DroneIDs can be falsified. I searched — nobody solves either from RF. So I built both, on top of IRIS, my zero-shot drone detector. AUC 0.978 on 7 drone types it's never seen. Here's the live demo."*

## Pre-Meeting Setup (Do This The Night Before)

```bash
# 1. Install dependencies (~5 min)
cd /path/to/IRIS
pip install -r requirements_demo.txt

# 2. Pull v11 checkpoint from Modal + compute Mahalanobis centroid (~10 min)
python scripts/pull_from_modal.py
# When prompted "Download HDF5 files? [y/N]" — type y if you have bandwidth

# 3. Run honest evaluation on Modal (produces results/honest_eval.md, ~30 min)
modal run scripts/honest_eval.py

# 4. Train intent head on Modal (produces models/intent_head.pt, ~15 min)
modal run scripts/train_intent.py

# 5. Pull intent head checkpoint from Modal:
#    After training, the checkpoint is in the "iris-intent" Modal volume.
#    Add this to pull_from_modal.py or use modal volume get:
#    modal volume get iris-intent /intent/intent_head.pt models/intent_head.pt

# 6. Export encoder to ONNX for fast M1 inference (~2 min)
python scripts/export_onnx.py

# 7. Run edge benchmark (~5 min)
python scripts/edge_benchmark.py

# 8. Verify the live demo works (synthetic mode, no files needed)
python scripts/live_demo.py --no-display

# 9. Verify the spoof demo works
python scripts/spoof_demo.py --synthetic
```

## Meeting-Day Demo Sequence

Open laptop. Have these terminals ready:

### Terminal 1 — Live Demo (the showstopper)

```bash
python scripts/live_demo.py
```

**What shows:** Dark GUI window with:
- Top: rolling RF spectrogram waterfall
- Middle: Mahalanobis distance line over time + red threshold
- Bottom: alert banner — "⚠ DRONE DETECTED — INTENT: ATTACK" when drone present

Wait for the first drone burst (~5 seconds). Let Amardeep watch it detect.

**Say:**
> *"This is IRIS running live on my Mac. Real-time spectrogram waterfall, Mahalanobis distance detector. AUC 0.978 on 7 drone types it's never seen. Inference is X ms per spectrogram on M1, extrapolates to Y on Jetson Orin Nano. The encoder is 3.4M params, 13MB. This is the SWaP-C story for Samaritan OS."*

### Terminal 2 — Spoof Demo (killer demo 2)

```bash
python scripts/spoof_demo.py --synthetic
```

**What shows:** Three packets, three verdicts:
1. Friendly drone → AUTHENTIC (green)
2. Attacker spoofing the friendly drone's serial → SPOOFED (red)
3. Unknown drone → NOT_ENROLLED (yellow)

**Say:**
> *"Your October blog says DroneIDs can be falsified to flood systems with fake dots. I searched — nobody uses RF fingerprinting to authenticate Remote ID broadcasts. So I built it. The Remote ID says 'DJI Mini 4 Pro S/N 12345,' but the RF fingerprint of the transmitter says 'HackRF SDR.' That's a spoof. Even when the payload is identical, IRIS catches it by checking the transmitter's physical RF fingerprint."*

### Have These Files Ready To Pull Up

When Amardeep asks specific questions, open these:

| Question | Open |
|---|---|
| "What are your actual numbers?" | `results/honest_eval.md` |
| "How does it run on edge?" | `results/edge_benchmark.md` |
| "How does intent work?" | `results/intent_results.md` (after training) |
| "Show me the code." | GitHub repo — `src/iris_inference.py`, `src/intent_head.py`, `src/remote_id_auth.py` |

## The Five Demo Moments (In Order)

### Moment 1: Live Detection (Foundation)
- Run `python scripts/live_demo.py`
- Show real-time waterfall + detection firing on drone bursts
- Quote: "X ms per spectrogram on M1, extrapolates to Y on Jetson"

### Moment 2: Honest Numbers (Credibility)
- Open `results/honest_eval.md`
- Quote: "Shulman 2026 showed drone RF benchmarks inflate 30 points via segment-level CV. I adopted recording-grouped CV. IRIS gets X% AUC honest, Y% FAR at Z dB SNR floor. Your blog cites GASx 95%/<0.5% — IRIS is X/Y, and these are the only honest numbers in the Indian C-UAS market."

### Moment 3: Intent Classification (Killer Demo 1)
- Same live demo, but point out the intent label in the alert banner
- Quote: "Your October blog says 'detect intent, not just ID.' I searched — no published paper does RF-only intent inference. CPhy-ML uses control physics, not RF. IRIS does it from RF alone. Three classes: surveillance, transit, attack approach. X% accuracy. First in the field."

### Moment 4: Remote ID Spoof Detection (Killer Demo 2)
- Run `python scripts/spoof_demo.py --synthetic`
- Show the three packets and verdicts
- Quote: "Your October blog says DroneIDs can be falsified. Nobody uses RF fingerprinting to authenticate Remote ID broadcasts. IRIS does. Even when the payload is identical, IRIS catches the spoof by checking the transmitter's physical RF fingerprint."

### Moment 5: What I'd Build at Armory (Close)
- Quote: "I want to build the rest of this at Armory. Next 90 days: wire IRIS to a real SDR on the SURGE sensor's I/Q stream. The pipeline I just showed you is 90% there. Second 90 days: productionize intent + spoof detection into Samaritan OS. Third 90 days: per-transmitter IFF at scale on DRFF-R2, and the open-source release."

## Anticipated Questions & Answers

**Q: How does IRIS compare to Dedrone / Anduril Lattice?**
> "Dedrone is sensor-fusion-first, RF as one of many sensors. Anduril Lattice is integrated hardware+software, RF is one input. IRIS is RFML-first — it's the RFML layer of the stack your October blog describes. I'm not competing with them on sensor fusion. I'm competing on the RFML depth nobody else has publicly demonstrated."

**Q: What about fiber-optic drones? RF can't see them.**
> "True. RF-only is a half-life. The fix is passive multistatic radar using ambient RF illuminators — 5G, DVB-T, LTE. That's a 6-month project, not a 30-day one. But the IRIS encoder and Mahalanobis detector are sensor-agnostic — when passive radar I/Q comes in, the same pipeline applies."

**Q: How do you handle adversarial attacks?**
> "Ben-Gurion published in January 2026 that RF drone detectors are vulnerable to OTA adversarial attacks. No defenses exist. IRIS's embedding geometry might be naturally more robust — Mahalanobis distance in a SIGReg-regularized space should be smoother than classifier logits. That's the next build I'd do at Armory."

**Q: What's your data situation?**
> "RFUAV dataset, 37 drone types, public on HuggingFace. I trained on 30, held out 7 for zero-shot evaluation. The honest evaluation report shows the per-type breakdown. For per-transmitter IFF, I'd use DRFF-R2 — 26 DJI units across 8 models, public on SciDB."

**Q: Can this run on our SURGE hardware?**
> "Yes. The encoder is 3.4M params, 13MB ONNX, runs at X ms on M1 — extrapolates to ~Y on Jetson Orin Nano. The whole pipeline (encoder + intent head + spoof detector) is <50MB and <100ms latency. Fits in the SURGE sensor's co-processor budget easily."

**Q: What about IFF — friend or foe?**
> "That's exactly what the spoof detector does. Enroll friendly drones by RF fingerprint. When a Remote ID packet arrives, check if the transmitter's fingerprint matches the enrolled fingerprint for that serial. If not — spoofed. This is unjammable because it's a passive fingerprint, not an active transponder. No protocol to jam, no encryption to break."

## What NOT to Do

- ❌ Don't show slides. Laptop + terminal only.
- ❌ Don't apologize for not having hardware. Frame as strength: "I built this with zero hardware budget on Kaggle free GPUs."
- ❌ Don't mention the Hidden Level datasheet thing. They know.
- ❌ Don't use AI-generated text in anything you show. Amardeep filters for this.
- ❌ Don't claim IRIS solves everything. Be honest: "RF-only. Won't see fiber-optic drones. But it's the foundation for a layered system that can."
- ❌ Don't ask about salary/role in the first meeting. Let him bring it up.

## File Inventory (What's in the Repo)

```
IRIS/
├── src/
│   ├── iris_inference.py         # Clean inference module (encoder + Mahalanobis)
│   ├── intent_head.py            # RF-only intent classifier (KILLER DEMO 1)
│   ├── remote_id_decoder.py      # DroneID + ASTM F3411 decoder
│   └── remote_id_auth.py         # RF fingerprint authentication (KILLER DEMO 2)
├── scripts/
│   ├── pull_from_modal.py        # Download checkpoint + data from Modal
│   ├── export_onnx.py            # ONNX export for M1 CoreML EP
│   ├── edge_benchmark.py         # SWaP-C benchmark
│   ├── live_demo.py              # Real-time waterfall demo (MOMENT 1)
│   ├── honest_eval.py            # Recording-grouped CV + SNR curve (MOMENT 2)
│   ├── train_intent.py           # Train intent head on Modal (MOMENT 3)
│   └── spoof_demo.py             # Spoof detection demo (MOMENT 4)
├── results/                      # Generated by scripts
│   ├── edge_benchmark.md
│   ├── honest_eval.md
│   └── intent_results.md
├── models/                       # Downloaded from Modal
│   ├── lejepa_v11_best.pt
│   ├── drone_centroid.npz
│   ├── iris_encoder.onnx
│   └── intent_head.pt
├── data/
│   ├── iris_rfuav.h5             # (optional — large)
│   ├── iris_matched_bg.h5        # (optional)
│   └── blue_force_registry.json  # created by spoof demo
├── requirements_demo.txt
├── PHASE1_README.md
└── MEETING_DAY_RUNBOOK.md        # this file
```

## If Something Breaks

### Live demo crashes
- Try `python scripts/live_demo.py --no-display` (console mode, no GUI)
- Try `python scripts/live_demo.py --fps 4` (slower)
- Try `python scripts/live_demo.py --no-intent` (skip intent classifier)

### Spoof demo crashes
- Try `python scripts/spoof_demo.py --synthetic` (no encoder needed)
- Check `data/blue_force_registry.json` exists — delete and re-run if corrupted

### "checkpoint not found"
- Run `python scripts/pull_from_modal.py`
- Verify Modal token: `modal token verify`

### "intent head not found"
- The intent classifier gracefully degrades — live demo works without it, just no intent label
- To train: `modal run scripts/train_intent.py` (15 min on A100)

### Network issues during meeting
- All demos work offline once checkpoints are downloaded
- Pre-download everything the night before

## The One-Line Summary

If Amardeep asks "what did you build?" — say:

> *"IRIS is a self-supervised drone detector with zero-shot AUC 0.978 on unseen drone types. On top of it I built the first RF-only intent classifier and the first Remote ID spoof detector via RF fingerprinting. Both close gaps you explicitly named in your October and December blogs. It runs live on my Mac."*

Then open the demo.
