# Muscle Memory

**Search hundreds of hours of skilled human work by motion alone.** A small IMU encoder, trained contrastively against frozen CLIP video embeddings of the World Context egocentric dataset, lands raw inertial signals in the same space as images — so a text query, a video moment, or a phone waved in your hand all retrieve matching moments, laid out on an explorable atlas.

**Track:** Visualization. 📋 Active plan: **[VIZ_PLAN.md](VIZ_PLAN.md)** · 🧾 [BUILD_CARD.md](BUILD_CARD.md) · 🦾 (on ice: [HACKATHON_PLAN.md](HACKATHON_PLAN.md), the SO-101 hardware option)

## Quickstart — no dataset needed

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
bash scripts/fixture_demo.sh          # synthetic data end-to-end, then open http://localhost:7860
```

You get the full product on synthetic data: the atlas, the inspector (frame + IMU trace + contact spikes + chapters), text search (stub), and live phone-motion search (`/phone` on a phone on the same network) via the handcrafted-feature space.

## With the World Context flash drive

```bash
python -m musclememory.probe                                        # verify the real API first
uv pip install -r requirements-ml.txt                               # torch, open-clip, umap
python -m musclememory.ingest --source worldcontext --embedder clip # GPU recommended
python -m musclememory.index
python -m musclememory.imu_encoder train && python -m musclememory.imu_encoder export
python -m musclememory.index && python -m musclememory.eval         # honest numbers
python -m musclememory.server --source worldcontext --embedder clip
```

## How it works

```
clips ──▶ 4s windows ──▶ frozen CLIP video embedding ─────────┐
              │                                               ▼
              └─▶ IMU (resampled + features + events)   contrastive
                          │                              InfoNCE ──▶ IMU tower in CLIP space
                          ▼                                               │
              contact spikes · chapters · rhythm                          ▼
                                                     text / example / LIVE PHONE MOTION
                                                            all query one atlas
```

Pipeline details, venue runbook, eval methodology, and the demo script live in [VIZ_PLAN.md](VIZ_PLAN.md).
