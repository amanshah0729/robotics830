# Build Card — Muscle Memory

> One page. Keep this updated as we build — not at 4 a.m. before submission.

## What existed before the hackathon

- The World Context egocentric dataset (video + IMU) and its `worldcontext` explorer library (organizer-provided)
- OpenCLIP pretrained ViT-B/32 (frozen; we train nothing of it)
- The IMU→CLIP alignment idea has prior art: IMU2CLIP (Meta AI, 2022, on Ego4D)
- This repo's pipeline scaffold (adapter/ingest/encoder/index/eval/server/UI), written with AI coding tools pre-kickoff and smoke-tested **only on synthetic fixture data** — no World Context data touched before the event; disclosed here at kickoff

## What the team added during the event

<!-- Update as each lands -->
- [ ] `probe.py` run on the real dataset; adapter corrections for the actual API
- [ ] Real ingest: N clips / H hours → W windows (CLIP embeddings + IMU features)
- [ ] IMU encoder trained from scratch on World Context (contrastive, InfoNCE vs frozen CLIP targets)
- [ ] Atlas of the full ingest (UMAP), cluster labels
- [ ] Live phone-motion search demo hardened for the venue
- [ ] Held-out evaluation (numbers below)

## Central result / claim

**Motion alone retrieves what a person was doing.** An IMU encoder trained at the event on World Context places inertial windows in CLIP space: held-out motion retrieves its own video moment far above chance and groups by task above a handcrafted-feature baseline — including live motion from a phone the model never saw.

**Evidence (from `work/eval.json`, held-out clips, full-bank pool, same-clip neighbors excluded):**
- IMU→video retrieval: R@1 = __, R@5 = __, R@10 = __ (random: __), median rank __ of __
- Task-consistency@5: learned __ vs. handcrafted baseline __ (chance __)

## Limitations, failures, unfinished work

- Window-level (4 s) granularity; task labels are coarse — fine-grained actions cluster but aren't named
- Phone IMU differs from the capture device (placement, axes); per-window z-norm helps but transfer is imperfect — live demo is qualitative
- Encoder trained for hours, not days; numbers are a floor, not a ceiling
- (fill in what actually broke)

## External code, models, datasets, APIs, assets

World Context dataset + `worldcontext` lib · OpenCLIP (ViT-B/32, laion2b) · PyTorch · umap-learn · FastAPI/uvicorn · numpy/scipy/Pillow · Gemini API (cluster captions only, if used) · IMU2CLIP (idea citation, no code used)
