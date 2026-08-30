# Build Card — Muscle Memory

> One page. Keep this updated as we build — not at 4 a.m. before submission.

## What existed before the hackathon

- The World Context egocentric dataset (video + IMU) and its `worldcontext` explorer library (organizer-provided)
- OpenCLIP pretrained ViT-B/32 (frozen; we train nothing of it)
- The IMU→CLIP alignment idea has prior art: IMU2CLIP (Meta AI, 2022, on Ego4D)
- This repo's pipeline scaffold (adapter/ingest/encoder/index/eval/server/UI), written with AI coding tools pre-kickoff and smoke-tested **only on synthetic fixture data** — no World Context data touched before the event; disclosed here at kickoff

## What the team added during the event

<!-- Update as each lands -->
- [x] `probe.py` run on the real dataset — adapter guesses all held (424 clips, 50 tasks, 201.8 Hz IMU with gyro + timestamps)
- [x] 6× ingest speedup: threaded sequential h264 decode + per-clip IMU cache (~50s → ~8s/clip; full dataset in 48 min on an M4 laptop, no CUDA)
- [x] Real ingest: 424 clips / ~35 h → 62,974 windows (CLIP ViT-B/32 embeddings + IMU features), zero failures
- [x] IMU encoder trained from scratch on World Context (contrastive, InfoNCE vs frozen CLIP targets; 15 epochs, ~2 min on Apple MPS)
- [x] Atlas of the full ingest (UMAP, 40k points)
- [ ] Cluster labels
- [x] Voice search (Web Speech → CLIP text tower) and live vision search (`/api/search/image`: camera or screen-captured smart-glasses video call → CLIP image tower)
- [x] Apple Watch app streaming real wrist IMU into the live motion search (`watch/`)
- [ ] Live phone/watch-motion demo hardened for the venue
- [x] Held-out evaluation (numbers below)

## Central result / claim

**Motion alone retrieves what a person was doing.** An IMU encoder trained at the event on World Context places inertial windows in CLIP space: held-out motion retrieves its own video moment far above chance and groups by task above a handcrafted-feature baseline — including live motion from a phone the model never saw.

**Evidence (from `work/eval.json`, 39 held-out clips = 5,793 query windows, full 62,974-window pool, same-clip neighbors excluded):**
- IMU→video retrieval: R@1 = 0.14%, R@5 = 0.55%, R@10 = 1.24% (random R@10 = 0.016% — **~78× chance**), median rank 978 of 62,974 (top 1.6%)
- Task-consistency@5: learned **0.480** vs. handcrafted baseline 0.184 (chance 0.020) — **2.6× the baseline, 24× chance**

## Limitations, failures, unfinished work

- Window-level (4 s) granularity; task labels are coarse — fine-grained actions cluster but aren't named
- Phone/watch IMU differs from the capture device (placement, axes); per-window z-norm helps but transfer is imperfect — live demo is qualitative
- Encoder trained for ~2 minutes (15 epochs, loss still falling); numbers are a floor, not a ceiling
- One clip (`clip_niryrodh42oqs`, electrical-wiring-assembly) has a flat IMU recording — 119 near-constant windows kept in the bank; a transient NaN warning appears in train/eval matmuls (stored data verified finite; metrics are computed over all queries, so this depresses rather than inflates them)
- Vision search is plain CLIP nearest-neighbor (no training) — the trained contribution is the IMU tower; text/vision going through frozen CLIP is standard

## External code, models, datasets, APIs, assets

World Context dataset + `worldcontext` lib · OpenCLIP (ViT-B/32, laion2b) · PyTorch · umap-learn · FastAPI/uvicorn · numpy/scipy/Pillow · Gemini API (cluster captions only, if used) · IMU2CLIP (idea citation, no code used)
