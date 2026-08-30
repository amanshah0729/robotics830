---
title: Muscle Memory — Visualization Track Master Plan
type: plan
status: active
tags: [hackathon, visualization, world-context, imu, retrieval]
created: 2026-08-30
updated: 2026-08-30
---

# Muscle Memory — search hundreds of hours of skilled work by motion alone

**One line:** We teach an IMU to see. A small motion encoder, trained contrastively at the event against frozen CLIP video embeddings of the World Context dataset, lands raw inertial signals in the same space as images — so a text query, a video moment, or **a judge shaking a phone** all retrieve matching moments from hundreds of hours of skilled human work, laid out on an explorable atlas.

**Track:** Visualization ($1,000 / $500). Hardware idea stays on ice (see `HACKATHON_PLAN.md`), and this codebase is deliberately robot-free.

---

## 1. Track expectations, extracted

What the organizers published, condensed to what scores:

**Rubric (all four must land):**
1. **Does it work?** — live, in front of judges, not a screenshot.
2. **How difficult / clever?** — technical novelty, not code volume, model size, or API count.
3. **Is it original?** — most teams will build video browsers and dashboards; the anti-list explicitly calls out "generic dashboards without a clear point."
4. **Does it land?** — a demo moment + a claim the data actually supports (they call out cherry-picking).

**Their own idea ladder** (we hit both rungs): "make hundreds of hours easier to **search, browse, segment, understand**" and "**derive something new** — depth, hand pose, contact, camera motion, scene structure."

**Hard requirements:** built substantially at the event (disclose everything pre-existing at kickoff); public GitHub repo; ≤3-min YouTube demo; one-page build card (what existed / what we added / central claim + evidence / limitations / external assets); 3-min live demo + 2-min Q&A; no slide deck needed.

**Explicitly encouraged:** heavy AI-coding-tool use, getting something working early then improving, honesty about limitations.

## 2. Why this wins

| Criterion | Our answer |
|---|---|
| Works | Every layer ships independently: explorer/atlas works with zero ML; text search works once CLIP ingest runs; motion search works with handcrafted features before the encoder ever trains; the learned encoder is the cherry on top. Fixture mode proves the plumbing before the flash drive is even mounted. |
| Difficult | Cross-modal self-supervised training **at the event**: an IMU tower distilled into CLIP space (IMU2CLIP-style, trained from scratch on this dataset). Plus derived events (contact spikes, chapter boundaries) from the IMU. |
| Original | Everyone else will search the video. We search the **motion** — the dataset's second modality that nobody opens — and close the loop with a live phone as a motion query device. |
| Lands | Judge holds the phone, mimes hammering, and the wall of matching hammering moments appears — from motion alone, no camera on them. The atlas of "the shape of human work" is the poster shot. Honest metrics on a held-out split back the claim. |

**Prior art disclosure (put this in the build card — it reads as scholarship, not weakness):** IMU→CLIP alignment was proposed by Meta's IMU2CLIP (2022) on Ego4D. What's ours, done at the event: training it from scratch on World Context, the handcrafted-baseline comparison, the contact/chapter derivations, the atlas + retrieval product, and the live phone-query loop.

## 3. What's already built (this repo, disclose at kickoff)

A complete scaffold, smoke-tested end to end on synthetic data — **no World Context data has been touched yet**:

```
musclememory/
  probe.py        run FIRST at the venue: verifies the real worldcontext API
  wc_adapter.py   uniform clip/frame/IMU access (real dataset or fixture)
  fixture.py      synthetic 8-task dataset so everything runs without the drive
  ingest.py       clips -> 4s windows: CLIP embedding + resampled IMU + features
  imu_features.py handcrafted features, contact-spike + chapter detection
  imu_encoder.py  the novel bit: contrastive IMU tower -> CLIP space (torch)
  index.py        search banks + 2D atlas (UMAP, PCA fallback)
  eval.py         held-out R@k + task-consistency vs baseline — build-card numbers
  server.py       FastAPI: atlas UI, text/motion search, phone websocket bridge
web/              atlas frontend, inspector, live view, phone sensor page
```

Verified in fixture mode: ingest → index → train (loss ↓, recall ↑) → export → eval → server; synthetic "hammering" motion retrieves hammer moments; a simulated phone stream retrieves whisking through the learned space and broadcasts to the live view; UI rendered in Chromium with zero console errors.

## 4. Dependencies

| Layer | Needs |
|---|---|
| Data | `WORLD_CONTEXT_EXPLORER_V3` **copied off the flash drive**; its `worldcontext` lib importable in our venv |
| Core (any laptop) | Python 3.11/3.12 + `requirements.txt` (numpy, scipy, fastapi, uvicorn, pillow) — fixture mode, explorer, feature-baseline motion search all run on this alone |
| ML | `requirements-ml.txt` (torch, open-clip-torch, umap-learn). **One CUDA machine** (gaming laptop or Colab) makes ingest ~20× faster and trains the encoder in minutes-to-an-hour |
| Demo | A phone: Android/Chrome works over plain LAN; iOS needs HTTPS → `cloudflared tunnel --url http://localhost:7860`. Phone hotspot beats venue Wi-Fi |
| Optional garnish | Gemini credits: auto-caption atlas clusters (sample frames near cluster centroids → short labels rendered on the map) |

## 5. Venue-day runbook

```bash
# 0. copy WORLD_CONTEXT_EXPLORER_V3 off the drive; cd there; uv sync; uv run wcx info
# 1. make worldcontext importable in our venv, then VERIFY THE API:
python -m musclememory.probe                  # fix wc_adapter.py where it complains
# 2. tiny real ingest to validate end to end (CPU is fine for 3 clips):
python -m musclememory.ingest --source worldcontext --embedder clip --max-clips 3
# 3. real ingest on the GPU box — shard/resume friendly, run big overnight:
python -m musclememory.ingest --source worldcontext --embedder clip --max-clips 200
python -m musclememory.index                   # UMAP atlas + search banks
python -m musclememory.server --source worldcontext --embedder clip
# 4. the novel bit (GPU, ~minutes on fixture-scale, ~1h at 100k windows):
python -m musclememory.imu_encoder train
python -m musclememory.imu_encoder export && python -m musclememory.index
python -m musclememory.eval                    # build-card numbers, honest split
```

Throughput planning: ViT-B/32 ≈ 500+ frames/s on a modest GPU, ~15/s on CPU. At 1 frame per 2 s stride, one hour of video ≈ 1,800 frames — so **~50 hours of dataset ≈ 90k windows ≈ 3 min GPU / 1.7 h CPU** of embedding time. IMU I/O will likely dominate; measure with the 3-clip ingest, then choose scale. Ingest is resumable and shardable across laptops (`--skip-clips/--max-clips`); npz outputs merge by directory.

## 6. Timeline (~30 h)

| Window | Focus | Gate |
|---|---|---|
| T+0–1 | Copy drive, `probe`, fix adapter guesses | probe clean |
| T+1–3 | 3-clip real ingest → index → server on real data | real atlas on screen |
| T+3–8 | Full ingest running on GPU box; meanwhile UI polish on partial index | ≥20 h of data indexed |
| T+8–14 | Train encoder v1; eval; iterate (epochs, window size) | beats feature baseline on task-consistency |
| T+14–20 | Phone demo hardening (hotspot, tunnel), cluster labels via Gemini, atlas polish | full demo loop 5× in a row |
| T+20–26 | Scale ingest overnight; final eval; **record YouTube video at peak** | video uploaded |
| Last 6 h | Build card, README, rehearse ×5, submit with 15 min spare | submitted |

Ruthless gates: encoder not beating baseline by T+16 → demo ships with feature-baseline motion search (it already works) and the encoder becomes "in progress" in limitations. Never hold the demo hostage to the research result.

## 7. Risk ladder

| Risk | Fallback (already built) |
|---|---|
| `worldcontext` API differs from guesses | `probe.py` pinpoints it; adapter isolates every touchpoint in one file |
| No CUDA machine materializes | Colab: zip repo + run ingest/train there, download `work/`; or CPU-ingest a curated 10–20 h subset overnight |
| Encoder underwhelms on real data | Feature-baseline motion search already demos; report both numbers honestly |
| Venue Wi-Fi / iPhone HTTPS | Phone hotspot; Android phone; cloudflared tunnel; worst case `POST /api/search/motion` from a laptop replaying a recorded IMU file |
| UMAP too slow / ugly | `--proj pca` flag exists; UMAP fit capped at 30k samples |
| Dataset too big for laptop disk | Ingest reads clips streaming; derived npz ≈ a few MB per clip; cap with `--max-clips` |

## 8. The 3:00 demo

- **0:00–0:30** — Atlas up. "Hundreds of hours of skilled work. Every dot is a moment. Colors are trades. We didn't organize it — the embeddings did."
- **0:30–1:00** — Text search ("pouring", "hammering") → moments light up. Click one: frame + IMU trace with contact spikes + chapters. "The accelerometer sees every strike."
- **1:00–2:15** — The kicker. Hand a judge the phone. They mime whisking / hammering / wiping. The wall updates live with matching moments — **from motion alone**. "The IMU learned to see. Nothing about you is on camera."
- **2:15–2:45** — The receipts: eval slide in the UI or README — held-out R@k vs. random, task-consistency vs. the handcrafted baseline. "Same numbers in the repo, same split, no cherry-picking."
- **2:45–3:00** — Limits, honestly: window-level granularity, tasks ≠ fine actions, encoder trained hours ago on one dataset.

## 9. Submission checklist

- [ ] Repo public, README quickstart runs
- [ ] YouTube ≤3:00 recorded at peak reliability (not 4 a.m.)
- [ ] Build card: pre-existing = this scaffold (pre-event, disclosed) + LeRobot-free stack + CLIP + IMU2CLIP idea; added at event = all real-data ingest, trained encoder, eval numbers, cluster labels, demo
- [ ] `work/eval.json` numbers pasted into build card verbatim
- [ ] Submission form (allow 15 min)
