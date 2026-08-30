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

## On Meta Ray-Ban Display

The dataset was captured on head-mounted cameras, so head-mounted IMU is the
one live query device with no placement gap to the training distribution — the
phone and watch clients have to fight a wrist/pocket domain shift that the
glasses simply don't have.

```bash
python -m musclememory.server --source fixture --embedder fake   # or worldcontext
./scripts/tunnel.sh                                              # glasses require HTTPS
```

In the Meta AI app: Settings > App Info > tap **App version** five times to turn
on Developer Mode, then App Settings > App Connections > **Web Apps > Add a Web
App** pointing at `<tunnel-url>/glasses`. It appears in the glasses app grid
immediately. Swipe to move focus, index-pinch to select. No Wearables Device
Access Toolkit, org, or app ID is involved — Web Apps skip all of it.

Needs glasses software `v125`+ and Meta AI app `v272`+. The **Diagnostics**
screen reports websocket state, IMU permission and sample rate, a rolling log
of gesture-to-key events, and an image-refresh probe, so a dead link is one
glance to identify rather than a guess.

### Camera into the glasses

MRBD Web Apps cannot open a camera, and the paired phone is a transport relay
rather than something the glasses page can address — so frames travel out to
the server and back. Open `<tunnel-url>/camera` in a normal phone browser,
press Start, then pick **Camera feed** on the glasses.

```
phone /camera  ──POST jpeg──▶  /api/cam/push  ──▶  newest frame only
                                                        │
glasses /glasses  ◀──GET /api/cam/latest, ~5 fps ───────┘
```

The glasses side swaps `<img>.src` rather than using `<video>`, which is
undocumented on MRBD; frame-at-a-time also suits a high-latency link, since
each frame stands alone and a dropped one costs nothing. Swapping the phone
for the arm's camera means changing only what POSTs to `/api/cam/push`.

Expect a photographic feed to look washed out: the waveguide adds light rather
than painting it, so mid-tones compete with the room. It is the right way to
prove the transport, but bright vector overlays on black are what actually
read on this display.

## On the SO-101 arm

Both robot tools run in a **LeRobot venv, not this repo's** — install once per machine:
[huggingface/lerobot](https://github.com/huggingface/lerobot) `v0.6.1` with the `feetech`
extra, plus `placo` for IK (`uv pip install placo` inside that venv). Calibrate your own
arm once (`lerobot-calibrate --robot.type=so101_follower --robot.port=<PORT> --robot.id=<ID>`);
find `<PORT>` with `lerobot-find-port` — it changes on every replug.

**Drive the arm like a video game** (WASD + IK claw mode, TAB for direct joint mode,
H glides home, ESC quits — full keymap in the script header):

```bash
<lerobot-venv>/bin/python scripts/keyboard_ee_teleop.py --port <PORT> --id <ID>
```

First run fetches the SO-ARM100 URDF (the script prints the exact command if it's
missing). On macOS, grant your terminal app Accessibility permission or the keys
silently do nothing.

**Perform a retrieved moment on the robot** — the muscle-memory bridge polls the demo
server for perform-jobs and re-synthesizes the matched window's motion character on
the arm (rhythm and shape, not learned manipulation — say so on stage):

```bash
<lerobot-venv>/bin/python -m musclememory.perform --port <PORT> --robot-id <ID>   # --dry-run to preview
```

### When it doesn't work

`GET /api/health` reports every hop at once, so a failure can be located
instead of guessed at:

```bash
curl -s <tunnel>/api/health | python3 -m json.tool
```

| Symptom | Look at | Means |
| --- | --- | --- |
| Glasses feed blank | `camera.frames_pushed` = 0 | phone isn't pushing; reload `/camera` and press Start |
| Feed frozen | `camera.last_frame_age_s` large | phone stopped capturing — iOS suspends it when the screen sleeps |
| Feed blank, frames arriving | `camera.pulls_by_viewer` climbing | glasses are asking and not painting; check the feed footer's `req/ok/err` |
| Swipes do nothing | `jog.drivers` = 0 | nothing is driving the arm; the teleop screen says **NO ROBOT** in red |
| Arm moves wrong way | `jog.last.axis` | command is arriving; flip that pair in `GLASSES_AXES` |
| Swipes register, arm still | — | his loop is paused (P), or `--glasses-hold` is too short to register a tick |

On the glasses themselves: **Diagnostics** covers the passive layers (websocket,
IMU permission and rate, gesture log, image refresh), the **Feed** footer shows
`fps`, `req/ok/err` and `ws`/`poll`, and **Teleop** shows sent, acked, and
whether any robot is listening. Between them, every hop reports its own state
without a laptop.

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
