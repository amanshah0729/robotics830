---
title: Apprentice — SO-101 Hackathon Master Plan
type: plan
status: active
tags: [hackathon, SO101, LeRobot, elevenlabs, gemini, world-context]
created: 2026-08-30
updated: 2026-08-30
---

# Apprentice — an SO-101 that watches humans work, then does the work

**One line:** A voice-driven SO-101 that ingests egocentric human video (Meta glasses live capture + the World Context dataset), uses the IMU + a VLM to turn what the human did into a step plan grounded in the robot's own learned skills, explains the plan out loud, and then executes it with ACT policies trained during the hackathon.

**Track strategy:** Enter **Open Hardware** (general track) + **both bonus tracks** (Human Data Transfers to Hardware, Voice-Enabled Robotics). This is the only configuration where one project is eligible for everything: $1,000 (1st) + $500 (bonus) + $1k ElevenLabs credits. Visualization cannot stack with the Human Data bonus (it requires an Open Hardware project).

---

## 1. Why this wins

Judging criteria → how we score on each:

| Criterion | Our answer |
|---|---|
| Does it work? | Every layer demos independently (teleop → trained skill → voice → video-to-plan), so partial failure still leaves a working demo. Graceful-degradation ladder in §8. |
| Difficult / clever? | Full pipeline: human egocentric video → IMU changepoint segmentation → VLM skill grounding → learned policy execution on real hardware. The IMU use is the clever bit — almost no one will touch the dataset's second modality. |
| Original? | Most teams will do "voice + teleop" or "plain ACT pick-and-place." The bridge from the sponsor's actual human-experience dataset to robot action is the story the sponsors themselves care about. |
| Does it land? | The demo has a magic moment: a teammate does a task wearing the glasses, asks the robot "what did you see?", the robot explains it and then does it. |

It also respects the explicit rules: policies are trained **at the event** from demos we record **at the event**; the World Context data **actually drives the plan that moves the robot** (bonus-track requirement: data used for "perception, planning, control… a real part of the physical system"); everything preexisting is disclosed in the build card.

**What we deliberately avoid** (from the organizers' anti-list): no hardcoded demo passed off as learned behavior (anything scripted gets labeled as scripted), no dashboard-for-its-own-sake, no API pile without an idea.

---

## 2. The concept

Three layers, one loop:

```
        WATCH                      THINK                       ACT
┌─────────────────────┐   ┌──────────────────────┐   ┌─────────────────────┐
│ Meta glasses clip    │   │ IMU changepoints →    │   │ Orchestrator runs    │
│ or World Context     │──▶│ keyframes → Gemini    │──▶│ ACT checkpoints      │
│ clip (video + IMU)   │   │ grounds each segment  │   │ per step on SO-101,  │
└─────────────────────┘   │ into robot skills →   │   │ narrates progress    │
                          │ ordered step plan     │   └─────────────────────┘
                          └──────────────────────┘             ▲
                                     ▲                          │
                          ┌──────────┴──────────────────────────┴──────┐
                          │  VOICE (ElevenLabs agent): "what did you   │
                          │  see?" / "go ahead" / "stop" / "skip 2"    │
                          └────────────────────────────────────────────┘
```

**Skills** (the robot's vocabulary — keep it to 2–3, trained during the event):

1. `pick_place_red` — pick red block, drop in left bin
2. `pick_place_blue` — pick blue block, drop in right bin
3. (stretch) `push_to_zone` — push the sponge/eraser to a taped zone

**The "analogy" framing** (say this on stage, it's honest and it lands): when the human video shows something the robot can't literally do — a World Context clip of skilled work — the planner maps each human action onto the *nearest skill the robot knows*. Hundreds of hours of human experience become robot-executable plans, at the resolution of the robot's current vocabulary. That's the entire thesis of human-data-to-hardware, demonstrated end to end.

---

## 3. Architecture & interface contracts

Write these contracts in hour 1 and integrate against mocks, so the robot, voice, and planner people never block each other.

```
[Mic/speaker] ⇄ ElevenLabs Agent (STT + LLM + TTS, client tools)
                     │ client tool calls
                     ▼
             Orchestrator — FastAPI on the robot laptop
             state machine: IDLE → PLANNING → READY → RUNNING(step i) → DONE/STOPPED
                     │                                    ⇄  Dashboard (WebSocket)
                     ▼
             Skill Runner (Python, LeRobot API, preloaded ACT checkpoints)
                     ▼
             SO-101 follower + front (+ wrist) camera
```

HTTP contract (mock this first):

```
GET  /skills                    → [{"id": "pick_place_red", "desc": "..."}]
POST /plan   {"video_path" | "wc_clip_id"}
                                → {"steps": [{"skill": "...", "evidence_ts": 12.4,
                                              "human_action": "picked up the red mug"}]}
GET  /status                    → {"state": "RUNNING", "step": 1, "of": 3}
POST /run    {"steps": [...]}   → 200 (async; progress via /status + WS)
POST /stop                      → 200 (immediate halt, torque hold)
```

Voice intents map 1:1 onto these: DESCRIBE_PLAN, RUN, STOP, RUN_STEP(n), STATUS.

---

## 4. Exact dependencies

### 4.1 Hardware

| Item | Spec / notes |
|---|---|
| SO-101 leader + follower | Our kit; venue arms are hot spares (calibrate a spare follower early). |
| Power supplies | Match servo voltage to PSU per workshop table: leader 7.4 V motors → 5 V supply; follower 7.4 V → 5 V; follower 12 V → 12 V. Stop if unknown. |
| USB cables ×2 + (ideally) powered hub | Port names change on replug — re-run find-port ritual each session. |
| Front workspace camera | Any USB webcam, 640×480@30. Clamp it; tape the clamp position. |
| Wrist camera (strongly recommended) | Second webcam on the SO-101 wrist mount. Biggest single ACT reliability boost. |
| Training GPU | Teammate's NVIDIA laptop, or Colab (T4 is enough), or any cloud GPU. ACT on 30–50 episodes ≈ 1.5–3 h to a usable checkpoint. Mac MPS works but is an overnight run. |
| Meta glasses + phone | Record → auto-sync to Meta AI app → AirDrop/Drive to laptop. **Do not attempt live streaming** (SDK rabbit hole). Fallback: phone video held at chest height. |
| Props | 2 colors of foam blocks, 2 bins, gaff tape for reset marks, a desk lamp (lighting consistency between training and demo is a top-3 failure cause), extension cord. |
| Mic | AirPods or any near-field mic; venue noise kills far-field STT. Push-to-talk. |

### 4.2 Software (pinned)

| Layer | Dependency |
|---|---|
| Robot | LeRobot `v0.6.1`, Python 3.12, uv, ffmpeg, extras `[core_scripts,feetech]` (exactly per our workshop guide) |
| Dataset | `WORLD_CONTEXT_EXPLORER_V3` folder **copied off the flash drive**, `uv sync`, `worldcontext` Python lib (`Dataset.open()`, `clip.frame()`, `clip.imu()`) |
| Planner | `google-genai` (direct Gemini key) or OpenAI-compatible client via Concentrate; ffmpeg frame sampling; numpy/scipy (+`ruptures` optional) for IMU changepoints |
| Voice | `elevenlabs` Python SDK (Agents/Conversational AI + client tools; needs `pyaudio`), agent configured at elevenlabs.io |
| Orchestrator | FastAPI + uvicorn + websockets |
| Dashboard | Local single-page HTML over the orchestrator WS (default) or Lovable (only if someone is free — hosted Lovable needs a tunnel like `cloudflared` to reach the laptop) |
| Data shuttle | Hugging Face account + `huggingface-cli login` (write token) to move recorded datasets to the training GPU |

### 4.3 Credits/accounts to claim in the first hour

- **ElevenLabs**: Discord → #coupon-codes → Start Redemption → registration email → coupon.
- **Concentrate**: invite already in the Luma-registration inbox; generate key ($25 any-model). Gemini $100 keys handed out at the event.
- **Lovable**: code `COMM-BER-R2N7` at checkout on Pro Plan 1 monthly (redeem before event ends).
- **Hugging Face**: everyone logs in; create org or share one namespace for datasets/checkpoints.

---

## 5. Build phases (~30 h; adapt to the real clock)

The single biggest scheduling trick: **pipeline overlap** — train skill A on the GPU while recording skill B on the robot.

| Window | Robot person | Voice/orchestrator person | Planner/data person | Integrator/demo person |
|---|---|---|---|---|
| T+0–2 | Finish workshop guide: calibrate both arms, verify teleop | Claim all credits; write API contracts; **mock robot server** | Copy WC folder; `uv sync`; open explorer; shortlist 3–5 demo-relevant clips | Scene build: clamp arms, mount cameras, tape reset marks, lamp |
| T+2–4 | Freeze task design; 5 throwaway episodes to tune camera/scene | ElevenLabs agent talking to mock (all 5 intents) | IMU segmentation on a WC clip (plot it — this is also a demo visual) | Repo scaffolding, `.env`, dashboard skeleton |
| T+4–7 | Record skill A (~30 eps, 15 s each), push to HF, **start training A**; record skill B | Orchestrator state machine real (still mock runner) | Gemini prompt → step-plan JSON from keyframes; iterate on 2 clips | Help recording (resetting scene is a 2-person job) |
| T+7–12 | Train B; eval A checkpoints from 20k steps; more episodes if <50% success | Wire runner: load checkpoint, run policy loop, timeouts, /stop | Glasses ingestion path (record → phone → laptop → /plan) | First end-to-end attempt with whatever works |
| T+12–18 | Best-checkpoint selection; small scene randomization only if reliable | Narration polish; barge-in "stop"; failure phrases | WC-clip → plan → robot "analogy" path | Dashboard shows plan/step/camera; **record b-roll now** |
| T+18–24 | Reliability drills: 10 full runs, count successes, fix flakiest link | Same | Same | Same |
| Last 6 h | **Freeze features.** Recalibrate nothing unless broken | Record the ≤3-min YouTube video at peak reliability | Build card + README finalized | Rehearse live demo ×5 with roles; stage fallback assets |

Checkpoint gates (be ruthless): **T+7**: if teleop/recording still fights us → drop to 1 skill. **T+12**: if no ACT checkpoint ever succeeds → pivot ladder §8. **T+24**: feature freeze, no exceptions.

---

## 6. Recipes

### 6.1 Skills: record → train → eval

Record (per skill — matches our workshop §7, scaled up; add the wrist camera if mounted):

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=hack_follower \
  --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},
                    wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}' \
  --teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=hack_leader \
  --dataset.repo_id=<hf_user>/so101_pick_place_red \
  --dataset.single_task="Pick up the red block and place it in the left bin" \
  --dataset.num_episodes=30 --dataset.episode_time_s=15 --dataset.reset_time_s=8 \
  --dataset.fps=30 --dataset.push_to_hub=true
```

Recording discipline (this is where the win is actually made): same operator for all episodes; smooth, unhurried motions; object placed within a taped ~5×5 cm region (small randomization, not none); always finish in the same rest pose; right-arrow to end an episode early, left-arrow to discard a bad one — **discard aggressively**.

Train (on the GPU machine; verify flags against `lerobot-train --help` on v0.6.1):

```bash
lerobot-train \
  --dataset.repo_id=<hf_user>/so101_pick_place_red \
  --policy.type=act --policy.device=cuda \
  --output_dir=outputs/train/act_pick_red --job_name=act_pick_red \
  --steps=60000 --save_freq=10000 --batch_size=8 --wandb.enable=false
```

Evaluate checkpoints early (20k is often already usable for a fixed scene) by running the policy for real:

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=hack_follower \
  --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
  --policy.path=outputs/train/act_pick_red/checkpoints/020000/pretrained_model \
  --dataset.repo_id=<hf_user>/eval_pick_red --dataset.single_task="eval" \
  --dataset.num_episodes=5 --dataset.episode_time_s=20 --dataset.push_to_hub=false
```

Golden rule: the demo table, camera positions, lighting, and calibration at judging must be **identical** to training time. Tape everything; bring the lamp; don't replug cameras without re-checking indices.

### 6.2 Planner: video (+ IMU) → step plan

IMU changepoint segmentation on a World Context clip (also our best dashboard visual):

```python
import numpy as np
from scipy.ndimage import uniform_filter1d
from worldcontext import Dataset

data = Dataset.open()
clip = data.clips[0]                       # or filter by clip.task_id
imu = clip.imu(start_s=0.0, end_s=None)    # verify exact end-of-clip API in README_FIRST.md
energy = np.linalg.norm(imu.acceleration, axis=1)
smooth = uniform_filter1d(energy, size=int(1.0 * imu_rate))
# sustained low-motion valleys = action boundaries
threshold = smooth.mean() - 0.5 * smooth.std()
boundaries = find_valley_centers(smooth < threshold, min_gap_s=2.0, rate=imu_rate)
keyframes = [clip.frame(t, width=640).image for t in segment_midpoints(boundaries)]
```

Grounding via Gemini (frame-sampling keeps it working through Concentrate's OpenAI-compatible router too; use inline video only with a direct Gemini key):

```
System: You map human actions in egocentric video onto a fixed robot skill list.
Skills: [{"id": "pick_place_red", "desc": "pick a red object, drop in left bin"}, ...]
Input: keyframe images in temporal order (timestamps attached), one per detected action segment.
Output STRICT JSON: {"steps": [{"skill": "...", "evidence_ts": <sec>,
                     "human_action": "<what the human actually did>"}]}
Rules: choose the closest skill by analogy; if no skill applies, use "skill": "none" and
say why in human_action. Never invent skills.
```

For a Meta-glasses clip (no IMU): sample frames at 1 fps with ffmpeg and let Gemini both segment and ground — same output schema, so downstream code doesn't care.

### 6.3 Voice: ElevenLabs agent

Configure an agent at elevenlabs.io (Agents/Conversational AI): low-latency voice (e.g. Flash), system prompt = "You are the voice of Apprentice, a robot arm…", and **client tools**: `describe_plan`, `run_plan`, `stop`, `run_step`, `status`, each just calling the orchestrator's HTTP API. Python side (verify against current SDK docs):

```python
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ClientTools
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

tools = ClientTools()
tools.register("run_plan",  lambda p: requests.post(f"{ORCH}/run",  json=cached_plan).json())
tools.register("stop",      lambda p: requests.post(f"{ORCH}/stop").json())
tools.register("describe_plan", lambda p: requests.get(f"{ORCH}/status").json())

conv = Conversation(ElevenLabs(api_key=KEY), AGENT_ID, requires_auth=True,
                    audio_interface=DefaultAudioInterface(), client_tools=tools)
conv.start_session()
```

Fallback voice path (2 h, zero-magic, keep in back pocket): push-to-talk mic → ElevenLabs STT (Scribe) → regex/LLM intent → orchestrator → ElevenLabs TTS for replies. Same demo, more manual.

`stop` must not go through the LLM round-trip alone — also bind a keyboard kill (spacebar) in the orchestrator, and keep the follower PSU switch within reach. Narrate steps with short pre-generated TTS lines cached to disk so narration survives venue Wi-Fi.

### 6.4 Skill runner

Custom loop (module paths moved between LeRobot versions — check imports against v0.6.1):

```python
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.policies.act.modeling_act import ACTPolicy

robot = SO101Follower(SO101FollowerConfig(port=PORT, id="hack_follower", cameras=CAMS))
robot.connect()
policies = {sid: ACTPolicy.from_pretrained(path) for sid, path in CHECKPOINTS.items()}

def run_skill(sid, max_s=25):
    policy = policies[sid]; policy.reset(); t0 = time.time()
    while time.time() - t0 < max_s and not STOP.is_set():
        obs = robot.get_observation()
        action = policy.select_action(prepare(obs))
        robot.send_action(action)
    return_to_rest(robot)
```

Success detection is timeout-based ("run the skill for N seconds, return to rest") — fine for the demo, and we say so honestly in limitations.

### 6.5 Dashboard

Default: one static HTML page served by the orchestrator — plan steps with live highlighting, current state, camera snapshot, transcript. Judges see the robot, not the screen; the screen just proves the pipeline is real (show the IMU segmentation plot!). Lovable version only if someone is idle — it needs a `cloudflared` tunnel to reach the laptop, and it also makes a great submission landing page.

---

## 7. Team split (4 people; collapse Integrator into Voice at 3, drop dashboard at 2)

- **A — Robot**: calibration, scene, all recording, training runs, checkpoint evals. Owns reliability.
- **B — Voice + Orchestrator**: contracts, state machine, ElevenLabs agent, safety stop.
- **C — Planner/Data**: World Context ingestion, IMU segmentation, Gemini grounding, glasses path.
- **D — Integrator/Demo**: scene logistics, dashboard, glue, b-roll, build card, YouTube video, rehearsal direction.

---

## 8. Risk ladder — every risk has a prebuilt fallback

| Risk | Mitigation / fallback |
|---|---|
| ACT never succeeds | Tighter object placement (tape a single position); 20 more episodes; 1 skill instead of 3. Last resort: demo teleop + the full watch→plan→voice pipeline, and show the policy attempting — labeled honestly. **Never pass scripted motion off as learned.** |
| Policy flaky at demo time | It's almost always scene drift: lighting, camera bump, table height. Tape, lamp, no replugging. Re-run 10× drill after any change. |
| Glasses transfer slow/broken | Pre-recorded glasses clip staged; attempt live second. Phone video is plan C — same pipeline. |
| Venue noise kills STT | Near-field push-to-talk mic; typed command box hidden in dashboard. |
| Venue Wi-Fi dies | Phone hotspot; pre-generated TTS narration clips on disk; pre-computed plan JSON for the WC clip cached. |
| No CUDA machine | Colab T4 (datasets are small, upload is fast); start training by T+7 so overnight is the buffer. |
| USB/camera indices shift | Re-run `lerobot-find-port` / `lerobot-find-cameras` ritual at every session start; keep a `ports.env`. |
| Calibration mismatch | Calibration lives on the robot laptop keyed by `--robot.id` — keep the robot on ONE laptop; recalibrating a spare arm takes ~10 min, do it early, not at 2 am. |
| Total hardware failure by T+12 | Pivot to **Visualization track**: the planner + IMU segmentation becomes "a semantic action-browser over the World Context dataset" — same code, different framing. |

---

## 9. The 3:00 demo

- **0:00–0:20** — Teammate wearing Meta glasses sorts the blocks by hand. "Nobody programmed what happens next."
- **0:20–0:50** — Video lands in the pipeline; dashboard shows segments + extracted plan. Voice: "Apprentice, what did you just watch?" → robot explains the steps in its own voice.
- **0:50–2:20** — "Go ahead." Robot executes with narration. Mid-run: "Stop." (it stops — rehearse this) … "Continue."
- **2:20–2:45** — The kicker: load a real World Context clip of skilled human work → plan appears mapped onto the robot's skill vocabulary → robot performs the analog. "Hundreds of hours of human experience, compiled to robot action."
- **2:45–3:00** — Honest limits: 2–3 skills, fixed scene, timeout success. "The vocabulary is small. The bridge is real."

Q&A prep (2 min): exactly what is learned vs. scripted; how the dataset mattered (segmentation + grounding source); failure modes; what we'd do with a week.

Record the YouTube video **before** the final hours; live demo gets the same script with fallback assets staged.

---

## 10. Submission checklist

- [ ] Repo **public** (this repo) — code, README with run instructions, this plan
- [ ] YouTube demo ≤ 3:00
- [ ] One-page build card (`BUILD_CARD.md` — keep it updated as we go, not at 4 am)
- [ ] Disclose preexisting: SO-101 hardware + LeRobot v0.6.1, World Context dataset + `worldcontext` lib, ElevenLabs/Gemini/Concentrate APIs, Lovable
- [ ] Built-at-event list: recorded datasets, trained ACT checkpoints, orchestrator, planner (IMU segmentation + grounding), voice agent + tools, dashboard, demo
- [ ] Submission form (allow 15 min)

---

## 11. Alternatives we considered (and why not)

- **Visualization-track search engine over the dataset** — solid and lower-risk, but wastes our hardware advantage and is ineligible for the $500 bonus. Kept as the §8 pivot.
- **Bimanual dual-SO-101 coordination** — flashy, but no data story and no voice leverage; pure mechanics. Optional garnish only if everything else is done (venue has spare arms).
- **Hand-tracking teleop from the glasses ("your hand is the leader arm")** — genuinely original; kept as a stretch goal if we're ahead at T+18 (MediaPipe hand → wrist pose → IK → follower). Demo line: "we replaced the leader arm with your hand."
