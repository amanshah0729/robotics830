# Build Card — Apprentice

> One page. Keep this updated as we build — not at 4 am before submission.

## What existed before the hackathon

- SO-101 leader/follower hardware (our kit + venue-provided arms); Feetech STS3215 servos
- LeRobot `v0.6.1` (calibration, teleop, recording, ACT policy implementation, training scripts)
- World Context egocentric dataset (video + IMU) and its `worldcontext` explorer library
- ElevenLabs Agents platform, Gemini API (via Concentrate credits), Lovable
- No project code existed at kickoff (this repo started empty)

## What the team added during the event

<!-- Update as each lands -->
- [ ] Teleop demo datasets recorded at the event: `<hf_user>/so101_pick_place_red` (N eps), `…_blue` (N eps)
- [ ] ACT policy checkpoints trained at the event (per skill), plus checkpoint eval results
- [ ] Orchestrator: FastAPI state machine + skill runner driving the SO-101
- [ ] Planner: IMU changepoint segmentation of World Context clips → keyframes → Gemini grounding into the robot's skill vocabulary → executable step plan
- [ ] Meta-glasses ingestion path (record → phone → pipeline)
- [ ] ElevenLabs voice agent with client tools (describe / run / stop / step / status) + narration
- [ ] Dashboard + demo scene

## Central result / claim

A single pipeline turns egocentric human video into actions a real robot performs:
World Context clip or fresh Meta-glasses video → IMU/VLM step plan → ACT skills execute on the SO-101, all controlled by voice.

**Evidence:** live demo; N/10 full-pipeline success runs in rehearsal (fill in real number); recorded runs in the demo video; eval episode success rates per skill (fill in).

## Limitations, failures, unfinished work

<!-- Be specific and honest — judges reward this -->
- Skill vocabulary is small (2–3 skills); "analogy mode" maps unfamiliar human actions to the nearest known skill rather than truly imitating them
- Success detection is timeout-based, not perception-based
- Fixed scene: policies are sensitive to lighting/camera geometry changes
- (fill in what actually broke)

## External code, models, datasets, APIs, assets

LeRobot v0.6.1 (Apache-2.0) · ACT (Zhao et al.) · World Context dataset + `worldcontext` lib · Gemini (via Concentrate) · ElevenLabs Agents/STT/TTS · MediaPipe (only if hand-teleop stretch used) · FastAPI/uvicorn · numpy/scipy
