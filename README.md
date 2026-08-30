# Apprentice

A voice-driven SO-101 arm that watches a human do a task through egocentric video (Meta glasses / World Context dataset), turns it into a step plan grounded in its own learned skills, explains the plan out loud, and executes it with ACT policies trained during the hackathon.

**Tracks:** Open Hardware + Human Data Transfers to Hardware (bonus) + Voice-Enabled Robotics (bonus).

- 📋 **[HACKATHON_PLAN.md](HACKATHON_PLAN.md)** — full strategy, architecture, dependencies, hour-by-hour build plan, demo script
- 🧾 **[BUILD_CARD.md](BUILD_CARD.md)** — required one-page build card (keep updated as we build)

## Repo layout (target)

```
orchestrator/   FastAPI state machine + skill runner (LeRobot v0.6.1)
planner/        World Context / glasses video → IMU segmentation → Gemini → step plan
voice/          ElevenLabs agent config + client tools
dashboard/      status page (plan, live step, camera, transcript)
scripts/        record / train / eval wrappers, port + camera setup ritual
```

## Quick start

See the workshop-pinned setup in [HACKATHON_PLAN.md §4–6](HACKATHON_PLAN.md): LeRobot `v0.6.1`, Python 3.12, `uv`, extras `[core_scripts,feetech]`.
