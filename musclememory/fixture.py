"""Synthetic stand-in for the World Context dataset.

Eight fake 'tasks', a few clips each. Each task has a distinct motion signature
(oscillation frequency/amplitude, impacts, drift) and a distinct visual style
(background hue, moving blob), so embeddings cluster by task and retrieval is
meaningfully testable end-to-end without the flash drive.
"""

from __future__ import annotations

import numpy as np

from . import config
from .wc_adapter import ClipHandle

TASKS = {
    # task_id: (osc_freq_hz, osc_amp, impact_rate_hz, hue_deg)
    "whisk_batter": (4.5, 3.0, 0.0, 20),
    "hammer_nail": (1.2, 1.0, 1.2, 60),
    "pour_coffee": (0.2, 0.4, 0.0, 100),
    "sand_wood": (2.8, 2.0, 0.0, 140),
    "screw_bolt": (1.8, 1.2, 0.0, 190),
    "wipe_table": (1.0, 1.5, 0.0, 230),
    "knead_dough": (0.7, 2.2, 0.0, 280),
    "solder_joint": (0.1, 0.2, 0.0, 330),
}
CLIPS_PER_TASK = 5
IMU_RATE = 100.0


def _hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
    i = int(h / 60.0) % 6
    f = h / 60.0 - int(h / 60.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return np.array(rgb)


class _FixtureBackend:
    def __init__(self, seed: int, task_id: str, duration_s: float):
        self.rng = np.random.default_rng(seed)
        self.task_id = task_id
        self.duration_s = duration_s
        self.freq, self.amp, self.impacts, self.hue = TASKS[task_id]
        self.phase = self.rng.uniform(0, 2 * np.pi)

    def frame(self, t: float, width: int) -> np.ndarray:
        h, w = int(width * 0.75), int(width)
        base = _hsv_to_rgb(self.hue, 0.45, 0.55)
        img = np.ones((h, w, 3), dtype=np.float32) * base
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # a blob that moves with the task's rhythm — gives time structure to embeddings
        cx = w * (0.5 + 0.3 * np.sin(2 * np.pi * self.freq * 0.25 * t + self.phase))
        cy = h * (0.5 + 0.25 * np.cos(2 * np.pi * self.freq * 0.17 * t))
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (w * 0.08) ** 2)))
        img += blob[..., None] * (1.0 - base) * 0.9
        img += self.rng.normal(0, 0.02, img.shape).astype(np.float32)
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    def imu_window(self, t0: float, t1: float):
        n = max(2, int((t1 - t0) * IMU_RATE))
        t = t0 + np.arange(n) / IMU_RATE
        osc = self.amp * np.sin(2 * np.pi * self.freq * t + self.phase)
        acc = np.stack(
            [
                osc + self.rng.normal(0, 0.3, n),
                0.5 * self.amp * np.sin(2 * np.pi * self.freq * t * 1.3) + self.rng.normal(0, 0.3, n),
                9.81 + 0.2 * osc + self.rng.normal(0, 0.3, n),
            ],
            axis=1,
        ).astype(np.float32)
        if self.impacts > 0:  # hammer-style spikes, deterministic in time
            period = 1.0 / self.impacts
            for k in range(int(t0 / period), int(t1 / period) + 1):
                ts = k * period + 0.3
                idx = int(round((ts - t0) * IMU_RATE))
                if 0 <= idx < n:
                    acc[idx : idx + 2, :] += 25.0
        gyro = np.stack(
            [
                2.0 * np.cos(2 * np.pi * self.freq * t + self.phase),
                self.rng.normal(0, 0.2, n),
                0.8 * np.sin(2 * np.pi * self.freq * 0.5 * t),
            ],
            axis=1,
        ).astype(np.float32)
        return t, np.concatenate([acc, gyro], axis=1), ["ax", "ay", "az", "gx", "gy", "gz"]


def fixture_clips() -> list[ClipHandle]:
    clips = []
    for ti, task_id in enumerate(TASKS):
        for c in range(CLIPS_PER_TASK):
            seed = ti * 100 + c
            duration = 40.0 + (seed * 7919) % 40  # 40–80 s, deterministic
            backend = _FixtureBackend(seed, task_id, duration)
            clips.append(ClipHandle(f"fx_{task_id}_{c:02d}", task_id, duration, backend))
    return clips
