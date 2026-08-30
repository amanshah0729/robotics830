"""Muscle memory -> robot: perform a retrieved moment's motion on an SO-101.

Runs in the LEROBOT venv (not this repo's): it polls the demo server for
perform-jobs, loads the matched window's raw IMU from work/derived, extracts a
motion signature (dominant frequency, amplitude, principal axis, twistiness),
and synthesizes a clamped joint-space oscillation with the same character.
The robot mimics the *rhythm and shape* of the remembered motion — retrieval
-> re-synthesis, not learned manipulation (be honest about this on stage).

    cd robotics830
    ../lerobot/.venv/bin/python -m musclememory.perform \
        --port /dev/cu.usbmodem5AE60798061 --robot-id hack_follower
    # add --dry-run to print trajectories without touching the robot
    # add --once ROW to perform one row and exit

Safety: amplitudes clamped to +/-12 deg, frequency to 2.2 Hz, hann ramps at
both ends, return-to-start after each move, and lerobot's max_relative_target
caps every step. Keep the workspace in front of the arm clear anyway.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

CONTROL_HZ = 50
MAX_AMP_DEG = 12.0
MIN_AMP_DEG = 4.0
MAX_FREQ = 2.2
MIN_FREQ = 0.5
MOVE_SECONDS = 6.0
IMU_HZ = 50.0  # windows are resampled to 200 samples / 4 s


# ---------------------------------------------------------------- data access

class Bank:
    """row -> (clip_id, t, imu window) via the built index + derived npz."""

    def __init__(self, work: Path):
        rows = np.load(work / "index" / "rows.npz")
        self.clip_idx, self.t = rows["clip_idx"], rows["t"]
        self.clips = json.loads((work / "index" / "clips.json").read_text())["clips"]
        self.tasks = json.loads((work / "index" / "tasks.json").read_text())
        self.task_idx = rows["task_idx"]
        self.derived = work / "derived"
        self._cache: dict[str, dict] = {}

    def window(self, row: int):
        clip = self.clips[int(self.clip_idx[row])]
        d = self._cache.get(clip)
        if d is None:
            z = np.load(self.derived / f"{clip}.npz")
            d = {"t": z["t"], "imu": z["imu"]}
            self._cache.clear()          # keep at most one clip resident
            self._cache[clip] = d
        i = int(np.argmin(np.abs(d["t"] - self.t[row])))
        task = self.tasks[int(self.task_idx[row])]
        return clip, float(self.t[row]), task, d["imu"][i].astype(np.float32)


# ------------------------------------------------------------- motion analysis

def signature(imu: np.ndarray) -> dict:
    """Extract the performable character of a 4 s IMU window (200 x 6)."""
    acc_raw = imu[:, :3]
    gyro = imu[:, 3:6] if imu.shape[1] >= 6 else np.zeros_like(acc_raw)

    g_dir = acc_raw.mean(axis=0)
    g_norm = np.linalg.norm(g_dir)
    g_dir = g_dir / g_norm if g_norm > 1e-6 else np.array([0.0, 0.0, 1.0])

    # high-pass: subtract a ~1s moving average so slow posture drift doesn't
    # mask the stroke rhythm (windows are 4s @ 50Hz)
    k = 51
    pad = np.pad(acc_raw, ((k // 2, k // 2), (0, 0)), mode="edge")
    trend = np.stack([np.convolve(pad[:, c], np.ones(k) / k, mode="valid")
                      for c in range(3)], axis=1)
    acc = acc_raw - trend[:len(acc_raw)]
    # principal direction of the oscillation
    _, s, vt = np.linalg.svd(acc, full_matrices=False)
    axis = vt[0]
    proj = acc @ axis

    # dominant frequency of the 1-D projected motion, 0.3..3.5 Hz band
    spec = np.abs(np.fft.rfft(proj * np.hanning(len(proj))))
    freqs = np.fft.rfftfreq(len(proj), 1.0 / IMU_HZ)
    band = (freqs >= 0.3) & (freqs <= 3.5)
    freq = float(freqs[band][np.argmax(spec[band])]) if band.any() and spec[band].max() > 0 else 1.0

    acc_amp = float(np.percentile(np.abs(proj), 90))
    gyro_amp = float(np.percentile(np.linalg.norm(gyro - gyro.mean(0), axis=1), 90))
    verticality = float(abs(axis @ g_dir))          # 1 = along gravity
    twistiness = gyro_amp / (acc_amp + 1e-6)

    if twistiness > 0.55:
        style = "stir"          # rotation-dominant: whisk / screw / polish
    elif verticality > 0.6:
        style = "chop"          # gravity-aligned strokes: hammer / press / knead
    else:
        style = "sweep"         # horizontal strokes: wipe / fold / place
    return {"style": style, "freq": freq, "acc_amp": acc_amp,
            "gyro_amp": gyro_amp, "verticality": round(verticality, 2),
            "twistiness": round(twistiness, 2)}


def trajectory(sig: dict, center: dict[str, float]) -> list[dict[str, float]]:
    """Joint-space frames (degrees) with the signature's rhythm, clamped."""
    freq = float(np.clip(sig["freq"], MIN_FREQ, MAX_FREQ))
    amp = float(np.clip(sig["acc_amp"] * 4.0, MIN_AMP_DEG, MAX_AMP_DEG))
    n = int(MOVE_SECONDS * CONTROL_HZ)
    ramp = np.ones(n)
    r = int(0.15 * n)
    ramp[:r] = np.linspace(0, 1, r)
    ramp[-r:] = np.linspace(1, 0, r)

    frames = []
    for k in range(n):
        th = 2 * np.pi * freq * k / CONTROL_HZ
        e = ramp[k]
        f = dict(center)
        if sig["style"] == "chop":
            f["shoulder_lift"] += e * amp * np.sin(th)
            f["elbow_flex"] -= e * 0.6 * amp * np.sin(th)
            f["wrist_flex"] += e * 0.3 * amp * np.sin(th)
        elif sig["style"] == "sweep":
            f["shoulder_pan"] += e * amp * np.sin(th)
            f["wrist_flex"] += e * 0.2 * amp * np.sin(2 * th)
        else:  # stir
            f["wrist_roll"] += e * amp * np.sin(th)
            f["shoulder_pan"] += e * 0.45 * amp * np.sin(th)
            f["shoulder_lift"] += e * 0.45 * amp * np.cos(th)
        frames.append(f)
    frames.append(dict(center))
    return frames


# ------------------------------------------------------------------ the robot

class Arm:
    def __init__(self, port: str, robot_id: str, dry: bool):
        self.dry = dry
        self.robot = None
        if dry:
            return
        from lerobot.robots.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

        cfg = SOFollowerRobotConfig(port=port, id=robot_id, max_relative_target=12.0)
        self.robot = SOFollower(cfg)
        self.robot.connect(calibrate=False)
        print("[arm] connected — torque on, holding pose")

    def center(self) -> dict[str, float]:
        if self.dry:
            return {m: 0.0 for m in
                    ("shoulder_pan", "shoulder_lift", "elbow_flex",
                     "wrist_flex", "wrist_roll", "gripper")}
        obs = self.robot.get_observation()
        return {k.removesuffix(".pos"): float(v)
                for k, v in obs.items() if k.endswith(".pos")}

    def play(self, frames: list[dict[str, float]]):
        if self.dry:
            a = frames[len(frames) // 2]
            print(f"[dry] {len(frames)} frames @ {CONTROL_HZ} Hz; mid-frame: "
                  + ", ".join(f"{k}={v:.1f}" for k, v in a.items()))
            return
        period = 1.0 / CONTROL_HZ
        for f in frames:
            t0 = time.perf_counter()
            self.robot.send_action({f"{m}.pos": v for m, v in f.items()})
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)


# ------------------------------------------------------------------- the loop

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", default="http://localhost:7860")
    ap.add_argument("--work", default="work")
    ap.add_argument("--port", default="/dev/cu.usbmodem5AE60798061")
    ap.add_argument("--robot-id", default="hack_follower")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", type=int, default=None, help="perform this row and exit")
    args = ap.parse_args()

    bank = Bank(Path(args.work))
    arm = Arm(args.port, args.robot_id, args.dry_run)

    def perform(row: int):
        clip, t, task, imu = bank.window(row)
        sig = signature(imu)
        print(f"[perform] row {row}: {task} ({clip} @ {t:.0f}s) -> "
              f"style={sig['style']} f={sig['freq']:.2f}Hz amp~{sig['acc_amp']:.2f} "
              f"(vert {sig['verticality']}, twist {sig['twistiness']})")
        arm.play(trajectory(sig, arm.center()))
        print("[perform] done")

    if args.once is not None:
        perform(args.once)
        return

    print(f"[perform] polling {args.server}/api/robot/next — queue a move from the UI")
    while True:
        try:
            with urllib.request.urlopen(f"{args.server}/api/robot/next", timeout=5) as r:
                job = json.loads(r.read())
        except Exception as e:
            print(f"[perform] server unreachable ({e}); retrying…")
            time.sleep(2)
            continue
        if job.get("row") is not None:
            try:
                perform(int(job["row"]))
            except Exception as e:
                print(f"[perform] FAILED: {e}")
        else:
            time.sleep(0.4)


if __name__ == "__main__":
    main()
