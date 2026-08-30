"""Run this FIRST at the venue, from inside the WORLD_CONTEXT_EXPLORER_V3 env
(or any venv that can `import worldcontext`):

    python -m musclememory.probe [--wc-root /path/to/data]

It reports exactly what the real API exposes (durations, IMU channels, rates,
frame shapes, call latency) and names the wc_adapter.py spots to fix if any
guess is wrong. Nothing is written.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wc-root", default=None)
    args = ap.parse_args()

    try:
        from worldcontext import Dataset  # type: ignore
    except ImportError as e:
        raise SystemExit(f"cannot import worldcontext: {e}\n"
                         "-> run inside the flash drive's env (`uv sync` there first)")

    data = Dataset.open(args.wc_root) if args.wc_root else Dataset.open()
    clips = list(data.clips)
    print(f"clips: {len(clips)}")
    if not clips:
        return

    clip = clips[0]
    public = [a for a in dir(clip) if not a.startswith("_")]
    print(f"clip attrs: {public}")
    print(f"clip.id={clip.id!r} task_id={getattr(clip, 'task_id', '<missing>')!r}")
    for attr in ("duration_s", "duration", "length_s", "seconds"):
        if hasattr(clip, attr):
            print(f"duration attr found: clip.{attr} = {getattr(clip, attr)}")
            break
    else:
        print("!! no duration attribute — check export_metadata output, "
              "then fix _wc_duration() in musclememory/wc_adapter.py")

    t0 = time.time()
    fr = clip.frame(5.0, width=256)
    img = np.asarray(getattr(fr, "image", fr))
    print(f"frame(5.0, width=256): shape={img.shape} dtype={img.dtype} "
          f"({time.time() - t0:.2f}s/call)")

    t0 = time.time()
    imu = clip.imu(start_s=5.0, end_s=10.0)
    acc = np.asarray(imu.acceleration)
    print(f"imu(5..10s): acceleration shape={acc.shape} ({time.time() - t0:.2f}s/call)")
    print(f"imu attrs: {[a for a in dir(imu) if not a.startswith('_')]}")
    for attr in ("timestamps", "times", "t", "timestamps_s"):
        ts = getattr(imu, attr, None)
        if ts is not None:
            ts = np.asarray(ts)
            rate = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 else float("nan")
            print(f"IMU timestamps via .{attr}: n={len(ts)}, rate≈{rate:.1f} Hz")
            break
    else:
        print(f"!! no IMU timestamps attr — implied rate {acc.shape[0] / 5.0:.1f} Hz "
              "over 5 s; set ASSUMED_IMU_RATE in musclememory/config.py accordingly")
    for attr in ("angular_velocity", "gyroscope", "gyro", "rotation_rate"):
        if getattr(imu, attr, None) is not None:
            print(f"gyro found via .{attr}")
            break
    else:
        print("no gyro channel — pipeline will run accel-only (zero-filled)")

    tasks: dict[str, int] = {}
    for c in clips:
        tid = str(getattr(c, "task_id", "unknown"))
        tasks[tid] = tasks.get(tid, 0) + 1
    print(f"tasks: {len(tasks)}")
    for tid, n in sorted(tasks.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {n:4d}  {tid}")


if __name__ == "__main__":
    main()
