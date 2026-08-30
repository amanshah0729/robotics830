"""Uniform access to clips, frames, and IMU — real World Context data or the
synthetic fixture.

The real `worldcontext` library ships on the hackathon flash drive and its API
is only partially documented (Dataset.open, data.clips, clip.id, clip.task_id,
clip.frame(t, width).image, clip.imu(start_s, end_s).acceleration). Everything
else is probed defensively; run `python -m musclememory.probe` against the real
drive FIRST and fix the marked spots if the probe complains.
"""

from __future__ import annotations

import numpy as np

from . import config

_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[wc_adapter] {msg}")


class ClipHandle:
    """One clip: identity, duration, frames on demand, IMU on demand."""

    def __init__(self, clip_id: str, task_id: str, duration_s: float, backend):
        self.id = str(clip_id)
        self.task_id = str(task_id)
        self.duration_s = float(duration_s)
        self._backend = backend

    def frame(self, t: float, width: int = config.FRAME_WIDTH) -> np.ndarray:
        """RGB uint8 array (H, W, 3) at time t seconds."""
        return self._backend.frame(t, width)

    def imu_window(self, t0: float, t1: float):
        """(times (m,), data (m, C), channel_names). Accel always the first 3
        channels, in m/s^2 or g (consistent within a dataset is all we need)."""
        return self._backend.imu_window(t0, t1)


# --------------------------------------------------------------------------- #
# Real World Context backend
# --------------------------------------------------------------------------- #

class _WCBackend:
    def __init__(self, clip):
        self._clip = clip

    def frame(self, t: float, width: int) -> np.ndarray:
        fr = self._clip.frame(t, width=width)
        img = getattr(fr, "image", fr)
        img = np.asarray(img)
        if img.ndim == 2:  # grayscale -> RGB
            img = np.stack([img] * 3, axis=-1)
        return img.astype(np.uint8)

    def imu_window(self, t0: float, t1: float):
        imu = self._clip.imu(start_s=t0, end_s=t1)
        acc = np.asarray(imu.acceleration, dtype=np.float32)
        if acc.ndim == 1:
            acc = acc.reshape(-1, 1)
        cols = [acc[:, :3] if acc.shape[1] >= 3 else np.pad(acc, ((0, 0), (0, 3 - acc.shape[1])))]
        names = ["ax", "ay", "az"]

        gyro = None
        for attr in ("angular_velocity", "gyroscope", "gyro", "rotation_rate"):
            g = getattr(imu, attr, None)
            if g is not None:
                gyro = np.asarray(g, dtype=np.float32)
                break
        if gyro is not None and gyro.ndim == 2 and gyro.shape[0] == acc.shape[0]:
            cols.append(gyro[:, :3])
            names += ["gx", "gy", "gz"]
        else:
            _warn_once("gyro", "no gyroscope channel found; using accel-only (3ch)")

        data = np.concatenate(cols, axis=1)

        times = None
        for attr in ("timestamps", "times", "t", "timestamps_s"):
            ts = getattr(imu, attr, None)
            if ts is not None:
                times = np.asarray(ts, dtype=np.float64).reshape(-1)
                break
        if times is None or len(times) != len(data):
            _warn_once(
                "imu_times",
                f"no per-sample IMU timestamps; assuming uniform {config.ASSUMED_IMU_RATE} Hz "
                "(verify with `python -m musclememory.probe`)",
            )
            times = t0 + np.arange(len(data)) / config.ASSUMED_IMU_RATE
        return times, data, names


def _wc_duration(clip, meta_by_id: dict) -> float | None:
    for attr in ("duration_s", "duration", "length_s", "seconds"):
        v = getattr(clip, attr, None)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    m = meta_by_id.get(str(getattr(clip, "id", "")), {})
    for key in ("duration_s", "duration", "length_s"):
        v = m.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


def _open_worldcontext(root: str | None) -> list[ClipHandle]:
    try:
        from worldcontext import Dataset  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "worldcontext is not importable. Run from inside the copied "
            "WORLD_CONTEXT_EXPLORER_V3 env (`uv sync` there), or `uv pip install -e` "
            f"that folder into this venv. ({e})"
        )

    data = Dataset.open(root) if root else Dataset.open()

    # Duration often lives in exported metadata rather than on the clip object.
    meta_by_id: dict = {}
    try:
        import json, tempfile, os

        tmp = os.path.join(tempfile.gettempdir(), "mm_wc_metadata.json")
        data.export_metadata(tmp, format="json")
        raw = json.load(open(tmp))
        rows = raw if isinstance(raw, list) else raw.get("clips", [])
        for r in rows:
            if isinstance(r, dict) and "id" in r:
                meta_by_id[str(r["id"])] = r
    except Exception as e:  # metadata export is best-effort
        _warn_once("meta", f"export_metadata failed ({e}); falling back to clip attrs")

    handles = []
    for clip in data.clips:
        dur = _wc_duration(clip, meta_by_id)
        if dur is None:
            # Last resort: probe IMU length. Marked spot — fix after running probe.
            _warn_once("dur", "no duration attribute found; probing IMU extent per clip (slow)")
            try:
                imu = clip.imu(start_s=0.0, end_s=10_000.0)
                n = len(np.asarray(imu.acceleration))
                dur = n / config.ASSUMED_IMU_RATE
            except Exception:
                continue
        handles.append(
            ClipHandle(clip.id, getattr(clip, "task_id", "unknown"), dur, _WCBackend(clip))
        )
    return handles


# --------------------------------------------------------------------------- #

def open_clips(source: str, root: str | None = None) -> list[ClipHandle]:
    if source == "fixture":
        from .fixture import fixture_clips

        return fixture_clips()
    if source == "worldcontext":
        return _open_worldcontext(root)
    raise ValueError(f"unknown source {source!r} (use 'fixture' or 'worldcontext')")
