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


def _to_rgb_uint8(img) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:  # grayscale -> RGB
        img = np.stack([img] * 3, axis=-1)
    return img.astype(np.uint8)


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

    def frames_at(self, times, width: int = config.FRAME_WIDTH) -> list[np.ndarray]:
        """RGB frames at each of `times` (ascending). Uses one sequential decode
        pass when times are uniformly spaced and the backend supports it —
        per-frame random seeks cost ~0.35s each on the real dataset."""
        times = np.asarray(times, dtype=np.float64)
        seq = getattr(self._backend, "frames_at", None)
        if seq is not None and len(times) > 1:
            step = float(times[1] - times[0])
            if step > 0 and np.allclose(np.diff(times), step, atol=1e-6):
                out = seq(times, step, width)
                if out is not None:
                    return out
                _warn_once("seq_frames", "sequential frame decode came up short; "
                           "falling back to per-frame seeks")
        return [self.frame(float(t), width) for t in times]

    def imu_window(self, t0: float, t1: float):
        """(times (m,), data (m, C), channel_names). Accel always the first 3
        channels, in m/s^2 or g (consistent within a dataset is all we need)."""
        return self._backend.imu_window(t0, t1)


# --------------------------------------------------------------------------- #
# Real World Context backend
# --------------------------------------------------------------------------- #

class _WCBackend:
    # Single-slot IMU cache: ingest walks clips sequentially, and one clip's
    # full IMU is ~1.5MB — caching all 424 would cost real memory for nothing.
    _IMU_SLOT: tuple[str, tuple] | None = None

    def __init__(self, clip):
        self._clip = clip

    def frame(self, t: float, width: int) -> np.ndarray:
        fr = self._clip.frame(t, width=width)
        return _to_rgb_uint8(getattr(fr, "image", fr))

    @staticmethod
    def _reformat(decoded, width: int) -> np.ndarray:
        h, w = decoded.height, decoded.width
        if width and w > width:
            new_h = max(2, int(round(h * width / w / 2)) * 2)
            fr = decoded.reformat(width=width, height=new_h, format="rgb24")
        else:
            fr = decoded.reformat(format="rgb24")
        return fr.to_ndarray()

    def frames_at(self, times, step: float, width: int) -> list[np.ndarray] | None:
        """One threaded sequential decode pass over [times[0], times[-1]].
        The worldcontext lib decodes single-threaded (~300fps on 1080p h264);
        thread_type=AUTO reaches ~1200fps on the same stream. Returns None on
        any mismatch so the caller can fall back to per-frame seeks."""
        try:
            import av

            with av.open(str(self._clip.video_path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                tb = float(stream.time_base)
                container.seek(max(0, int(float(times[0]) / tb)),
                               stream=stream, backward=True, any_frame=False)
                fps = float(stream.average_rate or 30.0)
                tol = 0.5 / fps
                out, ti = [], 0
                for decoded in container.decode(stream):
                    if ti >= len(times):
                        break
                    if decoded.pts is None:
                        continue
                    t = float(decoded.pts * tb)
                    if t < times[ti] - tol:
                        continue
                    img = None
                    while ti < len(times) and t >= times[ti] - tol:
                        if img is None:
                            img = self._reformat(decoded, width)
                        out.append(img)
                        ti += 1
        except Exception as e:
            _warn_once("seq_frames_err", f"threaded sequential decode failed ({e})")
            return None
        return out if len(out) == len(times) else None

    def imu_window(self, t0: float, t1: float):
        times, data, names = self._imu_full()
        m = (times >= t0) & (times <= t1)
        return times[m], data[m], names

    def _imu_full(self):
        """Load the clip's entire IMU once and slice windows in numpy — the
        lib re-parses the whole IMU file on every .imu() call."""
        slot = _WCBackend._IMU_SLOT
        if slot is not None and slot[0] == str(self._clip.id):
            return slot[1]
        imu = self._clip.imu(start_s=0.0)
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
            times = np.arange(len(data)) / config.ASSUMED_IMU_RATE
        full = (times, data, names)
        _WCBackend._IMU_SLOT = (str(self._clip.id), full)
        return full


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
