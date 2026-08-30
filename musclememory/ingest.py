"""Stage 1: slice every clip into overlapping windows and derive, per window,
a video embedding + resampled raw IMU + handcrafted IMU features.

Output: one .npz + .json pair per clip under --out. Resumable — existing clips
are skipped, so you can Ctrl-C anytime and shard across machines by --max-clips
/ --skip-clips or --tasks.

    python -m musclememory.ingest --source fixture --embedder fake
    python -m musclememory.ingest --source worldcontext --embedder clip --max-clips 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import config
from .embed_video import make_embedder
from .imu_features import FEATURE_NAMES, featurize, resample_window
from .wc_adapter import open_clips


def ingest_clip(clip, embedder, out_dir: Path, window_s: float, stride_s: float,
                frames_per_window: int, frame_width: int) -> int:
    centers = np.arange(window_s / 2, clip.duration_s - window_s / 2 + 1e-6, stride_s)
    if len(centers) == 0:
        return 0

    imus, feats = [], []
    for tc in centers:
        t0, t1 = tc - window_s / 2, tc + window_s / 2
        times, data, _names = clip.imu_window(t0, t1)
        imus.append(resample_window(times, data, t0, t1))
        feats.append(featurize(times, data))

    if frames_per_window == 1:
        # one sequential decode pass per clip instead of a seek per window
        frame_batches = [[im] for im in clip.frames_at(centers, width=frame_width)]
    else:
        frame_batches = []
        for tc in centers:
            t0, t1 = tc - window_s / 2, tc + window_s / 2
            ts = np.linspace(t0 + 0.2, t1 - 0.2, frames_per_window)
            frame_batches.append([clip.frame(t, width=frame_width) for t in ts])

    # embed in flat batches of 64 frames, then mean-pool back per window
    flat = [im for batch in frame_batches for im in batch]
    embs = []
    for i in range(0, len(flat), 64):
        embs.append(embedder.encode_images(flat[i : i + 64]))
    flat_emb = np.concatenate(embs, axis=0)
    per_window = flat_emb.reshape(len(centers), frames_per_window, -1).mean(axis=1)
    per_window /= np.linalg.norm(per_window, axis=1, keepdims=True) + 1e-9

    np.savez_compressed(
        out_dir / f"{clip.id}.npz",
        t=centers.astype(np.float32),
        vemb=per_window.astype(np.float16),
        imu=np.stack(imus).astype(np.float16),
        feat=np.stack(feats).astype(np.float32),
    )
    (out_dir / f"{clip.id}.json").write_text(
        json.dumps(
            {
                "clip_id": clip.id,
                "task_id": clip.task_id,
                "duration_s": clip.duration_s,
                "n_windows": int(len(centers)),
                "window_s": window_s,
                "stride_s": stride_s,
                "embedder": embedder.name,
                "feature_names": FEATURE_NAMES,
            }
        )
    )
    return len(centers)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["fixture", "worldcontext"], required=True)
    ap.add_argument("--wc-root", default=None, help="path for worldcontext Dataset.open()")
    ap.add_argument("--out", default=config.DERIVED_DIR)
    ap.add_argument("--embedder", choices=["fake", "clip"], default="clip")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--window", type=float, default=config.WINDOW_S)
    ap.add_argument("--stride", type=float, default=config.STRIDE_S)
    ap.add_argument("--frames-per-window", type=int, default=1)
    ap.add_argument("--frame-width", type=int, default=config.FRAME_WIDTH)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--skip-clips", type=int, default=0, help="offset for sharding across machines")
    ap.add_argument("--tasks", default=None, help="comma-separated task_id filter")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = open_clips(args.source, args.wc_root)
    if args.tasks:
        keep = {t.strip() for t in args.tasks.split(",")}
        clips = [c for c in clips if c.task_id in keep]
    clips = clips[args.skip_clips :]
    if args.max_clips:
        clips = clips[: args.max_clips]

    embedder = make_embedder(args.embedder, args.device)
    total, t_start = 0, time.time()
    for i, clip in enumerate(clips):
        if not args.overwrite and (out_dir / f"{clip.id}.npz").exists():
            print(f"[{i + 1}/{len(clips)}] {clip.id}: exists, skipping")
            continue
        t0 = time.time()
        try:
            n = ingest_clip(clip, embedder, out_dir, args.window, args.stride,
                            args.frames_per_window, args.frame_width)
        except Exception as e:
            print(f"[{i + 1}/{len(clips)}] {clip.id}: FAILED ({e}); continuing")
            continue
        total += n
        print(f"[{i + 1}/{len(clips)}] {clip.id} ({clip.task_id}): "
              f"{n} windows in {time.time() - t0:.1f}s")
    print(f"done: {total} new windows in {time.time() - t_start:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
