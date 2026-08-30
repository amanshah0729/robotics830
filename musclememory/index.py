"""Stage 2: fuse all per-clip derivations into one searchable bank + a 2D atlas.

Outputs under --out:
  bank_video.npy   (N, 512) float16, L2-normed   — text/frame search space
  bank_imu.npy     (N, 512) float16, optional    — learned IMU search space
  feats.npy        (N, F)  float32               — handcrafted-feature space
  feat_stats.npz   mean/std for z-norming feature queries
  rows.npz         clip_idx (u32), t (f32), task_idx (u16)
  clips.json / tasks.json                        — id lookups
  atlas.json       2D map points for the frontend

    python -m musclememory.index --derived work/derived --out work/index
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config


def load_derived(derived: Path):
    metas = sorted(derived.glob("*.json"))
    if not metas:
        raise SystemExit(f"no derived clips in {derived}; run musclememory.ingest first")
    vembs, imu_embs, feats, clip_idx, task_idx, ts = [], [], [], [], [], []
    clips, tasks, task_of_clip = [], [], []
    task_lookup: dict[str, int] = {}
    has_iemb = True
    for meta_path in metas:
        meta = json.loads(meta_path.read_text())
        npz = np.load(meta_path.with_suffix(".npz"))
        n = len(npz["t"])
        ci = len(clips)
        clips.append(meta["clip_id"])
        tid = meta["task_id"]
        if tid not in task_lookup:
            task_lookup[tid] = len(tasks)
            tasks.append(tid)
        task_of_clip.append(task_lookup[tid])
        vembs.append(npz["vemb"].astype(np.float32))
        feats.append(npz["feat"])
        ts.append(npz["t"])
        clip_idx.append(np.full(n, ci, dtype=np.uint32))
        task_idx.append(np.full(n, task_lookup[tid], dtype=np.uint16))
        if "iemb" in npz.files:
            imu_embs.append(npz["iemb"].astype(np.float32))
        else:
            has_iemb = False
    bank_v = np.concatenate(vembs)
    bank_v /= np.linalg.norm(bank_v, axis=1, keepdims=True) + 1e-9
    bank_i = None
    if has_iemb and imu_embs:
        bank_i = np.concatenate(imu_embs)
        bank_i /= np.linalg.norm(bank_i, axis=1, keepdims=True) + 1e-9
    return {
        "bank_v": bank_v,
        "bank_i": bank_i,
        "feats": np.concatenate(feats),
        "clip_idx": np.concatenate(clip_idx),
        "task_idx": np.concatenate(task_idx),
        "t": np.concatenate(ts),
        "clips": clips,
        "tasks": tasks,
        "task_of_clip": task_of_clip,
    }


def project_2d(emb: np.ndarray, method: str, fit_max: int = 30000, seed: int = 0):
    rng = np.random.default_rng(seed)
    fit_idx = (
        rng.choice(len(emb), fit_max, replace=False) if len(emb) > fit_max
        else np.arange(len(emb))
    )
    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                                random_state=seed)
            xy = reducer.fit_transform(emb[fit_idx])
            if len(fit_idx) < len(emb):
                xy_all = reducer.transform(emb)
            else:
                xy_all = xy
            return xy_all.astype(np.float32), "umap"
        except ImportError:
            print("[index] umap-learn not installed; falling back to PCA")
    # PCA via SVD on the fit sample
    sample = emb[fit_idx]
    mean = sample.mean(axis=0)
    _u, _s, vt = np.linalg.svd(sample - mean, full_matrices=False)
    xy_all = (emb - mean) @ vt[:2].T
    return xy_all.astype(np.float32), "pca"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--derived", default=config.DERIVED_DIR)
    ap.add_argument("--out", default=config.INDEX_DIR)
    ap.add_argument("--proj", choices=["umap", "pca"], default="umap")
    ap.add_argument("--atlas-max", type=int, default=40000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load_derived(Path(args.derived))
    n = len(d["bank_v"])
    print(f"[index] {n} windows / {len(d['clips'])} clips / {len(d['tasks'])} tasks; "
          f"imu embeddings: {'yes' if d['bank_i'] is not None else 'no (train encoder later)'}")

    np.save(out / "bank_video.npy", d["bank_v"].astype(np.float16))
    if d["bank_i"] is not None:
        np.save(out / "bank_imu.npy", d["bank_i"].astype(np.float16))
    np.save(out / "feats.npy", d["feats"])
    np.savez(out / "feat_stats.npz",
             mean=d["feats"].mean(axis=0), std=d["feats"].std(axis=0) + 1e-9)
    np.savez(out / "rows.npz", clip_idx=d["clip_idx"], task_idx=d["task_idx"], t=d["t"])
    (out / "clips.json").write_text(json.dumps(
        {"clips": d["clips"], "task_of_clip": d["task_of_clip"]}))
    (out / "tasks.json").write_text(json.dumps(d["tasks"]))

    # Atlas: stratified subsample by task so small tasks stay visible
    rng = np.random.default_rng(0)
    if n > args.atlas_max:
        keep: list[np.ndarray] = []
        per_task = max(50, args.atlas_max // max(len(d["tasks"]), 1))
        for ti in range(len(d["tasks"])):
            idx = np.where(d["task_idx"] == ti)[0]
            if len(idx) > per_task:
                idx = rng.choice(idx, per_task, replace=False)
            keep.append(idx)
        atlas_idx = np.sort(np.concatenate(keep))
    else:
        atlas_idx = np.arange(n)

    xy, method = project_2d(d["bank_v"][atlas_idx], args.proj)
    xy -= xy.mean(axis=0)
    scale = np.abs(xy).max() + 1e-9
    xy /= scale
    points = [
        [round(float(x), 4), round(float(y), 4), int(d["task_idx"][i]), int(i)]
        for (x, y), i in zip(xy, atlas_idx)
    ]
    (out / "atlas.json").write_text(json.dumps({
        "projection": method,
        "n_total": n,
        "tasks": d["tasks"],
        "task_counts": np.bincount(d["task_idx"], minlength=len(d["tasks"])).tolist(),
        "points": points,  # [x, y, task_idx, row_idx]
    }))
    print(f"[index] atlas: {len(points)} points via {method} -> {out / 'atlas.json'}")


if __name__ == "__main__":
    main()
