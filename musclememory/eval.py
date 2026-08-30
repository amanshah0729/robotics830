"""Honest numbers for the build card. No cherry-picking:

  * Queries come ONLY from held-out clips the IMU encoder never trained on.
  * The retrieval pool is the FULL bank (train + held-out) — the deployment
    scenario: a fresh motion query searching the whole dataset.
  * Same-clip neighbors never count toward task-consistency, so temporal
    autocorrelation can't inflate the score.

Reports:
  * IMU->video retrieval R@1/5/10 (learned encoder): given a held-out window's
    IMU alone, how often is its exact video moment ranked in the top k of the
    entire bank? Random baseline = k/N.
  * Task-consistency@5: do a query's top-5 cross-clip neighbors share its task?
    Computed for the learned space AND the handcrafted-feature baseline the
    encoder has to beat. Scored only for queries whose task appears in 2+ clips.

    python -m musclememory.eval --derived work/derived
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config
from .imu_encoder import clip_is_val


def _l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _load_all(derived: Path):
    iemb, vemb, feat, task, clip, is_val = [], [], [], [], [], []
    has_iemb = True
    for meta_path in sorted(derived.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        npz = np.load(meta_path.with_suffix(".npz"))
        n = len(npz["t"])
        vemb.append(npz["vemb"].astype(np.float32))
        feat.append(npz["feat"])
        task += [meta["task_id"]] * n
        clip += [meta["clip_id"]] * n
        is_val += [clip_is_val(meta["clip_id"])] * n
        if "iemb" in npz.files:
            iemb.append(npz["iemb"].astype(np.float32))
        else:
            has_iemb = False
    if not vemb:
        raise SystemExit("no derived clips found — run musclememory.ingest first")
    return (
        np.concatenate(iemb) if has_iemb and iemb else None,
        _l2(np.concatenate(vemb)),
        np.concatenate(feat),
        np.array(task),
        np.array(clip),
        np.array(is_val),
    )


def retrieval_recall(q_iemb, bank_v, q_rows, ks=(1, 5, 10)) -> dict:
    """Rank of each held-out window's own video embedding, queried by its IMU
    embedding, against the entire bank."""
    sims = _l2(q_iemb) @ bank_v.T
    out = {}
    ranks = np.array([int((sims[i] > sims[i, r]).sum()) for i, r in enumerate(q_rows)])
    for k in ks:
        out[f"R@{k}"] = round(float((ranks < k).mean()), 4)
    out["median_rank"] = int(np.median(ranks)) + 1
    out["random_baseline"] = {f"R@{k}": round(k / bank_v.shape[0], 6) for k in ks}
    out["pool_size"] = int(bank_v.shape[0])
    return out


def task_consistency(q_space, q_task, q_clip, bank_space, task, clip, k=5) -> dict:
    """Mean fraction of top-k cross-clip bank neighbors sharing the query's task.
    Queries whose task lives in a single clip are excluded (no fair answer)."""
    multi_clip_tasks = {
        t for t in set(task.tolist())
        if len(set(clip[task == t].tolist())) >= 2
    }
    eligible = np.array([t in multi_clip_tasks for t in q_task])
    if not eligible.any():
        return {"task_consistency@5": None, "n_eligible_queries": 0,
                "note": "every task lives in a single clip; metric undefined"}
    sims = _l2(q_space) @ _l2(bank_space).T
    hits = []
    for qi in np.where(eligible)[0]:
        s = sims[qi].copy()
        s[clip == q_clip[qi]] = -np.inf  # never reward same-clip neighbors
        top = np.argsort(-s)[:k]
        hits.append(float((task[top] == q_task[qi]).mean()))
    return {"task_consistency@5": round(float(np.mean(hits)), 4),
            "n_eligible_queries": int(eligible.sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--derived", default=config.DERIVED_DIR)
    ap.add_argument("--out", default="work/eval.json")
    args = ap.parse_args()

    iemb, bank_v, feat, task, clip, is_val = _load_all(Path(args.derived))
    q = np.where(is_val)[0]
    if len(q) == 0:
        raise SystemExit("no held-out clips in this ingest — add more clips")

    stats = {
        "n_windows_total": int(len(task)),
        "n_heldout_query_windows": int(len(q)),
        "n_heldout_clips": int(len(set(clip[q].tolist()))),
        "task_chance_level": round(
            float(np.mean([np.mean(task == t) for t in set(task[q].tolist())])), 4),
    }

    zfeat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-9)
    stats["feature_baseline"] = task_consistency(
        zfeat[q], task[q], clip[q], zfeat, task, clip)

    if iemb is not None:
        stats["learned_imu_encoder"] = {
            "imu_to_video_retrieval": retrieval_recall(iemb[q], bank_v, q),
            **task_consistency(iemb[q], task[q], clip[q], bank_v, task, clip),
        }
    else:
        stats["learned_imu_encoder"] = "not trained yet (run musclememory.imu_encoder)"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
