"""Stage 3 (the novel bit): a small 1D-CNN IMU tower trained contrastively to
land IMU windows in the SAME embedding space as the frozen video tower
(InfoNCE against per-window CLIP embeddings — IMU2CLIP-style, trained from
scratch on this dataset at the event).

Once trained, motion alone — including live phone motion — retrieves moments
from hundreds of hours of video.

    python -m musclememory.imu_encoder train  --derived work/derived --out work/models
    python -m musclememory.imu_encoder export --derived work/derived --model work/models/imu_encoder.pt

Requires the [ml] extras (torch). Everything else in the repo runs without it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import config
from .imu_features import resample_window


def clip_is_val(clip_id: str) -> bool:
    """Stable ~10% validation split by clip (shared with eval.py)."""
    return int(hashlib.sha256(clip_id.encode()).hexdigest()[:8], 16) % 10 == 0


def znorm_windows(x: np.ndarray) -> np.ndarray:
    """Per-window per-channel z-norm — units cancel, so a phone's IMU can query
    a model trained on the dataset's IMU."""
    mean = x.mean(axis=-2, keepdims=True)
    std = x.std(axis=-2, keepdims=True) + 1e-6
    return (x - mean) / std


def _torch():
    try:
        import torch
        import torch.nn as nn

        return torch, nn
    except ImportError:
        raise SystemExit("torch is required here: uv pip install -r requirements-ml.txt")


def build_model(nn, in_ch: int = config.IMU_CHANNELS, dim: int = config.EMB_DIM):
    return nn.Sequential(
        nn.Conv1d(in_ch, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.GELU(),
        nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.GELU(),
        nn.Conv1d(128, 256, 5, stride=2, padding=2), nn.BatchNorm1d(256), nn.GELU(),
        nn.Conv1d(256, 512, 3, stride=2, padding=1), nn.BatchNorm1d(512), nn.GELU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        nn.Linear(512, dim),
    )


def _load_split(derived: Path):
    tr_x, tr_y, va_x, va_y, va_clip = [], [], [], [], []
    for meta_path in sorted(derived.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        npz = np.load(meta_path.with_suffix(".npz"))
        x = npz["imu"].astype(np.float32)
        y = npz["vemb"].astype(np.float32)
        if clip_is_val(meta["clip_id"]):
            va_x.append(x); va_y.append(y)
            va_clip += [meta["clip_id"]] * len(x)
        else:
            tr_x.append(x); tr_y.append(y)
    if not tr_x or not va_x:
        raise SystemExit("need at least a few clips on both sides of the split — ingest more")
    return (np.concatenate(tr_x), np.concatenate(tr_y),
            np.concatenate(va_x), np.concatenate(va_y), va_clip)


def _recall(torch, model, device, va_x, va_y, ks=(1, 10)) -> dict:
    with torch.no_grad():
        q = model(torch.from_numpy(znorm_windows(va_x)).permute(0, 2, 1).to(device))
        q = torch.nn.functional.normalize(q, dim=1).cpu().numpy()
    tgt = va_y / (np.linalg.norm(va_y, axis=1, keepdims=True) + 1e-9)
    sims = q @ tgt.T                       # (n, n): IMU query vs every val video emb
    order = np.argsort(-sims, axis=1)
    truth = np.arange(len(q))[:, None]
    return {f"R@{k}": float((order[:, :k] == truth).any(axis=1).mean()) for k in ks}


def cmd_train(args) -> None:
    torch, nn = _torch()
    device = ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else args.device
    tr_x, tr_y, va_x, va_y, _ = _load_split(Path(args.derived))
    print(f"[train] {len(tr_x)} train / {len(va_x)} val windows on {device}")

    model = build_model(nn).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    temp = 0.07
    ys = torch.from_numpy(tr_y / (np.linalg.norm(tr_y, axis=1, keepdims=True) + 1e-9))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(tr_x))
        losses = []
        for i in range(0, len(perm), args.batch):
            idx = perm[i : i + args.batch]
            if len(idx) < 8:
                continue
            xb = torch.from_numpy(znorm_windows(tr_x[idx])).permute(0, 2, 1).to(device)
            yb = ys[idx].to(device)
            zi = torch.nn.functional.normalize(model(xb), dim=1)
            logits = zi @ yb.T / temp
            labels = torch.arange(len(idx), device=device)
            loss = (torch.nn.functional.cross_entropy(logits, labels)
                    + torch.nn.functional.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        model.eval()
        rec = _recall(torch, model, device, va_x, va_y)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {np.mean(losses):.3f}  "
              f"val R@1 {rec['R@1']:.3f}  R@10 {rec['R@10']:.3f}")
        if rec["R@10"] > best:
            best = rec["R@10"]
            torch.save(model.state_dict(), out / "imu_encoder.pt")
            (out / "imu_encoder.json").write_text(json.dumps(
                {"in_ch": config.IMU_CHANNELS, "dim": config.EMB_DIM,
                 "imu_len": config.IMU_LEN, "val": rec, "epochs_done": epoch + 1}))
    print(f"[train] best val R@10 {best:.3f} -> {out / 'imu_encoder.pt'}")


def cmd_export(args) -> None:
    """Write learned IMU embeddings (`iemb`) back into each derived npz, then
    re-run musclememory.index to make motion the primary search space."""
    torch, nn = _torch()
    model = build_model(nn)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()
    for meta_path in sorted(Path(args.derived).glob("*.json")):
        npz_path = meta_path.with_suffix(".npz")
        d = dict(np.load(npz_path))
        x = znorm_windows(d["imu"].astype(np.float32))
        with torch.no_grad():
            z = model(torch.from_numpy(x).permute(0, 2, 1))
            z = torch.nn.functional.normalize(z, dim=1).numpy()
        d["iemb"] = z.astype(np.float16)
        np.savez_compressed(npz_path, **d)
        print(f"[export] {npz_path.name}: iemb {z.shape}")


class QueryEncoder:
    """Server-side: raw (t, 6ch) samples -> 512-d vector in video space."""

    def __init__(self, model_path: str):
        torch, nn = _torch()
        self.torch = torch
        self.model = build_model(nn)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def encode(self, times: np.ndarray, data: np.ndarray) -> np.ndarray:
        t1 = times[-1]
        win = resample_window(times, data, t1 - config.WINDOW_S, t1)
        x = znorm_windows(win[None])
        with self.torch.no_grad():
            z = self.model(self.torch.from_numpy(x).permute(0, 2, 1))
            z = self.torch.nn.functional.normalize(z, dim=1).numpy()[0]
        return z.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--derived", default=config.DERIVED_DIR)
    tr.add_argument("--out", default=config.MODELS_DIR)
    tr.add_argument("--epochs", type=int, default=15)
    tr.add_argument("--batch", type=int, default=256)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--device", default="auto")
    ex = sub.add_parser("export")
    ex.add_argument("--derived", default=config.DERIVED_DIR)
    ex.add_argument("--model", default=f"{config.MODELS_DIR}/imu_encoder.pt")
    args = ap.parse_args()
    (cmd_train if args.cmd == "train" else cmd_export)(args)


if __name__ == "__main__":
    main()
