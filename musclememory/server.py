"""Stage 4: the demo. Serves the atlas UI, text search, per-window detail
(frame + IMU trace + contact events + chapters), and LIVE phone-motion search
over the websocket bridge.

    python -m musclememory.server --source fixture --embedder fake
    python -m musclememory.server --source worldcontext --embedder clip \
        --imu-model work/models/imu_encoder.pt --port 7860

Open http://<laptop-ip>:7860 on the big screen and http://<laptop-ip>:7860/phone
on the phone (same Wi-Fi / hotspot). iOS Safari requires HTTPS for motion
sensors — use an Android phone or `cloudflared tunnel --url http://localhost:7860`.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config
from .imu_features import chapter_boundaries, detect_spikes, featurize
from .wc_adapter import open_clips

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class State:
    """Everything the endpoints share, loaded once at startup."""

    def __init__(self, args):
        idx = Path(args.index)
        if not (idx / "rows.npz").exists():
            raise SystemExit(f"no index at {idx}; run musclememory.index first")
        # nan_to_num: windows with degenerate IMU (one dataset clip has a flat
        # recording) embed to NaN; zeroed rows get cosine 0 and never rank.
        self.bank_v = np.nan_to_num(np.load(idx / "bank_video.npy").astype(np.float32))
        p = idx / "bank_imu.npy"
        self.bank_i = np.nan_to_num(np.load(p).astype(np.float32)) if p.exists() else None
        self.feats = np.load(idx / "feats.npy")
        st = np.load(idx / "feat_stats.npz")
        self.f_mean, self.f_std = st["mean"], st["std"]
        self.zfeats = (self.feats - self.f_mean) / self.f_std
        self.zfeats /= np.linalg.norm(self.zfeats, axis=1, keepdims=True) + 1e-9
        rows = np.load(idx / "rows.npz")
        self.clip_idx, self.task_idx, self.t = rows["clip_idx"], rows["task_idx"], rows["t"]
        cj = json.loads((idx / "clips.json").read_text())
        self.clips: list[str] = cj["clips"]
        self.tasks: list[str] = json.loads((idx / "tasks.json").read_text())
        self.atlas_path = idx / "atlas.json"

        self.handles = {c.id: c for c in open_clips(args.source, args.wc_root)}
        self.embedder = None
        self.embedder_kind = args.embedder
        self._embedder_device = args.device

        self.query_encoder = None
        if args.imu_model and Path(args.imu_model).exists():
            from .imu_encoder import QueryEncoder

            self.query_encoder = QueryEncoder(args.imu_model)
            print(f"[server] learned IMU encoder loaded: {args.imu_model}")
        elif self.bank_i is None:
            print("[server] no IMU model yet -> motion search uses the "
                  "handcrafted-feature baseline space")
        self.live: set[WebSocket] = set()

    def get_embedder(self):
        if self.embedder is None:
            from .embed_video import make_embedder

            self.embedder = make_embedder(self.embedder_kind, self._embedder_device)
        return self.embedder

    # ---------------- search helpers ----------------

    def topk(self, bank: np.ndarray, q: np.ndarray, k: int,
             per_clip_cap: int = 2) -> list[dict]:
        sims = bank @ q
        order = np.argsort(-sims)
        out: list[dict] = []
        seen: dict[int, int] = {}
        for i in order:
            ci = int(self.clip_idx[i])
            if seen.get(ci, 0) >= per_clip_cap:
                continue
            seen[ci] = seen.get(ci, 0) + 1
            out.append(self.row_info(int(i), score=float(sims[i])))
            if len(out) >= k:
                break
        return out

    def row_info(self, row: int, score: float | None = None) -> dict:
        d = {
            "row": row,
            "clip": self.clips[int(self.clip_idx[row])],
            "task": self.tasks[int(self.task_idx[row])],
            "t": round(float(self.t[row]), 2),
        }
        if score is not None:
            d["score"] = round(score, 4)
        return d

    def motion_query(self, times: np.ndarray, data: np.ndarray, k: int = 8):
        """Route a raw motion window to whichever space is available."""
        if self.query_encoder is not None:
            q = self.query_encoder.encode(times, data)
            bank = self.bank_i if self.bank_i is not None else self.bank_v
            return self.topk(bank, q, k), "learned"
        f = featurize(times, data)
        q = (f - self.f_mean) / self.f_std
        q /= np.linalg.norm(q) + 1e-9
        return self.topk(self.zfeats, q.astype(np.float32), k), "feature-baseline"


def build_app(args) -> FastAPI:
    S = State(args)
    app = FastAPI(title="Muscle Memory")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    @app.get("/")
    def home():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/phone")
    def phone():
        return FileResponse(WEB_DIR / "phone.html")

    # FileResponse sends no Cache-Control, which lets a client heuristically
    # cache the page off Last-Modified. The glasses WebView did exactly that
    # and kept serving a stale build after a Restart, which is indistinguishable
    # from the new code being broken.
    NOCACHE = {"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"}

    @app.get("/camera")
    def camera():
        """Capture source: runs in a normal phone browser, not on the glasses.
        MRBD Web Apps have no camera, and the paired phone is a relay rather
        than something the glasses page can reach — so frames have to make the
        trip out to this server and back."""
        return FileResponse(WEB_DIR / "camera.html", headers=NOCACHE)

    @app.get("/glasses")
    def glasses():
        """Meta Ray-Ban Display client. Same /ws/phone motion contract, but the
        IMU is head-mounted like the World Context capture rig, so queries hit
        the bank without the wrist/pocket placement gap."""
        return FileResponse(WEB_DIR / "glasses.html", headers=NOCACHE)

    @app.get("/api/meta")
    def meta():
        return {
            "n_windows": int(len(S.t)),
            "n_clips": len(S.clips),
            "tasks": S.tasks,
            "embedder": S.embedder_kind,
            "motion_space": "learned" if S.query_encoder else "feature-baseline",
            "text_search_real": S.embedder_kind == "clip",
        }

    @app.get("/api/atlas")
    def atlas():
        return FileResponse(S.atlas_path, media_type="application/json")

    @app.get("/api/window/{row}")
    def window(row: int):
        if not (0 <= row < len(S.t)):
            raise HTTPException(404)
        info = S.row_info(row)
        clip = S.handles.get(info["clip"])
        if clip is not None:
            t = info["t"]
            t0, t1 = max(0.0, t - 15), min(clip.duration_s, t + 15)
            times, data, _ = clip.imu_window(t0, t1)
            info["spikes"] = detect_spikes(times, data)
            info["chapters"] = chapter_boundaries(times, data)
            info["duration_s"] = clip.duration_s
        return info

    @app.get("/api/frame")
    def frame(row: int, w: int = 320):
        if not (0 <= row < len(S.t)):
            raise HTTPException(404)
        return Response(_frame_jpeg(row, w), media_type="image/jpeg")

    @lru_cache(maxsize=1024)
    def _frame_jpeg(row: int, w: int) -> bytes:
        from PIL import Image

        info = S.row_info(row)
        clip = S.handles[info["clip"]]
        img = clip.frame(info["t"], width=w)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=82)
        return buf.getvalue()

    @app.get("/api/imu")
    def imu_trace(row: int, span: float = 12.0):
        info = S.row_info(row)
        clip = S.handles.get(info["clip"])
        if clip is None:
            raise HTTPException(404)
        t = info["t"]
        t0, t1 = max(0.0, t - span / 2), min(clip.duration_s, t + span / 2)
        times, data, _ = clip.imu_window(t0, t1)
        mag = np.linalg.norm(data[:, :3], axis=1)
        step = max(1, len(mag) // 400)
        return {
            "t": np.round(times[::step], 3).tolist(),
            "mag": np.round(mag[::step], 3).tolist(),
            "center": t,
            "spikes": detect_spikes(times, data),
        }

    @app.get("/api/row_at")
    def row_at(clip: str, t: float):
        try:
            ci = S.clips.index(clip)
        except ValueError:
            raise HTTPException(404)
        idx = np.where(S.clip_idx == ci)[0]
        if len(idx) == 0:
            raise HTTPException(404)
        row = int(idx[np.argmin(np.abs(S.t[idx] - t))])
        return S.row_info(row)

    @app.get("/api/search/text")
    def search_text(q: str, k: int = 24):
        emb = S.get_embedder()
        vec = emb.encode_text(q)
        return {
            "stub": S.embedder_kind != "clip",
            "results": S.topk(S.bank_v, vec.astype(np.float32), k),
        }

    @app.get("/api/search/window")
    def search_window(row: int, k: int = 16):
        q = S.bank_v[row] / (np.linalg.norm(S.bank_v[row]) + 1e-9)
        results = [r for r in S.topk(S.bank_v, q, k + 4) if r["row"] != row][:k]
        return {"results": results}

    @app.post("/api/search/image")
    async def search_image(payload: dict):
        """Visual query: one base64 frame (camera, screen capture of a glasses
        video call, phone) -> CLIP image tower -> nearest video moments."""
        b64 = payload.get("image", "")
        if "," in b64[:64]:  # strip data-URL prefix
            b64 = b64.split(",", 1)[1]
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception:
            raise HTTPException(400, "image must be base64 JPEG/PNG (data URL ok)")
        arr = np.asarray(img)
        k = int(payload.get("k", 12))
        loop = asyncio.get_event_loop()

        def run():
            vec = S.get_embedder().encode_images([arr])[0]
            return S.topk(S.bank_v, vec.astype(np.float32), k)

        return {"stub": S.embedder_kind != "clip",
                "results": await loop.run_in_executor(None, run)}

    @app.post("/api/search/motion")
    async def search_motion(payload: dict):
        samples = np.asarray(payload.get("samples", []), dtype=np.float32)
        if samples.ndim != 2 or samples.shape[0] < 20:
            raise HTTPException(400, "need samples: [[t,ax,ay,az,gx,gy,gz], ...]")
        times, data = samples[:, 0], samples[:, 1:7]
        matches, space = S.motion_query(times, data, k=int(payload.get("k", 8)))
        return {"space": space, "matches": matches}

    # ---------------- camera relay ----------------
    # Deliberately last-frame-wins with no queue: a stalled consumer should see
    # a fresh frame when it comes back, never work through a backlog of stale
    # ones. Frames are small and disposable, so nothing is persisted.
    cam = {"jpeg": None, "at": 0.0, "n": 0, "pulls": 0, "last_pull_ua": ""}

    @app.post("/api/cam/push")
    async def cam_push(request: Request):
        body = await request.body()
        if not body:
            raise HTTPException(400, "empty frame")
        cam["jpeg"], cam["at"], cam["n"] = body, time.time(), cam["n"] + 1
        return {"ok": True, "n": cam["n"], "bytes": len(body)}

    @app.get("/api/cam/latest")
    def cam_latest(request: Request):
        # Counted so a consumer that never asks can be told apart from one that
        # asks and fails to render -- the two look identical on the glasses.
        cam["pulls"] += 1
        cam["last_pull_ua"] = request.headers.get("user-agent", "")[:120]
        if cam["jpeg"] is None:
            raise HTTPException(404, "no frame pushed yet")
        return Response(cam["jpeg"], media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/cam/status")
    def cam_status():
        return {"frames": cam["n"],
                "age_s": round(time.time() - cam["at"], 2) if cam["n"] else None,
                "bytes": len(cam["jpeg"]) if cam["jpeg"] else 0,
                "pulls": cam["pulls"], "last_pull_ua": cam["last_pull_ua"]}

    @app.websocket("/ws/cam")
    async def ws_cam(ws: WebSocket):
        """Push frames instead of letting the client poll for them.

        Polling costs a full round trip per frame, so the frame rate collapses
        to 1/latency no matter how small the JPEG is -- which is why shrinking
        the payload changed nothing. Awaiting each send paces this loop to
        whatever the client can actually drain, so a slow link drops frames
        rather than building a backlog of stale ones.
        """
        await ws.accept()
        sent_n = -1
        try:
            while True:
                if cam["jpeg"] is not None and cam["n"] != sent_n:
                    sent_n = cam["n"]
                    await ws.send_bytes(cam["jpeg"])
                else:
                    await asyncio.sleep(0.02)
        except (WebSocketDisconnect, RuntimeError):
            pass

    # ---------------- live phone bridge ----------------

    async def _broadcast(msg: dict):
        dead = []
        for ws in S.live:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            S.live.discard(ws)

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        S.live.add(ws)
        try:
            while True:
                await ws.receive_text()  # keepalive pings; content ignored
        except WebSocketDisconnect:
            S.live.discard(ws)

    @app.websocket("/ws/phone")
    async def ws_phone(ws: WebSocket):
        await ws.accept()
        buf: deque = deque(maxlen=2000)
        last = 0.0
        await _broadcast({"type": "phone", "connected": True})
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                for s in msg.get("samples", []):
                    if len(s) >= 7:
                        buf.append(s)
                now = time.time()
                if len(buf) < 30 or now - last < 0.7:
                    continue
                arr = np.asarray(buf, dtype=np.float32)
                times, data = arr[:, 0], arr[:, 1:7]
                span = times[-1] - times[0]
                if span < config.WINDOW_S * 0.8:
                    continue
                last = now  # only rate-limit searches that actually ran
                loop = asyncio.get_event_loop()
                matches, space = await loop.run_in_executor(
                    None, S.motion_query, times, data)
                energy = float(np.linalg.norm(data[:, :3], axis=1).std())
                out = {"type": "matches", "space": space,
                       "energy": round(energy, 3), "matches": matches}
                await ws.send_json(out)
                await _broadcast(out)
        except WebSocketDisconnect:
            await _broadcast({"type": "phone", "connected": False})

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default=config.INDEX_DIR)
    ap.add_argument("--source", choices=["fixture", "worldcontext"], required=True)
    ap.add_argument("--wc-root", default=None)
    ap.add_argument("--embedder", choices=["fake", "clip"], default="fake",
                    help="text-search tower; must match what ingest used")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--imu-model", default=f"{config.MODELS_DIR}/imu_encoder.pt")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    import uvicorn

    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
