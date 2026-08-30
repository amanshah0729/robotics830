"""Video/text embedding towers.

`clip` mode: frozen OpenCLIP ViT-B/32 — real semantics, GPU strongly
recommended for full ingests, and it gives free-text search via the text tower.
`fake` mode: deterministic pixel projection — no torch needed, used for the
fixture smoke test and any machine without ML deps.
"""

from __future__ import annotations

import hashlib

import numpy as np

from . import config


class FakeEmbedder:
    """Deterministic 512-d embedding from downsampled pixels. Not semantic, but
    consistent: visually similar frames land close, which is all the pipeline
    test needs. Text encoding is a stub (hash-seeded) and labeled as such."""

    name = "fake"

    def __init__(self, dim: int = config.EMB_DIM, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((768, dim)).astype(np.float32) / 27.7

    def _pool(self, img: np.ndarray) -> np.ndarray:
        gray = img.astype(np.float32).mean(axis=2)
        h, w = gray.shape
        gh, gw = 24, 32
        ys = (np.linspace(0, h, gh + 1)).astype(int)
        xs = (np.linspace(0, w, gw + 1)).astype(int)
        cells = np.empty((gh, gw), dtype=np.float32)
        for i in range(gh):
            for j in range(gw):
                cells[i, j] = gray[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean()
        return (cells.flatten() - 127.5) / 128.0

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        feats = np.stack([self._pool(im) for im in images]) @ self._proj
        return _l2(feats)

    def encode_text(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.lower().encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).standard_normal(config.EMB_DIM).astype(np.float32)
        return _l2(v[None])[0]


class ClipEmbedder:
    """Frozen OpenCLIP tower. Lazy import so `fake` mode needs no torch."""

    name = "clip"

    def __init__(self, device: str = "auto",
                 model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        import torch
        import open_clip

        self.torch = torch
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available()
            else ("mps" if device == "auto" and torch.backends.mps.is_available() else
                  ("cpu" if device == "auto" else device))
        )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()
        print(f"[embed] OpenCLIP {model_name}/{pretrained} on {self.device}")

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        with self.torch.no_grad():
            batch = self.torch.stack(
                [self.preprocess(Image.fromarray(im)) for im in images]
            ).to(self.device)
            emb = self.model.encode_image(batch).float().cpu().numpy()
        return _l2(emb)

    def encode_text(self, text: str) -> np.ndarray:
        with self.torch.no_grad():
            tok = self.tokenizer([text]).to(self.device)
            emb = self.model.encode_text(tok).float().cpu().numpy()
        return _l2(emb)[0]


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def make_embedder(kind: str, device: str = "auto"):
    if kind == "fake":
        return FakeEmbedder()
    if kind == "clip":
        return ClipEmbedder(device=device)
    raise ValueError(f"unknown embedder {kind!r} (use 'fake' or 'clip')")
