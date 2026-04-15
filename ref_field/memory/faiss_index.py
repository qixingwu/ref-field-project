from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class ImageIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._xb = None
        self._index = None

    def build(self, xb: np.ndarray) -> None:
        xb = xb.astype(np.float32)
        self._xb = xb
        if faiss is not None:
            index = faiss.IndexFlatIP(self.dim)
            index.add(xb)
            self._index = index
        else:
            self._index = None

    def search(self, xq: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        xq = xq.astype(np.float32)
        if self._index is not None:
            scores, inds = self._index.search(xq, topk)
            return scores, inds
        assert self._xb is not None, "Index not built."
        sim = xq @ self._xb.T
        inds = np.argsort(-sim, axis=1)[:, :topk]
        rows = np.arange(xq.shape[0])[:, None]
        scores = sim[rows, inds]
        return scores, inds

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self._xb)

    def load(self, path: str | Path) -> None:
        xb = np.load(path).astype(np.float32)
        self.build(xb)
