from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


class TokenStore:
    def __init__(self):
        self.global_descs: List[np.ndarray] = []
        self.token_bank: List[np.ndarray] = []
        self.image_paths: List[str] = []

    def add(self, image_path: str, global_desc: np.ndarray, tokens: np.ndarray) -> None:
        self.image_paths.append(image_path)
        self.global_descs.append(global_desc.astype(np.float32))
        self.token_bank.append(tokens.astype(np.float32))

    def stack_globals(self) -> np.ndarray:
        return np.stack(self.global_descs, axis=0)

    def gather_tokens(self, indices: np.ndarray) -> np.ndarray:
        gathered = [self.token_bank[int(i)] for i in indices]
        return np.concatenate(gathered, axis=0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            image_paths=np.asarray(self.image_paths, dtype=object),
            global_descs=np.asarray(self.global_descs, dtype=object),
            token_bank=np.asarray(self.token_bank, dtype=object),
        )

    def load(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=True)
        self.image_paths = data["image_paths"].tolist()
        self.global_descs = data["global_descs"].tolist()
        self.token_bank = data["token_bank"].tolist()
