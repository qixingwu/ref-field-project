from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def upsample_score_map(score_map: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    # score_map: [B, Hp, Wp]
    score_map = score_map.unsqueeze(1)
    out = F.interpolate(score_map, size=size, mode="bilinear", align_corners=False)
    return out.squeeze(1)


def gaussian_blur(score_map: torch.Tensor, sigma: float = 1.0, kernel_size: int = 5) -> torch.Tensor:
    if sigma <= 0:
        return score_map
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2
    x = torch.arange(-radius, radius + 1, device=score_map.device, dtype=score_map.dtype)
    kernel_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    x = score_map.unsqueeze(1)
    x = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    x = F.conv2d(x, kernel_2d)
    return x.squeeze(1)


def topk_image_score(score_map: torch.Tensor, ratio: float = 0.01) -> torch.Tensor:
    # score_map: [B,H,W]
    b = score_map.shape[0]
    flat = score_map.view(b, -1)
    k = max(1, int(flat.shape[1] * ratio))
    topk, _ = torch.topk(flat, k=k, dim=1)
    return topk.mean(dim=1)


def save_checkpoint(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
