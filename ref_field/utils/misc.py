from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


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


def gaussian_blur(score_map: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """Apply Gaussian blur using scipy.ndimage for PatchCore-style smoothing.

    Args:
        score_map: [B, H, W] tensor
        sigma: Standard deviation for Gaussian kernel (default 4.0 like PatchCore)

    Returns:
        Blurred score_map with same shape and device as input
    """
    if sigma <= 0:
        return score_map

    device = score_map.device
    dtype = score_map.dtype
    b, h, w = score_map.shape

    # Convert to numpy, apply filter per-image, convert back
    score_np = score_map.cpu().numpy()
    blurred_np = np.empty_like(score_np)

    for i in range(b):
        blurred_np[i] = gaussian_filter(score_np[i], sigma=sigma, mode='reflect')

    return torch.from_numpy(blurred_np).to(device=device, dtype=dtype)


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
