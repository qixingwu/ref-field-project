from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_overlay(image: np.ndarray, score_map: np.ndarray, out_path: str | Path, gt_mask: np.ndarray | None = None) -> None:
    """
    Save visualization with ground truth mask for defect images.

    For defect images (gt_mask provided): shows 1x3 grid (Image, GT, Prediction)
    For normal images (no gt_mask): shows 1x3 grid (Image, Mask(empty), Prediction)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_gt = gt_mask is not None

    # Always use 1x3 layout for consistency
    fig = plt.figure(figsize=(15, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image)
    ax1.axis("off")
    ax1.set_title("Image")

    ax2 = fig.add_subplot(1, 3, 2)
    if has_gt:
        ax2.imshow(gt_mask, cmap='gray')
        ax2.set_title("Ground Truth")
    else:
        ax2.imshow(np.zeros_like(score_map), cmap='gray', vmin=0, vmax=1)
        ax2.set_title("Ground Truth (Normal)")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(image)
    ax3.imshow(score_map, cmap='jet', alpha=0.5)
    ax3.axis("off")
    ax3.set_title("Anomaly Map")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
