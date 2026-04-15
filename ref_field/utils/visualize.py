from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_overlay(image: np.ndarray, score_map: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.axis("off")
    ax1.set_title("Image")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(image)
    ax2.imshow(score_map, alpha=0.5)
    ax2.axis("off")
    ax2.set_title("Anomaly map")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
