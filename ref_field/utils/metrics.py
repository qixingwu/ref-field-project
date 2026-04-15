from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def image_level_metrics(labels: List[int], scores: List[float]) -> Dict[str, float]:
    y_true = np.asarray(labels)
    y_score = np.asarray(scores)
    return {
        "image_roc_auc": safe_roc_auc(y_true, y_score),
        "image_ap": safe_ap(y_true, y_score),
    }


def pixel_level_metrics(masks: List[np.ndarray], score_maps: List[np.ndarray]) -> Dict[str, float]:
    y_true = np.concatenate([m.astype(np.uint8).reshape(-1) for m in masks], axis=0)
    y_score = np.concatenate([s.astype(np.float32).reshape(-1) for s in score_maps], axis=0)
    return {
        "pixel_roc_auc": safe_roc_auc(y_true, y_score),
        "pixel_ap": safe_ap(y_true, y_score),
    }
