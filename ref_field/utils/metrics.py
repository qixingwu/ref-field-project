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


def compute_pro_auc(masks: List[np.ndarray], score_maps: List[np.ndarray], fpr_limit: float = 0.3) -> float:
    """
    Compute PRO (Per-Region Overpass) metric AUC.
    Reference: https://github.com/optimass-continual-learning/continual_anomaly_detection

    Args:
        masks: List of ground truth masks
        score_maps: List of anomaly score maps
        fpr_limit: False positive rate limit for PRO curve (default 0.3)
    """
    # Flatten and concatenate all masks and score maps
    y_true = np.concatenate([m.astype(np.float32).reshape(-1) for m in masks], axis=0)
    y_score = np.concatenate([s.astype(np.float32).reshape(-1) for s in score_maps], axis=0)

    # Sort scores descending
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Calculate true positives at each threshold
    tp_cumsum = np.cumsum(y_true_sorted)
    total_positive = np.sum(y_true)

    if total_positive == 0:
        return float("nan")

    # Calculate FPR and TPR
    num_pixels = len(y_true)
    num_negative = num_pixels - total_positive

    fp_cumsum = np.arange(1, num_pixels + 1) - tp_cumsum
    fpr = fp_cumsum / num_negative
    tpr = tp_cumsum / total_positive

    # Filter by FPR limit
    valid_mask = fpr <= fpr_limit

    if not np.any(valid_mask):
        return float("nan")

    fpr_valid = fpr[valid_mask]
    tpr_valid = tpr[valid_mask]

    # Compute AUC using trapezoidal rule
    pro_auc = np.trapz(tpr_valid, fpr_valid) / fpr_limit

    return float(pro_auc)


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

    pro_auc = compute_pro_auc(masks, score_maps)

    return {
        "pixel_roc_auc": safe_roc_auc(y_true, y_score),
        "pixel_ap": safe_ap(y_true, y_score),
        "pixel_pro": pro_auc,
    }
