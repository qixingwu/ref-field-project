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


def _connected_components(binary_mask: np.ndarray) -> List[np.ndarray]:
    """Extract 4-connected components from a binary mask using pure numpy.

    Args:
        binary_mask: 2D binary array (H, W) where True indicates foreground

    Returns:
        List of boolean masks, one per connected component
    """
    components = []
    visited = np.zeros_like(binary_mask, dtype=bool)
    h, w = binary_mask.shape

    # 4-connected directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(h):
        for j in range(w):
            if binary_mask[i, j] and not visited[i, j]:
                # BFS to find this component
                component = np.zeros_like(binary_mask, dtype=bool)
                stack = [(i, j)]
                visited[i, j] = True

                while stack:
                    ci, cj = stack.pop()
                    component[ci, cj] = True

                    for di, dj in directions:
                        ni, nj = ci + di, cj + dj
                        if (0 <= ni < h and 0 <= nj < w and
                            binary_mask[ni, nj] and not visited[ni, nj]):
                            visited[ni, nj] = True
                            stack.append((ni, nj))

                components.append(component)

    return components


def compute_pro_auc(masks: List[np.ndarray], score_maps: List[np.ndarray], fpr_limit: float = 0.3) -> float:
    """
    Compute PRO (Per-Region Overlap) metric AUC at region level.

    For each threshold, we compute:
    - PRO: average overlap between predicted regions and ground truth regions
    - FPR: false positive rate on normal pixels

    Args:
        masks: List of ground truth masks
        score_maps: List of anomaly score maps
        fpr_limit: False positive rate limit for PRO curve (default 0.3)
    """
    if len(masks) == 0 or len(score_maps) == 0:
        return float("nan")

    # Binarize masks and collect scores
    bin_masks = [(m > 0.5).astype(np.uint8) for m in masks]

    # Count total normal pixels
    total_normal = sum(np.sum(1 - m) for m in bin_masks)
    if total_normal == 0:
        return float("nan")

    # Collect all scores for threshold sampling
    all_scores = np.concatenate([s.astype(np.float32).reshape(-1) for s in score_maps])
    num_thresholds = min(512, len(np.unique(all_scores)))
    thresholds = np.percentile(all_scores, np.linspace(100, 0, num_thresholds))

    # Initialize arrays for PRO and FPR at each threshold
    pro_values = []
    fpr_values = []

    for thresh in thresholds:
        # Initialize accumulators
        total_regions = 0
        sum_overlap = 0.0
        fp_count = 0

        for mask, score_map in zip(bin_masks, score_maps):
            pred = (score_map >= thresh).astype(np.uint8)

            # Count FP on normal pixels
            normal_mask = (1 - mask)
            fp_count += np.sum(pred & normal_mask)

            # Extract connected components from GT
            components = _connected_components(mask.astype(bool))

            if len(components) == 0:
                continue

            # Compute overlap for each region
            for comp in components:
                region_pixels = comp.sum()
                if region_pixels == 0:
                    continue
                overlap = np.mean(pred[comp])
                sum_overlap += overlap
                total_regions += 1

        if total_regions == 0:
            continue

        pro = sum_overlap / total_regions
        fpr = fp_count / total_normal

        pro_values.append(pro)
        fpr_values.append(fpr)

    if len(pro_values) == 0:
        return float("nan")

    # Convert to arrays and sort by FPR
    pro_values = np.array(pro_values)
    fpr_values = np.array(fpr_values)

    # Sort by FPR ascending
    sort_idx = np.argsort(fpr_values)
    fpr_values = fpr_values[sort_idx]
    pro_values = pro_values[sort_idx]

    # Filter to FPR <= fpr_limit
    valid_mask = fpr_values <= fpr_limit
    if not np.any(valid_mask):
        return float("nan")

    fpr_valid = fpr_values[valid_mask]
    pro_valid = pro_values[valid_mask]

    # Linear interpolation at fpr_limit if needed
    if fpr_valid[-1] < fpr_limit and len(fpr_valid) > 1:
        # Extrapolate to fpr_limit
        last_fpr = fpr_valid[-1]
        last_pro = pro_valid[-1]
        second_last_fpr = fpr_valid[-2]
        second_last_pro = pro_valid[-2]

        # Linear interpolation
        if last_fpr > second_last_fpr:
            slope = (last_pro - second_last_pro) / (last_fpr - second_last_fpr)
            pro_at_limit = last_pro + slope * (fpr_limit - last_fpr)
            fpr_valid = np.append(fpr_valid, fpr_limit)
            pro_valid = np.append(pro_valid, pro_at_limit)

    # Compute AUC using trapezoidal rule, normalized by fpr_limit
    pro_auc = np.trapz(pro_valid, fpr_valid) / fpr_limit

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
