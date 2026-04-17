from __future__ import annotations

from typing import Dict, List, Union

import cv2
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from skimage import measure


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights: Union[np.ndarray, List[float]],
    anomaly_ground_truth_labels: Union[np.ndarray, List[int]],
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute image-level retrieval metrics following reference implementation.

    Args:
        anomaly_prediction_weights: Array of anomaly scores for each image
        anomaly_ground_truth_labels: Array of ground truth labels (0=normal, 1=anomaly)

    Returns:
        Dictionary containing:
            - auroc: Area under ROC curve
            - fpr: False positive rates at each threshold
            - tpr: True positive rates at each threshold
            - threshold: Threshold values for ROC curve
            - precision: Precision values at each threshold
            - recall: Recall values at each threshold
            - pr_threshold: Threshold values for PR curve
            - auc_pr: Area under Precision-Recall curve
    """
    y_true = np.asarray(anomaly_ground_truth_labels)
    y_score = np.asarray(anomaly_prediction_weights)

    # Handle edge cases
    if len(y_true) == 0 or len(y_score) == 0:
        return {
            "auroc": float("nan"),
            "fpr": np.array([]),
            "tpr": np.array([]),
            "threshold": np.array([]),
            "precision": np.array([]),
            "recall": np.array([]),
            "pr_threshold": np.array([]),
            "auc_pr": float("nan"),
        }

    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        # Return empty arrays with nan values for single-class case
        return {
            "auroc": float("nan"),
            "fpr": np.array([]),
            "tpr": np.array([]),
            "threshold": np.array([]),
            "precision": np.full(len(y_true), np.nan),
            "recall": np.full(len(y_true), np.nan),
            "pr_threshold": np.full(len(y_true), np.nan),
            "auc_pr": float("nan"),
        }

    # Compute ROC curve
    fpr, tpr, threshold = roc_curve(y_true, y_score)
    auroc = float(roc_auc_score(y_true, y_score))

    # Compute Precision-Recall curve
    precision, recall, pr_threshold = precision_recall_curve(y_true, y_score)
    auc_pr = float(auc(recall, precision))

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "pr_threshold": pr_threshold,
        "auc_pr": auc_pr,
    }


def compute_pixelwise_retrieval_metrics(
    anomaly_segmentations: Union[np.ndarray, List[np.ndarray]],
    ground_truth_masks: Union[np.ndarray, List[np.ndarray]],
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute pixel-level retrieval metrics following reference implementation.

    Args:
        anomaly_segmentations: Predicted anomaly score maps
        ground_truth_masks: Ground truth binary masks

    Returns:
        Dictionary containing:
            - auroc: Area under ROC curve
            - fpr: False positive rates at each threshold
            - tpr: True positive rates at each threshold
            - optimal_threshold: Threshold that maximizes F1 score
            - optimal_fpr: False positive rate at optimal threshold
            - optimal_fnr: False negative rate at optimal threshold
            - precision: Precision values at each threshold
            - recall: Recall values at each threshold
            - pr_threshold: Threshold values for PR curve
            - auc_pr: Area under Precision-Recall curve
            - f1_scores: F1 scores at each threshold
    """
    # Convert lists to stacked arrays
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations, axis=0)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks, axis=0)

    # Flatten arrays
    flat_anomaly_segmentations = anomaly_segmentations.reshape(-1)
    flat_ground_truth_masks = ground_truth_masks.reshape(-1)

    # Handle edge cases
    if len(flat_ground_truth_masks) == 0 or len(flat_anomaly_segmentations) == 0:
        return {
            "auroc": float("nan"),
            "fpr": np.array([]),
            "tpr": np.array([]),
            "optimal_threshold": float("nan"),
            "optimal_fpr": float("nan"),
            "optimal_fnr": float("nan"),
            "precision": np.array([]),
            "recall": np.array([]),
            "pr_threshold": np.array([]),
            "auc_pr": float("nan"),
            "f1_scores": np.array([]),
        }

    # Check if we have both classes
    if len(np.unique(flat_ground_truth_masks)) < 2:
        return {
            "auroc": float("nan"),
            "fpr": np.array([]),
            "tpr": np.array([]),
            "optimal_threshold": float("nan"),
            "optimal_fpr": float("nan"),
            "optimal_fnr": float("nan"),
            "precision": np.full(1, np.nan),
            "recall": np.full(1, np.nan),
            "pr_threshold": np.full(1, np.nan),
            "auc_pr": float("nan"),
            "f1_scores": np.full(1, np.nan),
        }

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(flat_ground_truth_masks, flat_anomaly_segmentations)
    auroc = float(roc_auc_score(flat_ground_truth_masks, flat_anomaly_segmentations))

    # Compute Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(
        flat_ground_truth_masks, flat_anomaly_segmentations
    )
    auc_pr = float(auc(recall, precision))

    # Compute F1 scores with zero division protection
    # Avoid division by zero: if precision + recall == 0, F1 = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * precision * recall / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores, nan=0.0)

    # Select optimal threshold (max F1, excluding last element which has threshold=0)
    optimal_threshold = float(pr_thresholds[np.argmax(f1_scores[:-1])])

    # Compute predictions at optimal threshold
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)

    # Compute optimal FPR and FNR
    optimal_fpr = float(np.mean(predictions > flat_ground_truth_masks))
    optimal_fnr = float(np.mean(predictions < flat_ground_truth_masks))

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": optimal_fpr,
        "optimal_fnr": optimal_fnr,
        "precision": precision,
        "recall": recall,
        "pr_threshold": pr_thresholds,
        "auc_pr": auc_pr,
        "f1_scores": f1_scores,
    }


def compute_pro(
    masks: Union[np.ndarray, List[np.ndarray]],
    amaps: Union[np.ndarray, List[np.ndarray]],
    num_th: int = 200,
) -> float:
    """
    Compute PRO (Per-Region Overlap) AUC following reference implementation.

    This implements the exact logic from the reference metrics.py:
    - Uses cv2.dilate with 5x5 rectangular kernel
    - Uses skimage.measure.regionprops for connected components
    - Normalizes FPR by its maximum value after filtering fpr < 0.3
    - Uses 200 thresholds by default

    Args:
        masks: Ground truth binary masks (list or array)
        amaps: Anomaly score maps (list or array)
        num_th: Number of thresholds to evaluate (default 200)

    Returns:
        PRO AUC score (float)
    """
    # Convert lists to arrays
    if isinstance(masks, list):
        masks = np.stack(masks, axis=0)
    if isinstance(amaps, list):
        amaps = np.stack(amaps, axis=0)

    # Handle edge cases
    if len(masks) == 0 or len(amaps) == 0:
        return float("nan")

    # Get score range
    min_th = amaps.min()
    max_th = amaps.max()

    # Handle constant score maps
    delta = (max_th - min_th) / num_th
    if delta == 0:
        return float("nan")

    # Thresholds
    thresholds = np.arange(min_th, max_th, delta)

    # Structuring element for dilation
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    results = []

    for th in thresholds:
        # Create binary anomaly maps at this threshold
        binary_amaps = (amaps >= th).astype(np.uint8)

        # Dilate binary maps
        dilated_amaps = []
        for i in range(len(binary_amaps)):
            dilated = cv2.dilate(binary_amaps[i], k)
            dilated_amaps.append(dilated)
        dilated_amaps = np.stack(dilated_amaps, axis=0)

        # Compute PRO score
        pro_scores = []
        for mask, dilated_amap in zip(masks, dilated_amaps):
            # Get connected components from ground truth
            regions = measure.regionprops(measure.label(mask))

            if len(regions) == 0:
                continue

            # Compute overlap for each region
            for region in regions:
                # Get region mask
                region_mask = (measure.label(mask) == region.label)
                tp_pixels = np.logical_and(region_mask, dilated_amap).sum()
                pro_scores.append(tp_pixels / region.area)

        if len(pro_scores) == 0:
            pro = 0.0
        else:
            pro = np.mean(pro_scores)

        # Compute FPR
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        results.append({
            "pro": pro,
            "fpr": fpr,
            "threshold": th,
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Filter to FPR < 0.3
    df = df[df["fpr"] < 0.3]

    # Handle empty dataframe after filtering
    if len(df) == 0:
        return float("nan")

    # Normalize FPR by its maximum
    if df["fpr"].max() == 0:
        return float("nan")

    df["fpr"] = df["fpr"] / df["fpr"].max()

    # Compute AUC
    pro_auc = metrics.auc(df["fpr"], df["pro"])

    return float(pro_auc)


def image_level_metrics(
    labels: List[int],
    scores: List[float],
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute image-level metrics with backward-compatible field names.

    This wrapper maintains compatibility with existing code while exposing
    the full set of metrics from the reference implementation.

    Returns:
        Dictionary with both legacy and new field names:
            - Legacy: image_roc_auc, image_ap
            - New: image_auc_pr, image_fpr, image_tpr, image_threshold,
                   image_precision, image_recall, image_pr_threshold
    """
    result = compute_imagewise_retrieval_metrics(scores, labels)

    # Add backward-compatible aliases
    result["image_roc_auc"] = result["auroc"]
    result["image_ap"] = result["auc_pr"]
    result["image_auc_pr"] = result["auc_pr"]
    result["image_fpr"] = result["fpr"]
    result["image_tpr"] = result["tpr"]
    result["image_threshold"] = result["threshold"]
    result["image_precision"] = result["precision"]
    result["image_recall"] = result["recall"]
    result["image_pr_threshold"] = result["pr_threshold"]

    return result


def pixel_level_metrics(
    masks: List[np.ndarray],
    score_maps: List[np.ndarray],
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute pixel-level metrics with backward-compatible field names.

    This wrapper maintains compatibility with existing code while exposing
    the full set of metrics from the reference implementation.

    Returns:
        Dictionary with both legacy and new field names:
            - Legacy: pixel_roc_auc, pixel_ap, pixel_pro
            - New: pixel_auc_pr, pixel_fpr, pixel_tpr, pixel_precision,
                   pixel_recall, pixel_pr_threshold, pixel_f1_scores,
                   pixel_optimal_threshold, pixel_optimal_fpr, pixel_optimal_fnr
    """
    # Compute PRO using reference implementation
    pro_score = compute_pro(masks, score_maps)

    # Compute other pixel metrics
    result = compute_pixelwise_retrieval_metrics(score_maps, masks)

    # Add backward-compatible aliases
    result["pixel_roc_auc"] = result["auroc"]
    result["pixel_ap"] = result["auc_pr"]
    result["pixel_pro"] = pro_score
    result["pixel_auc_pr"] = result["auc_pr"]
    result["pixel_fpr"] = result["fpr"]
    result["pixel_tpr"] = result["tpr"]
    result["pixel_precision"] = result["precision"]
    result["pixel_recall"] = result["recall"]
    result["pixel_pr_threshold"] = result["pr_threshold"]
    result["pixel_f1_scores"] = result["f1_scores"]
    result["pixel_optimal_threshold"] = result["optimal_threshold"]
    result["pixel_optimal_fpr"] = result["optimal_fpr"]
    result["pixel_optimal_fnr"] = result["optimal_fnr"]

    return result
