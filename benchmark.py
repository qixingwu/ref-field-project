from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ref_field.utils.config import load_config

MVTec_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]

PRIMARY_METRICS = [
    "image_roc_auc",
    "image_ap",
    "pixel_roc_auc",
    "pixel_ap",
    "pixel_pro",
]


def load_metrics_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        print(f"WARNING: metrics.json not found: {path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"WARNING: failed to parse metrics.json: {path} ({exc})")
    except OSError as exc:
        print(f"WARNING: failed to read metrics.json: {path} ({exc})")
    return None


def metric_value(metrics: Dict[str, Any], key: str) -> float:
    summary = metrics.get("summary")
    value = None
    if isinstance(summary, dict):
        value = summary.get(key)
    if value is None:
        value = metrics.get(key)

    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def summary_value(metrics: Dict[str, Any], key: str, default: Any = "nan") -> Any:
    summary = metrics.get("summary")
    if isinstance(summary, dict) and summary.get(key) is not None:
        return summary[key]
    if metrics.get(key) is not None:
        return metrics[key]
    return default


def extract_primary_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {key: metric_value(metrics, key) for key in PRIMARY_METRICS}


def format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def nanmean(values: Iterable[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    if not valid:
        return math.nan
    return sum(valid) / len(valid)


def print_category_table(rows: List[Dict[str, Any]]) -> None:
    print("\nBenchmark results summary")
    print("-" * 78)
    print(f"{'Category':<14} {'I-ROC':>10} {'I-AP':>10} {'P-ROC':>10} {'P-AP':>10} {'PRO':>10}")
    print("-" * 78)
    for row in rows:
        print(
            f"{row['category']:<14} "
            f"{format_metric(row['image_roc_auc']):>10} "
            f"{format_metric(row['image_ap']):>10} "
            f"{format_metric(row['pixel_roc_auc']):>10} "
            f"{format_metric(row['pixel_ap']):>10} "
            f"{format_metric(row['pixel_pro']):>10}"
        )
    print("-" * 78)

    mean_row = {key: nanmean(row[key] for row in rows) for key in PRIMARY_METRICS}
    print("Mean over categories summary")
    print(
        f"{'Mean':<14} "
        f"{format_metric(mean_row['image_roc_auc']):>10} "
        f"{format_metric(mean_row['image_ap']):>10} "
        f"{format_metric(mean_row['pixel_roc_auc']):>10} "
        f"{format_metric(mean_row['pixel_ap']):>10} "
        f"{format_metric(mean_row['pixel_pro']):>10}"
    )
    print("-" * 78)


def print_unified_summary(metrics: Dict[str, Any]) -> None:
    primary = extract_primary_metrics(metrics)

    print("\nUnified multi-class benchmark summary")
    print("-" * 50)
    print(f"Mode: {summary_value(metrics, 'mode')}")
    print(f"Context scope: {summary_value(metrics, 'context_scope')}")
    print(f"Gate scope: {summary_value(metrics, 'gate_scope')}")
    print(f"Test images: {summary_value(metrics, 'num_test_images')}")
    print(f"Memory images: {summary_value(metrics, 'num_train_memory_images')}")
    print(f"Image ROC-AUC: {format_metric(primary['image_roc_auc'])}")
    print(f"Image AP:      {format_metric(primary['image_ap'])}")
    print(f"Pixel ROC-AUC: {format_metric(primary['pixel_roc_auc'])}")
    print(f"Pixel AP:      {format_metric(primary['pixel_ap'])}")
    print(f"Pixel PRO:     {format_metric(primary['pixel_pro'])}")
    print(f"Result dir: {summary_value(metrics, 'result_dir')}")
    print("-" * 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--resume_context", type=str, default="")
    ap.add_argument("--context_scope", type=str, choices=["category", "dataset_shared"], default="category")
    ap.add_argument("--gate_scope", type=str, choices=["category", "dataset_shared"], default="category")
    ap.add_argument("--save_vis", action="store_true", help="Save visualization images")
    args = ap.parse_args()

    print(f"gate_scope: {args.gate_scope}")
    print(f"context_scope: {args.context_scope}")
    if args.resume_context:
        print(f"resume_context: {args.resume_context}")
    if args.checkpoint:
        print(f"checkpoint: {args.checkpoint}")

    cfg = load_config(args.config)
    work_dir = Path(cfg["work_dir"])

    if args.gate_scope == "dataset_shared":
        print("benchmark will run a single unified multi-class inference call")
        print("unified mode: --category bottle is only for CLI compatibility and will be ignored by infer.py")
        cmd = [
            "python", "infer.py",
            "--config", args.config,
            "--category", "bottle",
            "--split", "test",
            "--gate_scope", args.gate_scope,
            "--context_scope", args.context_scope,
        ]
        if args.resume_context:
            cmd += ["--resume_context", args.resume_context]
        if args.checkpoint:
            cmd += ["--checkpoint", args.checkpoint]
        if args.save_vis:
            cmd += ["--save_vis"]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=False)
        metrics_path = work_dir / "_shared_gate" / "mvtec" / "metrics.json"
        metrics = load_metrics_json(metrics_path)
        if metrics is not None:
            print_unified_summary(metrics)
        return

    print("benchmark will iterate over all MVTec categories")
    rows = []
    for cat in MVTec_CATEGORIES:
        cmd = [
            "python", "infer.py",
            "--config", args.config,
            "--category", cat,
            "--split", "test",
            "--context_scope", args.context_scope,
        ]
        if args.resume_context:
            cmd += ["--resume_context", args.resume_context]
        if args.checkpoint:
            cmd += ["--checkpoint", args.checkpoint]
        if args.save_vis:
            cmd += ["--save_vis"]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=False)
        metrics_path = work_dir / cat / "metrics.json"
        metrics = load_metrics_json(metrics_path)
        if metrics is None:
            continue

        row = {"category": cat, **extract_primary_metrics(metrics)}
        rows.append(row)
        print(
            f"{cat}: "
            f"I-ROC={format_metric(row['image_roc_auc'])}, "
            f"I-AP={format_metric(row['image_ap'])}, "
            f"P-ROC={format_metric(row['pixel_roc_auc'])}, "
            f"P-AP={format_metric(row['pixel_ap'])}, "
            f"PRO={format_metric(row['pixel_pro'])}"
        )

    if rows:
        print_category_table(rows)
    else:
        print("WARNING: no metrics.json files were loaded; benchmark summary is unavailable.")


if __name__ == "__main__":
    main()
