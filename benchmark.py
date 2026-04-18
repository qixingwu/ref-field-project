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


def metric_value(metrics: Dict[str, Any], key: str, prefer_top_level: bool = False) -> float:
    summary = metrics.get("summary")
    value = None
    if prefer_top_level and metrics.get(key) is not None:
        value = metrics.get(key)
    elif isinstance(summary, dict):
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


def extract_primary_metrics(metrics: Dict[str, Any], prefer_top_level: bool = False) -> Dict[str, float]:
    return {key: metric_value(metrics, key, prefer_top_level=prefer_top_level) for key in PRIMARY_METRICS}


def format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def nanmean(values: Iterable[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    if not valid:
        return math.nan
    return sum(valid) / len(valid)


def serialize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [serialize_for_json(item) for item in value]
    return value


def write_benchmark_summary(path: Path, summary: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialize_for_json(summary), f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        print(f"Saved benchmark summary to: {path}")
    except (OSError, TypeError, ValueError) as exc:
        print(f"WARNING: failed to save benchmark summary: {path} ({exc})")


def mean_over_category_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return {key: nanmean(row[key] for row in rows) for key in PRIMARY_METRICS}


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

    mean_row = mean_over_category_rows(rows)
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


def print_metric_row(name: str, metrics: Dict[str, float]) -> None:
    print(
        f"{name:<14} "
        f"{format_metric(metrics['image_roc_auc']):>10} "
        f"{format_metric(metrics['image_ap']):>10} "
        f"{format_metric(metrics['pixel_roc_auc']):>10} "
        f"{format_metric(metrics['pixel_ap']):>10} "
        f"{format_metric(metrics['pixel_pro']):>10}"
    )


def print_unified_summary(metrics: Dict[str, Any]) -> None:
    primary = extract_primary_metrics(metrics, prefer_top_level=True)

    print("\nUnified multi-class benchmark summary")
    print("-" * 50)
    print(f"Mode: {summary_value(metrics, 'mode')}")
    print(f"Context scope: {summary_value(metrics, 'context_scope')}")
    print(f"Gate scope: {summary_value(metrics, 'gate_scope')}")
    print(f"Test images: {summary_value(metrics, 'num_test_images')}")
    print(f"Memory images: {summary_value(metrics, 'num_train_memory_images')}")
    print("\nUnified overall metrics")
    print(f"Image ROC-AUC: {format_metric(primary['image_roc_auc'])}")
    print(f"Image AP:      {format_metric(primary['image_ap'])}")
    print(f"Pixel ROC-AUC: {format_metric(primary['pixel_roc_auc'])}")
    print(f"Pixel AP:      {format_metric(primary['pixel_ap'])}")
    print(f"Pixel PRO:     {format_metric(primary['pixel_pro'])}")
    print(f"Result dir: {summary_value(metrics, 'result_dir')}")
    print("-" * 50)


def print_unified_per_category_table(per_category: Any) -> None:
    if not isinstance(per_category, dict):
        print("WARNING: unified per-category breakdown not found")
        return

    print("\nUnified per-category metrics")
    print("-" * 78)
    print(f"{'Category':<14} {'I-ROC':>10} {'I-AP':>10} {'P-ROC':>10} {'P-AP':>10} {'PRO':>10}")
    print("-" * 78)
    for category in sorted(per_category):
        category_metrics = per_category[category]
        if not isinstance(category_metrics, dict):
            category_metrics = {}
        print_metric_row(category, extract_primary_metrics(category_metrics))
    print("-" * 78)


def print_unified_mean_over_categories(metrics: Dict[str, Any]) -> None:
    summary = metrics.get("summary")
    mean_metrics = None
    if isinstance(summary, dict):
        mean_metrics = summary.get("mean_over_categories")
    if not isinstance(mean_metrics, dict):
        print("WARNING: unified mean_over_categories not found")
        return

    primary = extract_primary_metrics(mean_metrics)
    print("\nMean over categories")
    print("(simple mean of per-category metrics; not the same as unified overall metrics)")
    print("-" * 78)
    print(f"{'Metric set':<14} {'I-ROC':>10} {'I-AP':>10} {'P-ROC':>10} {'P-AP':>10} {'PRO':>10}")
    print("-" * 78)
    print_metric_row("Mean", primary)
    print("-" * 78)


def build_run_info(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "context_scope": args.context_scope,
        "gate_scope": args.gate_scope,
        "checkpoint": args.checkpoint,
        "resume_context": args.resume_context,
        "save_vis": args.save_vis,
        "config": args.config,
    }


def build_category_summary(
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    source_metrics_paths: Dict[str, Path],
) -> Dict[str, Any]:
    summary_rows = []
    for row in rows:
        category = row["category"]
        summary_rows.append({
            "category": category,
            **{key: row[key] for key in PRIMARY_METRICS},
            "metrics_path": source_metrics_paths[category],
        })

    return {
        "mode": "category",
        **build_run_info(args),
        "source_metrics_paths": source_metrics_paths,
        "rows": summary_rows,
        "mean_over_categories": mean_over_category_rows(rows),
    }


def build_unified_summary(args: argparse.Namespace, metrics_path: Path, metrics: Dict[str, Any] | None) -> Dict[str, Any]:
    summary = {
        "mode": "dataset_shared",
        **build_run_info(args),
        "source_metrics_paths": str(metrics_path),
        "overall_available": metrics is not None,
    }
    if metrics is None:
        return summary

    raw_summary = metrics.get("summary")
    mean_metrics = raw_summary.get("mean_over_categories") if isinstance(raw_summary, dict) else None
    per_category = metrics.get("per_category")
    if isinstance(per_category, dict):
        per_category = {
            category: extract_primary_metrics(category_metrics if isinstance(category_metrics, dict) else {})
            for category, category_metrics in sorted(per_category.items())
        }
    else:
        per_category = {}

    summary.update({
        "overall": extract_primary_metrics(metrics, prefer_top_level=True),
        "per_category": per_category,
        "mean_over_categories": (
            extract_primary_metrics(mean_metrics) if isinstance(mean_metrics, dict) else {}
        ),
    })
    if isinstance(raw_summary, dict):
        summary["summary"] = raw_summary
    return summary


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
            print_unified_per_category_table(metrics.get("per_category"))
            print_unified_mean_over_categories(metrics)
        write_benchmark_summary(
            work_dir / "_shared_gate" / "mvtec" / "benchmark_summary.json",
            build_unified_summary(args, metrics_path, metrics),
        )
        return

    print("benchmark will iterate over all MVTec categories")
    rows = []
    source_metrics_paths = {}
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
        source_metrics_paths[cat] = metrics_path
        metrics = load_metrics_json(metrics_path)
        if metrics is None:
            continue

        row = {"category": cat, **extract_primary_metrics(metrics), "metrics_path": metrics_path}
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
        print("WARNING: no metrics.json files were loaded; category metrics table is unavailable.")
    write_benchmark_summary(
        work_dir / "benchmark_summary.json",
        build_category_summary(args, rows, source_metrics_paths),
    )


if __name__ == "__main__":
    main()
