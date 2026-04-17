from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from ref_field.datasets.mvtec import MVTecADDataset
from ref_field.datasets.transforms import denormalize_image
from ref_field.memory.faiss_index import ImageIndex
from ref_field.memory.token_store import TokenStore
from ref_field.models.ref_field import RefField
from ref_field.utils.config import load_config
from ref_field.utils.metrics import image_level_metrics, pixel_level_metrics
from ref_field.utils.misc import ensure_dir, gaussian_blur, get_device, set_seed, upsample_score_map
from ref_field.utils.visualize import save_overlay


def serialize_for_json(obj: Any) -> Any:
    """Convert numpy/pandas objects to JSON-serializable types.

    This is a minimal helper for saving metrics to JSON. It does not modify
    the original metrics computation logic, only converts types for serialization.

    Args:
        obj: Any object (numpy scalar, ndarray, DataFrame, dict, list, or primitive)

    Returns:
        JSON-serializable version of the input:
            - numpy scalar -> Python scalar (float/int)
            - numpy ndarray -> list
            - pandas DataFrame -> list of dicts (records format)
            - Other types returned as-is
    """
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        # Handle NaN values
        if obj.dtype == bool or obj.dtype == np.bool_:
            return [bool(x) if not np.isnan(x) else None for x in obj]
        return [float(x) if not np.isnan(x) else None for x in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    else:
        return obj


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--category", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--resume_context", type=str, default="")
    ap.add_argument("--context_scope", type=str, choices=["category", "dataset_shared"], default="category")
    ap.add_argument("--gate_scope", type=str, choices=["category", "dataset_shared"], default="category")
    ap.add_argument("--save_vis", action="store_true")
    return ap.parse_args()


def resolve_context_checkpoint(cfg, args):
    if args.resume_context:
        return Path(args.resume_context), "explicit path"

    if args.context_scope == "category":
        return Path(cfg["work_dir"]) / args.category / "checkpoints" / "context_last.pt", "category-specific auto path"

    return Path(cfg["work_dir"]) / "_shared_context" / "mvtec" / "checkpoints" / "context_last.pt", "shared auto path"


def discover_mvtec_categories(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"MVTec root does not exist: {root}")

    categories = [
        p.name
        for p in sorted(root.iterdir())
        if p.is_dir() and (p / "train" / "good").is_dir()
    ]
    if not categories:
        raise ValueError(f"No MVTec categories with train/good found under: {root}")
    return categories


def build_mvtec_infer_datasets(cfg, args):
    if args.gate_scope == "category":
        train_ds = MVTecADDataset(
            cfg["data"]["root"],
            args.category,
            split="train",
            image_size=cfg["data"]["image_size"],
            good_only=True,
        )
        test_ds = MVTecADDataset(
            cfg["data"]["root"],
            args.category,
            split=args.split,
            image_size=cfg["data"]["image_size"],
            good_only=False,
        )
        return train_ds, test_ds, [args.category]

    categories = discover_mvtec_categories(cfg["data"]["root"])
    train_sets = [
        MVTecADDataset(
            cfg["data"]["root"],
            category,
            split="train",
            image_size=cfg["data"]["image_size"],
            good_only=True,
        )
        for category in categories
    ]
    test_sets = [
        MVTecADDataset(
            cfg["data"]["root"],
            category,
            split=args.split,
            image_size=cfg["data"]["image_size"],
            good_only=False,
        )
        for category in categories
    ]
    return ConcatDataset(train_sets), ConcatDataset(test_sets), categories


def resolve_gate_checkpoint(cfg, args):
    if args.checkpoint:
        return Path(args.checkpoint), "explicit path"

    if args.gate_scope == "category":
        return Path(cfg["work_dir"]) / args.category / "checkpoints" / "gate_last.pt", "category-specific auto path"

    return Path(cfg["work_dir"]) / "_shared_gate" / "mvtec" / "checkpoints" / "gate_last.pt", "shared auto path"


def make_vis_output_path(vis_dir: Path, image_path: Path, data_root: Path, unified: bool) -> Path:
    subfolder = image_path.parent.name
    if not unified:
        return vis_dir / f"{subfolder}_{image_path.stem}.png"

    try:
        category = image_path.resolve().relative_to(data_root.resolve()).parts[0]
    except (ValueError, IndexError):
        category = image_path.parents[2].name if len(image_path.parents) >= 3 else "unknown"
    return vis_dir / f"{category}_{subfolder}_{image_path.stem}.png"


def build_run_summary(
    args,
    summary_metrics: Dict[str, Any],
    results_dir: Path,
    metrics_path: Path,
    context_path: Path,
    checkpoint_path: Path,
    context_checkpoint_found: bool,
    gate_checkpoint_found: bool,
    num_train_memory_images: int,
    num_test_images: int,
) -> Dict[str, Any]:
    summary = {
        "mode": args.gate_scope,
        "category": args.category,
        "context_scope": args.context_scope,
        "gate_scope": args.gate_scope,
        "split": args.split,
        "num_test_images": num_test_images,
        "num_train_memory_images": num_train_memory_images,
        "result_dir": str(results_dir),
        "metrics_path": str(metrics_path),
        "context_checkpoint_path": str(context_path),
        "gate_checkpoint_path": str(checkpoint_path),
        "context_checkpoint_found": context_checkpoint_found,
        "gate_checkpoint_found": gate_checkpoint_found,
    }

    for key in (
        "image_roc_auc",
        "image_ap",
        "pixel_roc_auc",
        "pixel_ap",
        "pixel_pro",
        "pixel_optimal_threshold",
        "pixel_optimal_fpr",
        "pixel_optimal_fnr",
    ):
        if key in summary_metrics:
            summary[key] = summary_metrics[key]

    return summary


def print_run_summary(summary: Dict[str, Any]) -> None:
    context_ckpt_status = "loaded" if summary["context_checkpoint_found"] else "missing"
    gate_ckpt_status = "loaded" if summary["gate_checkpoint_found"] else "missing"

    print("\n" + "="*50)
    if summary["mode"] == "dataset_shared":
        print("Unified multi-class inference summary")
    else:
        print("Category-specific inference summary")
    print("="*50)
    print(f"Mode: {summary['mode']}")
    print(f"Category: {summary['category']}")
    print(f"Split: {summary['split']}")
    print(f"Context scope: {summary['context_scope']}")
    print(f"Gate scope: {summary['gate_scope']}")
    print(f"Test images: {summary['num_test_images']}")
    print(f"Memory images: {summary['num_train_memory_images']}")
    print(f"Result dir: {summary['result_dir']}")
    print(f"Context ckpt: {summary['context_checkpoint_path']} ({context_ckpt_status})")
    print(f"Gate ckpt: {summary['gate_checkpoint_path']} ({gate_ckpt_status})")
    print("Image-level Metrics:")
    print(f"  - ROC-AUC: {summary['image_roc_auc']:.4f}")
    print(f"  - AP:      {summary['image_ap']:.4f}")
    print("Pixel-level Metrics:")
    print(f"  - ROC-AUC: {summary['pixel_roc_auc']:.4f}")
    print(f"  - AP:      {summary['pixel_ap']:.4f}")
    if "pixel_pro" in summary:
        print(f"  - PRO:     {summary['pixel_pro']:.4f}")

    if "pixel_optimal_threshold" in summary:
        th = summary["pixel_optimal_threshold"]
        fpr = summary.get("pixel_optimal_fpr", float("nan"))
        fnr = summary.get("pixel_optimal_fnr", float("nan"))
        print(f"  - Optimal threshold: {th:.4f}")
        if not np.isnan(fpr):
            print(f"  - Optimal FPR:       {fpr:.4f}")
        if not np.isnan(fnr):
            print(f"  - Optimal FNR:       {fnr:.4f}")

    print("="*50 + "\n")
    print(f"Metrics saved to: {summary['metrics_path']}")


def build_memory(model: RefField, loader: DataLoader, device: torch.device):
    model.eval()
    store = TokenStore()
    with torch.no_grad():
        for batch in tqdm(loader, desc="build memory"):
            image = batch["image"].to(device)
            z, _ = model.extract_tokens(image)
            g = F.normalize(z.mean(dim=1), dim=-1)
            for i in range(image.shape[0]):
                store.add(batch["image_path"][i], g[i].cpu().numpy(), z[i].cpu().numpy())
    xb = np.stack(store.global_descs, axis=0).astype(np.float32)
    index = ImageIndex(dim=xb.shape[1])
    index.build(xb)
    return store, index


def retrieve_bank(model: RefField, image: torch.Tensor, store: TokenStore, index: ImageIndex, top_r: int, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        g = model.global_descriptor(image).detach().cpu().numpy().astype(np.float32)
    _, inds = index.search(g, top_r)
    banks = []
    for b in range(image.shape[0]):
        tokens = store.gather_tokens(inds[b])
        banks.append(torch.from_numpy(tokens).to(device))
    max_m = max(b.shape[0] for b in banks)
    dim = banks[0].shape[1]
    padded = []
    for b in banks:
        if b.shape[0] < max_m:
            pad = torch.zeros((max_m - b.shape[0], dim), device=device, dtype=b.dtype)
            b = torch.cat([b, pad], dim=0)
        padded.append(b)
    return torch.stack(padded, dim=0)


def make_nuisance_views(image: torch.Tensor):
    views = []
    for _ in range(2):
        v = image.clone()
        gamma = torch.empty((v.shape[0], 1, 1, 1), device=v.device).uniform_(0.9, 1.1)
        v = v.sign() * v.abs().pow(gamma)
        views.append(v)
    return views


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    train_ds, test_ds, categories = build_mvtec_infer_datasets(cfg, args)
    shared_gate = args.gate_scope == "dataset_shared"
    if shared_gate:
        results_dir = Path(cfg["work_dir"]) / "_shared_gate" / "mvtec"
    else:
        results_dir = Path(cfg["work_dir"]) / args.category
    vis_dir = ensure_dir(results_dir / "visualizations")
    checkpoint_path, checkpoint_source = resolve_gate_checkpoint(cfg, args)
    num_train_memory_images = len(train_ds)
    num_test_images = len(test_ds)

    print(f"[Inference] gate_scope: {args.gate_scope}")
    print(f"[Inference] context_scope: {args.context_scope}")
    if shared_gate:
        print(f"[Inference] --category={args.category} is ignored for dataset_shared gate/data/result scope.")
        print(f"[Inference] unified MVTec categories ({len(categories)}): {', '.join(categories)}")
    print(f"[Inference] gate checkpoint path ({checkpoint_source}): {checkpoint_path}")
    print(f"[Inference] results_dir: {results_dir}")

    train_dl = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)

    model = RefField(
        encoder_name=cfg["model"]["name"],
        pretrained=cfg["model"]["pretrained"],
        freeze_encoder=cfg["model"]["freeze_encoder"],
        proj_dim=cfg["model"]["proj_dim"],
        intra_top_k=cfg["memory"]["intra_top_k"],
        intra_exclusion_radius=cfg["memory"]["intra_exclusion_radius"],
        external_top_k=cfg["memory"]["patch_top_k"],
        mask_ratio=cfg["train"]["mask_ratio"],
    ).to(device)

    with torch.no_grad():
        batch = next(iter(train_dl))["image"].to(device)
        model.extract_tokens(batch)

    # Load context checkpoint: explicit path takes priority, otherwise use context_scope.
    context_path, context_source = resolve_context_checkpoint(cfg, args)
    context_checkpoint_found = context_path.exists()
    if context_checkpoint_found:
        print(f"[Inference] Loading context checkpoint via {context_source}: {context_path}")
        state = torch.load(context_path, map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
        print(f"[Inference] Context checkpoint loaded successfully.")
    else:
        if context_source == "shared auto path":
            attempted = "shared context"
        elif context_source == "category-specific auto path":
            attempted = "category-specific"
        else:
            attempted = "explicit"
        print(f"[Inference] WARNING: No {attempted} context checkpoint found at {context_path}")
        print(f"[Inference] Will run inference with context weights not explicitly restored from a context checkpoint.")

    # Load gate checkpoint: explicit path takes priority, otherwise use gate_scope.
    gate_checkpoint_found = checkpoint_path.exists()
    if not gate_checkpoint_found:
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            f"Please train the gate first or specify --checkpoint <path>."
        )

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    print(f"[Inference] Gate checkpoint loaded successfully.")

    store, index = build_memory(model, train_dl, device)
    model.eval()

    image_labels = []
    image_scores = []
    pixel_masks = []
    pixel_scores = []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="infer"):
            image = batch["image"].to(device)
            mask = batch["mask"].cpu().numpy()[0]
            label = int(batch["label"].item())
            retrieved = retrieve_bank(model, image, store, index, cfg["memory"]["image_top_r"], device)
            nuis = make_nuisance_views(image)
            out = model(image, retrieved, nuisance_views=nuis, image_topk_ratio=cfg["infer"]["topk_ratio"])

            # Image-level score: aggregate from original patch-level map (decoupled from segmentation smoothing)
            img_score = float(out["score_map"][0].max().item())

            # Pixel-level score: upsample then apply Gaussian smoothing
            pixel_map = upsample_score_map(out["score_map"], size=mask.shape[-2:])
            pixel_map = gaussian_blur(pixel_map, sigma=cfg["infer"]["smoothing_sigma"])
            score_np = pixel_map[0].cpu().numpy().astype(np.float32)

            image_labels.append(label)
            image_scores.append(img_score)
            pixel_masks.append(mask)
            pixel_scores.append(score_np)

            if args.save_vis:
                rgb = denormalize_image(image[0]).permute(1, 2, 0).cpu().numpy()
                img_path = Path(batch["image_path"][0])
                out_path = make_vis_output_path(vis_dir, img_path, Path(cfg["data"]["root"]), shared_gate)
                save_overlay(rgb, score_np, out_path, gt_mask=mask)

    img_metrics = image_level_metrics(image_labels, image_scores)
    px_metrics = pixel_level_metrics(pixel_masks, pixel_scores)

    # Combine all metrics
    all_metrics = {**img_metrics, **px_metrics}

    # Filter to only keep summary metrics (exclude large arrays like fpr, tpr, precision, recall)
    summary_metrics = {
        "image_roc_auc": all_metrics.get("image_roc_auc"),
        "image_ap": all_metrics.get("image_ap"),
        "pixel_roc_auc": all_metrics.get("pixel_roc_auc"),
        "pixel_ap": all_metrics.get("pixel_ap"),
        "pixel_pro": all_metrics.get("pixel_pro"),
    }
    for optional_key in (
        "pixel_optimal_threshold",
        "pixel_optimal_fpr",
        "pixel_optimal_fnr",
    ):
        if optional_key in all_metrics:
            summary_metrics[optional_key] = all_metrics[optional_key]

    # Save metrics to JSON file with serialization helper
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    run_summary = build_run_summary(
        args=args,
        summary_metrics=summary_metrics,
        results_dir=results_dir,
        metrics_path=metrics_path,
        context_path=context_path,
        checkpoint_path=checkpoint_path,
        context_checkpoint_found=context_checkpoint_found,
        gate_checkpoint_found=gate_checkpoint_found,
        num_train_memory_images=num_train_memory_images,
        num_test_images=num_test_images,
    )
    metrics_output = {
        **summary_metrics,
        "summary": run_summary,
    }
    with open(metrics_path, "w") as f:
        json.dump(serialize_for_json(metrics_output), f, indent=4)

    print_run_summary(run_summary)


if __name__ == "__main__":
    main()
