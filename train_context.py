from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from ref_field.datasets.mvtec import MVTecADDataset
from ref_field.models.ref_field import RefField
from ref_field.utils.config import load_config
from ref_field.utils.misc import ensure_dir, get_device, save_checkpoint, set_seed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--category", type=str, default=None)
    ap.add_argument("--context_scope", type=str, choices=["category", "dataset_shared"], default="category")
    return ap.parse_args()


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


def build_context_dataset(cfg, args):
    if args.context_scope == "category":
        if not args.category:
            raise ValueError("--category is required when --context_scope category")
        categories = [args.category]
        ds = MVTecADDataset(
            root=cfg["data"]["root"],
            category=args.category,
            split="train",
            image_size=cfg["data"]["image_size"],
            good_only=True,
        )
        out_dir = ensure_dir(Path(cfg["work_dir"]) / args.category / "checkpoints")
        return ds, categories, out_dir

    categories = discover_mvtec_categories(cfg["data"]["root"])
    datasets = [
        MVTecADDataset(
            root=cfg["data"]["root"],
            category=category,
            split="train",
            image_size=cfg["data"]["image_size"],
            good_only=True,
        )
        for category in categories
    ]
    ds = ConcatDataset(datasets)
    out_dir = ensure_dir(Path(cfg["work_dir"]) / "_shared_context" / "mvtec" / "checkpoints")
    return ds, categories, out_dir


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    ds, categories, out_dir = build_context_dataset(cfg, args)
    print(f"context_scope: {args.context_scope}")
    if args.context_scope == "dataset_shared":
        print(f"shared MVTec categories ({len(categories)}): {', '.join(categories)}")
    print(f"checkpoint_dir: {out_dir}")

    dl = DataLoader(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

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

    # initialize projector with one forward pass
    with torch.no_grad():
        batch = next(iter(dl))["image"].to(device)
        model.extract_tokens(batch)

    # Always train context_expert and projector (if exists)
    params = list(model.context_expert.parameters())
    if model.projector is not None:
        params += list(model.projector.parameters())
    # Only add encoder params if not frozen
    if not cfg["model"]["freeze_encoder"]:
        for p in model.encoder.parameters():
            if p.requires_grad:
                params.append(p)
    optimizer = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")

    model.train()
    for epoch in range(cfg["train"]["epochs_context"]):
        pbar = tqdm(dl, desc=f"context epoch {epoch+1}/{cfg['train']['epochs_context']}")
        total = 0.0
        count = 0
        for batch in pbar:
            image = batch["image"].to(device)
            with autocast(enabled=scaler.is_enabled()):
                z, _ = model.extract_tokens(image)
                loss = model.context_expert.training_loss(z, ratio=cfg["train"]["mask_ratio"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            count += 1
            pbar.set_postfix(loss=total / max(1, count))

        save_checkpoint(out_dir / "context_last.pt", {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "category": args.category,
            "context_scope": args.context_scope,
            "categories": categories,
        })


if __name__ == "__main__":
    main()
