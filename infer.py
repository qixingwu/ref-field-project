from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--category", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--save_vis", action="store_true")
    return ap.parse_args()


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

    train_ds = MVTecADDataset(cfg["data"]["root"], args.category, split="train", image_size=cfg["data"]["image_size"], good_only=True)
    test_ds = MVTecADDataset(cfg["data"]["root"], args.category, split=args.split, image_size=cfg["data"]["image_size"], good_only=False)

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

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"], strict=False)

    store, index = build_memory(model, train_dl, device)
    model.eval()

    image_labels = []
    image_scores = []
    pixel_masks = []
    pixel_scores = []

    vis_dir = ensure_dir(Path(cfg["work_dir"]) / args.category / "visualizations")

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="infer"):
            image = batch["image"].to(device)
            mask = batch["mask"].cpu().numpy()[0]
            label = int(batch["label"].item())
            retrieved = retrieve_bank(model, image, store, index, cfg["memory"]["image_top_r"], device)
            nuis = make_nuisance_views(image)
            out = model(image, retrieved, nuisance_views=nuis, image_topk_ratio=cfg["infer"]["topk_ratio"])

            score_map = upsample_score_map(out["score_map"], size=mask.shape[-2:])
            score_map = gaussian_blur(score_map, sigma=cfg["infer"]["smoothing_sigma"])
            score_np = score_map[0].cpu().numpy().astype(np.float32)
            img_score = float(score_np.reshape(-1)[np.argsort(-score_np.reshape(-1))[:max(1, int(score_np.size * cfg['infer']['topk_ratio']))]].mean())

            image_labels.append(label)
            image_scores.append(img_score)
            pixel_masks.append(mask)
            pixel_scores.append(score_np)

            if args.save_vis:
                rgb = denormalize_image(image[0]).permute(1, 2, 0).cpu().numpy()
                # Always pass the mask (empty for normal images)
                out_path = vis_dir / (Path(batch["image_path"][0]).stem + ".png")
                save_overlay(rgb, score_np, out_path, gt_mask=mask)

    img_metrics = image_level_metrics(image_labels, image_scores)
    px_metrics = pixel_level_metrics(pixel_masks, pixel_scores)

    # Combine all metrics
    all_metrics = {**img_metrics, **px_metrics}

    # Save metrics to JSON file
    results_dir = Path(cfg["work_dir"]) / args.category
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    # Print metrics in a more readable format
    print("\n" + "="*50)
    print("Anomaly Detection Results")
    print("="*50)
    print(f"Image-level Metrics:")
    print(f"  - ROC-AUC: {img_metrics['image_roc_auc']:.4f}")
    print(f"  - AP:      {img_metrics['image_ap']:.4f}")
    print(f"Pixel-level Metrics:")
    print(f"  - ROC-AUC: {px_metrics['pixel_roc_auc']:.4f}")
    print(f"  - AP:      {px_metrics['pixel_ap']:.4f}")
    if 'pixel_pro' in px_metrics:
        print(f"  - PRO:     {px_metrics['pixel_pro']:.4f}")
    print("="*50 + "\n")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
