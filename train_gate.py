from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ref_field.datasets.mvtec import MVTecADDataset
from ref_field.datasets.pseudo_anomaly import PseudoAnomalyGenerator
from ref_field.memory.faiss_index import ImageIndex
from ref_field.memory.token_store import TokenStore
from ref_field.models.ref_field import RefField
from ref_field.utils.config import load_config
from ref_field.utils.misc import ensure_dir, get_device, save_checkpoint, set_seed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--category", type=str, required=True)
    ap.add_argument("--resume_context", type=str, default="")
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
                store.add(
                    image_path=batch["image_path"][i],
                    global_desc=g[i].detach().cpu().numpy(),
                    tokens=z[i].detach().cpu().numpy(),
                )
    xb = np.stack(store.global_descs, axis=0).astype(np.float32)
    index = ImageIndex(dim=xb.shape[1])
    index.build(xb)
    return store, index


def make_nuisance_views(image: torch.Tensor, num_views: int) -> List[torch.Tensor]:
    views = []
    for _ in range(num_views):
        v = image.clone()
        gamma = torch.empty((v.shape[0], 1, 1, 1), device=v.device).uniform_(0.9, 1.1)
        v = v.sign() * v.abs().pow(gamma)
        if torch.rand(1).item() < 0.5:
            shift = int(torch.randint(-4, 5, (1,), device=v.device).item())
            v = torch.roll(v, shifts=shift, dims=3)
        views.append(v)
    return views


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


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    train_ds = MVTecADDataset(
        root=cfg["data"]["root"],
        category=args.category,
        split="train",
        image_size=cfg["data"]["image_size"],
        good_only=True,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    mem_dl = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
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

    with torch.no_grad():
        batch = next(iter(train_dl))["image"].to(device)
        model.extract_tokens(batch)

    if args.resume_context:
        state = torch.load(args.resume_context, map_location="cpu")
        model.load_state_dict(state["model"], strict=False)

    store, index = build_memory(model, mem_dl, device)

    pseudo_gen = PseudoAnomalyGenerator(cfg["data"]["image_size"])
    params = list(model.gate.parameters())
    scaler = GradScaler(enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")
    optimizer = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    out_dir = ensure_dir(Path(cfg["work_dir"]) / args.category / "checkpoints")

    model.train()
    for epoch in range(cfg["train"]["epochs_gate"]):
        pbar = tqdm(train_dl, desc=f"gate epoch {epoch+1}/{cfg['train']['epochs_gate']}")
        total = 0.0
        count = 0
        for batch in pbar:
            image = batch["image"].to(device)

            x_mix = []
            y_mask = []
            for i in range(image.shape[0]):
                x_corrupt, m = pseudo_gen.sample(image[i])
                x_mix.append(x_corrupt)
                y_mask.append(m)
            x_mix = torch.stack(x_mix, dim=0)
            y_mask = torch.stack(y_mask, dim=0)

            retrieved = retrieve_bank(model, x_mix, store, index, cfg["memory"]["image_top_r"], device)
            nuis = make_nuisance_views(x_mix, cfg["train"]["nuisance_views"])

            with autocast(enabled=scaler.is_enabled()):
                out = model(x_mix, retrieved, nuisance_views=nuis, image_topk_ratio=cfg["infer"]["topk_ratio"])
                score_map = out["score_map"]
                score_up = F.interpolate(score_map.unsqueeze(1), size=y_mask.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
                loss_bce = F.binary_cross_entropy_with_logits(score_up, y_mask)

                pos = score_up[y_mask > 0.5]
                neg = score_up[y_mask <= 0.5]
                if pos.numel() > 0:
                    loss_rank = F.relu(1.0 - pos.mean() + neg.mean())
                else:
                    loss_rank = torch.zeros((), device=device)
                loss = loss_bce + 0.5 * loss_rank

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total += float(loss.item())
            count += 1
            pbar.set_postfix(loss=total / max(1, count))

        save_checkpoint(out_dir / "gate_last.pt", {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "category": args.category,
        })


if __name__ == "__main__":
    main()
