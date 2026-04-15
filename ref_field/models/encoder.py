from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import timm


class PatchEncoder(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # Most ViT models in timm support forward_features.
        feats = self.backbone.forward_features(x)

        if isinstance(feats, (tuple, list)):
            feats = feats[-1]

        # ViT commonly returns [B, N+1, C] with cls token.
        if feats.dim() == 3:
            if hasattr(self.backbone, "num_prefix_tokens"):
                prefix = int(getattr(self.backbone, "num_prefix_tokens"))
            else:
                prefix = 1
            feats = feats[:, prefix:, :]
            B, N, C = feats.shape
            hp = wp = int(N ** 0.5)
            return feats, (hp, wp)

        # CNN-style fallback: [B, C, H, W]
        if feats.dim() == 4:
            B, C, H, W = feats.shape
            feats = feats.flatten(2).transpose(1, 2)
            return feats, (H, W)

        raise RuntimeError(f"Unsupported feature shape: {tuple(feats.shape)}")
