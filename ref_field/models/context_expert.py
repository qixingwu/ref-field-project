from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ContextExpert(nn.Module):
    def __init__(self, dim: int = 256, depth: int = 4, heads: int = 8, mask_ratio: float = 0.4):
        super().__init__()
        self.mask_ratio = mask_ratio
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True,
            dim_feedforward=dim * 4,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mu_head = nn.Linear(dim, dim)
        self.logvar_head = nn.Linear(dim, 1)

    def random_mask(self, b: int, n: int, device: torch.device, ratio: float | None = None) -> torch.Tensor:
        ratio = self.mask_ratio if ratio is None else ratio
        mask = torch.zeros((b, n), dtype=torch.bool, device=device)
        m = max(1, int(n * ratio))
        for i in range(b):
            idx = torch.randperm(n, device=device)[:m]
            mask[i, idx] = True
        return mask

    def predict(self, z: torch.Tensor, mask: torch.Tensor):
        z_in = z.clone()
        z_in[mask] = self.mask_token
        h = self.encoder(z_in)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).squeeze(-1).clamp(-5.0, 5.0)
        return mu, logvar

    def training_loss(self, z: torch.Tensor, ratio: float | None = None) -> torch.Tensor:
        mask = self.random_mask(z.shape[0], z.shape[1], z.device, ratio)
        mu, logvar = self.predict(z, mask)
        nll = ((z - mu) ** 2).sum(dim=-1) / torch.exp(logvar) + logvar
        return nll[mask].mean()

    def forward(self, z: torch.Tensor):
        # Inference: mask every token independently is expensive.
        # We approximate by using a random mask once and score all tokens.
        mask = self.random_mask(z.shape[0], z.shape[1], z.device, ratio=self.mask_ratio)
        mu, logvar = self.predict(z, mask)
        e = ((z - mu) ** 2).sum(dim=-1) / torch.exp(logvar) + logvar
        r = torch.exp(-logvar)
        return e, r
