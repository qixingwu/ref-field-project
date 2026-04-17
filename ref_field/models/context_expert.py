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

    def _infer_grid_shape(self, n: int) -> tuple[int, int]:
        """Infer 2D grid shape (H, W) from total token count N."""
        # Try square root first
        h = int(round(n ** 0.5))
        if h > 0 and n % h == 0:
            return (h, n // h)

        # Search for closest factor pair to sqrt(n)
        sqrt_n = int(n ** 0.5)
        for h in range(sqrt_n, 0, -1):
            if n % h == 0:
                return (h, n // h)

        # Fallback: should rarely happen for reasonable patch grids
        return (1, n)

    def forward(self, z: torch.Tensor):
        # Deterministic inference: 2D spatial uniform masking to avoid directional artifacts
        b, n, d = z.shape

        # Infer 2D grid shape from token count
        h, w = self._infer_grid_shape(n)

        # Number of groups for spatial masking
        num_groups = max(2, round(1 / self.mask_ratio))

        # Choose (gh, gw) to get roughly num_groups groups, prefer square-ish
        gh = int(num_groups ** 0.5)
        gw = (num_groups + gh - 1) // gh  # ceil division

        # Create coordinate grids for 2D indexing
        rows = torch.arange(h, device=z.device).view(-1, 1).expand(h, w)
        cols = torch.arange(w, device=z.device).view(1, -1).expand(h, w)

        # Initialize accumulators
        e_accum = torch.zeros((b, n), device=z.device)
        r_accum = torch.zeros((b, n), device=z.device)
        count = torch.zeros((b, n), dtype=torch.float32, device=z.device)

        # 2D checkerboard masking: each group masks tokens forming a spatial pattern
        for i in range(gh):
            for j in range(gw):
                # Create 2D mask: select tokens where (row % gh == i) AND (col % gw == j)
                mask_2d = (rows % gh == i) & (cols % gw == j)
                mask_flat = mask_2d.view(-1).unsqueeze(0).expand(b, n)

                # Skip empty masks (can happen when gh * gw > num_groups)
                if not mask_flat.any():
                    continue

                # Predict for this mask
                mu, logvar = self.predict(z, mask_flat)
                e = ((z - mu) ** 2).sum(dim=-1) / torch.exp(logvar) + logvar
                r = torch.exp(-logvar)

                # Accumulate only for masked positions
                e_accum += e * mask_flat
                r_accum += r * mask_flat
                count += mask_flat

        # Average (each token is masked exactly once in ideal case)
        e = e_accum / count.clamp(min=1.0)
        r = r_accum / count.clamp(min=1.0)
        return e, r
