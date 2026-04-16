from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntraExpert(nn.Module):
    def __init__(self, k: int = 5, exclusion_radius: int = 1, tau: float = 0.1):
        super().__init__()
        self.k = k
        self.exclusion_radius = exclusion_radius
        self.tau = tau

    def _coords(self, hp: int, wp: int, device: torch.device) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.arange(hp, device=device),
            torch.arange(wp, device=device),
            indexing="ij",
        )
        return torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=1).float()

    def _gather_neighbors(self, z: torch.Tensor, topi: torch.Tensor) -> torch.Tensor:
        # z: [B,N,D], topi: [B,N,K]
        B, N, D = z.shape
        K = topi.shape[-1]
        idx = topi.unsqueeze(-1).expand(B, N, K, D)
        src = z.unsqueeze(1).expand(B, N, N, D)
        return torch.gather(src, 2, idx)

    def forward(self, z: torch.Tensor, hp: int, wp: int):
        B, N, D = z.shape
        sim = z @ z.transpose(1, 2)
        eye = torch.eye(N, device=z.device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(eye, -1e4)

        coords = self._coords(hp, wp, z.device)
        dist = torch.cdist(coords, coords)
        local_mask = (dist <= float(self.exclusion_radius)).unsqueeze(0)
        sim = sim.masked_fill(local_mask, -1e4)

        k = min(self.k, max(1, N - 1))
        topv, topi = torch.topk(sim, k=k, dim=-1)
        neigh = self._gather_neighbors(z, topi)

        attn = F.softmax(topv / self.tau, dim=-1).unsqueeze(-1)
        z_hat = (attn * neigh).sum(dim=2)
        e = ((z - z_hat) ** 2).sum(dim=-1)

        d = ((z.unsqueeze(2) - neigh) ** 2).sum(dim=-1).sqrt()
        r = torch.exp(-d.std(dim=-1))
        return e, r
