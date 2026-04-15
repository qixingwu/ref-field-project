from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalExpert(nn.Module):
    def __init__(self, k: int = 5, tau: float = 0.1):
        super().__init__()
        self.k = k
        self.tau = tau

    def _gather_neighbors(self, bank: torch.Tensor, topi: torch.Tensor) -> torch.Tensor:
        # bank: [B,M,D], topi: [B,N,K]
        B, M, D = bank.shape
        N, K = topi.shape[1], topi.shape[2]
        idx = topi.unsqueeze(-1).expand(B, N, K, D)
        src = bank.unsqueeze(1).expand(B, N, M, D)
        return torch.gather(src, 2, idx)

    def forward(self, z: torch.Tensor, bank: torch.Tensor):
        sim = z @ bank.transpose(1, 2)
        k = min(self.k, bank.shape[1])
        topv, topi = torch.topk(sim, k=k, dim=-1)
        neigh = self._gather_neighbors(bank, topi)
        attn = F.softmax(topv / self.tau, dim=-1).unsqueeze(-1)
        z_hat = (attn * neigh).sum(dim=2)
        e = ((z - z_hat) ** 2).sum(dim=-1)

        d1 = (1.0 - topv[..., 0]).clamp(min=0.0)
        if k > 1:
            d2 = (1.0 - topv[..., 1]).clamp(min=0.0)
        else:
            d2 = d1 + 1e-3
        r = torch.exp(-d1) * torch.sigmoid((d2 - d1) * 10.0)
        return e, r
