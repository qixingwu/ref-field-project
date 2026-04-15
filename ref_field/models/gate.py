from __future__ import annotations

import torch
import torch.nn as nn


class ReliabilityGate(nn.Module):
    def __init__(self, in_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.alpha_head = nn.Linear(hidden_dim, 3)
        self.lambda_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.alpha_head(h), self.lambda_head(h)
