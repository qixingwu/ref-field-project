from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PatchEncoder
from .projector import Projector
from .intra_expert import IntraExpert
from .external_expert import ExternalExpert
from .context_expert import ContextExpert
from .gate import ReliabilityGate


@dataclass
class RefFieldOutputs:
    score_map: torch.Tensor
    image_score: torch.Tensor
    alpha: torch.Tensor
    e_in: torch.Tensor
    e_out: torch.Tensor
    e_ctx: torch.Tensor


class RefField(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        pretrained: bool,
        freeze_encoder: bool,
        proj_dim: int,
        intra_top_k: int,
        intra_exclusion_radius: int,
        external_top_k: int,
        mask_ratio: float,
    ):
        super().__init__()
        self.encoder = PatchEncoder(encoder_name, pretrained=pretrained, freeze=freeze_encoder)
        # infer feature dim lazily
        self.projector: Optional[Projector] = None
        self.intra_expert = IntraExpert(k=intra_top_k, exclusion_radius=intra_exclusion_radius)
        self.external_expert = ExternalExpert(k=external_top_k)
        self.context_expert = ContextExpert(dim=proj_dim, mask_ratio=mask_ratio)
        self.gate = ReliabilityGate(in_dim=10, hidden_dim=64)
        self.proj_dim = proj_dim

    def _ensure_projector(self, tokens: torch.Tensor):
        if self.projector is None:
            self.projector = Projector(tokens.shape[-1], self.proj_dim).to(tokens.device)

    def extract_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        tokens, (hp, wp) = self.encoder(x)
        self._ensure_projector(tokens)
        assert self.projector is not None
        tokens = self.projector(tokens)
        return tokens, (hp, wp)

    @torch.no_grad()
    def global_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.extract_tokens(x)
        return F.normalize(z.mean(dim=1), dim=-1)

    @torch.no_grad()
    def compute_invariance_reliability(self, views: List[torch.Tensor]) -> torch.Tensor:
        zs = []
        for v in views:
            z, _ = self.extract_tokens(v)
            zs.append(z)
        z_stack = torch.stack(zs, dim=0)  # [T,B,N,D]
        z_mean = z_stack.mean(dim=0)
        var = ((z_stack - z_mean.unsqueeze(0)) ** 2).mean(dim=(0, -1))
        return torch.exp(-var)

    def forward(
        self,
        x: torch.Tensor,
        retrieved_bank: torch.Tensor,
        nuisance_views: Optional[List[torch.Tensor]] = None,
        image_topk_ratio: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        z, (hp, wp) = self.extract_tokens(x)
        e_in, r_in = self.intra_expert(z, hp, wp)
        e_out, r_out = self.external_expert(z, retrieved_bank)
        e_ctx, r_ctx = self.context_expert(z)

        if nuisance_views is not None and len(nuisance_views) > 0:
            r_inv = self.compute_invariance_reliability(nuisance_views)
        else:
            r_inv = torch.ones_like(e_in)

        feat = torch.stack([
            e_in, e_out, e_ctx,
            r_in, r_out, r_ctx,
            r_inv,
            (e_in - e_out).abs(),
            (e_in - e_ctx).abs(),
            (e_out - e_ctx).abs(),
        ], dim=-1)

        alpha_logits, lambda_conf = self.gate(feat)
        alpha = torch.softmax(alpha_logits, dim=-1)
        lambda_conf = F.softplus(lambda_conf).squeeze(-1)

        rin = r_in * r_inv
        rout = r_out * r_inv
        rctx = r_ctx * r_inv

        b_in = alpha[..., 0] * rin * torch.exp(-e_in)
        b_out = alpha[..., 1] * rout * torch.exp(-e_out)
        b_ctx = alpha[..., 2] * rctx * torch.exp(-e_ctx)
        normality = b_in + b_out + b_ctx

        e_stack = torch.stack([e_in, e_out, e_ctx], dim=-1)
        e_std = (e_stack - e_stack.mean(dim=-1, keepdim=True)) / (e_stack.std(dim=-1, keepdim=True) + 1e-6)
        conflict = (
            alpha[..., 0] * alpha[..., 1] * (e_std[..., 0] - e_std[..., 1]).abs() +
            alpha[..., 0] * alpha[..., 2] * (e_std[..., 0] - e_std[..., 2]).abs() +
            alpha[..., 1] * alpha[..., 2] * (e_std[..., 1] - e_std[..., 2]).abs()
        )

        score = -torch.log(normality + 1e-8) + lambda_conf * conflict
        score_map = score.view(x.shape[0], hp, wp)

        flat = score_map.view(score_map.shape[0], -1)
        k = max(1, int(flat.shape[1] * image_topk_ratio))
        image_score = torch.topk(flat, k=k, dim=1).values.mean(dim=1)

        return {
            "score_map": score_map,
            "image_score": image_score,
            "alpha": alpha.view(x.shape[0], hp, wp, 3),
            "e_in": e_in.view(x.shape[0], hp, wp),
            "e_out": e_out.view(x.shape[0], hp, wp),
            "e_ctx": e_ctx.view(x.shape[0], hp, wp),
        }
