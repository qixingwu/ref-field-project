from __future__ import annotations

import random
from typing import Tuple

import torch
import torchvision.transforms.functional as TF


class PseudoAnomalyGenerator:
    def __init__(self, image_size: int, min_frac: float = 0.12, max_frac: float = 0.3):
        self.image_size = image_size
        self.min_frac = min_frac
        self.max_frac = max_frac

    def _random_box(self) -> Tuple[int, int, int, int]:
        size = random.randint(int(self.image_size * self.min_frac), int(self.image_size * self.max_frac))
        x = random.randint(0, self.image_size - size)
        y = random.randint(0, self.image_size - size)
        return x, y, size, size

    def _mask_from_box(self, box, device=None):
        x, y, w, h = box
        mask = torch.zeros((self.image_size, self.image_size), dtype=torch.float32, device=device)
        mask[y:y+h, x:x+w] = 1.0
        return mask

    def local_shuffle(self, image: torch.Tensor):
        box = self._random_box()
        x, y, w, h = box
        patch = image[:, y:y+h, x:x+w].clone()
        flat = patch.view(patch.shape[0], -1)
        idx = torch.randperm(flat.shape[1], device=image.device)
        patch = flat[:, idx].view_as(patch)
        out = image.clone()
        out[:, y:y+h, x:x+w] = patch
        return out, self._mask_from_box(box, image.device)

    def color_shift(self, image: torch.Tensor):
        box = self._random_box()
        x, y, w, h = box
        out = image.clone()
        scale = torch.empty((3, 1, 1), device=image.device).uniform_(0.6, 1.5)
        bias = torch.empty((3, 1, 1), device=image.device).uniform_(-0.15, 0.15)
        out[:, y:y+h, x:x+w] = (out[:, y:y+h, x:x+w] * scale + bias).clamp(-3.0, 3.0)
        return out, self._mask_from_box(box, image.device)

    def cutpaste(self, image: torch.Tensor):
        src = self._random_box()
        dst = self._random_box()
        sx, sy, sw, sh = src
        dx, dy, dw, dh = dst
        size = min(sw, sh, dw, dh)
        patch = image[:, sy:sy+size, sx:sx+size].clone()
        out = image.clone()
        out[:, dy:dy+size, dx:dx+size] = patch
        mask = torch.zeros((self.image_size, self.image_size), dtype=torch.float32, device=image.device)
        mask[dy:dy+size, dx:dx+size] = 1.0
        return out, mask

    def patch_relocate(self, image: torch.Tensor):
        src = self._random_box()
        dst = self._random_box()
        sx, sy, sw, sh = src
        dx, dy, dw, dh = dst
        size = min(sw, sh, dw, dh)
        out = image.clone()
        src_patch = out[:, sy:sy+size, sx:sx+size].clone()
        dst_patch = out[:, dy:dy+size, dx:dx+size].clone()
        out[:, sy:sy+size, sx:sx+size] = dst_patch
        out[:, dy:dy+size, dx:dx+size] = src_patch
        mask = torch.zeros((self.image_size, self.image_size), dtype=torch.float32, device=image.device)
        mask[sy:sy+size, sx:sx+size] = 1.0
        mask[dy:dy+size, dx:dx+size] = 1.0
        return out, mask

    def nuisance_only(self, image: torch.Tensor):
        out = image.clone()
        gamma = random.uniform(0.9, 1.1)
        out = out.sign() * out.abs().pow(gamma)
        if random.random() < 0.5:
            out = torch.roll(out, shifts=random.randint(-4, 4), dims=2)
        return out, torch.zeros((self.image_size, self.image_size), dtype=torch.float32, device=image.device)

    def sample(self, image: torch.Tensor, p_nuisance: float = 0.2):
        if random.random() < p_nuisance:
            return self.nuisance_only(image)
        ops = [self.local_shuffle, self.color_shift, self.cutpaste, self.patch_relocate]
        op = random.choice(ops)
        return op(image)
