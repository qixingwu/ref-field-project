from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .base_dataset import Sample
from .transforms import build_image_transform, build_mask_transform


class MVTecADDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str = "train",
        image_size: int = 448,
        good_only: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.category = category
        self.split = split
        self.good_only = good_only
        self.image_transform = build_image_transform(image_size)
        self.mask_transform = build_mask_transform(image_size)
        self.samples: List[Sample] = self._build_samples()

    def _build_samples(self) -> List[Sample]:
        cat_root = self.root / self.category
        split_root = cat_root / self.split
        samples: List[Sample] = []

        if self.split == "train":
            defect_dirs = [split_root / "good"]
        else:
            defect_dirs = sorted([p for p in split_root.iterdir() if p.is_dir()])

        for defect_dir in defect_dirs:
            defect_type = defect_dir.name
            if self.good_only and defect_type != "good":
                continue
            image_paths = sorted([p for p in defect_dir.glob("*.png")])
            for image_path in image_paths:
                if defect_type == "good":
                    label = 0
                    mask_path: Optional[Path] = None
                else:
                    label = 1
                    gt_root = cat_root / "ground_truth" / defect_type
                    mask_name = image_path.stem + "_mask.png"
                    mask_path = gt_root / mask_name
                samples.append(Sample(
                    image_path=image_path,
                    mask_path=mask_path,
                    label=label,
                    defect_type=defect_type,
                    category=self.category,
                ))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = Image.open(s.image_path).convert("RGB")
        image_t = self.image_transform(image)

        if s.mask_path is None or not s.mask_path.exists():
            mask_t = torch.zeros((1, image_t.shape[1], image_t.shape[2]), dtype=torch.uint8)
        else:
            mask = Image.open(s.mask_path).convert("L")
            mask_t = self.mask_transform(mask)
            mask_t = (mask_t > 0).to(torch.uint8)

        return {
            "image": image_t,
            "mask": mask_t.squeeze(0),
            "label": torch.tensor(s.label, dtype=torch.long),
            "image_path": str(s.image_path),
            "defect_type": s.defect_type,
            "category": s.category,
        }
