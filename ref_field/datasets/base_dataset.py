from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    image_path: Path
    mask_path: Optional[Path]
    label: int
    defect_type: str
    category: str


class BaseADDataset(Dataset):
    def __init__(self):
        super().__init__()
