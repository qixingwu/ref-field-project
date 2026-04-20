from __future__ import annotations

from typing import Tuple

from PIL import Image
import torchvision.transforms as T


def build_image_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def build_mask_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
        T.PILToTensor(),
    ])


def denormalize_image(tensor):
    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)
