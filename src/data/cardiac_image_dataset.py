"""
src/data/cardiac_image_dataset.py

2D CMR image dataset utilities for binary HCM classification.

Designed for nested directory layouts such as:

    data/raw/
      Normal/
        Directory_1/
          .../img_001.png
      Sick/
        Directory_17/
          .../img_102.jpg

Key properties
--------------
- Recursive image-path discovery.
- Lazy image loading (no full dataset preloading into RAM).
- Grayscale loading with optional grayscale→RGB channel replication.
- Training/validation transform builders for CMR classification.
- Weighted sampling helpers for class imbalance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T


class GaussianNoise:
    """Add Gaussian noise to a tensor image with probability ``p``.

    Expects a float tensor in range [0, 1] with shape (C, H, W).
    """

    def __init__(self, mean: float = 0.0, std: float = 0.01, p: float = 0.3) -> None:
        self.mean = float(mean)
        self.std = float(std)
        self.p = float(p)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std + self.mean
        return (tensor + noise).clamp(0.0, 1.0)


class RepeatChannels:
    """Repeat single-channel tensors to ``target_channels`` channels."""

    def __init__(self, target_channels: int = 3) -> None:
        self.target_channels = int(target_channels)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() != 3:
            raise ValueError(f"Expected tensor shape (C,H,W), got {tuple(tensor.shape)}")
        if tensor.size(0) == self.target_channels:
            return tensor
        if tensor.size(0) != 1:
            raise ValueError(
                f"RepeatChannels expects 1 or {self.target_channels} channels, "
                f"got {tensor.size(0)}"
            )
        return tensor.repeat(self.target_channels, 1, 1)


class CardiacCMRImageDataset(Dataset):
    """PyTorch Dataset for image-level CMR classification.

    Required CSV columns:
      - ``patient_id``
      - ``class``
      - ``image_path``

    Images are read lazily in ``__getitem__`` to keep memory usage bounded.
    """

    REQUIRED_COLUMNS = {"patient_id", "class", "image_path"}

    def __init__(
        self,
        split_csv: Path,
        class_to_idx: Dict[str, int],
        transform: Optional[T.Compose] = None,
        return_path: bool = False,
    ) -> None:
        if not split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")

        df = pd.read_csv(split_csv)
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Split CSV missing required columns: {missing}")

        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("L")
        tensor = self.transform(image) if self.transform is not None else T.ToTensor()(image)

        label = self.class_to_idx[row["class"]]
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_path:
            return tensor, label_tensor, str(image_path)
        return tensor, label_tensor

    @property
    def labels(self) -> np.ndarray:
        return np.array([self.class_to_idx[c] for c in self.df["class"]], dtype=np.int64)


def _is_valid_image(path: Path, valid_extensions: Sequence[str]) -> bool:
    return path.is_file() and path.suffix.lower() in {ext.lower() for ext in valid_extensions}


def scan_patient_image_table(
    data_root: Path,
    class_to_idx: Dict[str, int],
    valid_extensions: Sequence[str],
) -> pd.DataFrame:
    """Recursively scan class/patient folders and return an image table.

    Patient ID convention: ``<class_name>/<top_level_patient_dir_name>``.
    """
    rows: List[Dict[str, str]] = []

    for class_name in class_to_idx:
        class_dir = data_root / class_name
        if not class_dir.is_dir():
            continue

        patient_dirs = sorted([p for p in class_dir.iterdir() if p.is_dir()], key=lambda p: p.name)

        for patient_dir in patient_dirs:
            patient_id = f"{class_name}/{patient_dir.name}"
            images = sorted(
                [p for p in patient_dir.rglob("*") if _is_valid_image(p, valid_extensions)],
                key=lambda p: str(p),
            )

            for image_path in images:
                rows.append(
                    {
                        "patient_id": patient_id,
                        "class": class_name,
                        "image_path": str(image_path),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["patient_id", "class", "image_path"])

    return pd.DataFrame(rows, columns=["patient_id", "class", "image_path"])


def build_image_transforms(
    image_size: int,
    train: bool,
    replicate_to_rgb: bool,
    mean: Sequence[float],
    std: Sequence[float],
    rotation_degrees: float = 15.0,
    horizontal_flip_p: float = 0.5,
    affine_translate: Tuple[float, float] = (0.1, 0.1),
    sharpness_factor: float = 1.2,
    sharpness_p: float = 0.3,
    gaussian_noise_mean: float = 0.0,
    gaussian_noise_std: float = 0.01,
    gaussian_noise_p: float = 0.3,
) -> T.Compose:
    """Build train/val transforms for grayscale CMR image classification."""
    tfms: List = [T.Resize((int(image_size), int(image_size)))]

    if train:
        tfms.extend(
            [
                T.RandomRotation(degrees=float(rotation_degrees)),
                T.RandomHorizontalFlip(p=float(horizontal_flip_p)),
                T.RandomAffine(degrees=0, translate=affine_translate),
                T.RandomAdjustSharpness(sharpness_factor=float(sharpness_factor), p=float(sharpness_p)),
            ]
        )

    tfms.append(T.ToTensor())

    if train:
        tfms.append(
            GaussianNoise(
                mean=float(gaussian_noise_mean),
                std=float(gaussian_noise_std),
                p=float(gaussian_noise_p),
            )
        )

    if replicate_to_rgb:
        tfms.append(RepeatChannels(target_channels=3))

    tfms.append(T.Normalize(mean=list(mean), std=list(std)))
    return T.Compose(tfms)


def build_weighted_image_sampler(
    labels: np.ndarray,
    class_weights: Optional[Iterable[float]] = None,
) -> WeightedRandomSampler:
    """Create a weighted sampler for imbalanced binary classes.

    If ``class_weights`` is provided, it must align with class indices
    (e.g. [0.6, 1.0] for [Normal, Sick]).
    """
    labels = labels.astype(np.int64)

    if class_weights is not None:
        class_weights_arr = np.array(list(class_weights), dtype=np.float64)
    else:
        counts = np.bincount(labels)
        class_weights_arr = 1.0 / np.maximum(counts, 1)

    sample_weights = torch.tensor(class_weights_arr[labels], dtype=torch.double)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_class_weights_from_labels(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for loss weighting."""
    counts = np.bincount(labels.astype(np.int64), minlength=int(num_classes)).astype(float)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.sum() * float(num_classes)
    return torch.tensor(weights, dtype=torch.float32)
