"""
src/utils/losses.py

Custom loss functions for the HCM Multi-Agent pipeline.

Available losses
----------------
* :class:`FocalLoss`                  — Class-imbalance robust CE.
* :class:`WeightedCrossEntropyLoss`   — CE with pre-computed class weights.
* :class:`DiceLoss`                   — Overlap-based segmentation loss.
* :class:`DiceCELoss`                 — Combined Dice + Cross-Entropy.
* :func:`build_classification_loss`   — Factory keyed from config string.
* :func:`build_segmentation_loss`     — Factory for segmentation.

Usage
-----
    from src.utils.losses import build_classification_loss, build_segmentation_loss
    from omegaconf import OmegaConf

    cfg  = OmegaConf.load("src/config/base.yaml")
    loss = build_classification_loss(cfg, class_weights=weights)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Multi-class Focal Loss (Lin et al., 2017).

    Down-weights well-classified examples so the model focuses on hard cases.

    Args:
        gamma:        Focusing parameter (≥0).  ``gamma=0`` → standard CE.
        weight:       Optional per-class weight tensor (same semantics as
                      ``torch.nn.CrossEntropyLoss``).
        reduction:    ``"mean"`` | ``"sum"`` | ``"none"``.
        ignore_index: Class index to ignore (use ``-100`` to disable).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits:  Raw logits ``(B, C)`` or ``(B, C, D, H, W)`` for segmentation.
            targets: Integer class labels ``(B,)`` or ``(B, D, H, W)``.

        Returns:
            Scalar loss value.
        """
        weight = self.weight if hasattr(self, "weight") else None  # type: ignore[union-attr]
        ce = F.cross_entropy(
            logits,
            targets,
            weight=weight,
            reduction="none",
            ignore_index=self.ignore_index,
        )
        # p_t = exp(-CE)
        p_t = torch.exp(-ce)
        focal = (1.0 - p_t) ** self.gamma * ce

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


# ---------------------------------------------------------------------------
# WeightedCrossEntropyLoss
# ---------------------------------------------------------------------------


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-Entropy with per-class weights and label smoothing.

    Args:
        weight:         Pre-computed class weight tensor.
        label_smoothing: Smoothing factor in ``[0, 1)``.
        ignore_index:    Class index to ignore.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy loss.

        Args:
            logits:  Raw unnormalised scores.
            targets: Ground-truth class indices.

        Returns:
            Scalar loss.
        """
        weight = self.weight if hasattr(self, "weight") else None  # type: ignore[union-attr]
        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )


# ---------------------------------------------------------------------------
# DiceLoss
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    """Soft Dice Loss for multi-class 3-D segmentation.

    Measures voxel-level overlap between predicted and ground-truth masks.
    Robust to class imbalance.

    Args:
        smooth:        Small constant added to numerator/denominator to
                       prevent division by zero.
        include_background: Whether to include the background class in the
                       Dice average.
    """

    def __init__(
        self,
        smooth: float = 1e-5,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute mean soft Dice loss over all classes.

        Args:
            logits:  Raw logits ``(B, C, D, H, W)``.
            targets: Integer class labels ``(B, D, H, W)``.

        Returns:
            Scalar Dice loss (1 − mean Dice score).
        """
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)                         # (B,C,D,H,W)

        # One-hot encode targets → (B, C, D, H, W)
        one_hot = F.one_hot(targets.long(), num_classes)          # (B,D,H,W,C)
        one_hot = one_hot.permute(0, 4, 1, 2, 3).float()         # (B,C,D,H,W)

        start_cls = 0 if self.include_background else 1
        dice_scores = []
        for c in range(start_cls, num_classes):
            p = probs[:, c].contiguous().view(-1)
            g = one_hot[:, c].contiguous().view(-1)
            intersection = (p * g).sum()
            dice = (2.0 * intersection + self.smooth) / (
                p.sum() + g.sum() + self.smooth
            )
            dice_scores.append(dice)

        if not dice_scores:
            return torch.tensor(0.0, requires_grad=True)

        mean_dice = torch.stack(dice_scores).mean()
        return 1.0 - mean_dice


# ---------------------------------------------------------------------------
# DiceCELoss
# ---------------------------------------------------------------------------


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss for segmentation training.

    Args:
        dice_weight: Scalar weight for the Dice component.
        ce_weight:   Scalar weight for the CE component.
        smooth:      Dice smoothing constant.
        class_weight: Optional class weight for the CE term.
        include_background: Whether to include background in Dice.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1e-5,
        class_weight: Optional[torch.Tensor] = None,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        self.dice_loss = DiceLoss(smooth=smooth, include_background=include_background)
        self.ce_loss   = WeightedCrossEntropyLoss(weight=class_weight)
        self.dice_w = dice_weight
        self.ce_w   = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice + CE loss.

        Args:
            logits:  Segmentation logits ``(B, C, D, H, W)``.
            targets: Ground-truth labels ``(B, D, H, W)``.

        Returns:
            Weighted scalar loss.
        """
        return (
            self.dice_w * self.dice_loss(logits, targets)
            + self.ce_w  * self.ce_loss(logits, targets)
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def build_classification_loss(
    cfg: DictConfig,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Build the classification loss function specified in config.

    Args:
        cfg:           Root OmegaConf config (reads ``cfg.training``).
        class_weights: Optional float32 tensor ``(num_classes,)`` for
                       imbalance correction.

    Returns:
        An :class:`~torch.nn.Module` loss instance.

    Raises:
        ValueError: If the loss name is not recognised.
    """
    name: str = str(cfg.training.loss).lower()

    if name == "cross_entropy":
        return nn.CrossEntropyLoss()

    if name == "weighted_cross_entropy":
        return WeightedCrossEntropyLoss(weight=class_weights)

    if name == "focal":
        return FocalLoss(
            gamma=float(cfg.training.focal_gamma),
            weight=class_weights,
        )

    raise ValueError(
        f"Unknown classification loss '{name}'. "
        "Choose from: 'cross_entropy', 'weighted_cross_entropy', 'focal'."
    )


def build_segmentation_loss(
    cfg: DictConfig,
) -> nn.Module:
    """Build the segmentation loss function specified in config.

    Args:
        cfg: Root OmegaConf config (reads ``cfg.segmentation``).

    Returns:
        An :class:`~torch.nn.Module` loss instance.

    Raises:
        ValueError: If the loss name is not recognised.
    """
    name: str = str(cfg.segmentation.loss).lower()
    smooth: float = float(cfg.segmentation.dice_smooth)

    if name == "dice":
        return DiceLoss(smooth=smooth)

    if name == "dice_ce":
        return DiceCELoss(smooth=smooth)

    if name == "focal_dice":
        return DiceCELoss(
            dice_weight=1.0,
            ce_weight=1.0,
            smooth=smooth,
        )

    raise ValueError(
        f"Unknown segmentation loss '{name}'. "
        "Choose from: 'dice', 'dice_ce', 'focal_dice'."
    )
