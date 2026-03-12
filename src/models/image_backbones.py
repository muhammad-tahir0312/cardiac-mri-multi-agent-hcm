"""
src/models/image_backbones.py

2D image backbones for binary HCM classification.

Supported backbones:
- resnet18
- resnet50
- efficientnet_b0
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet18,
    resnet50,
)

BackboneName = Literal["resnet18", "resnet50", "efficientnet_b0"]


def _adapt_conv_weight(weight: torch.Tensor, new_in_channels: int) -> torch.Tensor:
    """Adapt pretrained Conv2d weights to a different input channel count."""
    old_in_channels = weight.shape[1]

    if new_in_channels == old_in_channels:
        return weight

    if new_in_channels == 1:
        return weight.mean(dim=1, keepdim=True)

    repeat_factor = int(math.ceil(new_in_channels / old_in_channels))
    expanded = weight.repeat(1, repeat_factor, 1, 1)[:, :new_in_channels, :, :]
    return expanded * (old_in_channels / float(new_in_channels))


def _replace_first_conv(module: nn.Conv2d, new_in_channels: int) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=(module.bias is not None),
        padding_mode=module.padding_mode,
    )

    with torch.no_grad():
        adapted = _adapt_conv_weight(module.weight.detach(), new_in_channels)
        new_conv.weight.copy_(adapted)
        if module.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(module.bias.detach())

    return new_conv


class ImageClassifier2D(nn.Module):
    """Configurable 2D classifier for CMR image-level binary prediction."""

    def __init__(
        self,
        backbone: BackboneName = "resnet18",
        pretrained: bool = True,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone
        self.model = self._build_model(
            backbone=backbone,
            pretrained=pretrained,
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    def _build_model(
        self,
        backbone: BackboneName,
        pretrained: bool,
        in_channels: int,
        num_classes: int,
        dropout: float,
    ) -> nn.Module:
        if backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet18(weights=weights)
            if in_channels != 3:
                model.conv1 = _replace_first_conv(model.conv1, in_channels)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=float(dropout)),
                nn.Linear(in_features, int(num_classes)),
            )
            return model

        if backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model = resnet50(weights=weights)
            if in_channels != 3:
                model.conv1 = _replace_first_conv(model.conv1, in_channels)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=float(dropout)),
                nn.Linear(in_features, int(num_classes)),
            )
            return model

        if backbone == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = efficientnet_b0(weights=weights)
            first_conv = model.features[0][0]
            if not isinstance(first_conv, nn.Conv2d):
                raise TypeError("Unexpected EfficientNet stem layer type.")
            if in_channels != 3:
                model.features[0][0] = _replace_first_conv(first_conv, in_channels)

            in_features = model.classifier[-1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=float(dropout), inplace=True),
                nn.Linear(in_features, int(num_classes)),
            )
            return model

        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose from: resnet18, resnet50, efficientnet_b0."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "ImageClassifier2D":
        m = cfg.model
        return cls(
            backbone=str(m.backbone),
            pretrained=bool(m.pretrained),
            in_channels=int(m.in_channels),
            num_classes=int(m.num_classes),
            dropout=float(m.dropout),
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
