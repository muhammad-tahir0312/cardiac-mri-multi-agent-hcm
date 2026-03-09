"""
src/models/resnet3d.py

3-D ResNet-18 / ResNet-50 for binary HCM classification.

Architecture
------------
Inflated 3-D ResNet following:
    Hara et al., "Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs
    and ImageNet?", CVPR 2018.

Key design choices
------------------
- ``BasicBlock3D``  → ResNet-18  (2 × 3×3×3 conv per block)
- ``Bottleneck3D``  → ResNet-50  (1×1×1 + 3×3×3 + 1×1×1 per block)
- Instance Normalisation for small-batch 3-D training stability.
- Global Average Pooling before the classification head.
- Optional dropout before the final linear layer.

Usage
-----
    from src.models.resnet3d import ResNet3D
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("src/config/base.yaml")
    model = ResNet3D.from_config(cfg)
    logits = model(torch.zeros(2, 1, 32, 224, 224))  # (2, 2)
"""

from __future__ import annotations

from typing import List, Optional, Type, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class BasicBlock3D(nn.Module):
    """Residual block with two 3×3×3 convolutions (ResNet-18/34 style).

    Args:
        in_planes:  Input channels.
        planes:     Intermediate / output channels.
        stride:     Stride for the first convolution (used for down-sampling).
        downsample: Optional projection shortcut.
    """

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(planes, affine=True)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(planes, affine=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.act(out + residual)


class Bottleneck3D(nn.Module):
    """Bottleneck block with 1×1×1 + 3×3×3 + 1×1×1 convolutions (ResNet-50+ style).

    Args:
        in_planes:  Input channels.
        planes:     Bottleneck channels; output is ``planes * expansion``.
        stride:     Stride for the 3×3×3 convolution.
        downsample: Optional projection shortcut.
    """

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        out_planes = planes * self.expansion
        self.conv1 = nn.Conv3d(in_planes, planes, 1, bias=False)
        self.norm1 = nn.InstanceNorm3d(planes, affine=True)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(planes, affine=True)
        self.conv3 = nn.Conv3d(planes, out_planes, 1, bias=False)
        self.norm3 = nn.InstanceNorm3d(out_planes, affine=True)
        self.act   = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.act(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.act(out + residual)


# ---------------------------------------------------------------------------
# ResNet3D
# ---------------------------------------------------------------------------

Block = Union[Type[BasicBlock3D], Type[Bottleneck3D]]

# Standard ResNet layer configurations
_CONFIGS: dict = {
    "resnet3d_18": (BasicBlock3D, [2, 2, 2, 2]),
    "resnet3d_34": (BasicBlock3D, [3, 4, 6, 3]),
    "resnet3d_50": (Bottleneck3D, [3, 4, 6, 3]),
    "resnet3d_101": (Bottleneck3D, [3, 4, 23, 3]),
}


class ResNet3D(nn.Module):
    """3-D ResNet for volumetric classification.

    Args:
        block:       Block class — :class:`BasicBlock3D` or :class:`Bottleneck3D`.
        layers:      Number of blocks at each of the 4 stages.
        in_channels: Input image channels.
        num_classes: Output class count.
        dropout:     Dropout probability before the classification head.

    Example::

        model = ResNet3D.from_name("resnet3d_18", in_channels=1, num_classes=2)
        logits = model(torch.zeros(2, 1, 32, 224, 224))
        # logits.shape → (2, 2)
    """

    def __init__(
        self,
        block: Block,
        layers: List[int],
        in_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_planes = 64

        # Stem: large receptive field, stride-2 to aggressively reduce spatial dim
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False),
            nn.InstanceNorm3d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Head
        self.gap     = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_layer(
        self,
        block: Block,
        planes: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Build one residual stage.

        Args:
            block:      Block class.
            planes:     Channel width for this stage.
            num_blocks: Number of blocks in this stage.
            stride:     Stride for the first block (performs spatial downsampling).

        Returns:
            :class:`~torch.nn.Sequential` stage module.
        """
        downsample = None
        out_planes = planes * block.expansion
        if stride != 1 or self.in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, out_planes, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_planes, affine=True),
            )
        layers = [block(self.in_planes, planes, stride=stride, downsample=downsample)]
        self.in_planes = out_planes
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Kaiming / zero initialisation for Conv and BN-like layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass.

        Args:
            x: Float tensor with shape ``(B, C, D, H, W)``.

        Returns:
            Raw logits with shape ``(B, num_classes)``.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the global average-pooled feature vector (before FC + dropout).

        Useful for embedding-space analysis and interpretability.

        Args:
            x: Input tensor ``(B, C, D, H, W)``.

        Returns:
            Feature tensor ``(B, 512 * expansion)``.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.gap(x).flatten(1)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_name(
        cls,
        name: str,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> "ResNet3D":
        """Construct a ResNet3D by name.

        Args:
            name:        One of ``"resnet3d_18"``, ``"resnet3d_34"``,
                         ``"resnet3d_50"``, ``"resnet3d_101"``.
            in_channels: Input image channels.
            num_classes: Number of output classes.
            dropout:     Dropout probability.

        Returns:
            Instantiated :class:`ResNet3D`.

        Raises:
            ValueError: If ``name`` is not recognised.
        """
        if name not in _CONFIGS:
            raise ValueError(
                f"Unknown model name '{name}'. "
                f"Choose from: {list(_CONFIGS.keys())}"
            )
        block, layers = _CONFIGS[name]
        return cls(block, layers, in_channels=in_channels,
                   num_classes=num_classes, dropout=dropout)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "ResNet3D":
        """Construct from the root OmegaConf config.

        Args:
            cfg: Root config object (reads ``cfg.classification``).

        Returns:
            Instantiated :class:`ResNet3D`.
        """
        c = cfg.classification
        return cls.from_name(
            name=str(c.model),
            in_channels=int(c.in_channels),
            num_classes=int(c.num_classes),
            dropout=float(c.dropout),
        )

    def count_parameters(self) -> int:
        """Return trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
