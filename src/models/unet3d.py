"""
src/models/unet3d.py

3-D U-Net for cardiac structure segmentation (LV, RV, Myocardium).

Architecture reference
-----------------------
Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation from Sparse
Annotation", MICCAI 2016.

Modifications vs. original
---------------------------
- Configurable feature pyramid via ``features`` list.
- Instance Normalisation (better than BatchNorm for small 3-D batches).
- Dropout inserted at the bottleneck.
- Bilinear / trilinear up-sampling with 1×1×1 projection as an alternative
  to transposed convolutions (reduces checkerboard artefacts).

Usage
-----
    from src.models.unet3d import UNet3D
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("src/config/base.yaml")
    model = UNet3D.from_config(cfg)
    x = torch.zeros(1, 1, 32, 224, 224)
    out = model(x)   # (1, 4, 32, 224, 224)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class DoubleConv(nn.Module):
    """Two consecutive Conv3d → InstanceNorm → LeakyReLU blocks.

    Args:
        in_channels:  Number of input feature channels.
        out_channels: Number of output feature channels.
        mid_channels: Optional intermediate channel count; defaults to
                      ``out_channels``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        mid = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(mid, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(mid, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Feature map after two convolutional blocks.
        """
        return self.block(x)


class Down(nn.Module):
    """Max-pool 2×2×2 followed by DoubleConv.

    Args:
        in_channels:  Input channels.
        out_channels: Output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample and extract features."""
        return self.pool_conv(x)


class Up(nn.Module):
    """Trilinear up-sampling + skip-connection concatenation + DoubleConv.

    Args:
        in_channels:  Channels coming from the deeper path (before concat).
        out_channels: Output channels after DoubleConv.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 1×1×1 to halve channels before concat  (avoids transposed conv artefacts)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=1),
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_deep: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """Upsample ``x_deep``, concatenate with ``x_skip``, then convolve.

        Args:
            x_deep: Tensor from the decoder path.
            x_skip: Skip-connection tensor from the encoder path.

        Returns:
            Merged feature tensor.
        """
        x = self.up(x_deep)
        # Pad if spatial dims differ by 1 (odd input sizes)
        diff = [x_skip.size(i) - x.size(i) for i in range(2, 5)]
        x = nn.functional.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
        x = torch.cat([x_skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# UNet3D
# ---------------------------------------------------------------------------


class UNet3D(nn.Module):
    """3-D U-Net for semantic segmentation of cardiac MRI volumes.

    Args:
        in_channels:  Number of input image channels (1 for grayscale MRI).
        out_channels: Number of output segmentation classes.
        features:     Number of feature maps at each encoder level.
                      The decoder mirrors this list in reverse.
        dropout:      Dropout probability applied at the bottleneck.

    Example::

        model = UNet3D(in_channels=1, out_channels=4,
                       features=[16, 32, 64, 128, 256], dropout=0.1)
        out = model(torch.zeros(1, 1, 32, 224, 224))
        # out.shape → (1, 4, 32, 224, 224)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        features: List[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if features is None:
            features = [16, 32, 64, 128, 256]

        # ---- Encoder ----
        self.enc1 = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.enc2 = DoubleConv(features[1], features[1])
        self.down2 = Down(features[1], features[2])
        self.enc3 = DoubleConv(features[2], features[2])
        self.down3 = Down(features[2], features[3])
        self.enc4 = DoubleConv(features[3], features[3])
        self.down4 = Down(features[3], features[4])

        # ---- Bottleneck ----
        self.bottleneck = nn.Sequential(
            DoubleConv(features[4], features[4]),
            nn.Dropout3d(p=dropout),
        )

        # ---- Decoder ----
        self.up4 = Up(features[4], features[3])
        self.up3 = Up(features[3], features[2])
        self.up2 = Up(features[2], features[1])
        self.up1 = Up(features[1], features[0])

        # ---- Output ----
        self.out_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-uniform initialisation for all Conv3d layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a full U-Net forward pass.

        Args:
            x: Input tensor with shape ``(B, C, D, H, W)``.

        Returns:
            Raw logit tensor with shape ``(B, num_classes, D, H, W)``.
        """
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.down1(s1))
        s3 = self.enc3(self.down2(s2))
        s4 = self.enc4(self.down3(s3))

        # Bottleneck
        b = self.bottleneck(self.down4(s4))

        # Decoder
        d4 = self.up4(b, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)

        return self.out_conv(d1)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "UNet3D":
        """Construct a UNet3D from an OmegaConf config.

        Args:
            cfg: Root config object (reads ``cfg.segmentation``).

        Returns:
            Instantiated :class:`UNet3D`.
        """
        seg = cfg.segmentation
        return cls(
            in_channels=int(seg.in_channels),
            out_channels=int(seg.out_channels),
            features=list(seg.features),
            dropout=float(seg.dropout),
        )

    def count_parameters(self) -> int:
        """Return total trainable parameter count.

        Returns:
            Integer parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
