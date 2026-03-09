"""
src/models/slice_aggregation.py

Slice Aggregation Model for 3-D volume classification via 2-D slice features.

Motivation
----------
3-D CNNs have high memory footprint and relatively limited pre-training data.
An alternative is to:
1. Extract per-slice 2-D features with a lightweight pre-trained backbone.
2. Aggregate them across the depth axis with an attention mechanism.
3. Classify the aggregated descriptor.

This yields strong performance while enabling transfer learning from
ImageNet-pre-trained 2D weights via the ``timm`` library.

Architecture
------------
``SliceAggregationModel(backbone_2d, aggregation_mode, num_classes)``

    Input  (B, 1, D, H, W)
      ↓ reshape to (B*D, 3, H, W)   [replicate grayscale to 3 channels for timm]
    2D backbone  →  (B*D, feat_dim)
      ↓ reshape to (B, D, feat_dim)
    Aggregation  →  (B, feat_dim)   [mean | max | attention]
    MLP head     →  (B, num_classes)

Aggregation modes
-----------------
* ``"mean"``      — Simple averaged pooling across depth.
* ``"max"``       — Max-pooled across depth.
* ``"attention"`` — Learned scalar weight per slice via a small MLP,
                    softmax-normalised, then a weighted sum.

Usage
-----
    from src.models.slice_aggregation import SliceAggregationModel
    model = SliceAggregationModel(backbone_2d="resnet18", aggregation="attention")
    out = model(torch.zeros(2, 1, 32, 224, 224))  # (2, 2)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Attention aggregator
# ---------------------------------------------------------------------------


class SliceAttention(nn.Module):
    """Compute a normalised attention score for each depth slice.

    Args:
        feat_dim:    Dimension of per-slice feature vectors.
        hidden_dim:  Hidden dimension of the attention MLP.
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted sum of slice features using learned attention.

        Args:
            x: Slice feature tensor with shape ``(B, D, feat_dim)``.

        Returns:
            Aggregated tensor with shape ``(B, feat_dim)``.
        """
        # x: (B, D, F) → scores: (B, D, 1)
        scores = self.attn(x)
        weights = F.softmax(scores, dim=1)          # (B, D, 1)
        return (weights * x).sum(dim=1)             # (B, F)


# ---------------------------------------------------------------------------
# SliceAggregationModel
# ---------------------------------------------------------------------------


class SliceAggregationModel(nn.Module):
    """2-D backbone + depth aggregation for 3-D cardiac MRI classification.

    Args:
        backbone_2d:  ``timm`` model name for the 2-D feature extractor.
        aggregation:  Aggregation strategy — ``"mean"``, ``"max"``,
                      or ``"attention"``.
        num_classes:  Number of output classes.
        pretrained:   Load ImageNet-pre-trained backbone weights.
        dropout:      Dropout probability before the final FC.
        in_channels:  Input MRI channels (grayscale = 1).

    Raises:
        ImportError: If ``timm`` is not installed.
    """

    def __init__(
        self,
        backbone_2d: str = "resnet18",
        aggregation: Literal["mean", "max", "attention"] = "attention",
        num_classes: int = 2,
        pretrained: bool = False,
        dropout: float = 0.5,
        in_channels: int = 1,
    ) -> None:
        super().__init__()

        if not _TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for SliceAggregationModel.\n"
                "Install with: pip install timm"
            )

        # 2-D backbone — extract features, strip classifier head
        self.backbone: nn.Module = timm.create_model(
            backbone_2d,
            pretrained=pretrained,
            num_classes=0,           # remove classification head
            global_pool="avg",       # global average pooling → (B, feat_dim)
            in_chans=3,              # always 3 channels (grayscale → replicate)
        )
        feat_dim: int = self.backbone.num_features

        # Aggregation module
        self.aggregation_mode = aggregation
        if aggregation == "attention":
            self.aggregator = SliceAttention(feat_dim)
        elif aggregation in ("mean", "max"):
            self.aggregator = None
        else:
            raise ValueError(
                f"Unknown aggregation mode '{aggregation}'. "
                "Choose from: 'mean', 'max', 'attention'."
            )

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )

        self.in_channels = in_channels
        self._feat_dim = feat_dim

    def _extract_slice_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pass all slices through the 2-D backbone in a single batch.

        Args:
            x: Input volume tensor ``(B, 1, D, H, W)``.

        Returns:
            Per-slice feature tensor ``(B, D, feat_dim)``.
        """
        B, C, D, H, W = x.shape

        # Reshape: (B, 1, D, H, W) → (B*D, 1, H, W) → (B*D, 3, H, W)
        slices = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        if C == 1:
            slices = slices.repeat(1, 3, 1, 1)       # grayscale → pseudo-RGB

        feats = self.backbone(slices)                # (B*D, feat_dim)
        return feats.view(B, D, self._feat_dim)      # (B, D, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass.

        Args:
            x: Float tensor with shape ``(B, C, D, H, W)``.

        Returns:
            Logit tensor with shape ``(B, num_classes)``.
        """
        slice_feats = self._extract_slice_features(x)  # (B, D, F)

        if self.aggregation_mode == "attention":
            agg = self.aggregator(slice_feats)         # (B, F)
        elif self.aggregation_mode == "mean":
            agg = slice_feats.mean(dim=1)              # (B, F)
        else:  # "max"
            agg = slice_feats.max(dim=1).values        # (B, F)

        return self.head(agg)                          # (B, num_classes)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-slice attention weights for interpretability.

        Only valid when ``aggregation == "attention"``.

        Args:
            x: Input tensor ``(B, C, D, H, W)``.

        Returns:
            Attention weight tensor ``(B, D)`` summing to 1 per sample.

        Raises:
            RuntimeError: If aggregation mode is not ``"attention"``.
        """
        if self.aggregation_mode != "attention":
            raise RuntimeError("get_attention_weights requires aggregation='attention'.")

        slice_feats = self._extract_slice_features(x)   # (B, D, F)
        scores = self.aggregator.attn(slice_feats)       # (B, D, 1)
        weights = F.softmax(scores, dim=1).squeeze(-1)   # (B, D)
        return weights

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "SliceAggregationModel":
        """Construct from the root OmegaConf config.

        Args:
            cfg: Root config object (reads ``cfg.classification``).

        Returns:
            Instantiated :class:`SliceAggregationModel`.
        """
        c = cfg.classification
        return cls(
            backbone_2d=str(c.backbone_2d),
            aggregation=str(c.aggregation),
            num_classes=int(c.num_classes),
            pretrained=bool(c.pretrained),
            dropout=float(c.dropout),
            in_channels=int(c.in_channels),
        )

    def count_parameters(self) -> int:
        """Return trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
