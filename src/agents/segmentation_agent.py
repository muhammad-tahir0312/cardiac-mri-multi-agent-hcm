"""
src/agents/segmentation_agent.py

Segmentation Agent — wraps the 3-D U-Net to produce per-voxel cardiac
structure masks from a pre-processed volume tensor.

Responsibilities
----------------
* Model construction from config.
* Inference (with optional sliding-window for large volumes).
* Checkpoint loading / saving.
* Producing probability maps and hard argmax segmentation masks.

Usage
-----
    from pathlib import Path
    from omegaconf import OmegaConf
    from src.agents.segmentation_agent import SegmentationAgent

    cfg   = OmegaConf.load("src/config/base.yaml")
    agent = SegmentationAgent(cfg)

    # volume tensor: (1, 1, D, H, W)
    probs, mask = agent.predict(volume_tensor)
    # probs: (1, num_classes, D, H, W)  — soft probabilities
    # mask:  (1, D, H, W)               — hard segmentation mask
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.unet3d import UNet3D
from src.utils.losses import build_segmentation_loss
from src.utils.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


class SegmentationAgent:
    """Manages the 3-D U-Net segmentation model lifecycle.

    Args:
        cfg:    Root OmegaConf config (reads ``cfg.segmentation``).
        device: Torch device string or :class:`~torch.device`.  Auto-detects
                CUDA if ``None``.

    Attributes:
        model  (UNet3D):   The underlying segmentation network.
        device (torch.device): Device where the model resides.
        loss_fn (nn.Module):   Configured segmentation loss.
        metrics (SegmentationMetrics): Running metrics accumulator.
    """

    def __init__(
        self,
        cfg: DictConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Build model
        self.model: UNet3D = UNet3D.from_config(cfg).to(self.device)
        logger.info(
            "SegmentationAgent: UNet3D loaded on %s | params=%s",
            self.device,
            f"{self.model.count_parameters():,}",
        )

        self.loss_fn: nn.Module = build_segmentation_loss(cfg)
        self.metrics = SegmentationMetrics(
            num_classes=int(cfg.segmentation.out_channels),
        )

        # Optionally load checkpoint
        ckpt_path: Optional[str] = cfg.segmentation.get("checkpoint", None)
        if ckpt_path and Path(ckpt_path).exists():
            self.load_checkpoint(Path(ckpt_path))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        volume: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run segmentation inference on a pre-processed volume.

        Args:
            volume: Float32 tensor with shape ``(B, 1, D, H, W)``.

        Returns:
            Tuple of:
            - ``probs`` ``(B, num_classes, D, H, W)`` — per-class softmax
              probabilities.
            - ``mask``  ``(B, D, H, W)``              — argmax hard labels.
        """
        self.model.eval()
        volume = volume.to(self.device)
        logits = self.model(volume)                                   # (B,C,D,H,W)
        probs  = torch.softmax(logits, dim=1)                         # (B,C,D,H,W)
        mask   = logits.argmax(dim=1)                                 # (B,D,H,W)
        return probs.cpu(), mask.cpu()

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Forward-pass loss computation (training use).

        Args:
            logits:  Segmentation logits ``(B, C, D, H, W)``.
            targets: Integer ground-truth masks ``(B, D, H, W)``.

        Returns:
            Scalar loss tensor.
        """
        return self.loss_fn(logits, targets)

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path, extra: Optional[dict] = None) -> None:
        """Save model weights and optional metadata to disk.

        Args:
            path:  Destination ``.pt`` file path.
            extra: Optional dict merged into the checkpoint (e.g. epoch, loss).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {"model_state_dict": self.model.state_dict()}
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)
        logger.info("SegmentationAgent checkpoint saved: %s", path)

    def load_checkpoint(self, path: Path, strict: bool = True) -> None:
        """Load model weights from a checkpoint file.

        Args:
            path:   Path to a ``.pt`` checkpoint.
            strict: Require exact key matching (default ``True``).

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state, strict=strict)
        logger.info("SegmentationAgent checkpoint loaded: %s", path)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def train_mode(self) -> None:
        """Set the model to training mode."""
        self.model.train()

    def eval_mode(self) -> None:
        """Set the model to evaluation mode."""
        self.model.eval()

    def parameters(self):
        """Expose model parameters for optimizer construction."""
        return self.model.parameters()
