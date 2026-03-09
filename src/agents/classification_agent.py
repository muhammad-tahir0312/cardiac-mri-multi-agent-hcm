"""
src/agents/classification_agent.py

Classification Agent — wraps the 3-D ResNet or Slice Aggregation model to
predict HCM (Sick) vs Normal from a pre-processed cardiac MRI volume.

Responsibilities
----------------
* Model construction from config (``resnet3d_18`` / ``resnet3d_50`` /
  ``slice_aggregation``).
* Inference: returns class probabilities and predicted label.
* Gradient-weighted Class Activation Mapping (Grad-CAM) for interpretability.
* Checkpoint loading / saving.

Usage
-----
    from pathlib import Path
    from omegaconf import OmegaConf
    from src.agents.classification_agent import ClassificationAgent

    cfg   = OmegaConf.load("src/config/base.yaml")
    agent = ClassificationAgent(cfg)

    # volume: (1, 1, 32, 224, 224) — output of PreprocessingAgent
    probs, pred_label = agent.predict(volume)
    # probs:      (1, 2) softmax probabilities
    # pred_label: int in {0, 1}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.resnet3d import ResNet3D
from src.models.slice_aggregation import SliceAggregationModel
from src.utils.losses import build_classification_loss
from src.utils.metrics import ClassificationMetrics

logger = logging.getLogger(__name__)

# Model name → builder mapping
_RESNET_NAMES = {"resnet3d_18", "resnet3d_34", "resnet3d_50", "resnet3d_101"}


def build_classifier(cfg: DictConfig) -> nn.Module:
    """Instantiate the classification model specified in config.

    Args:
        cfg: Root OmegaConf config (reads ``cfg.classification``).

    Returns:
        Instantiated :class:`~torch.nn.Module`.

    Raises:
        ValueError: If the model name is not recognised.
    """
    model_name: str = str(cfg.classification.model).lower()

    if model_name in _RESNET_NAMES:
        model = ResNet3D.from_config(cfg)
        logger.info("Built ResNet3D '%s' (%s params)", model_name,
                    f"{model.count_parameters():,}")
        return model

    if model_name == "slice_aggregation":
        model = SliceAggregationModel.from_config(cfg)
        logger.info("Built SliceAggregationModel (%s params)",
                    f"{model.count_parameters():,}")
        return model

    raise ValueError(
        f"Unknown classification model '{model_name}'. "
        "Choose from: resnet3d_18, resnet3d_34, resnet3d_50, "
        "resnet3d_101, slice_aggregation."
    )


class ClassificationAgent:
    """Manages the cardiac MRI classification model lifecycle.

    Args:
        cfg:          Root OmegaConf config.
        device:       Torch device; auto-detects CUDA if ``None``.
        class_weights: Optional tensor ``(num_classes,)`` for weighted loss.

    Attributes:
        model  (nn.Module):           The wrapped classification network.
        device (torch.device):        Computation device.
        loss_fn (nn.Module):          Configured classification loss.
        metrics (ClassificationMetrics): Per-epoch metric accumulator.
        idx_to_class (Dict[int,str]):  Reverse label mapping.
    """

    def __init__(
        self,
        cfg: DictConfig,
        device: Optional[torch.device] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.model: nn.Module = build_classifier(cfg).to(self.device)

        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.loss_fn: nn.Module = build_classification_loss(cfg, class_weights)

        self.metrics = ClassificationMetrics(
            num_classes=int(cfg.classification.num_classes),
        )
        self.idx_to_class: Dict[int, str] = {
            v: k for k, v in dict(cfg.data.class_to_idx).items()
        }

        ckpt_path: Optional[str] = cfg.classification.get("checkpoint", None)
        if ckpt_path and Path(ckpt_path).exists():
            self.load_checkpoint(Path(ckpt_path))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        volume: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Classify a pre-processed volume tensor.

        Args:
            volume: Float32 tensor ``(B, 1, D, H, W)`` or ``(1, D, H, W)``
                    (channel dim will be added automatically).

        Returns:
            Tuple of:
            - ``probs`` ``(B, num_classes)`` — softmax class probabilities.
            - ``pred_label`` ``int``          — argmax prediction for the
                                               first sample in the batch.
        """
        self.model.eval()
        if volume.dim() == 4:
            volume = volume.unsqueeze(0)

        volume     = volume.to(self.device)
        logits     = self.model(volume)                  # (B, C)
        probs      = torch.softmax(logits, dim=1).cpu()  # (B, C)
        pred_label = int(logits[0].argmax().item())
        return probs, pred_label

    @torch.no_grad()
    def predict_proba(self, volume: torch.Tensor) -> np.ndarray:
        """Return class probability array for a single volume.

        Args:
            volume: Float32 tensor ``(1, D, H, W)`` or ``(1, 1, D, H, W)``.

        Returns:
            Float32 numpy array ``(num_classes,)``.
        """
        probs, _ = self.predict(volume)
        return probs[0].numpy()

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classification loss (used during training).

        Args:
            logits:  Logits ``(B, num_classes)``.
            targets: Integer labels ``(B,)``.

        Returns:
            Scalar loss tensor.
        """
        return self.loss_fn(logits, targets.to(self.device))

    # ------------------------------------------------------------------
    # Grad-CAM (ResNet3D only)
    # ------------------------------------------------------------------

    def grad_cam(
        self,
        volume: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Gradient-weighted Class Activation Map for a single volume.

        Hooks into the last residual stage (``layer4``) of a
        :class:`~src.models.resnet3d.ResNet3D`.  Returns ``None`` when
        called with a ``SliceAggregationModel`` (use
        :meth:`~src.models.slice_aggregation.SliceAggregationModel.get_attention_weights`
        instead).

        Args:
            volume:       Float32 tensor ``(1, 1, D, H, W)``.
            target_class: Class index to generate the map for.  Defaults
                          to the model's predicted class.

        Returns:
            3-D numpy array ``(D, H, W)`` with non-negative activation values,
            or ``None`` if the model does not support Grad-CAM.
        """
        if not isinstance(self.model, ResNet3D):
            logger.warning(
                "Grad-CAM is only implemented for ResNet3D. "
                "Use get_attention_weights() for SliceAggregationModel."
            )
            return None

        self.model.eval()
        volume = volume.to(self.device)
        if volume.dim() == 4:
            volume = volume.unsqueeze(0)

        activations: torch.Tensor = torch.empty(0)
        gradients:   torch.Tensor = torch.empty(0)

        def _fwd_hook(_, _inp, output):
            nonlocal activations
            activations = output

        def _bwd_hook(_, _grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0]

        fwd_handle = self.model.layer4.register_forward_hook(_fwd_hook)
        bwd_handle = self.model.layer4.register_full_backward_hook(_bwd_hook)

        logits = self.model(volume)           # (1, C)
        if target_class is None:
            target_class = int(logits[0].argmax().item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        fwd_handle.remove()
        bwd_handle.remove()

        # Global average-pool gradients → (C,)
        weights = gradients.mean(dim=[0, 2, 3, 4])          # (C,)
        cam = (weights[:, None, None, None] * activations[0]).sum(dim=0)
        cam = torch.relu(cam)

        # Normalise to [0, 1]
        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path, extra: Optional[dict] = None) -> None:
        """Save model weights to disk.

        Args:
            path:  Destination ``.pt`` file.
            extra: Optional metadata dict merged into the checkpoint.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {"model_state_dict": self.model.state_dict()}
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)
        logger.info("ClassificationAgent checkpoint saved: %s", path)

    def load_checkpoint(
        self,
        path: Path,
        strict: bool = True,
    ) -> None:
        """Load model weights from a checkpoint file.

        Args:
            path:   Path to a ``.pt`` checkpoint.
            strict: Require exact key matching.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state, strict=strict)
        logger.info("ClassificationAgent checkpoint loaded: %s", path)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def train_mode(self) -> None:
        """Set model to training mode."""
        self.model.train()

    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()

    def parameters(self):
        """Expose model parameters for optimizer construction."""
        return self.model.parameters()

    @property
    def class_names(self) -> Dict[int, str]:
        """Integer → class name mapping from config."""
        return self.idx_to_class
