"""
src/utils/metrics.py

Evaluation metrics for classification (HCM vs Normal) and segmentation
(LV / RV / Myocardium Dice scores).

All metrics operate on CPU tensors or numpy arrays so they can be called
either inside a training loop or during standalone evaluation.

Classification metrics
----------------------
* :class:`ClassificationMetrics` — AUC, F1, sensitivity, specificity, accuracy.

Segmentation metrics
--------------------
* :func:`dice_score`           — Per-class volumetric Dice.
* :func:`hausdorff_distance_95` — 95th-percentile Hausdorff distance (requires scipy).
* :class:`SegmentationMetrics` — Running buffer for multi-batch aggregation.

Usage
-----
    from src.utils.metrics import ClassificationMetrics

    metric = ClassificationMetrics(num_classes=2)
    metric.update(logits, labels)   # call per batch
    results = metric.compute()      # call at epoch end
    metric.reset()
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------------


class ClassificationMetrics:
    """Accumulates batch predictions and computes epoch-level metrics.

    Metrics computed at :meth:`compute`:
    - **Accuracy**          — fraction of correct predictions.
    - **AUROC**             — macro-averaged one-vs-rest ROC AUC.
    - **F1 (macro)**        — harmonic mean of per-class F1 scores.
    - **Sensitivity**       — recall for the positive (Sick) class.
    - **Specificity**       — true-negative rate for the negative (Normal) class.
    - **Precision**         — positive predictive value for the Sick class.

    Args:
        num_classes: Number of output classes.
        pos_class:   Index of the positive (disease) class.  Used for
                     binary sensitivity / specificity.
    """

    def __init__(self, num_classes: int = 2, pos_class: int = 1) -> None:
        self.num_classes = num_classes
        self.pos_class = pos_class
        self._logits:  List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []

    def reset(self) -> None:
        """Clear accumulated predictions for a new epoch."""
        self._logits.clear()
        self._targets.clear()

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Accumulate a batch of predictions.

        Args:
            logits:  Raw model logits ``(B, num_classes)``.  Softmax is
                     applied internally.
            targets: Ground-truth integer labels ``(B,)``.
        """
        self._logits.append(logits.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute and return all metrics over accumulated predictions.

        Returns:
            Dict with keys: ``accuracy``, ``auroc``, ``f1``,
            ``sensitivity``, ``specificity``, ``precision``.

        Raises:
            RuntimeError: If no predictions have been accumulated.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        if not self._logits:
            raise RuntimeError("No predictions accumulated. Call update() first.")

        all_logits  = torch.cat(self._logits,  dim=0)   # (N, C)
        all_targets = torch.cat(self._targets, dim=0)   # (N,)

        probs = F.softmax(all_logits, dim=1).numpy()    # (N, C)
        preds = all_logits.argmax(dim=1).numpy()        # (N,)
        trues = all_targets.numpy()                     # (N,)

        acc = float(accuracy_score(trues, preds))

        # AUROC — binary vs multi-class
        if self.num_classes == 2:
            auroc = float(roc_auc_score(trues, probs[:, self.pos_class]))
        else:
            try:
                auroc = float(
                    roc_auc_score(trues, probs, multi_class="ovr", average="macro")
                )
            except ValueError:
                auroc = float("nan")

        f1 = float(f1_score(trues, preds, average="macro", zero_division=0))

        # Binary-specific: sensitivity = recall of pos class
        sensitivity = float(recall_score(
            trues, preds, pos_label=self.pos_class,
            average="binary", zero_division=0,
        ))
        precision = float(precision_score(
            trues, preds, pos_label=self.pos_class,
            average="binary", zero_division=0,
        ))

        # Specificity = recall of negative class
        neg_class = 1 - self.pos_class
        specificity = float(recall_score(
            trues, preds, pos_label=neg_class,
            average="binary", zero_division=0,
        ))

        return {
            "accuracy":    acc,
            "auroc":       auroc,
            "f1":          f1,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision":   precision,
        }

    def compute_confusion_matrix(self) -> np.ndarray:
        """Return an (num_classes × num_classes) confusion matrix.

        Returns:
            Integer numpy array.
        """
        from sklearn.metrics import confusion_matrix

        all_logits  = torch.cat(self._logits,  dim=0)
        all_targets = torch.cat(self._targets, dim=0)
        preds = all_logits.argmax(dim=1).numpy()
        trues = all_targets.numpy()
        return confusion_matrix(trues, preds, labels=list(range(self.num_classes)))


# ---------------------------------------------------------------------------
# Segmentation: per-volume Dice
# ---------------------------------------------------------------------------


def dice_score(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    num_classes: int,
    smooth: float = 1e-6,
    include_background: bool = False,
) -> Dict[str, float]:
    """Compute per-class Dice scores for a single predicted segmentation.

    Args:
        pred_mask:          Integer mask ``(D, H, W)`` with predicted class IDs.
        true_mask:          Integer mask ``(D, H, W)`` with ground-truth IDs.
        num_classes:        Total number of classes (including background).
        smooth:             Smoothing constant.
        include_background: Whether to include class 0 in the average.

    Returns:
        Dict ``{class_i: dice, ..., "mean": mean_dice}``.
    """
    scores: Dict[str, float] = {}
    start = 0 if include_background else 1

    for c in range(start, num_classes):
        pred_c = (pred_mask == c).astype(float)
        true_c = (true_mask == c).astype(float)
        intersection = (pred_c * true_c).sum()
        denom = pred_c.sum() + true_c.sum()
        scores[f"dice_class_{c}"] = float(
            (2.0 * intersection + smooth) / (denom + smooth)
        )

    class_vals = [v for k, v in scores.items() if k.startswith("dice_class_")]
    scores["mean_dice"] = float(np.mean(class_vals)) if class_vals else 0.0
    return scores


def hausdorff_distance_95(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    class_id: int,
    voxel_spacing: Optional[List[float]] = None,
) -> float:
    """95th-percentile Hausdorff distance for a single class.

    Requires ``scipy``.

    Args:
        pred_mask:     Predicted integer mask ``(D, H, W)``.
        true_mask:     Ground-truth integer mask ``(D, H, W)``.
        class_id:      Target class index.
        voxel_spacing: Physical voxel size ``[dz, dy, dx]`` in mm.
                       Defaults to isotropic 1 mm.

    Returns:
        HD95 value in mm (or voxels if spacing not provided).
        Returns ``float("nan")`` if either mask is empty.
    """
    from scipy.ndimage import distance_transform_edt

    if voxel_spacing is None:
        voxel_spacing = [1.0, 1.0, 1.0]

    p = (pred_mask == class_id)
    t = (true_mask == class_id)

    if p.sum() == 0 or t.sum() == 0:
        return float("nan")

    dt_p = distance_transform_edt(~p, sampling=voxel_spacing)
    dt_t = distance_transform_edt(~t, sampling=voxel_spacing)

    surface_dist_to_t = dt_t[p]
    surface_dist_to_p = dt_p[t]
    all_distances = np.concatenate([surface_dist_to_t, surface_dist_to_p])
    return float(np.percentile(all_distances, 95))


# ---------------------------------------------------------------------------
# Segmentation Metrics accumulator
# ---------------------------------------------------------------------------


class SegmentationMetrics:
    """Accumulates per-volume segmentation metrics across a dataset split.

    Args:
        num_classes:        Total classes (including background).
        include_background: Whether to include class 0 in Dice average.
    """

    def __init__(
        self,
        num_classes: int = 4,
        include_background: bool = False,
    ) -> None:
        self.num_classes = num_classes
        self.include_background = include_background
        self._per_volume: List[Dict[str, float]] = []

    def reset(self) -> None:
        """Clear accumulated metrics."""
        self._per_volume.clear()

    def update(
        self,
        pred_logits: torch.Tensor,
        true_masks: torch.Tensor,
    ) -> None:
        """Add a batch of segmentation predictions.

        Args:
            pred_logits: Segmentation logits ``(B, C, D, H, W)``.
            true_masks:  Integer ground-truth masks ``(B, D, H, W)``.
        """
        preds = pred_logits.argmax(dim=1).cpu().numpy()   # (B, D, H, W)
        trues = true_masks.cpu().numpy()                  # (B, D, H, W)

        for b in range(preds.shape[0]):
            scores = dice_score(
                preds[b], trues[b],
                self.num_classes,
                include_background=self.include_background,
            )
            self._per_volume.append(scores)

    def compute(self) -> Dict[str, float]:
        """Aggregate per-volume metrics across all accumulated volumes.

        Returns:
            Dict of mean metric values across the split.
        """
        if not self._per_volume:
            return {}

        keys = self._per_volume[0].keys()
        return {
            k: float(np.mean([v[k] for v in self._per_volume]))
            for k in keys
        }
