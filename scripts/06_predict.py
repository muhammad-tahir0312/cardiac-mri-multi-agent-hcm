"""
scripts/06_predict.py

Phase 4 — Inference & Prediction Export.

Loads a trained ClassificationAgent checkpoint and runs inference over
preprocessed NIfTI volumes listed in ``data/processed/manifest.csv``.

Outputs (all written to ``results/predictions/<run_name>/``)
-----------------------------------------------------------
* ``predictions.csv``      — per-patient predictions + probabilities.
* ``metrics.json``         — accuracy, AUROC, F1, sensitivity, specificity.
* ``classification_report.txt`` — full sklearn classification report.
* ``confusion_matrix.png`` — annotated confusion matrix figure.
* ``roc_curve.png``        — ROC curve with AUC annotation.
* ``gradcam/``             — per-patient Grad-CAM mid-slice PNGs (opt-in).

Usage
-----
    # Evaluate test split with the latest checkpoint (auto-detected)
    python scripts/06_predict.py

    # Specify a checkpoint and evaluate all splits
    python scripts/06_predict.py \\
        --checkpoint results/models/cls_resnet3d_18_20260309_204355/best.pt \\
        --splits all

    # Lower decision threshold for Sick class (improves sensitivity)
    python scripts/06_predict.py --threshold 0.3

    # Include Grad-CAM heatmaps
    python scripts/06_predict.py --gradcam
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
)

from src.agents.classification_agent import ClassificationAgent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES   = {0: "Normal", 1: "Sick"}
IDX_TO_CLASS  = {0: "Normal", 1: "Sick"}
CLASS_TO_IDX  = {"Normal": 0, "Sick": 1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(models_dir: Path) -> Optional[Path]:
    """Return the most recently created best.pt checkpoint."""
    candidates = sorted(models_dir.glob("**/best.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def load_volume(nifti_path: Path, device: torch.device) -> torch.Tensor:
    """Load a preprocessed NIfTI volume and return a ``(1,1,D,H,W)`` tensor.

    Args:
        nifti_path: Path to the ``.nii.gz`` file.
        device:     Target PyTorch device.

    Returns:
        Float32 tensor ``(1, 1, D, H, W)``.
    """
    import nibabel as nib

    img  = nib.load(str(nifti_path))
    data = img.get_fdata(dtype=np.float32)          # (H, W, D)  or (D, H, W)

    # nibabel returns (H, W, D); convert to (D, H, W) if needed
    if data.ndim == 3 and data.shape[2] < data.shape[0]:
        data = np.transpose(data, (2, 0, 1))        # (H,W,D) → (D,H,W)

    tensor = torch.from_numpy(data).unsqueeze(0)    # → (1, D, H, W)
    return tensor.unsqueeze(0).to(device)           # → (1, 1, D, H, W)


def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    out_path: Path,
    class_names: List[str],
) -> None:
    """Plot and save a labelled confusion matrix PNG.

    Args:
        y_true:      Ground-truth integer labels.
        y_pred:      Predicted integer labels.
        out_path:    Destination PNG file.
        class_names: List of class name strings.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )

    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", out_path)


def save_roc_curve(
    y_true: List[int],
    y_prob_pos: List[float],
    out_path: Path,
) -> None:
    """Plot and save a ROC curve PNG.

    Args:
        y_true:      Ground-truth integer labels.
        y_prob_pos:  Predicted probability for the positive (Sick) class.
        out_path:    Destination PNG file.
    """
    if len(set(y_true)) < 2:
        logger.warning("Only one class present in y_true — skipping ROC curve.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
    auc_val      = roc_auc_score(y_true, y_prob_pos)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, color="steelblue", label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], lw=1, color="grey", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — HCM Classification", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved: %s", out_path)


def save_gradcam(
    agent: ClassificationAgent,
    volume: torch.Tensor,
    patient_id: str,
    pred_label: int,
    out_dir: Path,
) -> Optional[Path]:
    """Generate and save a mid-slice Grad-CAM overlay.

    Args:
        agent:      Loaded :class:`ClassificationAgent`.
        volume:     Preprocessed volume tensor ``(1,1,D,H,W)``.
        patient_id: Used for the output filename.
        pred_label: Predicted class index (CAM targets this class).
        out_dir:    Directory to write the PNG into.

    Returns:
        Path to the saved PNG, or ``None`` if CAM generation fails.
    """
    cam = agent.grad_cam(volume, target_class=pred_label)
    if cam is None:
        return None

    mid = cam.shape[0] // 2
    vol_slice = volume[0, 0, mid].cpu().numpy()
    cam_slice = cam[mid]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    axes[0].imshow(vol_slice, cmap="gray")
    axes[0].set_title("MRI slice (mid)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(cam_slice, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(vol_slice, cmap="gray")
    axes[2].imshow(cam_slice, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Overlay", fontsize=10)
    axes[2].axis("off")

    safe_name = patient_id.replace("/", "_").replace(" ", "_")
    plt.suptitle(
        f"{patient_id}  →  pred: {IDX_TO_CLASS[pred_label]}",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    out_path = out_dir / f"{safe_name}_gradcam.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Core predict function
# ---------------------------------------------------------------------------

def run_predictions(
    cfg,
    checkpoint: Path,
    manifest_df: pd.DataFrame,
    splits: List[str],
    threshold: float,
    out_dir: Path,
    generate_gradcam: bool,
    device: torch.device,
) -> pd.DataFrame:
    """Run inference on patients from the requested splits.

    Args:
        cfg:             OmegaConf config.
        checkpoint:      Path to the ``.pt`` model checkpoint.
        manifest_df:     Loaded manifest CSV DataFrame.
        splits:          List of splits to include, e.g. ``["test"]`` or
                         ``["train", "val", "test"]``.
        threshold:       Decision threshold for the Sick class.
        out_dir:         Directory to write outputs.
        generate_gradcam: Whether to produce Grad-CAM PNGs.
        device:          PyTorch device.

    Returns:
        DataFrame with columns: patient_id, true_label, true_class,
        prob_Normal, prob_Sick, pred_class, pred_label, correct.
    """
    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    cfg.classification.checkpoint = str(checkpoint)
    agent = ClassificationAgent(cfg, device=device)
    agent.eval_mode()
    logger.info("Loaded checkpoint: %s", checkpoint)
    logger.info("Device: %s | Threshold: %.2f", device, threshold)

    # ------------------------------------------------------------------
    # 2. Filter manifest by requested splits
    # ------------------------------------------------------------------
    if "all" not in splits:
        manifest_df = manifest_df[manifest_df["split"].isin(splits)].copy()

    manifest_df = manifest_df[manifest_df["status"] == "ok"].copy()
    logger.info(
        "Running inference on %d patients (%s split(s))",
        len(manifest_df), ", ".join(splits),
    )

    # ------------------------------------------------------------------
    # 3. Inference loop
    # ------------------------------------------------------------------
    gradcam_dir = out_dir / "gradcam"
    records: List[Dict] = []

    for _, row in manifest_df.iterrows():
        patient_id    = row["patient_id"]
        true_class    = row["class"]
        true_label    = CLASS_TO_IDX[true_class]
        nifti_path    = PROJECT_ROOT / row["processed_path"]

        if not nifti_path.exists():
            logger.warning("NIfTI not found, skipping: %s", nifti_path)
            continue

        t0 = time.monotonic()
        volume = load_volume(nifti_path, device)

        with torch.no_grad():
            probs, _ = agent.predict(volume)           # (1, 2)

        probs_np   = probs[0].numpy()                  # (2,)
        prob_sick  = float(probs_np[CLASS_TO_IDX["Sick"]])
        prob_normal= float(probs_np[CLASS_TO_IDX["Normal"]])

        # Apply custom threshold for Sick class
        pred_label = 1 if prob_sick >= threshold else 0
        pred_class = IDX_TO_CLASS[pred_label]
        elapsed    = round(time.monotonic() - t0, 3)

        records.append({
            "patient_id":  patient_id,
            "split":       row["split"],
            "true_class":  true_class,
            "true_label":  true_label,
            "prob_Normal": round(prob_normal, 6),
            "prob_Sick":   round(prob_sick, 6),
            "pred_class":  pred_class,
            "pred_label":  pred_label,
            "correct":     pred_label == true_label,
            "elapsed_s":   elapsed,
        })

        status_icon = "✓" if pred_label == true_label else "✗"
        logger.info(
            "  %s  %-28s  true=%-6s  pred=%-6s  P(Sick)=%.3f  [%.2fs]",
            status_icon, patient_id, true_class, pred_class, prob_sick, elapsed,
        )

        # Grad-CAM (ResNet3D only, keep gradients enabled)
        if generate_gradcam:
            gradcam_path = save_gradcam(
                agent, volume, patient_id, pred_label, gradcam_dir
            )
            if gradcam_path:
                logger.debug("  Grad-CAM: %s", gradcam_path)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Metrics & reporting
# ---------------------------------------------------------------------------

def compute_and_save_metrics(
    df: pd.DataFrame,
    out_dir: Path,
    class_names: List[str],
) -> Dict:
    """Compute metrics from the predictions DataFrame and save reports.

    Args:
        df:          Predictions DataFrame produced by :func:`run_predictions`.
        out_dir:     Directory to write ``metrics.json``,
                     ``classification_report.txt``, ``confusion_matrix.png``,
                     and ``roc_curve.png``.
        class_names: Ordered list of class name strings.

    Returns:
        Dict of metric name → value.
    """
    y_true     = df["true_label"].tolist()
    y_pred     = df["pred_label"].tolist()
    y_prob_pos = df["prob_Sick"].tolist()

    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    if len(set(y_true)) >= 2:
        auroc = float(roc_auc_score(y_true, y_prob_pos))
    else:
        auroc = float("nan")
        logger.warning("Only one class in ground truth — AUROC set to NaN.")

    # Sensitivity / Specificity from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    else:
        sensitivity = specificity = precision = float("nan")

    metrics = {
        "n_patients":   len(df),
        "accuracy":     round(acc, 4),
        "auroc":        round(auroc, 4) if not np.isnan(auroc) else None,
        "f1_macro":     round(f1, 4),
        "sensitivity":  round(float(sensitivity), 4),
        "specificity":  round(float(specificity), 4),
        "precision":    round(float(precision), 4),
    }

    # Print summary
    print("\n" + "=" * 55)
    print("  PREDICTION METRICS")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
    print("=" * 55 + "\n")

    # classification_report.txt
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    logger.info("Classification report:\n%s", report)
    (out_dir / "classification_report.txt").write_text(report)

    # metrics.json
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved: %s", out_dir / "metrics.json")

    # Plots
    save_confusion_matrix(
        y_true, y_pred,
        out_dir / "confusion_matrix.png",
        class_names,
    )
    save_roc_curve(y_true, y_prob_pos, out_dir / "roc_curve.png")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run HCM classification inference and export predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to a best.pt checkpoint. Auto-detects latest if omitted.",
    )
    p.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "src" / "config" / "base.yaml",
        help="Path to OmegaConf config YAML.",
    )
    p.add_argument(
        "--manifest", type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "manifest.csv",
        help="Path to the preprocessed manifest CSV.",
    )
    p.add_argument(
        "--splits", nargs="+", default=["test"],
        choices=["train", "val", "test", "all"],
        help="Which splits to run inference on. Use 'all' for every split.",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for predicting 'Sick' (0.0–1.0).",
    )
    p.add_argument(
        "--output_dir", type=Path,
        default=PROJECT_ROOT / "results" / "predictions",
        help="Root directory to write prediction outputs.",
    )
    p.add_argument(
        "--gradcam", action="store_true",
        help="Generate and save Grad-CAM mid-slice overlays.",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Torch device string (e.g. 'cpu', 'cuda'). Auto-detected if omitted.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    cfg = OmegaConf.load(args.config)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    checkpoint = args.checkpoint
    if checkpoint is None:
        models_dir = PROJECT_ROOT / "results" / "models"
        checkpoint = find_latest_checkpoint(models_dir)
        if checkpoint is None:
            logger.error(
                "No best.pt checkpoint found under %s. "
                "Train the model first with scripts/05_train.py",
                models_dir,
            )
            sys.exit(1)
    logger.info("Using checkpoint: %s", checkpoint)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest)
        sys.exit(1)
    manifest_df = pd.read_csv(args.manifest)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    # Use the checkpoint run name so predictions are paired to the model
    run_name = checkpoint.parent.name
    out_dir  = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving outputs to: %s", out_dir)

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    splits = ["train", "val", "test"] if "all" in args.splits else args.splits

    predictions_df = run_predictions(
        cfg            = cfg,
        checkpoint     = checkpoint,
        manifest_df    = manifest_df,
        splits         = splits,
        threshold      = args.threshold,
        out_dir        = out_dir,
        generate_gradcam = args.gradcam,
        device         = device,
    )

    if predictions_df.empty:
        logger.error("No predictions generated — check manifest and split names.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Save predictions CSV
    # ------------------------------------------------------------------
    csv_path = out_dir / "predictions.csv"
    predictions_df.to_csv(csv_path, index=False)
    logger.info("Predictions saved: %s  (%d rows)", csv_path, len(predictions_df))

    # ------------------------------------------------------------------
    # Metrics & plots
    # ------------------------------------------------------------------
    compute_and_save_metrics(
        df          = predictions_df,
        out_dir     = out_dir,
        class_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
    )

    # ------------------------------------------------------------------
    # Also copy confusion matrix and ROC to results/figures
    # ------------------------------------------------------------------
    import shutil
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for plot_name in ("confusion_matrix.png", "roc_curve.png"):
        src = out_dir / plot_name
        if src.exists():
            shutil.copy2(src, figures_dir / f"{run_name}_{plot_name}")

    logger.info("Done. All outputs written to: %s", out_dir)


if __name__ == "__main__":
    main()
