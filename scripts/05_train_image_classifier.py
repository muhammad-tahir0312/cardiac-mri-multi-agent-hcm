"""
scripts/09_train_image_classifier.py

Train a 2D image-level HCM classifier on recursively discovered CMR slices.

Outputs
-------
- results/models/image2d/<run_name>/best.pt
- results/logs/image2d/<run_name>/history.csv
- results/logs/image2d/<run_name>/test_metrics.json
- results/logs/image2d/<run_name>/test_predictions.csv

Metrics focus
-------------
- Primary: Recall/Sensitivity
- Secondary: Specificity, F1, AUC-ROC
- Confusion matrices reported at threshold=0.5 and optimized threshold.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cardiac_image_dataset import (
    CardiacCMRImageDataset,
    build_image_transforms,
    build_weighted_image_sampler,
    compute_class_weights_from_labels,
)
from src.models.image_backbones import ImageClassifier2D
from src.utils.losses import FocalLoss


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve(path_like: Path) -> Path:
    return path_like if path_like.is_absolute() else (PROJECT_ROOT / path_like)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 2D CMR image classifier for HCM detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/image2d.yaml"),
        help="Path to 2D image training config YAML.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. training.batch_size=32",
    )
    return parser.parse_args()


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob_sick: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (y_prob_sick >= float(threshold)).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

    if len(np.unique(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, y_prob_sick))
    else:
        auc = float("nan")

    return {
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_prob_sick: np.ndarray,
    metric: str,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> Tuple[float, float]:
    thresholds = np.linspace(float(threshold_min), float(threshold_max), int(threshold_steps))

    best_threshold = 0.5
    best_score = -np.inf

    for thr in thresholds:
        y_pred = (y_prob_sick >= float(thr)).astype(np.int64)

        if metric == "youden_j":
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            score = tpr + tnr - 1.0
        elif metric == "f1":
            score = f1_score(y_true, y_pred, average="binary", zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(
                f"Unknown threshold optimization metric '{metric}'. "
                "Choose from: youden_j, f1, balanced_accuracy."
            )

        if score > best_score or (
            np.isclose(score, best_score) and abs(float(thr) - 0.5) < abs(best_threshold - 0.5)
        ):
            best_score = float(score)
            best_threshold = float(thr)

    return best_threshold, best_score


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    use_amp: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true_all: List[np.ndarray] = []
    y_prob_all: List[np.ndarray] = []

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(amp_device, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss.item())

            probs = torch.softmax(logits.detach(), dim=1)[:, 1]
            y_true_all.append(labels.detach().cpu().numpy())
            y_prob_all.append(probs.detach().cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([], dtype=np.float32)
    return avg_loss, y_true, y_prob


def build_loss(cfg: DictConfig, train_labels: np.ndarray, device: torch.device) -> nn.Module:
    loss_name = str(cfg.training.loss).lower()
    class_weights_cfg = cfg.training.get("class_weights", None)

    if class_weights_cfg is not None:
        class_weights = torch.tensor(list(class_weights_cfg), dtype=torch.float32, device=device)
    else:
        class_weights = compute_class_weights_from_labels(
            labels=train_labels,
            num_classes=int(cfg.model.num_classes),
        ).to(device)

    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    if loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_name == "focal":
        return FocalLoss(gamma=float(cfg.training.focal_gamma), weight=class_weights)

    raise ValueError("training.loss must be one of: ce, weighted_ce, focal")


def build_loaders(cfg: DictConfig, use_rgb: bool) -> Dict[str, DataLoader]:
    splits_dir = _resolve(Path(cfg.paths.splits_image))
    class_to_idx = dict(cfg.data.class_to_idx)

    if use_rgb:
        mean = list(cfg.data.normalization.mean_rgb)
        std = list(cfg.data.normalization.std_rgb)
    else:
        mean = list(cfg.data.normalization.mean_grayscale)
        std = list(cfg.data.normalization.std_grayscale)

    aug = cfg.augmentation

    train_transform = build_image_transforms(
        image_size=int(cfg.data.image_size),
        train=True,
        replicate_to_rgb=use_rgb,
        mean=mean,
        std=std,
        rotation_degrees=float(aug.rotation_degrees),
        horizontal_flip_p=float(aug.horizontal_flip_p),
        affine_translate=(float(aug.affine_translate[0]), float(aug.affine_translate[1])),
        sharpness_factor=float(aug.sharpness_factor),
        sharpness_p=float(aug.sharpness_p),
        gaussian_noise_mean=float(aug.gaussian_noise_mean),
        gaussian_noise_std=float(aug.gaussian_noise_std),
        gaussian_noise_p=float(aug.gaussian_noise_p),
    )

    eval_transform = build_image_transforms(
        image_size=int(cfg.data.image_size),
        train=False,
        replicate_to_rgb=use_rgb,
        mean=mean,
        std=std,
    )

    train_ds = CardiacCMRImageDataset(
        split_csv=splits_dir / "train.csv",
        class_to_idx=class_to_idx,
        transform=train_transform,
    )
    val_ds = CardiacCMRImageDataset(
        split_csv=splits_dir / "val.csv",
        class_to_idx=class_to_idx,
        transform=eval_transform,
    )
    test_ds = CardiacCMRImageDataset(
        split_csv=splits_dir / "test.csv",
        class_to_idx=class_to_idx,
        transform=eval_transform,
        return_path=True,
    )

    imbalance_strategy = str(cfg.training.imbalance_strategy).lower()
    if imbalance_strategy == "sampler":
        class_weights = cfg.training.get("class_weights", None)
        sampler = build_weighted_image_sampler(train_ds.labels, class_weights=class_weights)
        shuffle = False
    elif imbalance_strategy in {"loss_weights", "none"}:
        sampler = None
        shuffle = True
    else:
        raise ValueError("training.imbalance_strategy must be one of: sampler, loss_weights, none")

    common_kwargs = {
        "batch_size": int(cfg.training.batch_size),
        "num_workers": int(cfg.training.num_workers),
        "pin_memory": bool(cfg.training.pin_memory) and torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_ds,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=True,
        **common_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common_kwargs)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_labels": train_ds.labels,
    }


def main() -> None:
    args = parse_args()

    cfg_path = _resolve(args.config)
    cfg: DictConfig = OmegaConf.load(cfg_path)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    set_seed(int(cfg.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.training.amp) and device.type == "cuda"

    use_rgb = bool(cfg.data.replicate_to_rgb_if_pretrained) and bool(cfg.model.pretrained)
    effective_in_channels = 3 if use_rgb else int(cfg.model.in_channels)

    loaders = build_loaders(cfg, use_rgb=use_rgb)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    train_labels = loaders["train_labels"]

    model = ImageClassifier2D(
        backbone=str(cfg.model.backbone),
        pretrained=bool(cfg.model.pretrained),
        in_channels=effective_in_channels,
        num_classes=int(cfg.model.num_classes),
        dropout=float(cfg.model.dropout),
    ).to(device)

    criterion = build_loss(cfg, train_labels=train_labels, device=device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    scaler = GradScaler("cuda", enabled=use_amp)

    run_name = f"img2d_{cfg.model.backbone}_{time.strftime('%Y%m%d_%H%M%S')}"
    model_dir = _resolve(Path(cfg.paths.models_2d)) / run_name
    log_dir = _resolve(Path(cfg.paths.logs_2d)) / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, log_dir / "config.yaml")

    best_score = -np.inf
    best_epoch = -1
    patience_counter = 0
    patience = int(cfg.training.early_stopping_patience)
    save_best_metric = str(cfg.training.save_best_metric)

    history: List[Dict[str, float]] = []

    logger.info("Training on %s | model=%s | params=%s", device, cfg.model.backbone, f"{model.count_parameters():,}")
    logger.info("Input channels=%d (replicate_to_rgb=%s)", effective_in_channels, use_rgb)

    for epoch in range(1, int(cfg.training.epochs) + 1):
        train_loss, train_y, train_prob = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        train_metrics = compute_binary_metrics(train_y, train_prob, threshold=0.5)

        val_loss, val_y, val_prob = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
        )
        val_metrics = compute_binary_metrics(val_y, val_prob, threshold=0.5)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_recall": train_metrics["recall"],
            "train_specificity": train_metrics["specificity"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["auc"],
            "val_recall": val_metrics["recall"],
            "val_specificity": val_metrics["specificity"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
        }
        history.append(row)

        metric_value = row.get(save_best_metric, float("nan"))
        if np.isnan(metric_value):
            metric_value = -val_loss

        if metric_value > best_score:
            best_score = float(metric_value)
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "effective_in_channels": effective_in_channels,
                    "use_rgb": use_rgb,
                },
                model_dir / "best.pt",
            )
        else:
            patience_counter += 1

        logger.info(
            "Epoch %3d | train_loss=%.4f val_loss=%.4f | val_recall=%.4f val_spec=%.4f val_f1=%.4f val_auc=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_metrics["recall"],
            val_metrics["specificity"],
            val_metrics["f1"],
            val_metrics["auc"],
        )

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    pd.DataFrame(history).to_csv(log_dir / "history.csv", index=False)

    ckpt = torch.load(model_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded best checkpoint from epoch %d", int(ckpt["epoch"]))

    _, val_y, val_prob = run_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=None,
        use_amp=use_amp,
    )

    opt_cfg = cfg.threshold_optimization
    best_threshold, best_threshold_score = optimize_threshold(
        y_true=val_y,
        y_prob_sick=val_prob,
        metric=str(opt_cfg.metric),
        threshold_min=float(opt_cfg.min),
        threshold_max=float(opt_cfg.max),
        threshold_steps=int(opt_cfg.steps),
    )

    model.eval()
    test_loss = 0.0
    test_true: List[np.ndarray] = []
    test_prob: List[np.ndarray] = []
    test_paths: List[str] = []

    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast("cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)[:, 1]
            test_loss += float(loss.item())
            test_true.append(labels.cpu().numpy())
            test_prob.append(probs.cpu().numpy())
            test_paths.extend(list(paths))

    test_loss /= max(len(test_loader), 1)
    y_true_test = np.concatenate(test_true)
    y_prob_test = np.concatenate(test_prob)

    metrics_05 = compute_binary_metrics(y_true_test, y_prob_test, threshold=0.5)
    metrics_opt = compute_binary_metrics(y_true_test, y_prob_test, threshold=best_threshold)

    test_payload = {
        "run_name": run_name,
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "save_best_metric": save_best_metric,
        "val_optimized_threshold": float(best_threshold),
        "val_optimized_threshold_metric": str(opt_cfg.metric),
        "val_optimized_threshold_score": float(best_threshold_score),
        "test_loss": float(test_loss),
        "test_metrics_threshold_0_5": metrics_05,
        "test_metrics_threshold_optimized": metrics_opt,
        "confusion_matrix_threshold_0_5": [
            [metrics_05["tn"], metrics_05["fp"]],
            [metrics_05["fn"], metrics_05["tp"]],
        ],
        "confusion_matrix_threshold_optimized": [
            [metrics_opt["tn"], metrics_opt["fp"]],
            [metrics_opt["fn"], metrics_opt["tp"]],
        ],
    }

    with open(log_dir / "test_metrics.json", "w") as f:
        json.dump(test_payload, f, indent=2)

    test_pred_df = pd.DataFrame(
        {
            "image_path": test_paths,
            "true_label": y_true_test.astype(int),
            "prob_sick": y_prob_test.astype(float),
            "pred_label_thr_0_5": (y_prob_test >= 0.5).astype(int),
            "pred_label_thr_opt": (y_prob_test >= best_threshold).astype(int),
        }
    )
    test_pred_df.to_csv(log_dir / "test_predictions.csv", index=False)

    logger.info("Training complete.")
    logger.info("Best model: %s", model_dir / "best.pt")
    logger.info("Metrics: %s", log_dir / "test_metrics.json")


if __name__ == "__main__":
    main()
