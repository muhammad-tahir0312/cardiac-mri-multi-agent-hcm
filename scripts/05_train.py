"""
scripts/05_train.py

Phase 3 — Classification Training Loop.

Trains the :class:`~src.agents.classification_agent.ClassificationAgent`
(ResNet3D or SliceAggregation) on the pre-processed NIfTI dataset.

Features
--------
* AMP (automatic mixed precision) via ``torch.amp``.
* Cosine / step / ReduceLROnPlateau learning-rate schedulers.
* Gradient clipping.
* Linear warm-up epochs.
* WeightedRandomSampler for class imbalance.
* Checkpoint saving on best metric (AUROC, F1, or loss).
* TensorBoard + optional W&B logging.
* Full reproducibility (global seed).
* Early stopping.

Usage
-----
    # Default (ResNet3D-18, Z-score, 32×224×224, 100 epochs)
    python scripts/05_train.py

    # With OmegaConf overrides
    python scripts/05_train.py \\
        classification.model=resnet3d_50 \\
        training.batch_size=4 \\
        training.epochs=50 \\
        training.learning_rate=5e-5 \\
        logging.use_wandb=true
"""

import argparse
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    LambdaLR,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agents.classification_agent import ClassificationAgent
from src.data.cardiac_dataset import build_dataloaders, compute_class_weights
from src.utils.metrics import ClassificationMetrics

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
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# LR warm-up helper
# ---------------------------------------------------------------------------


def create_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    main_scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> LambdaLR:
    """Chain a linear warm-up with a main LR scheduler.

    Args:
        optimizer:       The optimiser instance.
        warmup_epochs:   Number of epochs to ramp LR from 0 to base_lr.
        main_scheduler:  The scheduler to use after warm-up.

    Returns:
        A :class:`~torch.optim.lr_scheduler.LambdaLR` that performs warm-up
        for ``warmup_epochs`` then delegates to ``main_scheduler``.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(warmup_epochs, 1))
        # Delegate: get the scale factor from the main scheduler
        return main_scheduler.get_last_lr()[0] / optimizer.defaults["lr"]

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# One epoch: train / validate
# ---------------------------------------------------------------------------


def run_epoch(
    agent: ClassificationAgent,
    loader: torch.utils.data.DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    cfg: DictConfig,
    phase: str = "train",
) -> Tuple[float, Dict[str, float]]:
    """Run one full epoch (train or eval).

    Args:
        agent:     The :class:`~src.agents.classification_agent.ClassificationAgent`.
        loader:    DataLoader for the current split.
        optimizer: Optimiser (only used in ``"train"`` phase).
        scaler:    AMP GradScaler (only used in ``"train"`` phase).
        cfg:       Root OmegaConf config.
        phase:     ``"train"`` | ``"val"`` | ``"test"``.

    Returns:
        Tuple of:
        - Average loss over the epoch (float).
        - Metrics dict from :class:`~src.utils.metrics.ClassificationMetrics`.
    """
    is_train = phase == "train"
    agent.train_mode() if is_train else agent.eval_mode()
    agent.metrics.reset()

    total_loss = 0.0
    device = agent.device
    use_amp = bool(cfg.training.amp) and device.type == "cuda"
    clip_norm = float(cfg.training.clip_grad_norm)
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for volumes, labels in tqdm(loader, desc=f"  {phase}", leave=False):
            volumes = volumes.to(device, non_blocking=True)   # (B,1,D,H,W)
            labels  = labels.to(device, non_blocking=True)    # (B,)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                with autocast(amp_device, enabled=use_amp):
                    logits = agent.model(volumes)
                    loss   = agent.compute_loss(logits, labels)

                if use_amp:
                    scaler.scale(loss).backward()
                    if clip_norm > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(agent.parameters(), clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_norm > 0:
                        nn.utils.clip_grad_norm_(agent.parameters(), clip_norm)
                    optimizer.step()
            else:
                with autocast(amp_device, enabled=use_amp):
                    logits = agent.model(volumes)
                    loss   = agent.compute_loss(logits, labels)

            total_loss += loss.item()
            agent.metrics.update(logits, labels)

    avg_loss = total_loss / max(len(loader), 1)
    metrics = agent.metrics.compute()
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Build optimizer + scheduler
# ---------------------------------------------------------------------------


def build_optimizer_scheduler(
    agent: ClassificationAgent,
    cfg: DictConfig,
    steps_per_epoch: int,
) -> Tuple[torch.optim.Optimizer, object]:
    """Construct AdamW optimizer and LR scheduler from config.

    Args:
        agent:           The classification agent.
        cfg:             Root OmegaConf config.
        steps_per_epoch: Used by certain schedulers.

    Returns:
        Tuple of ``(optimizer, scheduler)``.

    Raises:
        ValueError: If the scheduler name is not recognised.
    """
    t = cfg.training
    optimizer = AdamW(
        agent.parameters(),
        lr=float(t.learning_rate),
        weight_decay=float(t.weight_decay),
    )

    sched_name = str(t.lr_scheduler).lower()
    epochs = int(t.epochs)

    if sched_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    elif sched_name == "step":
        scheduler = StepLR(optimizer, step_size=int(t.step_size), gamma=float(t.gamma))
    elif sched_name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )
    else:
        raise ValueError(
            f"Unknown scheduler '{sched_name}'. Choose: cosine, step, plateau."
        )

    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log_epoch(
    writer: SummaryWriter,
    phase: str,
    loss: float,
    metrics: Dict[str, float],
    epoch: int,
    lr: float,
    wandb_run=None,
) -> None:
    """Write epoch metrics to TensorBoard (and optionally W&B).

    Args:
        writer:    TensorBoard SummaryWriter.
        phase:     ``"train"`` | ``"val"`` | ``"test"``.
        loss:      Epoch average loss.
        metrics:   Dict of metric name → value.
        epoch:     Current epoch index (0-based).
        lr:        Current learning rate.
        wandb_run: Optional W&B run handle.
    """
    writer.add_scalar(f"{phase}/loss", loss, epoch)
    for k, v in metrics.items():
        writer.add_scalar(f"{phase}/{k}", v, epoch)
    writer.add_scalar("train/lr", lr, epoch)

    if wandb_run is not None:
        wandb_run.log(
            {f"{phase}/{k}": v for k, v in {"loss": loss, **metrics}.items()}
            | {"epoch": epoch, "lr": lr}
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and OmegaConf overrides.

    Returns:
        Parsed namespace with ``config`` Path and ``overrides`` list.
    """
    parser = argparse.ArgumentParser(
        description="Train the HCM Classification Agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, default=Path("src/config/base.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="OmegaConf dot-notation overrides, e.g. training.batch_size=4",
    )
    return parser.parse_args()


def main() -> None:
    """Full training entry point."""
    args = parse_args()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        sys.exit(1)

    cfg: DictConfig = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    # Adjust logging level from config
    logging.getLogger().setLevel(getattr(logging, cfg.logging.level.upper(), logging.INFO))

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    set_seed(int(cfg.seed))

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    imbalance_strategy = str(cfg.training.get("imbalance_strategy", "loss_weights")).lower()
    if imbalance_strategy not in {"loss_weights", "sampler", "none"}:
        raise ValueError(
            f"Unknown training.imbalance_strategy '{imbalance_strategy}'. "
            "Choose from: loss_weights, sampler, none."
        )
    logger.info("Imbalance strategy: %s", imbalance_strategy)

    logger.info("Building DataLoaders …")
    loaders = build_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    # ------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------
    class_weights: Optional[torch.Tensor] = None
    if imbalance_strategy == "loss_weights" and cfg.training.compute_class_weights:
        import pandas as pd
        train_csv = Path(cfg.paths.splits) / "train.csv"
        if train_csv.exists():
            df = pd.read_csv(train_csv).drop_duplicates(subset=["patient_id"])
            labels_arr = np.array([
                dict(cfg.data.class_to_idx)[c] for c in df["class"]
            ])
            class_weights = compute_class_weights(
                labels_arr, int(cfg.classification.num_classes)
            )
            logger.info("Class weights: %s", class_weights.tolist())
    elif imbalance_strategy != "loss_weights":
        logger.info(
            "Class-weight computation skipped because imbalance_strategy='%s'.",
            imbalance_strategy,
        )

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------
    agent = ClassificationAgent(cfg, device=device, class_weights=class_weights)
    logger.info(
        "Model: %s | params: %s",
        cfg.classification.model,
        f"{sum(p.numel() for p in agent.parameters() if p.requires_grad):,}",
    )

    # ------------------------------------------------------------------
    # Optimizer + Scheduler + AMP
    # ------------------------------------------------------------------
    optimizer, scheduler = build_optimizer_scheduler(
        agent, cfg, len(train_loader)
    )
    scaler = GradScaler("cuda", enabled=bool(cfg.training.amp) and device.type == "cuda")

    # ------------------------------------------------------------------
    # Output dirs
    # ------------------------------------------------------------------
    run_ts  = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(cfg.paths.logs) / f"cls_{cfg.classification.model}_{run_ts}"
    ckpt_dir = Path(cfg.paths.models) / f"cls_{cfg.classification.model}_{run_ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    logger.info("TensorBoard logs: %s", log_dir)

    # Optional W&B
    wandb_run = None
    if cfg.logging.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.get("wandb_entity", None),
                config=OmegaConf.to_container(cfg, resolve=True),
                name=f"cls_{cfg.classification.model}_{run_ts}",
            )
        except Exception as e:
            logger.warning("W&B init failed: %s", e)

    # ------------------------------------------------------------------
    # Save config snapshot
    # ------------------------------------------------------------------
    OmegaConf.save(cfg, log_dir / "config.yaml")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    epochs            = int(cfg.training.epochs)
    patience          = int(cfg.training.patience)
    warmup_epochs     = int(cfg.training.warmup_epochs)
    save_metric       = str(cfg.training.save_best_metric)   # e.g. "val_auroc"
    metric_key        = save_metric.replace("val_", "")       # e.g. "auroc"

    best_val: float  = -math.inf
    patience_counter: int = 0

    logger.info("Starting training for %d epochs ...", epochs)

    for epoch in range(epochs):
        # -- Warm-up LR override
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / max(warmup_epochs, 1)
            for pg in optimizer.param_groups:
                pg["lr"] = float(cfg.training.learning_rate) * lr_scale

        # -- Train
        train_loss, train_metrics = run_epoch(
            agent, train_loader, optimizer, scaler, cfg, phase="train"
        )

        # -- Validate
        val_loss, val_metrics = run_epoch(
            agent, val_loader, None, None, cfg, phase="val"
        )

        # -- LR scheduler step
        cur_lr = optimizer.param_groups[0]["lr"]
        if epoch >= warmup_epochs:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics.get(metric_key, -val_loss))
            else:
                scheduler.step()

        # -- Log
        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
            "val_auroc=%.4f  val_f1=%.4f  lr=%.2e",
            epoch + 1, epochs,
            train_loss, val_loss,
            val_metrics.get("auroc", float("nan")),
            val_metrics.get("f1", float("nan")),
            cur_lr,
        )

        log_epoch(writer, "train", train_loss, train_metrics, epoch, cur_lr, wandb_run)
        log_epoch(writer, "val",   val_loss,   val_metrics,   epoch, cur_lr, wandb_run)

        # -- Checkpoint: best model
        val_score = val_metrics.get(metric_key, -val_loss)
        if val_score > best_val:
            best_val = val_score
            patience_counter = 0
            agent.save_checkpoint(
                ckpt_dir / "best.pt",
                extra={
                    "epoch": epoch,
                    "val_loss": val_loss,
                    metric_key: val_score,
                },
            )
            logger.info("  ✓ New best %s=%.4f — checkpoint saved.", save_metric, best_val)
        else:
            patience_counter += 1

        # -- Checkpoint: last model
        if cfg.training.save_last:
            agent.save_checkpoint(
                ckpt_dir / "last.pt",
                extra={"epoch": epoch, "val_loss": val_loss},
            )

        # -- Early stopping
        if patience_counter >= patience:
            logger.info(
                "Early stopping triggered after %d epochs without improvement.",
                patience,
            )
            break

    # ------------------------------------------------------------------
    # Final test evaluation using best checkpoint
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Loading best checkpoint for final test evaluation …")
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        agent.load_checkpoint(best_ckpt)

    test_loader = loaders["test"]
    test_loss, test_metrics = run_epoch(
        agent, test_loader, None, None, cfg, phase="test"
    )

    logger.info("TEST  loss=%.4f", test_loss)
    for k, v in test_metrics.items():
        logger.info("      %-14s = %.4f", k, v)

    log_epoch(writer, "test", test_loss, test_metrics, 0, cur_lr, wandb_run)

    # Save test results
    import json
    test_results = {"loss": test_loss, **test_metrics}
    with open(log_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    logger.info("Test results saved to: %s", log_dir / "test_results.json")

    writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Training complete. Best val_%s=%.4f", metric_key, best_val)
    logger.info("Checkpoints : %s", ckpt_dir)
    logger.info("Logs        : %s", log_dir)


if __name__ == "__main__":
    main()
