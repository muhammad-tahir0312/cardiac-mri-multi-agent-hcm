"""
scripts/07_kfold_evaluate.py

Phase 5 — Stratified K-Fold evaluation utilities.

This script creates patient-level stratified k-fold split CSVs and can
optionally run fold-wise training using ``scripts/05_train.py``.

Generated fold structure
------------------------
data/splits_kfold/
    fold_01/
        train.csv
        val.csv
        test.csv
    fold_02/
        ...
    folds_summary.json

Optional training outputs (when --run_training is enabled)
----------------------------------------------------------
results/models/kfold/fold_XX/
results/logs/kfold/fold_XX/

Usage
-----
    # Generate 5-fold patient splits only
    python scripts/07_kfold_evaluate.py

    # Generate splits and run training for each fold
    python scripts/07_kfold_evaluate.py --run_training \
        --epochs 30 --patience 10 --num_workers 0

    # Limit to first 2 folds for a quick smoke run
    python scripts/07_kfold_evaluate.py --run_training --max_folds 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


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
# Helpers
# ---------------------------------------------------------------------------


def _resolve_path(project_root: Path, path_like: Path) -> Path:
    """Resolve a path relative to the project root when not absolute."""
    return path_like if path_like.is_absolute() else (project_root / path_like)


def load_patient_table(manifest_path: Path, base_splits_dir: Path) -> pd.DataFrame:
    """Build a single patient-level table used for k-fold splitting.

    Args:
        manifest_path: Path to ``data/processed/manifest.csv``.
        base_splits_dir: Path to existing ``data/splits`` directory.

    Returns:
        DataFrame with one row per patient and columns:
        ``patient_id``, ``class``, ``series_path``, ``image_count``,
        ``processed_path``.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    for split_name in ("train", "val", "test"):
        split_csv = base_splits_dir / f"{split_name}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Base split CSV not found: {split_csv}")

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["status"] == "ok"].copy()
    if manifest.empty:
        raise ValueError("No status=ok rows found in manifest.")

    split_frames = [
        pd.read_csv(base_splits_dir / "train.csv"),
        pd.read_csv(base_splits_dir / "val.csv"),
        pd.read_csv(base_splits_dir / "test.csv"),
    ]
    split_df = pd.concat(split_frames, ignore_index=True)

    # Keep one representative series row per patient for compatibility with
    # existing dataset classes that require series_path in split CSVs.
    series_meta = (
        split_df[["patient_id", "series_path", "image_count"]]
        .drop_duplicates(subset=["patient_id"])
        .reset_index(drop=True)
    )

    patients = (
        manifest[["patient_id", "class", "processed_path"]]
        .drop_duplicates(subset=["patient_id"])
        .merge(series_meta, on="patient_id", how="left", validate="one_to_one")
    )

    missing_series = int(patients["series_path"].isna().sum())
    if missing_series > 0:
        missing_patients = patients.loc[patients["series_path"].isna(), "patient_id"].tolist()
        raise ValueError(
            f"{missing_series} patient(s) missing series_path mapping from base splits: "
            f"{missing_patients}"
        )

    patients = patients[["patient_id", "class", "series_path", "image_count", "processed_path"]]
    return patients.reset_index(drop=True)


def create_fold_splits(
    patients_df: pd.DataFrame,
    n_splits: int,
    seed: int,
    val_ratio: float,
    output_root: Path,
) -> List[Dict]:
    """Create stratified fold train/val/test CSV files.

    Args:
        patients_df: One-row-per-patient DataFrame.
        n_splits: Number of stratified folds.
        seed: Random seed.
        val_ratio: Validation ratio applied to each fold's train+val subset.
        output_root: Root output directory for generated fold CSVs.

    Returns:
        List of fold metadata dicts.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    y = patients_df["class"].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_records: List[Dict] = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(patients_df, y), start=1):
        fold_name = f"fold_{fold_idx:02d}"
        fold_dir = output_root / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        trainval_df = patients_df.iloc[trainval_idx].copy().reset_index(drop=True)
        test_df = patients_df.iloc[test_idx].copy().reset_index(drop=True)

        stratify_labels = trainval_df["class"]
        if trainval_df["class"].value_counts().min() < 2:
            # Fallback for very small edge cases.
            stratify_labels = None

        train_df, val_df = train_test_split(
            trainval_df,
            test_size=float(val_ratio),
            stratify=stratify_labels,
            random_state=seed + fold_idx,
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
            out_csv = fold_dir / f"{split_name}.csv"
            split_df.to_csv(out_csv, index=False)

        fold_info = {
            "fold": fold_name,
            "path": str(fold_dir),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "class_counts_train": train_df["class"].value_counts().to_dict(),
            "class_counts_val": val_df["class"].value_counts().to_dict(),
            "class_counts_test": test_df["class"].value_counts().to_dict(),
        }
        fold_records.append(fold_info)

    (output_root / "folds_summary.json").write_text(json.dumps(fold_records, indent=2))
    return fold_records


def run_fold_training(
    project_root: Path,
    config_path: Path,
    fold_info: Dict,
    args: argparse.Namespace,
) -> Dict:
    """Run training for one fold and collect its test metrics.

    Args:
        project_root: Project root path.
        config_path: Path to base YAML config.
        fold_info: Fold metadata dict from :func:`create_fold_splits`.
        args: Parsed CLI args.

    Returns:
        Dict with fold name and extracted test metrics.
    """
    fold_name = str(fold_info["fold"])
    fold_splits_dir = Path(fold_info["path"])

    models_root = project_root / "results" / "models" / "kfold" / fold_name
    logs_root = project_root / "results" / "logs" / "kfold" / fold_name
    models_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    python_exe = str(args.python_executable)
    cmd = [
        python_exe,
        "scripts/05_train.py",
        "--config",
        str(config_path),
        f"paths.splits={fold_splits_dir.as_posix()}",
        f"paths.models={models_root.as_posix()}",
        f"paths.logs={logs_root.as_posix()}",
        f"training.epochs={int(args.epochs)}",
        f"training.patience={int(args.patience)}",
        f"dataset.num_workers={int(args.num_workers)}",
    ]

    if args.overrides:
        cmd.extend(list(args.overrides))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    logger.info("[%s] Running training...", fold_name)
    subprocess.run(cmd, cwd=str(project_root), env=env, check=True)

    run_dirs = sorted(
        [d for d in logs_root.glob("cls_*") if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if not run_dirs:
        raise FileNotFoundError(f"[{fold_name}] No training run directory found in {logs_root}")

    latest_run = run_dirs[-1]
    test_results_path = latest_run / "test_results.json"
    if not test_results_path.exists():
        raise FileNotFoundError(f"[{fold_name}] Missing test results: {test_results_path}")

    metrics = json.loads(test_results_path.read_text())
    logger.info("[%s] Test metrics: %s", fold_name, metrics)

    return {
        "fold": fold_name,
        "run_dir": str(latest_run),
        "test_results": metrics,
    }


def aggregate_fold_metrics(fold_results: List[Dict]) -> Dict:
    """Compute mean/std summary across fold test metrics."""
    if not fold_results:
        return {"n_folds": 0, "mean": {}, "std": {}}

    metric_keys = sorted(
        {
            k
            for fr in fold_results
            for k, v in fr["test_results"].items()
            if isinstance(v, (int, float))
        }
    )

    mean_metrics: Dict[str, float] = {}
    std_metrics: Dict[str, float] = {}

    for key in metric_keys:
        vals = [float(fr["test_results"][key]) for fr in fold_results if key in fr["test_results"]]
        mean_metrics[key] = float(np.mean(vals))
        std_metrics[key] = float(np.std(vals))

    return {
        "n_folds": len(fold_results),
        "mean": mean_metrics,
        "std": std_metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stratified k-fold splits and optionally run fold-wise training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.csv"),
        help="Path to processed manifest CSV.",
    )
    parser.add_argument(
        "--base_splits",
        type=Path,
        default=Path("data/splits"),
        help="Path to existing train/val/test split directory for series metadata.",
    )
    parser.add_argument(
        "--output_splits",
        type=Path,
        default=Path("data/splits_kfold"),
        help="Output root for generated fold split CSV files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/base.yaml"),
        help="Config passed to scripts/05_train.py when --run_training is enabled.",
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Number of stratified folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation ratio inside each fold's train+val partition.",
    )

    parser.add_argument(
        "--run_training",
        action="store_true",
        help="Run scripts/05_train.py for each generated fold.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Epochs override for fold training.")
    parser.add_argument("--patience", type=int, default=10, help="Patience override for fold training.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers override.")
    parser.add_argument(
        "--python_executable",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used for fold training subprocesses.",
    )
    parser.add_argument(
        "--max_folds",
        type=int,
        default=None,
        help="Optional cap on number of folds to run when --run_training is enabled.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra OmegaConf overrides forwarded to scripts/05_train.py.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    manifest_path = _resolve_path(project_root, args.manifest)
    base_splits_dir = _resolve_path(project_root, args.base_splits)
    output_splits_root = _resolve_path(project_root, args.output_splits)
    config_path = _resolve_path(project_root, args.config)

    if args.n_splits < 2:
        raise ValueError("--n_splits must be >= 2")

    if not (0.0 < float(args.val_ratio) < 0.5):
        raise ValueError("--val_ratio must be in (0, 0.5)")

    patients_df = load_patient_table(manifest_path, base_splits_dir)
    logger.info(
        "Loaded %d patients for k-fold splitting (class counts: %s)",
        len(patients_df),
        patients_df["class"].value_counts().to_dict(),
    )

    fold_infos = create_fold_splits(
        patients_df=patients_df,
        n_splits=int(args.n_splits),
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        output_root=output_splits_root,
    )

    logger.info("Generated %d folds under %s", len(fold_infos), output_splits_root)

    if not args.run_training:
        logger.info("Split generation complete (training not requested).")
        return

    fold_results: List[Dict] = []
    selected_folds = fold_infos
    if args.max_folds is not None:
        selected_folds = fold_infos[: int(args.max_folds)]

    for fold_info in selected_folds:
        result = run_fold_training(
            project_root=project_root,
            config_path=config_path,
            fold_info=fold_info,
            args=args,
        )
        fold_results.append(result)

    aggregate = aggregate_fold_metrics(fold_results)
    output_payload = {
        "folds": fold_results,
        "aggregate": aggregate,
    }
    results_path = output_splits_root / "kfold_results.json"
    results_path.write_text(json.dumps(output_payload, indent=2))

    logger.info("K-fold training complete. Summary saved: %s", results_path)
    logger.info("Aggregate mean metrics: %s", aggregate["mean"])


if __name__ == "__main__":
    main()
