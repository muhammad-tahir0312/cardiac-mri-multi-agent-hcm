"""
scripts/08_create_image_splits.py

Create leakage-safe train/val/test splits for image-level CMR classification.

Split strategy
--------------
- Patient-level stratification using the top-level patient directory
  (e.g. Normal/Directory_1, Sick/Directory_17).
- Images are then assigned by patient membership.
- Ensures no patient appears across multiple splits.

Optional
--------
- Generate stratified k-fold patient splits (e.g., 10-fold).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cardiac_image_dataset import scan_patient_image_table


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _resolve(project_root: Path, path_like: Path) -> Path:
    return path_like if path_like.is_absolute() else (project_root / path_like)


def build_patient_table(image_df: pd.DataFrame) -> pd.DataFrame:
    """Create one-row-per-patient table with class labels."""
    return (
        image_df[["patient_id", "class"]]
        .drop_duplicates(subset=["patient_id"])
        .reset_index(drop=True)
    )


def stratified_patient_split(
    patients_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    total = float(train_ratio + val_ratio + test_ratio)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")

    patient_ids = patients_df["patient_id"].tolist()
    labels = patients_df["class"].tolist()

    val_test_ratio = val_ratio + test_ratio
    train_ids, val_test_ids, _, val_test_labels = train_test_split(
        patient_ids,
        labels,
        test_size=float(val_test_ratio),
        stratify=labels,
        random_state=seed,
    )

    relative_test = float(test_ratio) / float(val_test_ratio)
    val_ids, test_ids = train_test_split(
        val_test_ids,
        test_size=relative_test,
        stratify=val_test_labels,
        random_state=seed,
    )

    return list(train_ids), list(val_ids), list(test_ids)


def verify_no_leakage(train_ids: Sequence[str], val_ids: Sequence[str], test_ids: Sequence[str]) -> None:
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    if (train_set & val_set) or (train_set & test_set) or (val_set & test_set):
        raise AssertionError("Patient-level leakage detected between splits.")


def save_split_csvs(
    image_df: pd.DataFrame,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    output_dir: Path,
) -> Dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = image_df[image_df["patient_id"].isin(train_ids)].reset_index(drop=True)
    val_df = image_df[image_df["patient_id"].isin(val_ids)].reset_index(drop=True)
    test_df = image_df[image_df["patient_id"].isin(test_ids)].reset_index(drop=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    return {"train": train_df, "val": val_df, "test": test_df}


def print_split_summary(split_name: str, split_df: pd.DataFrame) -> None:
    n_images = len(split_df)
    n_patients = split_df["patient_id"].nunique()
    class_counts = split_df["class"].value_counts().to_dict()
    logger.info(
        "%s: patients=%d images=%d class_counts=%s",
        split_name.upper(),
        n_patients,
        n_images,
        class_counts,
    )


def create_kfold_splits(
    image_df: pd.DataFrame,
    output_root: Path,
    n_splits: int,
    val_ratio: float,
    seed: int,
) -> None:
    """Generate patient-level stratified k-fold train/val/test CSVs."""
    patients_df = build_patient_table(image_df)
    patient_ids = patients_df["patient_id"].to_numpy()
    labels = patients_df["class"].to_numpy()

    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    output_root.mkdir(parents=True, exist_ok=True)

    fold_records: List[Dict] = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(patient_ids, labels), start=1):
        fold_dir = output_root / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        trainval_patients = patients_df.iloc[trainval_idx].reset_index(drop=True)
        test_patients = patients_df.iloc[test_idx].reset_index(drop=True)

        stratify_labels = trainval_patients["class"]
        train_patients, val_patients = train_test_split(
            trainval_patients,
            test_size=float(val_ratio),
            stratify=stratify_labels,
            random_state=int(seed) + fold_idx,
        )

        train_ids = train_patients["patient_id"].tolist()
        val_ids = val_patients["patient_id"].tolist()
        test_ids = test_patients["patient_id"].tolist()

        verify_no_leakage(train_ids, val_ids, test_ids)

        fold_train = image_df[image_df["patient_id"].isin(train_ids)].reset_index(drop=True)
        fold_val = image_df[image_df["patient_id"].isin(val_ids)].reset_index(drop=True)
        fold_test = image_df[image_df["patient_id"].isin(test_ids)].reset_index(drop=True)

        fold_train.to_csv(fold_dir / "train.csv", index=False)
        fold_val.to_csv(fold_dir / "val.csv", index=False)
        fold_test.to_csv(fold_dir / "test.csv", index=False)

        fold_records.append(
            {
                "fold": fold_dir.name,
                "n_train_patients": int(len(train_ids)),
                "n_val_patients": int(len(val_ids)),
                "n_test_patients": int(len(test_ids)),
                "n_train_images": int(len(fold_train)),
                "n_val_images": int(len(fold_val)),
                "n_test_images": int(len(fold_test)),
            }
        )

    (output_root / "folds_summary.json").write_text(json.dumps(fold_records, indent=2))
    logger.info("Saved %d-fold image splits to %s", n_splits, output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create patient-level leakage-safe image splits for CMR classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits_image"))
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--kfolds", type=int, default=0, help="Set >1 to generate stratified patient k-fold splits.")
    parser.add_argument("--kfold_val_ratio", type=float, default=0.10)
    parser.add_argument("--kfold_output", type=Path, default=Path("data/splits_image_kfold"))

    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
        help="Valid image extensions.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    data_root = _resolve(PROJECT_ROOT, args.data_root)
    splits_dir = _resolve(PROJECT_ROOT, args.splits_dir)
    kfold_output = _resolve(PROJECT_ROOT, args.kfold_output)

    class_to_idx = {"Normal": 0, "Sick": 1}

    logger.info("Scanning images recursively from: %s", data_root)
    image_df = scan_patient_image_table(
        data_root=data_root,
        class_to_idx=class_to_idx,
        valid_extensions=args.extensions,
    )

    if image_df.empty:
        logger.error("No images found under %s", data_root)
        sys.exit(1)

    patients_df = build_patient_table(image_df)
    logger.info(
        "Discovered patients=%d images=%d class_counts=%s",
        len(patients_df),
        len(image_df),
        patients_df["class"].value_counts().to_dict(),
    )

    train_ids, val_ids, test_ids = stratified_patient_split(
        patients_df=patients_df,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )
    verify_no_leakage(train_ids, val_ids, test_ids)

    split_frames = save_split_csvs(
        image_df=image_df,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        output_dir=splits_dir,
    )

    logger.info("Saved leakage-safe splits to %s", splits_dir)
    print_split_summary("train", split_frames["train"])
    print_split_summary("val", split_frames["val"])
    print_split_summary("test", split_frames["test"])

    if int(args.kfolds) > 1:
        create_kfold_splits(
            image_df=image_df,
            output_root=kfold_output,
            n_splits=int(args.kfolds),
            val_ratio=float(args.kfold_val_ratio),
            seed=int(args.seed),
        )


if __name__ == "__main__":
    main()
