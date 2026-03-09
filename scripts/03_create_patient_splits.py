"""
scripts/03_create_patient_splits.py

Phase 1 — Data Integrity & Patient-Level Splits.

Scans data/raw/{Normal,Sick}, identifies each Directory_X as a unique
patient, then performs a **stratified split by patient** (80 / 10 / 10).

Outputs
-------
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv

Each CSV has columns:
    patient_id  | class  | series_path  | image_count

Usage
-----
    python scripts/03_create_patient_splits.py \
        --data_root data/raw \
        --splits_dir data/splits \
        --train_ratio 0.80 \
        --val_ratio   0.10 \
        --test_ratio  0.10 \
        --seed 42
"""

import argparse
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Random seed set to %d", seed)


def scan_class_dir(class_dir: Path, class_label: str) -> List[Dict]:
    """Scan a class directory and collect per-series records.

    Each top-level subdirectory of ``class_dir`` is treated as one patient
    (``Directory_X``).  Within each patient directory every subdirectory is a
    series.  The number of DICOM files in each series is recorded.

    Args:
        class_dir:   Absolute path to ``data/raw/{Normal|Sick}``.
        class_label: Human-readable label (``"Normal"`` or ``"Sick"``).

    Returns:
        A list of dicts with keys ``patient_id``, ``class``,
        ``series_path``, ``image_count``.
    """
    records: List[Dict] = []

    if not class_dir.is_dir():
        logger.warning("Class directory not found, skipping: %s", class_dir)
        return records

    patient_dirs = sorted(
        [d for d in class_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    logger.info("Found %d patient directories in %s", len(patient_dirs), class_dir)

    for patient_dir in patient_dirs:
        patient_id = f"{class_label}/{patient_dir.name}"

        # Collect series directories (immediate children that are dirs)
        series_dirs = sorted(
            [s for s in patient_dir.iterdir() if s.is_dir()],
            key=lambda p: p.name,
        )

        if not series_dirs:
            # Flat layout — patient dir itself contains images
            img_count = len(list(patient_dir.glob("*.jpg")))
            records.append(
                {
                    "patient_id": patient_id,
                    "class": class_label,
                    "series_path": str(patient_dir),
                    "image_count": img_count,
                }
            )
        else:
            for series_dir in series_dirs:
                # Count JPEG slices recursively within the series folder
                jpg_files = list(series_dir.rglob("*.jpg"))
                records.append(
                    {
                        "patient_id": patient_id,
                        "class": class_label,
                        "series_path": str(series_dir),
                        "image_count": len(jpg_files),
                    }
                )

    return records


def aggregate_patient_metadata(
    records: List[Dict],
) -> Tuple[List[str], List[str]]:
    """Build ordered lists of unique patient IDs and their class labels.

    Args:
        records: Flat list of series-level dicts from :func:`scan_class_dir`.

    Returns:
        Tuple of (patient_ids, class_labels) — one entry per patient.
    """
    seen: Dict[str, str] = {}
    for r in records:
        pid = r["patient_id"]
        if pid not in seen:
            seen[pid] = r["class"]
    patient_ids = list(seen.keys())
    class_labels = [seen[pid] for pid in patient_ids]
    return patient_ids, class_labels


def stratified_patient_split(
    patient_ids: List[str],
    class_labels: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Split patient IDs into train / val / test preserving class balance.

    The split is performed **at the patient level** to guarantee that all
    series from a single patient end up in the same partition (no leakage).

    Args:
        patient_ids:  List of unique patient identifier strings.
        class_labels: Corresponding class label for each patient ID.
        train_ratio:  Fraction of patients for training (e.g. 0.80).
        val_ratio:    Fraction of patients for validation (e.g. 0.10).
        test_ratio:   Fraction of patients for testing (e.g. 0.10).
        seed:         Random seed for reproducibility.

    Returns:
        Three lists (train_ids, val_ids, test_ids).

    Raises:
        ValueError: If ratios do not sum to 1.0 (±0.001 tolerance).
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.4f}")

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_ids, val_test_ids, _, val_test_labels = train_test_split(
        patient_ids,
        class_labels,
        test_size=val_test_ratio,
        stratify=class_labels,
        random_state=seed,
    )

    # Second split: val vs test (equal within the remainder)
    relative_test = test_ratio / val_test_ratio
    val_ids, test_ids = train_test_split(
        val_test_ids,
        test_size=relative_test,
        stratify=val_test_labels,
        random_state=seed,
    )

    return train_ids, val_ids, test_ids


def build_split_df(
    patient_ids: List[str],
    all_records: List[Dict],
) -> pd.DataFrame:
    """Filter the full records list to only rows belonging to given patients.

    Args:
        patient_ids: Patient IDs that belong to this split.
        all_records: Complete list of series-level dicts.

    Returns:
        DataFrame with columns: ``patient_id``, ``class``, ``series_path``,
        ``image_count``.
    """
    id_set = set(patient_ids)
    rows = [r for r in all_records if r["patient_id"] in id_set]
    df = pd.DataFrame(rows, columns=["patient_id", "class", "series_path", "image_count"])
    return df.reset_index(drop=True)


def print_split_statistics(
    name: str,
    df: pd.DataFrame,
    total_patients: int,
) -> None:
    """Log class-balance statistics for a given split dataframe.

    Args:
        name:           Split name (e.g. ``"TRAIN"``).
        df:             DataFrame for this split.
        total_patients: Total unique patients in this split.
    """
    logger.info("=" * 60)
    logger.info("Split: %s", name)
    logger.info("  Unique patients : %d", total_patients)
    logger.info("  Total series    : %d", len(df))
    logger.info("  Total images    : %d", df["image_count"].sum())
    logger.info("  Class balance (series):")
    counts = df.groupby("class")["series_path"].count()
    for cls, cnt in counts.items():
        pct = 100.0 * cnt / len(df) if len(df) > 0 else 0.0
        logger.info("    %-8s : %d  (%.1f%%)", cls, cnt, pct)
    logger.info("  Class balance (images):")
    img_counts = df.groupby("class")["image_count"].sum()
    total_imgs = img_counts.sum()
    for cls, cnt in img_counts.items():
        pct = 100.0 * cnt / total_imgs if total_imgs > 0 else 0.0
        logger.info("    %-8s : %d  (%.1f%%)", cls, cnt, pct)


def verify_no_leakage(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
) -> None:
    """Assert that no patient appears in more than one split.

    Args:
        train_ids: Patient IDs in the training split.
        val_ids:   Patient IDs in the validation split.
        test_ids:  Patient IDs in the test split.

    Raises:
        AssertionError: If any patient appears in more than one split.
    """
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set & test_set

    if tv or tt or vt:
        raise AssertionError(
            f"DATA LEAKAGE DETECTED!\n"
            f"  Train ∩ Val  : {tv}\n"
            f"  Train ∩ Test : {tt}\n"
            f"  Val   ∩ Test : {vt}"
        )

    logger.info("NO DATA LEAKAGE verified — all patient splits are disjoint.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Create patient-level stratified train/val/test splits."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing Normal/ and Sick/ subdirectories.",
    )
    parser.add_argument(
        "--splits_dir",
        type=Path,
        default=Path("data/splits"),
        help="Output directory for CSV split files.",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.80, help="Fraction for training."
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.10, help="Fraction for validation."
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.10, help="Fraction for test."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for patient split creation."""
    args = parse_args()
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Scan both class directories
    # ------------------------------------------------------------------
    logger.info("Scanning data root: %s", args.data_root.resolve())
    normal_records = scan_class_dir(args.data_root / "Normal", "Normal")
    sick_records = scan_class_dir(args.data_root / "Sick", "Sick")
    all_records = normal_records + sick_records

    if not all_records:
        logger.error("No DICOM files found under %s. Aborting.", args.data_root)
        sys.exit(1)

    logger.info(
        "Total series records: %d  (Normal: %d, Sick: %d)",
        len(all_records),
        len(normal_records),
        len(sick_records),
    )

    # ------------------------------------------------------------------
    # 2. Aggregate to patient level
    # ------------------------------------------------------------------
    patient_ids, class_labels = aggregate_patient_metadata(all_records)
    logger.info("Total unique patients: %d", len(patient_ids))

    normal_patients = sum(1 for c in class_labels if c == "Normal")
    sick_patients = sum(1 for c in class_labels if c == "Sick")
    logger.info("Patient class balance — Normal: %d | Sick: %d", normal_patients, sick_patients)

    # ------------------------------------------------------------------
    # 3. Stratified patient-level split
    # ------------------------------------------------------------------
    train_ids, val_ids, test_ids = stratified_patient_split(
        patient_ids,
        class_labels,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    logger.info(
        "Patient split sizes — Train: %d | Val: %d | Test: %d",
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )

    # ------------------------------------------------------------------
    # 4. Leakage check
    # ------------------------------------------------------------------
    verify_no_leakage(train_ids, val_ids, test_ids)

    # ------------------------------------------------------------------
    # 5. Build DataFrames and print statistics
    # ------------------------------------------------------------------
    train_df = build_split_df(train_ids, all_records)
    val_df = build_split_df(val_ids, all_records)
    test_df = build_split_df(test_ids, all_records)

    print_split_statistics("TRAIN", train_df, len(train_ids))
    print_split_statistics("VAL",   val_df,   len(val_ids))
    print_split_statistics("TEST",  test_df,  len(test_ids))

    # ------------------------------------------------------------------
    # 6. Save CSVs
    # ------------------------------------------------------------------
    args.splits_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(args.splits_dir / "train.csv", index=False)
    val_df.to_csv(args.splits_dir / "val.csv",   index=False)
    test_df.to_csv(args.splits_dir / "test.csv",  index=False)

    logger.info("Saved split CSVs to %s", args.splits_dir.resolve())
    logger.info(
        "Files: train.csv (%d rows), val.csv (%d rows), test.csv (%d rows)",
        len(train_df),
        len(val_df),
        len(test_df),
    )


if __name__ == "__main__":
    main()
