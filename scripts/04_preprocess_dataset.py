"""
scripts/04_preprocess_dataset.py

Phase 2 — Batch DICOM-to-NIfTI preprocessing across the entire dataset.

Reads the split CSVs produced by ``03_create_patient_splits.py``, runs the
RouterAgent + IngestionAgent + PreprocessingAgent pipeline for every patient,
and writes one compressed NIfTI file per patient to ``data/processed/``.

The output filename follows the convention:
    {processed_root}/{Normal|Sick}_{Directory_X}.nii.gz

A manifest CSV (``data/processed/manifest.csv``) is written upon completion
with columns:
    patient_id | class | split | processed_path | status | error_msg

This manifest can be used by :class:`~src.data.cardiac_dataset.CardiacMRIDataset`
via its ``processed_path`` column to skip the path-inference logic.

Usage
-----
    python scripts/04_preprocess_dataset.py \\
        --config   src/config/base.yaml \\
        --splits   data/splits \\
        --output   data/processed \\
        --workers  4 \\
        --overwrite

    # Dry-run: print what would be processed without writing files
    python scripts/04_preprocess_dataset.py --dry_run
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

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
# Worker function (runs in a subprocess)
# ---------------------------------------------------------------------------


def _process_one(
    patient_id: str,
    class_label: str,
    series_path_str: str,
    processed_root_str: str,
    config_path_str: str,
    overwrite: bool,
) -> Dict:
    """Process a single patient's best series and save as NIfTI.

    Designed to run in a :class:`~concurrent.futures.ProcessPoolExecutor`.
    All arguments are primitive types (pickle-safe).

    Args:
        patient_id:         Patient identifier string (e.g. ``"Normal/Directory_1"``).
        class_label:        ``"Normal"`` or ``"Sick"``.
        series_path_str:    String path to the patient directory (not series).
        processed_root_str: String path to the output root.
        config_path_str:    String path to the YAML config file.
        overwrite:          Re-process even if the output file already exists.

    Returns:
        Dict with keys ``patient_id``, ``class``, ``processed_path``,
        ``status`` (``"ok"`` / ``"skipped"`` / ``"error"``), ``error_msg``.
    """
    result: Dict = {
        "patient_id": patient_id,
        "class": class_label,
        "processed_path": "",
        "status": "error",
        "error_msg": "",
    }

    try:
        import sys
        from pathlib import Path as _Path
        _project_root = str(_Path(__file__).resolve().parents[1])
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)

        from src.agents.router_agent import RouterAgent
        from src.agents.ingestion_agent import IngestionAgent
        from src.agents.preprocessing_agent import PreprocessingAgent
        import nibabel as nib
        import numpy as np

        cfg: DictConfig = OmegaConf.load(config_path_str)

        # Determine output path
        safe_name = patient_id.replace("/", "_").replace("\\", "_")
        output_path = Path(processed_root_str) / f"{safe_name}.nii.gz"
        result["processed_path"] = str(output_path)

        if output_path.exists() and not overwrite:
            result["status"] = "skipped"
            return result

        # Patient directory = parent of series_path recorded in CSV
        patient_dir = Path(series_path_str)  # series_path col stores patient dir
        if not patient_dir.is_dir():
            patient_dir = patient_dir.parent

        # Router
        router = RouterAgent(
            min_slices=int(cfg.router.min_slices),
            priority_keywords=list(cfg.router.priority_keywords),
        )
        best_series = router.select_series(patient_dir)
        if best_series is None:
            # Fallback: use the provided path directly
            best_series = Path(series_path_str)

        # Ingest
        ingestor = IngestionAgent()
        volume, affine, _ = ingestor.load_series(best_series)

        # Preprocess (no augmentation during preprocessing)
        preprocessor = PreprocessingAgent(cfg)
        tensor = preprocessor.preprocess(volume, augment=False)  # (1, D, H, W)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_np = tensor.squeeze(0).numpy()  # (D, H, W)
        nib.save(nib.Nifti1Image(out_np, affine), str(output_path))

        result["status"] = "ok"

    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error_msg"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_all_patients(splits_dir: Path) -> pd.DataFrame:
    """Concatenate train/val/test CSVs into one deduplicated patient table.

    Args:
        splits_dir: Directory containing ``train.csv``, ``val.csv``,
                    ``test.csv``.

    Returns:
        DataFrame with columns ``patient_id``, ``class``, ``series_path``,
        ``split``.

    Raises:
        FileNotFoundError: If any split CSV is missing.
    """
    frames: List[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        csv = splits_dir / f"{split}.csv"
        if not csv.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {csv}. "
                "Run 03_create_patient_splits.py first."
            )
        df = pd.read_csv(csv)
        df["split"] = split
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # Deduplicate: keep one representative series per patient
    combined = combined.drop_duplicates(subset=["patient_id"]).reset_index(drop=True)
    logger.info(
        "Total unique patients to process: %d  "
        "(Normal: %d | Sick: %d)",
        len(combined),
        (combined["class"] == "Normal").sum(),
        (combined["class"] == "Sick").sum(),
    )
    return combined


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Batch-convert DICOM series to preprocessed NIfTI volumes."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("src/config/base.yaml"),
        help="Path to OmegaConf YAML config.",
    )
    parser.add_argument(
        "--splits", type=Path, default=Path("data/splits"),
        help="Directory containing train/val/test CSVs.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed"),
        help="Root output directory for .nii.gz files.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process patients whose output file already exists.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be processed without writing any files.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    # Validate config
    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        sys.exit(1)

    cfg: DictConfig = OmegaConf.load(args.config)
    logger.info("Loaded config: %s", args.config)

    # Load all patients
    try:
        patients_df = load_all_patients(args.splits)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if args.dry_run:
        logger.info("[DRY RUN] Would process %d patients → %s", len(patients_df), args.output)
        print(patients_df[["patient_id", "class", "split"]].to_string(index=False))
        return

    args.output.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks: List[Tuple] = [
        (
            row["patient_id"],
            row["class"],
            row["series_path"],
            str(args.output),
            str(args.config),
            args.overwrite,
        )
        for _, row in patients_df.iterrows()
    ]

    # Process
    results: List[Dict] = []
    start = time.monotonic()

    workers = min(args.workers, len(tasks))
    logger.info("Launching %d worker(s) for %d patients …", workers, len(tasks))

    if workers <= 1:
        # Sequential mode — easier to debug
        for task in tqdm(tasks, desc="Preprocessing"):
            results.append(_process_one(*task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, *t): t[0] for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
                results.append(fut.result())

    elapsed = time.monotonic() - start

    # Summary
    ok      = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors  = sum(1 for r in results if r["status"] == "error")

    logger.info("=" * 60)
    logger.info("Preprocessing complete in %.1f s", elapsed)
    logger.info("  OK      : %d", ok)
    logger.info("  Skipped : %d  (already exist, use --overwrite to redo)", skipped)
    logger.info("  Errors  : %d", errors)

    if errors:
        logger.warning("Errors encountered:")
        for r in results:
            if r["status"] == "error":
                logger.warning("  %s — %s", r["patient_id"], r["error_msg"])

    # Save manifest
    manifest_df = pd.DataFrame(results)
    # Merge split column back in
    split_map = dict(zip(patients_df["patient_id"], patients_df["split"]))
    manifest_df["split"] = manifest_df["patient_id"].map(split_map)
    manifest_path = args.output / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    logger.info("Manifest saved to: %s", manifest_path)


if __name__ == "__main__":
    main()
