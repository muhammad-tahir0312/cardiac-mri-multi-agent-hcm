"""
src/agents/ingestion_agent.py

Ingestion Agent — Loads a JPEG cardiac MRI series, builds a spatially-ordered
3-D volume, and persists it as a compressed NIfTI file (.nii.gz).

Data format
-----------
Each series directory contains grayscale JPEG slices named:

    imgNNNN-z_pos.jpg

where ``NNNN`` is a zero-padded frame index and ``z_pos`` is the slice
z-position in mm (may be negative, producing a double-dash: ``img0001--42.jpg``).

Pipeline per series
-------------------
1. Glob all ``.jpg`` files in the target directory (non-recursive).
2. Parse the frame index from each filename; skip unreadable files.
3. Sort slices by ascending frame index.
4. Load each image with Pillow (grayscale, float32).
5. Stack pixel arrays into a 3-D ``np.ndarray`` with shape (H, W, D).
6. Construct a 4×4 affine using z-positions parsed from filenames.
7. Save as ``.nii.gz`` via ``nibabel``.

Usage (programmatic)
---------------------
    from pathlib import Path
    from src.agents.ingestion_agent import IngestionAgent

    agent = IngestionAgent()
    volume, affine, meta = agent.load_series(
        Path("data/raw/Normal/Directory_1/series0001-Body")
    )
    agent.save_nifti(volume, affine, Path("data/processed/patient_001.nii.gz"))

Usage (CLI)
-----------
    python -m src.agents.ingestion_agent \\
        --series_dir  data/raw/Normal/Directory_1/series0001-Body \\
        --output_path data/processed/patient_001.nii.gz
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Volume = np.ndarray          # shape: (H, W, D), dtype: float32
Affine = np.ndarray          # shape: (4, 4),    dtype: float64
SeriesMetadata = Dict        # metadata dict


# ---------------------------------------------------------------------------
# Filename pattern
# ---------------------------------------------------------------------------

# Matches:  img0011-38.jpg  →  frame=11,  z= 38.0
# Matches:  img0001--42.jpg →  frame= 1,  z=-42.0
# Matches:  img0024--37.19.jpg → frame=24, z=-37.19
_IMG_RE = re.compile(
    r"img(\d+)-([-\d.]+)\.jpg",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# IngestionAgent
# ---------------------------------------------------------------------------


class IngestionAgent:
    """Converts a folder of JPEG cardiac MRI slices into a 3-D NIfTI volume.

    Slices are named ``imgNNNN-z_pos.jpg`` where ``NNNN`` is the zero-padded
    frame index and ``z_pos`` is the slice z-position in millimetres.

    Args:
        img_extension: File extension for image files (default ``".jpg"``).
        rescale:       Backward-compatible no-op flag kept for callers that
                       previously ingested DICOM with rescale options.

    Attributes:
        img_extension (str): Image file suffix.
    """

    def __init__(
        self,
        img_extension: str = ".jpg",
        rescale: Optional[bool] = None,
    ) -> None:
        self.img_extension = img_extension
        self.rescale = bool(rescale) if rescale is not None else False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_series(
        self,
        series_dir: Path,
    ) -> Tuple[Volume, Affine, SeriesMetadata]:
        """Load a JPEG series directory into a 3-D NumPy volume.

        Steps:
            1. Discover all JPEG files in ``series_dir``.
            2. Parse frame index and z-position from each filename.
            3. Sort slices by ascending frame index.
            4. Load each image as grayscale float32 with Pillow.
            5. Stack pixel data into shape ``(H, W, D)``.
            6. Build a NIfTI-compatible affine matrix from z-positions.
            7. Assemble and return a metadata summary dict.

        Args:
            series_dir: Path to the series folder containing ``.jpg`` files.

        Returns:
            Tuple of:
                - ``volume``: float32 array with shape ``(H, W, D)``.
                - ``affine``: 4×4 float64 affine matrix for NIfTI header.
                - ``metadata``: Dict with keys ``n_slices``, ``spacing``,
                  ``z_positions``, ``series_name``, ``rows``, ``columns``.

        Raises:
            FileNotFoundError: If ``series_dir`` does not exist.
            ValueError: If no valid image slices are found.
        """
        if not series_dir.exists():
            raise FileNotFoundError(f"Series directory not found: {series_dir}")

        img_paths = sorted(series_dir.glob(f"*{self.img_extension}"))
        if not img_paths:
            # Try recursive as fallback
            img_paths = sorted(series_dir.rglob(f"*{self.img_extension}"))
        if not img_paths:
            raise ValueError(f"No {self.img_extension} files found in: {series_dir}")

        logger.info("Loading %d image files from: %s", len(img_paths), series_dir)

        # ------------------------------------------------------------------
        # 1. Parse filenames → (frame_idx, z_pos, path)
        # ------------------------------------------------------------------
        parsed: List[Tuple[int, float, Path]] = []
        for p in img_paths:
            frame_idx, z_pos = self._parse_img_filename(p.name)
            parsed.append((frame_idx, z_pos, p))

        # ------------------------------------------------------------------
        # 2. Sort by frame index
        # ------------------------------------------------------------------
        parsed.sort(key=lambda x: x[0])

        # ------------------------------------------------------------------
        # 3. Load each slice (first pass to collect shapes)
        # ------------------------------------------------------------------
        raw_slices: List[Tuple[float, np.ndarray]] = []
        for frame_idx, z_pos, img_path in parsed:
            arr = self._load_jpg_slice(img_path)
            if arr is None:
                continue
            raw_slices.append((z_pos, arr))

        if not raw_slices:
            raise ValueError(f"No valid image slices loaded from: {series_dir}")

        # Determine the modal (most common) shape and resize outliers to it
        from collections import Counter
        shape_counts = Counter(arr.shape for _, arr in raw_slices)
        target_hw: Tuple[int, int] = shape_counts.most_common(1)[0][0]   # (H, W)

        pixel_arrays: List[np.ndarray] = []
        z_positions: List[float] = []
        for z_pos, arr in raw_slices:
            if arr.shape != target_hw:
                arr = self._resize_slice(arr, target_hw)
            pixel_arrays.append(arr)
            z_positions.append(z_pos)

        # ------------------------------------------------------------------
        # 4. Stack → (H, W, D)
        # ------------------------------------------------------------------
        volume: Volume = np.stack(pixel_arrays, axis=-1).astype(np.float32)
        logger.info("Volume shape: %s | dtype: %s", volume.shape, volume.dtype)

        # ------------------------------------------------------------------
        # 5. Build affine
        # ------------------------------------------------------------------
        affine = self._build_affine_from_z_positions(z_positions)

        # ------------------------------------------------------------------
        # 6. Metadata
        # ------------------------------------------------------------------
        H, W, D = volume.shape
        metadata: SeriesMetadata = {
            "n_slices":    D,
            "spacing":     [1.0, 1.0, _compute_slice_spacing(z_positions)],
            "z_positions": z_positions,
            "series_name": series_dir.name,
            "patient_id":  series_dir.parent.name,
            "rows":        H,
            "columns":     W,
        }

        return volume, affine, metadata

    def save_nifti(
        self,
        volume: Volume,
        affine: Affine,
        output_path: Path,
        dtype: type = np.float32,
    ) -> None:
        """Save a 3-D volume as a compressed NIfTI file.

        The output directory is created automatically if it does not exist.
        Pixel data is cast to ``dtype`` before saving (default ``float32``).

        Args:
            volume:      3-D array with shape ``(H, W, D)``.
            affine:      4×4 affine matrix for spatial registration.
            output_path: Destination file path (should end in ``.nii.gz``).
            dtype:       NumPy dtype for the stored image data.

        Raises:
            ValueError: If ``volume`` is not a 3-D array.
        """
        if volume.ndim != 3:
            raise ValueError(
                f"Expected 3-D volume, got shape {volume.shape}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        nifti_img = nib.Nifti1Image(volume.astype(dtype), affine)
        nib.save(nifti_img, str(output_path))
        logger.info("Saved NIfTI volume to: %s", output_path)

    def process_series(
        self,
        series_dir: Path,
        output_path: Path,
    ) -> Tuple[Volume, Affine, SeriesMetadata]:
        """Convenience method: load a series and immediately save as NIfTI.

        Args:
            series_dir:  Input JPEG series directory.
            output_path: Destination ``.nii.gz`` file path.

        Returns:
            Same tuple as :meth:`load_series`.
        """
        volume, affine, metadata = self.load_series(series_dir)
        self.save_nifti(volume, affine, output_path)
        return volume, affine, metadata

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_img_filename(filename: str) -> Tuple[int, float]:
        """Extract frame index and z-position from an image filename.

        Handles both positive and negative z-positions::

            img0011-38.jpg     → (11,   38.0)
            img0001--42.jpg    → ( 1,  -42.0)
            img0024--37.1961.jpg → (24, -37.1961)

        Falls back to ``(sequential_order, 0.0)`` for non-standard names.

        Args:
            filename: Bare filename (no directory component).

        Returns:
            Tuple of ``(frame_index, z_position_mm)``.
        """
        m = _IMG_RE.match(filename)
        if m:
            return int(m.group(1)), float(m.group(2))
        # Fallback: extract leading digit sequence
        m2 = re.match(r"img(\d+)", filename, re.IGNORECASE)
        if m2:
            return int(m2.group(1)), 0.0
        return 0, 0.0

    @staticmethod
    def _load_jpg_slice(img_path: Path) -> Optional[np.ndarray]:
        """Load a single JPEG slice as a grayscale float32 array.

        Args:
            img_path: Path to the ``.jpg`` file.

        Returns:
            2-D float32 array ``(H, W)`` or ``None`` on error.
        """
        try:
            with Image.open(img_path) as img:
                arr = np.array(img.convert("L"), dtype=np.float32)
            return arr
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", img_path.name, exc)
            return None

    @staticmethod
    def _resize_slice(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        """Resize a 2-D slice to ``target_hw = (H, W)`` using Pillow LANCZOS.

        Args:
            arr:       2-D float32 array ``(H, W)``.
            target_hw: Target ``(height, width)`` in pixels.

        Returns:
            Resized float32 array with shape ``target_hw``.
        """
        H, W = target_hw
        pil_img = Image.fromarray(arr.astype(np.uint8))
        pil_img = pil_img.resize((W, H), Image.LANCZOS)
        return np.array(pil_img, dtype=np.float32)

    @staticmethod
    def _build_affine_from_z_positions(z_positions: List[float]) -> Affine:
        """Build a 4×4 NIfTI affine matrix from parsed z-positions.

        Assumes 1 mm isotropic in-plane pixel spacing.  The slice axis
        direction is aligned with the z-axis; spacing is derived from the
        median inter-slice distance.

        Args:
            z_positions: Ordered list of z-coordinates (mm) for each slice.

        Returns:
            4×4 float64 identity-based affine.
        """
        slice_spacing = _compute_slice_spacing(z_positions)

        affine = np.eye(4, dtype=np.float64)
        affine[0, 0] = 1.0              # col direction: x
        affine[1, 1] = 1.0              # row direction: y
        affine[2, 2] = slice_spacing    # slice direction: z
        if z_positions:
            affine[2, 3] = z_positions[0]   # z origin = first slice

        return affine


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _compute_slice_spacing(z_positions: List[float]) -> float:
    """Compute median slice spacing from z-position list.

    Args:
        z_positions: Ordered list of z-coordinates in mm.

    Returns:
        Median absolute difference between consecutive z-positions (mm).
        Returns ``1.0`` if fewer than two positions are given.
    """
    if len(z_positions) < 2:
        return 1.0
    diffs = [abs(z_positions[i + 1] - z_positions[i]) for i in range(len(z_positions) - 1)]
    spacing = float(np.median(diffs))
    return spacing if spacing > 0 else 1.0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli() -> None:
    """Command-line interface for converting a single JPEG series to NIfTI."""
    parser = argparse.ArgumentParser(
        description="Ingest a JPEG series and save it as a NIfTI volume."
    )
    parser.add_argument(
        "--series_dir",
        type=Path,
        required=True,
        help="Path to the series directory containing .jpg slices.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Destination .nii.gz file path.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    agent = IngestionAgent()

    try:
        volume, affine, meta = agent.process_series(args.series_dir, args.output_path)
        print("\n=== Ingestion Summary ===")
        print(f"  Volume shape   : {volume.shape}    (H × W × D)")
        print(f"  NIfTI affine   :\n{affine}")
        print(f"  Patient ID     : {meta['patient_id']}")
        print(f"  Series name    : {meta['series_name']}")
        print(f"  Spacing (mm)   : {meta['spacing']}")
        print(f"  Slices loaded  : {meta['n_slices']}")
        print(f"  Output saved   : {args.output_path}")
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    _cli()

