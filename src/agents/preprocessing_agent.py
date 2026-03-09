"""
src/agents/preprocessing_agent.py

Preprocessing Agent — Normalises, resizes, and optionally augments a 3-D
cardiac MRI volume prior to model ingestion.

Pipeline
--------
1. **Intensity clipping** — percentile-based clamp to remove outlier voxels.
2. **Normalisation** — Z-score (μ=0, σ=1) or Min-Max rescaling.
3. **Spatial resampling** — trilinear interpolation to a fixed (D, H, W) grid.
4. **Augmentation** (training only) — TorchIO-based random spatial and
   intensity transforms controlled entirely via the config YAML.

All hyper-parameters are consumed from an :class:`omegaconf.DictConfig` object
(loaded from ``src/config/base.yaml``) so **nothing is hard-coded**.

Usage (programmatic)
---------------------
    from omegaconf import OmegaConf
    from src.agents.preprocessing_agent import PreprocessingAgent

    cfg = OmegaConf.load("src/config/base.yaml")
    agent = PreprocessingAgent(cfg)

    # volume: np.ndarray (H, W, D) float32 — raw from IngestionAgent
    processed = agent.preprocess(volume, augment=False)
    # processed: torch.Tensor  shape (1, D, H, W)

Usage (CLI — process a single NIfTI file)
------------------------------------------
    python -m src.agents.preprocessing_agent \\
        --input  data/processed/raw/patient_001.nii.gz \\
        --output data/processed/clean/patient_001.nii.gz \\
        --config src/config/base.yaml \\
        --augment
"""

import logging
from pathlib import Path
from typing import Literal, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

try:
    import torchio as tio
    _TIO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TIO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
NormMode = Literal["zscore", "minmax", "percentile"]


# ---------------------------------------------------------------------------
# PreprocessingAgent
# ---------------------------------------------------------------------------


class PreprocessingAgent:
    """Normalises and resizes a 3-D MRI volume; applies TorchIO augmentation.

    Args:
        cfg: OmegaConf config object.  Reads ``cfg.preprocessing`` and
             ``cfg.augmentation`` sub-trees.

    Attributes:
        target_shape (Tuple[int, int, int]): Target ``(D, H, W)`` after resize.
        norm_mode (str):                    Normalisation strategy.
        lower_pct (float):                  Lower percentile for intensity clip.
        upper_pct (float):                  Upper percentile for intensity clip.
        train_transforms: TorchIO ``Compose`` built from config (or ``None``).
    """

    def __init__(self, cfg: DictConfig) -> None:
        pp = cfg.preprocessing
        aug = cfg.augmentation

        self.target_shape: Tuple[int, int, int] = (
            int(pp.target_shape.depth),
            int(pp.target_shape.height),
            int(pp.target_shape.width),
        )
        self.norm_mode: NormMode = str(pp.normalization)
        self.lower_pct: float = float(pp.intensity_clip.lower_percentile)
        self.upper_pct: float = float(pp.intensity_clip.upper_percentile)

        self.train_transforms = (
            self._build_torchio_transforms(aug)
            if aug.enabled and _TIO_AVAILABLE
            else None
        )

        if aug.enabled and not _TIO_AVAILABLE:
            logger.warning(
                "augmentation.enabled=true but torchio is not installed. "
                "Augmentation will be skipped. Install with: pip install torchio"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(
        self,
        volume: np.ndarray,
        augment: bool = False,
    ) -> torch.Tensor:
        """Full preprocessing pipeline for a single 3-D volume.

        Steps:
            1. Ensure float32.
            2. Percentile intensity clipping.
            3. Normalisation (Z-score or Min-Max).
            4. Spatial resize to ``self.target_shape``.
            5. Optional TorchIO augmentation (training only).

        Args:
            volume:  Raw 3-D numpy array from :class:`~src.agents.ingestion_agent.IngestionAgent`,
                     shape ``(H, W, D)`` or ``(D, H, W)``.  The array is
                     treated as ``(H, W, D)`` and internally transposed to
                     ``(D, H, W)`` for PyTorch compatibility before resizing.
            augment: If ``True``, apply the TorchIO training transforms.

        Returns:
            Float32 tensor with shape ``(1, D, H, W)`` — channel-first,
            ready for model ingestion.
        """
        vol = volume.astype(np.float32)

        # Ingestion agent returns (H, W, D); convert → (D, H, W)
        if vol.ndim == 3:
            vol = np.transpose(vol, (2, 0, 1))  # (H,W,D) → (D,H,W)

        # 1. Clip
        vol = self._clip_percentile(vol)

        # 2. Normalise
        vol = self._normalise(vol, self.norm_mode)

        # 3. Resize → (1, 1, D, H, W) for F.interpolate, then squeeze
        tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        tensor = F.interpolate(
            tensor,
            size=self.target_shape,
            mode="trilinear",
            align_corners=False,
        )
        tensor = tensor.squeeze(0)  # (1, D, H, W)

        # 4. Augmentation
        if augment and self.train_transforms is not None:
            tensor = self._apply_torchio(tensor)

        return tensor  # (1, D, H, W)

    def preprocess_nifti(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        augment: bool = False,
    ) -> torch.Tensor:
        """Load a NIfTI file, run the full pipeline, optionally save output.

        Args:
            input_path:  Path to an existing ``.nii.gz`` file.
            output_path: If provided, save the processed volume as ``.nii.gz``.
            augment:     Whether to apply augmentation.

        Returns:
            Processed tensor with shape ``(1, D, H, W)``.

        Raises:
            FileNotFoundError: If ``input_path`` does not exist.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"NIfTI file not found: {input_path}")

        nifti = nib.load(str(input_path))
        volume: np.ndarray = nifti.get_fdata(dtype=np.float32)
        tensor = self.preprocess(volume, augment=augment)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_np = tensor.squeeze(0).numpy()           # (D, H, W)
            out_img = nib.Nifti1Image(out_np, nifti.affine)
            nib.save(out_img, str(output_path))
            logger.debug("Saved preprocessed NIfTI: %s", output_path)

        return tensor

    # ------------------------------------------------------------------
    # Private: normalisation helpers
    # ------------------------------------------------------------------

    def _clip_percentile(self, vol: np.ndarray) -> np.ndarray:
        """Clip voxel intensities at configured lower/upper percentiles.

        Args:
            vol: 3-D float32 array.

        Returns:
            Clipped float32 array with same shape.
        """
        lo = float(np.percentile(vol, self.lower_pct))
        hi = float(np.percentile(vol, self.upper_pct))
        return np.clip(vol, lo, hi)

    def _normalise(self, vol: np.ndarray, mode: NormMode) -> np.ndarray:
        """Apply intensity normalisation.

        Modes:

        * ``"zscore"``    — subtract mean, divide by std (clip std to ε).
        * ``"minmax"``    — linearly scale to [0, 1].
        * ``"percentile"``— alias for ``"minmax"`` after percentile clipping
                           (the clip is already done upstream).

        Args:
            vol:  3-D float32 array.
            mode: One of ``"zscore"``, ``"minmax"``, ``"percentile"``.

        Returns:
            Normalised float32 array.
        """
        if mode == "zscore":
            mu  = float(vol.mean())
            std = float(vol.std())
            eps = 1e-8
            return (vol - mu) / max(std, eps)

        if mode in ("minmax", "percentile"):
            vmin = float(vol.min())
            vmax = float(vol.max())
            denom = max(vmax - vmin, 1e-8)
            return (vol - vmin) / denom

        logger.warning("Unknown normalisation mode '%s'. Returning unchanged.", mode)
        return vol

    # ------------------------------------------------------------------
    # Private: augmentation
    # ------------------------------------------------------------------

    def _build_torchio_transforms(self, aug: DictConfig) -> "tio.Compose":
        """Build a TorchIO ``Compose`` transform from the config.

        Only transforms with ``probability > 0`` are included.

        Args:
            aug: ``cfg.augmentation`` sub-config.

        Returns:
            :class:`torchio.Compose` transform.
        """
        transforms = []

        # --- Spatial ---
        rf = aug.random_flip
        if rf.flip_probability > 0:
            transforms.append(
                tio.RandomFlip(
                    axes=list(rf.axes),
                    flip_probability=float(rf.flip_probability),
                )
            )

        ra = aug.random_affine
        if ra.probability > 0:
            transforms.append(
                tio.RandomAffine(
                    scales=tuple(ra.scales),
                    degrees=float(ra.degrees),
                    translation=float(ra.translation),
                    p=float(ra.probability),
                )
            )

        red = aug.random_elastic_deformation
        if red.probability > 0:
            transforms.append(
                tio.RandomElasticDeformation(
                    num_control_points=int(red.num_control_points),
                    max_displacement=float(red.max_displacement),
                    p=float(red.probability),
                )
            )

        # --- Intensity ---
        rno = aug.random_noise
        if rno.probability > 0:
            transforms.append(
                tio.RandomNoise(
                    mean=float(rno.mean),
                    std=tuple(rno.std),
                    p=float(rno.probability),
                )
            )

        rbf = aug.random_bias_field
        if rbf.probability > 0:
            transforms.append(
                tio.RandomBiasField(
                    coefficients=float(rbf.coefficients),
                    order=int(rbf.order),
                    p=float(rbf.probability),
                )
            )

        rg = aug.random_gamma
        if rg.probability > 0:
            transforms.append(
                tio.RandomGamma(
                    log_gamma=float(rg.log_gamma),
                    p=float(rg.probability),
                )
            )

        rm = aug.random_motion
        if rm.probability > 0:
            transforms.append(
                tio.RandomMotion(
                    degrees=float(rm.degrees),
                    translation=float(rm.translation),
                    num_transforms=int(rm.num_transforms),
                    p=float(rm.probability),
                )
            )

        logger.info("Built TorchIO pipeline with %d transforms.", len(transforms))
        return tio.Compose(transforms)

    def _apply_torchio(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the TorchIO training transforms to a channel-first tensor.

        TorchIO expects a :class:`torchio.Subject` / ``ScalarImage``.  We
        wrap the tensor, apply transforms, and unwrap.

        Args:
            tensor: Float32 tensor with shape ``(1, D, H, W)``.

        Returns:
            Augmented float32 tensor with shape ``(1, D, H, W)``.
        """
        # TorchIO ScalarImage expects (C, W, H, D) — note reversed depth axis.
        # We permute accordingly, augment, then permute back.
        tio_tensor = tensor.permute(0, 3, 2, 1)   # (1,D,H,W) → (1,W,H,D)
        subject = tio.Subject(image=tio.ScalarImage(tensor=tio_tensor))
        augmented = self.train_transforms(subject)
        out = augmented["image"].data                # (1, W, H, D)
        return out.permute(0, 3, 2, 1).contiguous()  # → (1, D, H, W)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    """Single-file CLI wrapper for quick testing."""
    import argparse
    import sys
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="PreprocessingAgent: normalise and resize a NIfTI volume."
    )
    parser.add_argument("--input",  type=Path, required=True,  help="Input .nii.gz path.")
    parser.add_argument("--output", type=Path, default=None,   help="Output .nii.gz path (optional).")
    parser.add_argument("--config", type=Path, default=Path("src/config/base.yaml"), help="Config YAML path.")
    parser.add_argument("--augment", action="store_true", help="Apply training augmentation.")
    args = parser.parse_args()

    try:
        cfg = OmegaConf.load(args.config)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    agent = PreprocessingAgent(cfg)

    try:
        tensor = agent.preprocess_nifti(args.input, args.output, args.augment)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    print(f"\nProcessed tensor  : shape={tuple(tensor.shape)}  dtype={tensor.dtype}")
    print(f"Intensity range   : [{tensor.min():.4f}, {tensor.max():.4f}]")
    if args.output:
        print(f"Saved to          : {args.output}")


if __name__ == "__main__":
    _cli()
