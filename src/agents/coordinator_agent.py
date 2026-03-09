"""
src/agents/coordinator_agent.py

Coordinator Agent — orchestrates the full HCM Multi-Agent pipeline from a
raw patient directory to a final HCM diagnosis with confidence score.

Pipeline order
--------------
1. **RouterAgent**       — select the best DICOM series in the patient dir.
2. **IngestionAgent**    — load DICOMs → 3-D numpy volume + NIfTI affine.
3. **PreprocessingAgent** — normalise + resize → ``(1, D, H, W)`` tensor.
4. **SegmentationAgent** (optional) — LV/RV/myocardium mask on the volume.
5. **ClassificationAgent** — HCM probability + predicted label.

All intermediate artefacts are optionally persisted to disk for
reproducibility and clinical audit.

Usage
-----
    from pathlib import Path
    from omegaconf import OmegaConf
    from src.agents.coordinator_agent import CoordinatorAgent

    cfg   = OmegaConf.load("src/config/base.yaml")
    coord = CoordinatorAgent(cfg)

    result = coord.run(Path("data/raw/Sick/Directory_17"))
    print(result)
    # {
    #   "patient_dir":  "...",
    #   "series_used":  "...",
    #   "prediction":   "Sick",
    #   "confidence":   0.91,
    #   "probabilities": {"Normal": 0.09, "Sick": 0.91},
    #   "seg_mask_path": "...",
    #   "status":       "ok",
    # }
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from src.agents.classification_agent import ClassificationAgent
from src.agents.ingestion_agent import IngestionAgent
from src.agents.preprocessing_agent import PreprocessingAgent
from src.agents.router_agent import RouterAgent
from src.agents.segmentation_agent import SegmentationAgent

logger = logging.getLogger(__name__)


class CoordinatorAgent:
    """Orchestrates the full HCM diagnostic pipeline.

    Args:
        cfg:    Root OmegaConf config.
        device: PyTorch device; auto-detects if ``None``.

    Attributes:
        router      (RouterAgent):         Series selection.
        ingestor    (IngestionAgent):      DICOM reader / stacker.
        preprocessor (PreprocessingAgent): Normalisation + resize.
        segmenter   (SegmentationAgent | None): Optional mask prediction.
        classifier  (ClassificationAgent): HCM vs Normal diagnosis.
    """

    def __init__(
        self,
        cfg: DictConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.router = RouterAgent(
            min_slices=int(cfg.router.min_slices),
            priority_keywords=list(cfg.router.priority_keywords),
        )
        self.ingestor     = IngestionAgent(rescale=bool(cfg.preprocessing.rescale_dicom))
        self.preprocessor = PreprocessingAgent(cfg)
        self.classifier   = ClassificationAgent(cfg, device=self.device)

        self.segmenter: Optional[SegmentationAgent] = (
            SegmentationAgent(cfg, device=self.device)
            if cfg.segmentation.enabled
            else None
        )

        logger.info(
            "CoordinatorAgent ready | device=%s | segmentation=%s",
            self.device,
            cfg.segmentation.enabled,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        patient_dir: Path,
        save_nifti: bool = False,
        save_seg_mask: bool = False,
        output_dir: Optional[Path] = None,
    ) -> Dict:
        """Run the full diagnostic pipeline on a single patient directory.

        Args:
            patient_dir:   Path to the patient folder (``Directory_X``).
            save_nifti:    Persist the pre-processed volume as ``.nii.gz``.
            save_seg_mask: Persist the segmentation mask as ``.nii.gz``
                           (only when segmentation is enabled).
            output_dir:    Root directory for saving artefacts.  Defaults to
                           ``cfg.paths.data_processed``.

        Returns:
            Result dict with keys:
            ``patient_dir``, ``series_used``, ``prediction``,
            ``confidence``, ``probabilities``, ``seg_mask_path``,
            ``elapsed_s``, ``status``, ``error_msg``.
        """
        t0 = time.monotonic()
        result: Dict = {
            "patient_dir":    str(patient_dir),
            "series_used":    None,
            "prediction":     None,
            "confidence":     None,
            "probabilities":  {},
            "seg_mask_path":  None,
            "elapsed_s":      None,
            "status":         "error",
            "error_msg":      "",
        }

        try:
            # ----------------------------------------------------------
            # 1. Route
            # ----------------------------------------------------------
            series_path = self.router.select_series(patient_dir)
            if series_path is None:
                raise ValueError(f"No valid series found in {patient_dir}")
            result["series_used"] = str(series_path)

            # ----------------------------------------------------------
            # 2. Ingest
            # ----------------------------------------------------------
            volume_np, affine, meta = self.ingestor.load_series(series_path)

            # ----------------------------------------------------------
            # 3. Preprocess → (1, D, H, W)
            # ----------------------------------------------------------
            tensor = self.preprocessor.preprocess(volume_np, augment=False)
            volume_tensor = tensor.unsqueeze(0)  # → (1, 1, D, H, W) for batch dim

            if save_nifti and output_dir is not None:
                nifti_path = (
                    Path(output_dir)
                    / f"{patient_dir.parent.name}_{patient_dir.name}_preprocessed.nii.gz"
                )
                import nibabel as nib
                nib.save(
                    nib.Nifti1Image(tensor.squeeze(0).numpy(), affine),
                    str(nifti_path),
                )
                logger.debug("Saved preprocessed NIfTI: %s", nifti_path)

            # ----------------------------------------------------------
            # 4. Segment (optional)
            # ----------------------------------------------------------
            seg_mask_path: Optional[Path] = None
            if self.segmenter is not None:
                probs_seg, mask = self.segmenter.predict(volume_tensor)

                if save_seg_mask and output_dir is not None:
                    seg_mask_path = (
                        Path(output_dir)
                        / f"{patient_dir.parent.name}_{patient_dir.name}_seg.nii.gz"
                    )
                    import nibabel as nib
                    nib.save(
                        nib.Nifti1Image(mask[0].numpy().astype(np.uint8), affine),
                        str(seg_mask_path),
                    )
                result["seg_mask_path"] = (
                    str(seg_mask_path) if seg_mask_path else None
                )

            # ----------------------------------------------------------
            # 5. Classify
            # ----------------------------------------------------------
            probs, pred_label = self.classifier.predict(volume_tensor)
            probs_np = probs[0].numpy()                # (num_classes,)

            result["prediction"]    = self.classifier.class_names[pred_label]
            result["confidence"]    = float(probs_np[pred_label])
            result["probabilities"] = {
                name: float(probs_np[idx])
                for idx, name in self.classifier.class_names.items()
            }
            result["status"] = "ok"

        except Exception as exc:  # noqa: BLE001
            logger.error("Pipeline failed for %s: %s", patient_dir, exc)
            result["status"]    = "error"
            result["error_msg"] = str(exc)

        result["elapsed_s"] = round(time.monotonic() - t0, 3)
        return result

    def run_batch(
        self,
        patient_dirs: List[Path],
        **kwargs,
    ) -> List[Dict]:
        """Run the pipeline on a list of patient directories.

        Args:
            patient_dirs: List of patient directory paths.
            **kwargs:     Additional keyword arguments forwarded to :meth:`run`.

        Returns:
            List of result dicts, one per patient.
        """
        results = []
        n = len(patient_dirs)
        for i, pdir in enumerate(patient_dirs, 1):
            logger.info("[%d/%d] Processing %s …", i, n, pdir.name)
            results.append(self.run(pdir, **kwargs))
        return results

    # ------------------------------------------------------------------
    # Pipeline introspection
    # ------------------------------------------------------------------

    def describe_pipeline(self) -> str:
        """Return a human-readable summary of the active pipeline.

        Returns:
            Formatted multi-line string.
        """
        lines = [
            "=" * 55,
            "  HCM Multi-Agent Diagnostic Pipeline",
            "=" * 55,
            f"  Device              : {self.device}",
            f"  Router keywords     : {self.cfg.router.priority_keywords}",
            f"  Min slices          : {self.cfg.router.min_slices}",
            f"  Pre-processing      : {self.cfg.preprocessing.normalization} norm, "
            f"resize to (D={self.cfg.preprocessing.target_shape.depth}, "
            f"H={self.cfg.preprocessing.target_shape.height}, "
            f"W={self.cfg.preprocessing.target_shape.width})",
            f"  Segmentation        : {'enabled' if self.segmenter else 'disabled'}",
            f"  Classification      : {self.cfg.classification.model}",
            "=" * 55,
        ]
        return "\n".join(lines)
