"""ACDC / M&Ms -> nnU-Net v2 raw format, in CANONICAL label order.

THE POINT OF THIS FILE: nnU-Net is trained on whatever integers we hand it and will
happily emit them back. ACDC natively stores 1=RV / 3=LV. If we dump ACDC's native
labels into nnU-Net, its predictions come out in ACDC order, every LV/RV Dice is
swapped, every LVEF is computed from the RV, and nothing crashes. So we do not touch
the raw files: we go through `cmr.data.load_acdc()`, which has already remapped, and
then we `assert_canonical` every single label volume on the way out. Belt and braces,
because the failure mode is silent.

`dataset.json` declares the SAME mapping (1=LV, 2=Myo, 3=RV), so nnU-Net's outputs
are canonical everywhere and drop straight into cmr's evaluation with no remap.

One subject yields two cases (ED and ES) — they are independent 3-D volumes as far as
segmentation is concerned.

    python -m baselines.nnunet_convert --dataset acdc --dataset-id 27
    python -m baselines.nnunet_convert --dataset mnms --dataset-id 28
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import nibabel as nib
import numpy as np

from cmr import config
from cmr.checks import assert_canonical
from cmr.data import load_acdc, load_mnms
from cmr.types import Subject

log = logging.getLogger("baselines.nnunet_convert")

# Declared in dataset.json AND asserted on every volume. These are the same numbers
# as cmr.types.LV/MYO/RV — if they ever diverge, the assertion below fails loudly.
LABELS = {"background": 0, "LV": 1, "Myo": 2, "RV": 3}

# nnU-Net splits on directory, not on a column, so the split has to be baked in here.
_ACDC_TS = "test"  # ACDC's `testing` split -> imagesTs. It HAS ground truth (rare, useful).
_MNMS_TS = ("val", "test")


def _phases(s: Subject) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    yield f"{s.subject_id}_ED", s.img_ed, s.gt_ed
    yield f"{s.subject_id}_ES", s.img_es, s.gt_es


def _affine(spacing: tuple[float, float, float]) -> np.ndarray:
    """A diagonal RAS affine carrying the true voxel spacing.

    nnU-Net self-configures its target spacing and patch size from the headers, so the
    spacing MUST be right; the rotation must merely be identical between image and
    label, which it trivially is here. We never map back to scanner space — evaluation
    happens in array space against the same loader — so a diagonal affine is sufficient
    and avoids importing per-file headers we do not otherwise need.
    """
    a = np.eye(4, dtype=np.float64)
    a[:3, :3] = np.diag(spacing)
    return a


def _write(arr: np.ndarray, spacing: tuple[float, float, float], path: Path, dtype) -> None:
    img = nib.Nifti1Image(arr.astype(dtype), _affine(spacing))
    img.header.set_zooms(spacing)
    nib.save(img, str(path))


def convert(
    cfg: config.Config,
    dataset: str,
    dataset_id: int,
    raw_root: Path,
    limit: int | None = None,
) -> Path:
    name = {"acdc": "ACDC", "mnms": "MnMs"}[dataset]
    out = raw_root / f"Dataset{dataset_id:03d}_{name}"
    for d in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (out / d).mkdir(parents=True, exist_ok=True)

    subjects = load_acdc(cfg) if dataset == "acdc" else load_mnms(cfg)
    n_tr = n_ts = 0

    for i, s in enumerate(subjects):
        if limit and i >= limit:
            break
        if not s.has_gt:
            log.warning("%s has no GT — skipped", s.subject_id)
            continue
        is_test = s.split == _ACDC_TS if dataset == "acdc" else s.split in _MNMS_TS
        img_dir, lab_dir = ("imagesTs", "labelsTs") if is_test else ("imagesTr", "labelsTr")

        for case, img, gt in _phases(s):
            # THE GUARD. cmr.data already remapped ACDC; this proves it, per volume.
            assert_canonical(gt, f"{dataset}/{case}")
            _write(img, s.spacing_mm, out / img_dir / f"{case}_0000.nii.gz", np.float32)
            _write(gt, s.spacing_mm, out / lab_dir / f"{case}.nii.gz", np.uint8)
            n_ts += is_test
            n_tr += not is_test

    meta = {
        "channel_names": {"0": "cineMRI"},
        "labels": LABELS,  # canonical, and identical to what we just asserted
        "numTraining": n_tr,
        "file_ending": ".nii.gz",
        "description": (
            f"{name} short-axis cine, ED+ES as independent 3-D cases. "
            "Labels are CANONICAL (1=LV, 2=Myo, 3=RV); ACDC's native 1=RV/3=LV order "
            "was remapped by cmr.data.load_acdc and asserted per volume."
        ),
    }
    (out / "dataset.json").write_text(json.dumps(meta, indent=2))
    log.info("%s: %d training + %d test cases -> %s", name, n_tr, n_ts, out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("acdc", "mnms"), default="acdc")
    p.add_argument("--dataset-id", type=int, default=27)
    p.add_argument("--config-file", default=None)
    p.add_argument("--raw", default=os.environ.get("nnUNet_raw"), help="default: $nnUNet_raw")
    p.add_argument("--limit", type=int, default=None, help="first N subjects (smoke tests)")
    a = p.parse_args()

    if not a.raw:
        p.error("set nnUNet_raw or pass --raw")
    convert(config.load(a.config_file), a.dataset, a.dataset_id, Path(a.raw), a.limit)


if __name__ == "__main__":
    main()
