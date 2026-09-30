"""The data spine. Three datasets, three completely different layouts, one record.

Every other module reads this module's output. Get it wrong and every number in the
thesis is wrong-but-plausible, so this file does two things and does them carefully:

  1. It CANONICALISES LABELS.  ACDC is 1=RV / 3=LV; M&Ms and M&Ms-2 are 1=LV / 3=RV.
     ACDC is remapped on read. After this module, 1=LV / 2=Myo / 3=RV. Everywhere.

  2. It reads ED/ES from the dataset, never guesses them. Three datasets, three
     mechanisms — which is the entire reason a loader abstraction exists at all:

        ACDC    Info.cfg           ED: / ES:  (1-indexed frame numbers)
        M&Ms    metadata CSV       ED / ES columns (0-indexed). The GT is a 4-D volume
                                   the same shape as the image, non-zero ONLY at those
                                   two frames. The indices are NOT in the file.
        M&Ms-2  pre-extracted      {SID}_SA_ED.nii / _ES.nii — nothing to look up.

     The proposal (S3.3.3 iv) proposes detecting ED/ES by "time gradient analysis".
     That is unnecessary work that injects unattributable error. We do not do it.

Counts, verified: ACDC 150 (100 train + 50 test, both with GT) + M&Ms 345 + M&Ms-2 360 = 855.
The proposal's "375 subjects / 6 centres" for M&Ms describes the full challenge cohort;
the open release is 345 across 5 centres.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .checks import assert_canonical, canonicalise
from .config import Config, artifacts
from .types import Subject

log = logging.getLogger("cmr.data")


def _load(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    zooms = img.header.get_zooms()[:3]
    return data, (float(zooms[0]), float(zooms[1]), float(zooms[2]))


# M&Ms and M&Ms-2 spell the same four vendors seven different ways. Cross-dataset
# vendor stratification (which is the whole of RQ3) is meaningless without this.
_VENDOR_CANON = {
    "siemens": "Siemens",
    "philips": "Philips",
    "philips medical systems": "Philips",
    "ge": "GE",
    "ge medical systems": "GE",
    "canon": "Canon",
}


def normalise_vendor(v: str | None) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return _VENDOR_CANON.get(str(v).strip().lower(), str(v).strip())


# ─────────────────────────────────────────────────────────────────────────────
# ACDC — 150 subjects. Labels are INVERTED relative to the other two.
# ─────────────────────────────────────────────────────────────────────────────
def _read_info_cfg(p: Path) -> dict[str, str]:
    out = {}
    for line in p.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def load_acdc(cfg: Config, splits=("training", "testing")) -> Iterator[Subject]:
    root = Path(cfg.paths.acdc)
    for split in splits:
        for pdir in sorted((root / split).glob("patient*")):
            info = _read_info_cfg(pdir / "Info.cfg")
            ed, es = int(info["ED"]), int(info["ES"])  # 1-indexed frame numbers
            sid = pdir.name

            img_ed, spacing = _load(pdir / f"{sid}_frame{ed:02d}.nii.gz")
            img_es, _ = _load(pdir / f"{sid}_frame{es:02d}.nii.gz")
            gt_ed, _ = _load(pdir / f"{sid}_frame{ed:02d}_gt.nii.gz")
            gt_es, _ = _load(pdir / f"{sid}_frame{es:02d}_gt.nii.gz")

            # THE REMAP. Everything downstream depends on this one line pair.
            gt_ed = canonicalise(gt_ed.astype(np.uint8), "acdc")
            gt_es = canonicalise(gt_es.astype(np.uint8), "acdc")

            yield Subject(
                subject_id=sid,
                dataset="ACDC",
                split="train" if split == "training" else "test",
                pathology=info["Group"],
                ed_idx=ed - 1,  # store 0-indexed, as everywhere else
                es_idx=es - 1,
                spacing_mm=spacing,
                img_ed=img_ed.astype(np.float32),
                img_es=img_es.astype(np.float32),
                gt_ed=gt_ed,
                gt_es=gt_es,
            )


# ─────────────────────────────────────────────────────────────────────────────
# M&Ms — 345 subjects, 4 vendors, 5 centres. GT is 4-D; frame indices live in a CSV.
# ─────────────────────────────────────────────────────────────────────────────
_MNMS_SPLITS = {
    "Training/Labeled": "train",
    "Training/Unlabeled": "train_unlabeled",
    "Validation": "val",
    "Testing": "test",
}


def _mnms_meta(cfg: Config) -> pd.DataFrame:
    root = Path(cfg.paths.mnms)
    csv = next(root.glob("*Dataset_information*.csv"))
    df = pd.read_csv(csv)
    return df.set_index("External code")


def load_mnms(cfg: Config, include_unlabeled: bool = False) -> Iterator[Subject]:
    root = Path(cfg.paths.mnms)
    meta = _mnms_meta(cfg)

    for rel, split in _MNMS_SPLITS.items():
        if split == "train_unlabeled" and not include_unlabeled:
            continue  # 25 subjects, deliberately unused. Stated explicitly, not ignored.
        d = root / rel
        if not d.is_dir():
            continue
        for sdir in sorted(p for p in d.iterdir() if p.is_dir()):
            sid = sdir.name
            if sid not in meta.index:
                log.warning("M&Ms subject %s not in metadata CSV — skipped", sid)
                continue
            row = meta.loc[sid]
            ed, es = int(row["ED"]), int(row["ES"])  # 0-indexed. M4P7Q6 has ED=2, not 0.

            img4, spacing = _load(sdir / f"{sid}_sa.nii.gz")  # (X, Y, Z, T)
            gt_path = sdir / f"{sid}_sa_gt.nii.gz"
            if gt_path.exists():
                gt4, _ = _load(gt_path)  # same shape as image; non-zero only at ED/ES
                gt_ed = gt4[..., ed].astype(np.uint8)
                gt_es = gt4[..., es].astype(np.uint8)
            else:
                gt_ed = gt_es = None

            yield Subject(
                subject_id=sid,
                dataset="MnMs",
                split=split,
                pathology=str(row["Pathology"]),
                ed_idx=ed,
                es_idx=es,
                spacing_mm=spacing,
                img_ed=img4[..., ed].astype(np.float32),
                img_es=img4[..., es].astype(np.float32),
                gt_ed=gt_ed,  # already canonical — no remap
                gt_es=gt_es,
                vendor=normalise_vendor(row["VendorName"]),
                centre=str(row["Centre"]),
            )


# ─────────────────────────────────────────────────────────────────────────────
# M&Ms-2 — 360 subjects. ED/ES pre-extracted. Short-axis AND long-axis.
# ─────────────────────────────────────────────────────────────────────────────
def _mnms2_meta(cfg: Config) -> pd.DataFrame:
    csv = Path(cfg.paths.mnms2) / "dataset_information.csv"
    df = pd.read_csv(csv, low_memory=False)
    # The file is an Excel export padded to 1,048,575 rows with trailing commas.
    # Only 360 rows carry data. Any naive len(df) is wrong.
    df = df.dropna(subset=["SUBJECT_CODE"])
    df["SUBJECT_CODE"] = df["SUBJECT_CODE"].astype(int).astype(str).str.zfill(3)
    return df.set_index("SUBJECT_CODE")


def load_mnms2(cfg: Config, view: str = "SA") -> Iterator[Subject]:
    root = Path(cfg.paths.mnms2) / "dataset"
    meta = _mnms2_meta(cfg)

    for sdir in sorted(p for p in root.iterdir() if p.is_dir()):
        sid = sdir.name
        row = meta.loc[sid] if sid in meta.index else None

        img_ed, spacing = _load(sdir / f"{sid}_{view}_ED.nii")
        img_es, _ = _load(sdir / f"{sid}_{view}_ES.nii")
        gt_ed, _ = _load(sdir / f"{sid}_{view}_ED_gt.nii")
        gt_es, _ = _load(sdir / f"{sid}_{view}_ES_gt.nii")

        # Long-axis volumes are (X, Y, 1); keep the 3-D shape for a uniform interface.
        if img_ed.ndim == 2:
            img_ed, img_es = img_ed[..., None], img_es[..., None]
            gt_ed, gt_es = gt_ed[..., None], gt_es[..., None]

        yield Subject(
            subject_id=sid,
            dataset="MnMs2",
            split="all",
            pathology=str(row["DISEASE"]) if row is not None else "UNKNOWN",
            ed_idx=0,  # pre-extracted; there is nothing to index into
            es_idx=0,
            spacing_mm=spacing,
            img_ed=img_ed.astype(np.float32),
            img_es=img_es.astype(np.float32),
            gt_ed=gt_ed.astype(np.uint8),  # already canonical
            gt_es=gt_es.astype(np.uint8),
            vendor=normalise_vendor(row["VENDOR"]) if row is not None else None,
            field_strength=float(row["FIELD"]) if row is not None else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
def load_all(cfg: Config, datasets: tuple[str, ...] = ("ACDC", "MnMs", "MnMs2")) -> Iterator[Subject]:
    if "ACDC" in datasets:
        yield from load_acdc(cfg)
    if "MnMs" in datasets:
        yield from load_mnms(cfg)
    if "MnMs2" in datasets:
        yield from load_mnms2(cfg)


def build_manifest(cfg: Config, verify: bool = True) -> pd.DataFrame:
    """Enumerate all 855 subjects, verify the canonical label order on every one,
    and write artifacts/manifest.parquet.

    `verify=True` runs the concentricity assertion over 1,710 ground-truth volumes.
    That is the whole point of this function; do not turn it off to make it faster.
    """
    rows, bad = [], []
    for s in load_all(cfg):
        if verify and s.has_gt:
            for phase, gt in (("ED", s.gt_ed), ("ES", s.gt_es)):
                try:
                    assert_canonical(gt, f"{s.dataset}/{s.subject_id}/{phase}")
                except AssertionError as e:
                    bad.append(str(e))
        rows.append(
            {
                "subject_id": s.subject_id,
                "dataset": s.dataset,
                "split": s.split,
                "pathology": s.pathology,
                "vendor": s.vendor,
                "centre": s.centre,
                "field_strength": s.field_strength,
                "ed_idx": s.ed_idx,
                "es_idx": s.es_idx,
                "sx": s.spacing_mm[0],
                "sy": s.spacing_mm[1],
                "sz": s.spacing_mm[2],
                "shape": str(s.img_ed.shape),
                "has_gt": s.has_gt,
            }
        )

    if bad:
        raise AssertionError(
            f"{len(bad)} volume(s) failed the canonical label check. "
            f"This means a remap is missing and every downstream number would be "
            f"silently wrong. First failure:\n  {bad[0]}"
        )

    df = pd.DataFrame(rows)
    out = artifacts(cfg) / "manifest.parquet"
    df.to_parquet(out, index=False)
    log.info("manifest: %d subjects -> %s", len(df), out)
    log.info("  by dataset: %s", df.dataset.value_counts().to_dict())
    return df
