"""Agent 2 — quantification. Pure arithmetic. No model, no training, no randomness.

Two subjects with the same mask get the same numbers, forever. That is the property
that lets `factcheck.py` call a number in the report "unsupported": there is exactly
one place a number can legitimately come from, and it is this file.

Two things here are corrections to the proposal, not implementation detail:

  1. DENSITY.  1.05 g/mL is Kawel-Boehm et al., the SCMR 2025 reference-ranges paper.
     The proposal cites Leiner (2019), which does not establish it. The constant is
     right; the citation is not. It lives in cfg.quantification.myocardial_density_g_per_ml.

  2. HEART-FAILURE CUT-POINTS.  The proposal writes (<40 = reduced, 40-50 = mildly
     reduced, >50 = preserved). That rule is undefined at LVEF == 40.0 and at
     LVEF == 50.0 — the two values a boundary-aware system will see most often, and
     the two it must get right. ESC 2021 and AHA/ACC/HFSA 2022 both close the
     intervals downward:

         LVEF <= 40          -> HFrEF
         40 <  LVEF <  50    -> HFmrEF
         LVEF >= 50          -> HFpEF

     Read from cfg. Tested at 40.0 / 40.1 / 49.9 / 50.0 in tests/test_quantify.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .checks import n_components, stray_fraction
from .config import Config, artifacts
from .data import load_all
from .types import LV, MYO, RV, GateResult, Measurements

log = logging.getLogger("cmr.quantify")

__all__ = ["measure", "gate", "run_groundtruth", "hf_category", "is_near_boundary"]


# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic
# ─────────────────────────────────────────────────────────────────────────────
def _volume_ml(mask: np.ndarray, label: int, voxel_ml: float) -> float:
    return float(np.count_nonzero(mask == label) * voxel_ml)


def _ef_pct(edv_ml: float, esv_ml: float) -> float:
    """Ejection fraction. An empty cavity gives 0.0, not a ZeroDivisionError — the
    gate is what rejects it, and the gate needs a number to reject."""
    return 100.0 * (edv_ml - esv_ml) / edv_ml if edv_ml > 0 else 0.0


def hf_category(lvef_pct: float, cfg: Config) -> str:
    """The corrected guideline rule. Both boundaries are closed; nothing falls through."""
    t = cfg.quantification.hf_thresholds
    if lvef_pct <= t.hfref_max:  # 40.0 -> HFrEF
        return "HFrEF"
    if lvef_pct >= t.hfpef_min:  # 50.0 -> HFpEF
        return "HFpEF"
    return "HFmrEF"


def is_near_boundary(lvef_pct: float, cfg: Config) -> bool:
    """Within `boundary_margin_pct` of a cut-point -> the H2 feedback loop fires."""
    t = cfg.quantification.hf_thresholds
    margin = cfg.quantification.boundary_margin_pct
    return any(abs(lvef_pct - cut) <= margin for cut in (t.hfref_max, t.hfpef_min))


def measure(
    mask_ed: np.ndarray,
    mask_es: np.ndarray,
    spacing_mm: tuple[float, float, float],
    subject_id: str,
    cfg: Config,
    source: str = "predicted",
) -> Measurements:
    """Simpson-free volumetry: count voxels, multiply by voxel volume. For a
    contiguous short-axis stack this IS the disc-summation integral, exactly."""
    voxel_ml = float(np.prod(spacing_mm)) / 1000.0

    edv = _volume_ml(mask_ed, LV, voxel_ml)
    esv = _volume_ml(mask_es, LV, voxel_ml)
    lvef = _ef_pct(edv, esv)

    rv_edv = _volume_ml(mask_ed, RV, voxel_ml)
    rv_esv = _volume_ml(mask_es, RV, voxel_ml)

    return Measurements(
        subject_id=subject_id,
        edv_ml=edv,
        esv_ml=esv,
        sv_ml=edv - esv,
        lvef_pct=lvef,
        lv_mass_g=_volume_ml(mask_ed, MYO, voxel_ml)
        * cfg.quantification.myocardial_density_g_per_ml,
        rv_edv_ml=rv_edv,
        rv_esv_ml=rv_esv,
        rv_ef_pct=_ef_pct(rv_edv, rv_esv),
        hf_category=hf_category(lvef, cfg),
        near_boundary=is_near_boundary(lvef, cfg),
        source=source,
    )


# ─────────────────────────────────────────────────────────────────────────────
# The plausibility gate (C10 / H3). Rule-based. Nothing is learned here.
# ─────────────────────────────────────────────────────────────────────────────
def _lv_slice_gap(mask: np.ndarray) -> bool:
    """A slice with no LV, sandwiched between two slices that have LV.

    The heart is one connected object from base to apex. A hole in the middle of the
    stack is a segmentation failure, never anatomy. (Missing slices at either END are
    normal — that is just where the stack stops — so we only look inside the span.)
    """
    present = [bool((mask[:, :, z] == LV).any()) for z in range(mask.shape[2])]
    if not any(present):
        return False  # an empty LV is caught by the LVEF range check, not here
    first, last = present.index(True), len(present) - 1 - present[::-1].index(True)
    return not all(present[first : last + 1])


def gate(m: Measurements, mask_ed: np.ndarray, mask_es: np.ndarray, cfg: Config) -> GateResult:
    """Refuse rather than report a confident diagnosis on a broken mask.

    CALIBRATION — this is the methodological point, and it is what makes H3 defensible.
    A plausibility gate is only meaningful if it accepts what expert annotation produces.
    So the thresholds are not guessed: they are set on the GROUND TRUTH of all three
    datasets so that >= 99% of expert annotations pass. Anything the gate then rejects is,
    by construction, less anatomically plausible than an expert's own annotation.

    Measured on ground truth (specks < min_component_px ignored):
        per-slice components   LV 1, RV 1, Myo <= 2   -> threshold 2      (100% GT pass)
        3-D stray fraction     99th pct <= 0.10       -> threshold 0.15   (>99% GT pass)

    An earlier version counted 1-2 px annotation islands as components and rejected 65%
    of the ground truth. A gate that refuses the ground truth measures nothing.

    Violation names are machine-readable on purpose: they are what the orchestrator routes
    on, and what the H3 ablation counts.
    """
    g = cfg.quantification.gate
    min_px = int(g.get("min_component_px", 10))
    max_stray = float(g.get("max_stray_fraction", 0.15))
    v: list[str] = []

    if not (g.lvef_min <= m.lvef_pct <= g.lvef_max):
        v.append("lvef_out_of_range")
    if g.require_esv_lt_edv and m.esv_ml >= m.edv_ml:
        v.append("esv_ge_edv")

    for phase, mask in (("ED", mask_ed), ("ES", mask_es)):
        for label, name in ((LV, "LV"), (MYO, "Myo"), (RV, "RV")):
            if n_components(mask, label, min_px) > g.max_components_per_slice:
                v.append(f"multiple_components_{name}_{phase}")
            if stray_fraction(mask, label) > max_stray:
                v.append(f"disconnected_{name}_{phase}")
        if _lv_slice_gap(mask):
            v.append(f"lv_slice_discontinuity_{phase}")

    return GateResult(passed=not v, violations=v)


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth — the denominator of every MAE in the thesis
# ─────────────────────────────────────────────────────────────────────────────
def run_groundtruth(cfg: Config) -> pd.DataFrame:
    """Measure every ground-truth subject once, cache to artifacts/gt_measurements.parquet.

    Metadata (vendor / centre / pathology) is taken from the live Subject, NOT from
    manifest.parquet — the manifest on disk predates data.normalise_vendor and still
    carries 'SIEMENS' and 'GE MEDICAL SYSTEMS' as distinct vendors, which would split
    every RQ3 stratum in two.
    """
    rows = []
    for s in load_all(cfg):
        if not s.has_gt:
            continue
        m = measure(s.gt_ed, s.gt_es, s.spacing_mm, s.subject_id, cfg, source="groundtruth")
        gr = gate(m, s.gt_ed, s.gt_es, cfg)
        rows.append(
            m.model_dump()
            | {
                "dataset": s.dataset,
                "split": s.split,
                "pathology": s.pathology,
                "vendor": s.vendor,
                "centre": s.centre,
                "field_strength": s.field_strength,
                "spacing_mm": str(s.spacing_mm),
                "voxel_ml": s.voxel_ml,
                "gate_passed": gr.passed,
                "gate_violations": ";".join(gr.violations),
            }
        )
        if len(rows) % 100 == 0:
            log.info("  measured %d subjects", len(rows))

    df = pd.DataFrame(rows)
    out = artifacts(cfg) / "gt_measurements.parquet"
    df.to_parquet(out, index=False)
    log.info("ground truth: %d subjects -> %s", len(df), out)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# `python -m cmr.quantify` — the week-2 deliverable report. print() is deliberate here.
# ─────────────────────────────────────────────────────────────────────────────
def _report(df: pd.DataFrame, cfg: Config) -> None:  # noqa: C901
    band = tuple(cfg.evaluation.boundary_band)
    pd.set_option("display.width", 120)

    print(f"\n{'=' * 78}\nGROUND-TRUTH MEASUREMENTS — {len(df)} subjects\n{'=' * 78}")
    print(df.groupby("dataset").size().to_string(), "\n")

    print("--- SANITY CHECK: DCM must eject less than NOR ---")
    for ds in ["ACDC", "MnMs"]:
        sub = df[df.dataset == ds]
        for path in ["DCM", "NOR"]:
            c = sub[sub.pathology == path]
            if len(c):
                print(f"  {ds:6s} {path:4s} n={len(c):3d}  mean LVEF = {c.lvef_pct.mean():5.2f}%")
    dcm = df[df.pathology == "DCM"].lvef_pct
    nor = df[df.pathology == "NOR"].lvef_pct
    ok = dcm.mean() < nor.mean()
    print(f"  POOLED DCM {dcm.mean():.2f}%  vs  NOR {nor.mean():.2f}%   -> {'PASS' if ok else 'FAIL'}")

    print(f"\n--- H2 SIZING: GT LVEF in [{band[0]}, {band[1]}]% ---")
    in_band = df[df.lvef_pct.between(*band)]
    for ds, n in in_band.groupby("dataset").size().items():
        tot = int((df.dataset == ds).sum())
        print(f"  {ds:6s} {n:4d} / {tot:4d}  ({100 * n / tot:4.1f}%)")
    print(f"  TOTAL  {len(in_band):4d} / {len(df):4d}  ({100 * len(in_band) / len(df):4.1f}%)")
    print(f"  of which near a cut-point (+/-{cfg.quantification.boundary_margin_pct}%): "
          f"{int(df.near_boundary.sum())}")

    print("\n--- DATA QUALITY: ESV >= EDV ---")
    bad = df[df.esv_ml >= df.edv_ml]
    print(f"  LV: {len(bad)} subject(s)")
    for _, r in bad.iterrows():
        print(f"    {r.dataset:6s} {r.subject_id:12s} EDV={r.edv_ml:7.2f} ESV={r.esv_ml:7.2f}")
    bad_rv = df[df.rv_esv_ml >= df.rv_edv_ml]
    print(f"  RV: {len(bad_rv)} subject(s)")
    for _, r in bad_rv.iterrows():
        print(f"    {r.dataset:6s} {r.subject_id:12s} "
              f"RV EDV={r.rv_edv_ml:7.2f} RV ESV={r.rv_esv_ml:7.2f}")

    print("\n--- GATE on GROUND TRUTH (these masks are the reference standard) ---")
    print(f"  passed: {int(df.gate_passed.sum())} / {len(df)}")
    viol = df[~df.gate_passed].gate_violations.str.split(";").explode().value_counts()
    print(viol.to_string() if len(viol) else "  (none)")

    print("\n--- LVEF by pathology ---")
    g = df.groupby(["dataset", "pathology"]).lvef_pct.agg(["count", "mean", "std", "min", "max"])
    print(g.round(1).to_string())

    print("\n--- HF category ---")
    print(pd.crosstab(df.dataset, df.hf_category).to_string())
    print("\n  overall:", df.hf_category.value_counts().to_dict())
    print()


def main() -> None:
    from . import config

    cfg = config.load()
    out = artifacts(cfg) / "gt_measurements.parquet"
    df = pd.read_parquet(out) if out.exists() else run_groundtruth(cfg)
    _report(df, cfg)


if __name__ == "__main__":
    main()
