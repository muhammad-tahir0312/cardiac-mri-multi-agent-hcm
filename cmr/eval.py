"""Evaluation. Everything the panel will ask for, computed once, written to JSON.

Read the split here honestly:

  * Dice / HD95 / Bland-Altman / ICC / Pearson r are TABLE STAKES. BAAI's Cardiac
    Agent already reports Bland-Altman and Pearson r. They are here because their
    absence would be noticed, not because they distinguish anything.

  * `hf_category_accuracy(band=(35,55))` and `pathology_top1` are the answer to panel
    comment C13 ("nothing in the evaluation measures diagnostic correctness"). They
    are the metrics that carry the thesis.

  * `stratify(..., by="vendor" | "centre")` is RQ3 — the cross-vendor generalisation
    gap. It is the only place the zero-shot claim can be falsified.

PATHOLOGY LABELS DO NOT LINE UP ACROSS THE THREE DATASETS and no amount of wishing
makes them. ACDC has 5 classes, M&Ms 9, M&Ms-2 8, and the overlap is partial. The
mapping below is explicit, and `pathology_top1` reports the comparable subset
separately from the dataset-specific one. Nothing is quietly merged.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, artifacts
from .data import normalise_vendor
from .types import LABEL_NAMES, Fidelity, Report

log = logging.getLogger("cmr.eval")

__all__ = [
    "segmentation_metrics",
    "agreement",
    "hf_category_accuracy",
    "pathology_top1",
    "fidelity_summary",
    "bootstrap_ci",
    "stratify",
    "evaluate_run",
    "PATHOLOGY_MAP",
]


# ─────────────────────────────────────────────────────────────────────────────
# The pathology vocabulary. (dataset, native label) -> canonical code.
# ─────────────────────────────────────────────────────────────────────────────
PATHOLOGY_MAP: dict[tuple[str, str], str] = {
    # ACDC — 5 classes, criteria published in Bernard et al. 2018
    ("ACDC", "NOR"): "NOR",
    ("ACDC", "DCM"): "DCM",
    ("ACDC", "HCM"): "HCM",
    ("ACDC", "MINF"): "MINF",
    ("ACDC", "RV"): "RV_ABN",
    # M&Ms — 9 labels in the released cohort (the challenge paper names 4)
    ("MnMs", "NOR"): "NOR",
    ("MnMs", "DCM"): "DCM",
    ("MnMs", "HCM"): "HCM",
    ("MnMs", "HHD"): "HHD",
    ("MnMs", "ARV"): "ARV",
    ("MnMs", "IHD"): "IHD",
    ("MnMs", "LVNC"): "LVNC",
    ("MnMs", "AHS"): "OTHER",
    ("MnMs", "Other"): "OTHER",
    # M&Ms-2 — 8 classes. Note "LV"/"RV" here mean DILATED LV / DILATED RV.
    ("MnMs2", "NOR"): "NOR",
    ("MnMs2", "HCM"): "HCM",
    ("MnMs2", "LV"): "DLV",
    ("MnMs2", "RV"): "RV_ABN",
    ("MnMs2", "ARR"): "ARR",
    ("MnMs2", "FALL"): "TOF",
    ("MnMs2", "CIA"): "CIA",
    ("MnMs2", "TRI"): "TRI",
}

# Which canonical codes mean the same thing in more than one dataset, and which do not.
CANONICAL_INFO: dict[str, dict[str, Any]] = {
    "NOR": {"datasets": ["ACDC", "MnMs", "MnMs2"], "comparable": True,
            "note": "normal; consistent across all three"},
    "HCM": {"datasets": ["ACDC", "MnMs", "MnMs2"], "comparable": True,
            "note": "hypertrophic cardiomyopathy; consistent across all three"},
    "DCM": {"datasets": ["ACDC", "MnMs"], "comparable": True,
            "note": "dilated cardiomyopathy; ACDC defines it as LVEDV>100 mL/m2 AND LVEF<40"},
    "RV_ABN": {"datasets": ["ACDC", "MnMs2"], "comparable": True,
               "note": "ACDC 'abnormal RV' (RVEDV>110 mL/m2 or RVEF<40) vs M&Ms-2 'dilated RV'. "
                       "Merged DELIBERATELY, but the operational definitions are not identical."},
    "DLV": {"datasets": ["MnMs2"], "comparable": False,
            "note": "M&Ms-2 'dilated left ventricle'. Phenotypically near-identical to DCM "
                    "(measured mean LVEF 33.2%), but it is a DIFFERENT LABEL and is NOT merged "
                    "by default. Use merge_dlv_as_dcm=True to see the lenient number."},
    "MINF": {"datasets": ["ACDC"], "comparable": False,
             "note": "ACDC prior myocardial infarction. Related to but not the same as M&Ms IHD."},
    "IHD": {"datasets": ["MnMs"], "comparable": False,
            "note": "M&Ms ischaemic heart disease. Related to but not the same as ACDC MINF."},
    "ARV": {"datasets": ["MnMs"], "comparable": False,
            "note": "arrhythmogenic RV cardiomyopathy. NOT the same as M&Ms-2 ARR despite the "
                    "similar abbreviation, and NOT the same as ACDC's 'abnormal RV'."},
    "ARR": {"datasets": ["MnMs2"], "comparable": False, "note": "congenital arrhythmogenesis"},
    "HHD": {"datasets": ["MnMs"], "comparable": False, "note": "hypertensive heart disease"},
    "LVNC": {"datasets": ["MnMs"], "comparable": False, "note": "LV non-compaction (n=2)"},
    "TOF": {"datasets": ["MnMs2"], "comparable": False, "note": "tetralogy of Fallot"},
    "CIA": {"datasets": ["MnMs2"], "comparable": False, "note": "interatrial communication"},
    "TRI": {"datasets": ["MnMs2"], "comparable": False, "note": "tricuspid regurgitation"},
    "OTHER": {"datasets": ["MnMs"], "comparable": False,
              "note": "M&Ms 'Other' + 1x 'AHS'. Not a diagnosis; a residual bucket. "
                      "Excluded from top-1 by default — a model cannot be right about it."},
}
COMPARABLE = {k for k, v in CANONICAL_INFO.items() if v["comparable"]}

# Free-text (an LLM's `diagnosis`) -> canonical. Most specific FIRST; order is load-bearing.
# Matched with a leading word boundary, so "dilated" still matches "dilatedcardio..."-free
# prose but "tri" cannot match "ven-TRI-cular". Bare abbreviations go in _CODES below and
# are matched as whole words only — that distinction is the whole reason there are two lists.
_PHRASES: list[tuple[str, str]] = [
    ("arrhythmogenic right ventricular", "ARV"),
    ("congenital arrhythmogenesis", "ARR"), ("arrhythmogenesis", "ARR"),
    ("hypertrophic cardiomyopathy", "HCM"), ("hypertrophic", "HCM"),
    ("dilated cardiomyopathy", "DCM"),
    ("dilated left ventricle", "DLV"), ("left ventricular dilat", "DLV"),
    ("dilated right ventricle", "RV_ABN"), ("abnormal right ventricle", "RV_ABN"),
    ("right ventricular dilat", "RV_ABN"),
    ("myocardial infarction", "MINF"), ("infarct", "MINF"),
    ("ischaemic heart disease", "IHD"), ("ischemic heart disease", "IHD"),
    ("coronary artery disease", "IHD"),
    ("hypertensive heart disease", "HHD"),
    ("non-compaction", "LVNC"), ("noncompaction", "LVNC"),
    ("tetralogy of fallot", "TOF"), ("fallot", "TOF"),
    ("interatrial communication", "CIA"), ("atrial septal defect", "CIA"),
    ("tricuspid regurgitation", "TRI"), ("tricuspidal", "TRI"),
    ("no significant abnormality", "NOR"), ("normal", "NOR"),
]
# Bare abbreviations. Whole-word match ONLY.
_CODES: dict[str, str] = {
    "arvc": "ARV", "hcm": "HCM", "dcm": "DCM", "minf": "MINF", "ihd": "IHD", "cad": "IHD",
    "hhd": "HHD", "lvnc": "LVNC", "tof": "TOF", "asd": "CIA", "cia": "CIA", "tri": "TRI",
    "arr": "ARR", "dlv": "DLV", "nor": "NOR",
}
_PHRASE_RE = [(re.compile(rf"\b{re.escape(p)}"), c) for p, c in _PHRASES]
_CODE_RE = re.compile(rf"\b({'|'.join(sorted(_CODES, key=len, reverse=True))})\b")


def canonical_pathology(dataset: str, native: str) -> str | None:
    return PATHOLOGY_MAP.get((str(dataset), str(native)))


def normalise_diagnosis(text: str) -> str | None:
    """Map an LLM's free-text diagnosis onto the canonical vocabulary. None = unmappable,
    which is itself a result: an unmappable top-1 is a wrong top-1, never a free pass."""
    t = (text or "").lower()
    for rx, code in _PHRASE_RE:
        if rx.search(t):
            return code
    hit = _CODE_RE.search(t)
    return _CODES[hit.group(1)] if hit else None


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────────────────────────────────────
def segmentation_metrics(
    pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]
) -> dict[str, float]:
    """Dice + HD95 per structure. HD95 comes from MONAI (spacing-aware, boundary-correct);
    a hand-rolled version would silently ignore the anisotropic z spacing, which on a
    short-axis stack is 5-10 mm and dominates the distance."""
    import torch
    from monai.metrics import compute_dice, compute_hausdorff_distance

    def onehot(m: np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(np.ascontiguousarray(m), dtype=torch.long)
        return torch.nn.functional.one_hot(t, 4).permute(3, 0, 1, 2)[None].float()

    p, g = onehot(pred), onehot(gt)
    dice = compute_dice(p, g, include_background=False, ignore_empty=True)[0]
    hd = compute_hausdorff_distance(
        p, g, include_background=False, percentile=95, spacing=list(spacing)
    )[0]

    out: dict[str, float] = {}
    for i, label in enumerate((1, 2, 3)):  # LV, Myo, RV — background excluded
        name = LABEL_NAMES[label]
        out[f"dice_{name}"] = float(dice[i])
        out[f"hd95_{name}"] = float(hd[i])
    out["dice_mean"] = float(np.nanmean([out[f"dice_{n}"] for n in LABEL_NAMES.values()]))
    out["hd95_mean"] = float(np.nanmean([out[f"hd95_{n}"] for n in LABEL_NAMES.values()]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Agreement — Bland-Altman, ICC(2,1), Pearson. Table stakes; report them anyway.
# ─────────────────────────────────────────────────────────────────────────────
AGREEMENT_COLS = ["edv_ml", "esv_ml", "sv_ml", "lvef_pct", "lv_mass_g",
                  "rv_edv_ml", "rv_esv_ml", "rv_ef_pct"]


def icc21(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement.

    Absolute agreement, not consistency — a systematic under-segmentation bias must
    cost us, and ICC(3,1)/consistency would forgive it.
    """
    x = np.column_stack([a, b]).astype(float)
    n, k = x.shape
    if n < 2:
        return float("nan")
    grand = x.mean()
    ms_r = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)  # between subjects
    ms_c = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)  # between raters
    ss_e = ((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0) + grand) ** 2).sum()
    ms_e = ss_e / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if denom else float("nan")


def agreement(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-measurement MAE, bias, 95% limits of agreement, ICC(2,1), Pearson r."""
    from scipy import stats

    m = pred_df.merge(gt_df, on="subject_id", suffixes=("_pred", "_gt"))
    if m.empty:
        log.warning("agreement(): no overlapping subject_id between prediction and GT")
        return {}

    out: dict[str, dict[str, float]] = {}
    for col in AGREEMENT_COLS:
        if f"{col}_pred" not in m or f"{col}_gt" not in m:
            continue
        p = m[f"{col}_pred"].to_numpy(float)
        g = m[f"{col}_gt"].to_numpy(float)
        ok = np.isfinite(p) & np.isfinite(g)
        p, g = p[ok], g[ok]
        if len(p) < 2:
            continue
        d = p - g
        bias, sd = float(d.mean()), float(d.std(ddof=1))
        out[col] = {
            "n": int(len(p)),
            "mae": float(np.abs(d).mean()),
            "rmse": float(np.sqrt((d**2).mean())),
            "bias": bias,  # Bland-Altman: mean pred - gt
            "loa_lower": bias - 1.96 * sd,
            "loa_upper": bias + 1.96 * sd,
            "sd_diff": sd,
            "icc_2_1": icc21(p, g),
            "pearson_r": float(stats.pearsonr(p, g).statistic) if np.ptp(p) and np.ptp(g)
            else float("nan"),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# C13 — diagnostic correctness
# ─────────────────────────────────────────────────────────────────────────────
_HF = ["HFrEF", "HFmrEF", "HFpEF"]


def hf_category_accuracy(
    pred_df: pd.DataFrame, gt_df: pd.DataFrame, band: tuple[float, float] | None = None
) -> dict[str, Any]:
    """Agreement on the ESC/AHA category. `band=(35,55)` is THE test for H2: outside it
    the category is over-determined by a huge LVEF margin and any method scores ~100%,
    so a headline number computed on the full cohort would hide the effect entirely.
    """
    m = pred_df.merge(gt_df, on="subject_id", suffixes=("_pred", "_gt"))
    if band is not None:
        m = m[m.lvef_pct_gt.between(*band)]
    if m.empty:
        return {"n": 0, "accuracy": float("nan"), "band": band}

    yp, yg = m.hf_category_pred, m.hf_category_gt
    cm = pd.crosstab(yg, yp).reindex(index=_HF, columns=_HF, fill_value=0)
    per_class = {
        c: {
            "support": int((yg == c).sum()),
            "recall": float(((yp == c) & (yg == c)).sum() / max((yg == c).sum(), 1)),
            "precision": float(((yp == c) & (yg == c)).sum() / max((yp == c).sum(), 1)),
        }
        for c in _HF
    }
    return {
        "n": int(len(m)),
        "band": list(band) if band else None,
        "accuracy": float((yp == yg).mean()),
        "confusion": cm.to_dict(),
        "per_class": per_class,
        "n_near_boundary_gt": int(m.get("near_boundary_gt", pd.Series(dtype=bool)).sum()),
    }


def pathology_top1(
    reports: list[Report] | dict[str, Report],
    manifest: pd.DataFrame,
    merge_dlv_as_dcm: bool = False,
) -> dict[str, Any]:
    """Agent 3's leading differential vs the dataset label.

    `OTHER` (M&Ms's residual bucket + its single AHS subject) is excluded from the
    denominator: it is not a diagnosis, so a model cannot be right about it. That
    exclusion is reported, not hidden.
    """
    rep = {r.subject_id: r for r in reports} if isinstance(reports, list) else dict(reports)
    meta = manifest.set_index("subject_id")

    rows = []
    for sid, r in rep.items():
        if sid not in meta.index:
            continue
        row = meta.loc[sid]
        truth = canonical_pathology(row["dataset"], row["pathology"])
        top = r.diagnosis or (r.differential[0] if r.differential else "")
        pred = normalise_diagnosis(top)
        if merge_dlv_as_dcm:
            truth = "DCM" if truth == "DLV" else truth
            pred = "DCM" if pred == "DLV" else pred
        rows.append({"subject_id": sid, "dataset": row["dataset"], "truth": truth,
                     "pred": pred, "raw": top})

    df = pd.DataFrame(rows)
    if df.empty:
        return {"n": 0, "top1": float("nan")}

    scored = df[df.truth.notna() & (df.truth != "OTHER")].copy()
    scored["hit"] = scored.truth == scored.pred  # pred=None never hits. Deliberate.

    comp = scored[scored.truth.isin(COMPARABLE)]
    return {
        "n_reports": int(len(df)),
        "n_scored": int(len(scored)),
        "n_excluded_other": int((df.truth == "OTHER").sum()),
        "n_unmappable_prediction": int(scored.pred.isna().sum()),
        "top1": float(scored.hit.mean()),
        "top1_by_dataset": scored.groupby("dataset").hit.mean().round(4).to_dict(),
        "top1_comparable_classes": float(comp.hit.mean()) if len(comp) else float("nan"),
        "n_comparable": int(len(comp)),
        "comparable_classes": sorted(COMPARABLE),
        "per_class_recall": scored.groupby("truth").hit.mean().round(4).to_dict(),
        "support": scored.truth.value_counts().to_dict(),
        "confusion": pd.crosstab(scored.truth, scored.pred.fillna("<unmapped>")).to_dict(),
        "merge_dlv_as_dcm": merge_dlv_as_dcm,
        "class_notes": CANONICAL_INFO,
    }


def fidelity_summary(fidelities: list[Fidelity]) -> dict[str, Any]:
    """The headline. BAAI claims 'zero hallucination'; nobody measures it. This does."""
    if not fidelities:
        return {"n": 0}
    num = np.array([f.numeric_fidelity for f in fidelities], float)
    cit = np.array([f.citation_fidelity for f in fidelities], float)
    return {
        "n": len(fidelities),
        "numeric_fidelity_mean": float(np.nanmean(num)),
        "numeric_fidelity_sd": float(np.nanstd(num, ddof=1)) if len(num) > 1 else 0.0,
        "citation_fidelity_mean": float(np.nanmean(cit)),
        "citation_fidelity_sd": float(np.nanstd(cit, ddof=1)) if len(cit) > 1 else 0.0,
        "pct_perfect_numeric": float((num >= 1.0).mean()),
        "pct_perfect_citation": float((cit >= 1.0).mean()),
        "n_reports_with_hallucinated_number": int(sum(bool(f.unsupported_numbers)
                                                      for f in fidelities)),
        "n_reports_with_dangling_citation": int(sum(bool(f.dangling_citations)
                                                    for f in fidelities)),
        "total_numbers": int(sum(f.n_numbers for f in fidelities)),
        "total_citations": int(sum(f.n_citations for f in fidelities)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    stat: Callable[[np.ndarray, np.ndarray], float],
    n: int = 10000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """PAIRED bootstrap: resample SUBJECTS, not the two arrays independently.

    Independent resampling would destroy the pairing and inflate the interval; every
    statistic here (MAE, bias, Dice difference between two configs) is a within-subject
    quantity. Returns (point, lo, hi).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    boot = np.array([stat(a[i], b[i]) for i in idx])
    lo, hi = np.nanpercentile(boot, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return float(stat(a, b)), float(lo), float(hi)


def stratify(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Group by vendor / centre / pathology / dataset. RQ3 lives here.

    Vendor is re-normalised on the way in: manifest.parquet on disk still contains
    'SIEMENS' and 'Siemens' as two different vendors, which would silently halve every
    Siemens stratum.
    """
    d = df.copy()
    if by == "vendor" and "vendor" in d:
        d["vendor"] = d["vendor"].map(normalise_vendor)
    if by not in d:
        raise KeyError(f"cannot stratify by {by!r}; columns are {sorted(d.columns)}")

    num = d.select_dtypes(include="number").columns
    g = d.groupby(by, dropna=False)
    out = g[list(num)].mean().round(4)
    out.insert(0, "n", g.size())
    return out.sort_values("n", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# One run -> one metrics.json
# ─────────────────────────────────────────────────────────────────────────────
def _json_safe(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _read_jsonl(path: Path, model) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return [model.model_validate_json(line) for line in f if line.strip()]


def evaluate_run(cfg: Config, config_id: str) -> dict[str, Any]:
    """Collect whatever the run produced and turn it into artifacts/runs/{id}/metrics.json.

    Deliberately tolerant of missing inputs: a segmentation-only ablation has no reports,
    and a template-LLM baseline has no citations. Absent != zero, so absent is absent.
    """
    run_dir = artifacts(cfg, "runs", config_id)
    gt_path = artifacts(cfg) / "gt_measurements.parquet"
    if not gt_path.exists():
        raise FileNotFoundError(f"{gt_path} missing — run cmr.quantify.run_groundtruth first")
    gt = pd.read_parquet(gt_path)
    band = tuple(cfg.evaluation.boundary_band)
    n_boot = int(cfg.evaluation.bootstrap_n)
    ci = float(cfg.evaluation.ci)

    metrics: dict[str, Any] = {"config_id": config_id, "n_gt": len(gt)}
    meta = gt[["subject_id", "dataset", "pathology", "vendor", "centre"]]

    # ── segmentation ──────────────────────────────────────────────────────────
    seg_path = run_dir / "seg_metrics.parquet"
    if seg_path.exists():
        seg = pd.read_parquet(seg_path).merge(meta, on="subject_id", how="left")
        dice_cols = [c for c in seg if c.startswith(("dice_", "hd95_"))]
        metrics["segmentation"] = {
            "n": len(seg),
            "overall": seg[dice_cols].mean().round(4).to_dict(),
            **{
                f"by_{by}": stratify(seg, by)[["n", *dice_cols]].to_dict("index")
                for by in cfg.evaluation.stratify_by
                if by in seg
            },
        }
        if "dice_LV" in seg:  # a CI on the headline Dice
            d = seg.dice_LV.to_numpy(float)
            metrics["segmentation"]["dice_LV_ci"] = bootstrap_ci(
                d, d, lambda x, _: float(np.nanmean(x)), n=n_boot, ci=ci
            )
    else:
        log.warning("no seg_metrics.parquet in %s — skipping segmentation block", run_dir)

    # ── measurements ──────────────────────────────────────────────────────────
    pred_path = run_dir / "measurements.parquet"
    if pred_path.exists():
        pred = pd.read_parquet(pred_path)
        metrics["agreement"] = agreement(pred, gt)
        metrics["hf_category"] = {
            "all": hf_category_accuracy(pred, gt),
            "band": hf_category_accuracy(pred, gt, band=band),
        }
        merged = pred.merge(gt, on="subject_id", suffixes=("_pred", "_gt"))
        if len(merged) > 1:  # the number the abstract quotes
            metrics["lvef_mae_ci"] = bootstrap_ci(
                merged.lvef_pct_pred.to_numpy(float),
                merged.lvef_pct_gt.to_numpy(float),
                lambda p, g: float(np.mean(np.abs(p - g))),
                n=n_boot,
                ci=ci,
            )
            metrics["lvef_mae_by_vendor"] = (
                stratify(merged.assign(abs_err=(merged.lvef_pct_pred - merged.lvef_pct_gt).abs()),
                         "vendor")[["n", "abs_err"]].to_dict("index")
            )
        metrics["gate"] = {
            "n_failed": int((~pred.gate_passed).sum()) if "gate_passed" in pred else None,
            "violations": (
                pred[~pred.gate_passed].gate_violations.str.split(";").explode()
                .value_counts().to_dict()
                if "gate_passed" in pred and "gate_violations" in pred and (~pred.gate_passed).any()
                else {}
            ),
        }
    else:
        log.warning("no measurements.parquet in %s — skipping agreement/HF blocks", run_dir)

    # ── reports + fidelity ────────────────────────────────────────────────────
    reports = _read_jsonl(run_dir / "reports.jsonl", Report)
    if reports:
        metrics["pathology"] = pathology_top1(reports, gt)
        metrics["pathology_lenient"] = pathology_top1(reports, gt, merge_dlv_as_dcm=True)
    fid = _read_jsonl(run_dir / "fidelity.jsonl", Fidelity)
    if fid:
        metrics["fidelity"] = fidelity_summary(fid)

    out = run_dir / "metrics.json"
    out.write_text(json.dumps(_json_safe(metrics), indent=2, sort_keys=True))
    log.info("metrics -> %s", out)
    return metrics
