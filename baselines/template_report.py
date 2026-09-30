"""The zero-AI floor: a fixed-template report. NO LLM, NO RETRIEVAL, NO MODEL.

This baseline exists to embarrass the full system on one metric, deliberately.

A template CANNOT hallucinate. Every number it prints is interpolated straight out of
`Measurements`, so `numeric_fidelity` is 1.0 BY CONSTRUCTION — not by being good, but
by being incapable of doing anything else. The full LLM system will score below it.
Reporting that is the point: it is what makes the rest of the evaluation credible, and
the gap between this floor and the full system is precisely the value the LLM adds.

What the template CANNOT do, and what the LLM is therefore buying:
  * it cites nothing (`citations == []`) — it has no retriever, so citation_fidelity is
    vacuous here, not perfect. Do not report it as 1.0; report it as N/A.
  * it cannot weigh a borderline case, reconcile conflicting guidance (C11), or say
    anything not already in the six numbers it was handed.
  * its differential is a lookup table, not reasoning.

It also earns its keep a second way: ACDC and M&Ms ship no free-text reports, so there
is no reference text for BERTScore. These templated reports ARE that reference corpus.
"""

from __future__ import annotations

import logging

from cmr.config import Config
from cmr.types import Measurements, Report

log = logging.getLogger("baselines.template_report")

# Conventional lower limit of normal for RV ejection fraction. Overridable via config;
# it is stated here rather than buried in prose so it can be cited or changed in one place.
_RV_EF_MIN_DEFAULT = 45.0

# A lookup table, not a diagnosis. Keyed by HF category; deliberately shallow, because
# pretending a template can reason is exactly the dishonesty this baseline guards against.
_DIFFERENTIAL = {
    "HFrEF": ["Dilated cardiomyopathy", "Ischaemic cardiomyopathy", "Myocarditis"],
    "HFmrEF": ["Early/recovering systolic dysfunction", "Ischaemic heart disease"],
    "HFpEF": ["Hypertensive heart disease", "Hypertrophic cardiomyopathy", "Normal study"],
}

_ACTIONS = {
    "HFrEF": [
        "Confirm LVEF with a second modality or repeat acquisition.",
        "Assess for ischaemic aetiology.",
        "Refer for guideline-directed medical therapy review.",
    ],
    "HFmrEF": [
        "Repeat imaging to confirm the ejection fraction; the value is in the mid-range band.",
        "Assess for ischaemic aetiology.",
    ],
    "HFpEF": [
        "Correlate with clinical presentation and diastolic function assessment.",
    ],
}


def _fmt(m: Measurements, rv_ef_min: float) -> str:
    """The findings paragraph. Every number here is a verbatim field of `m`."""
    lv = (
        f"Left ventricle: EDV {m.edv_ml:.1f} mL, ESV {m.esv_ml:.1f} mL, "
        f"stroke volume {m.sv_ml:.1f} mL, ejection fraction {m.lvef_pct:.1f}%. "
        f"Left ventricular myocardial mass {m.lv_mass_g:.1f} g."
    )
    rv = (
        f" Right ventricle: EDV {m.rv_edv_ml:.1f} mL, ESV {m.rv_esv_ml:.1f} mL, "
        f"ejection fraction {m.rv_ef_pct:.1f}%"
    )
    # Deliberately does NOT quote the numeric reference limit. The template has no
    # retrieval and therefore no citation to hang that threshold on, and factcheck.py
    # correctly scores an unsourced number as unsupported. Naming the finding
    # qualitatively is honest; quoting "45%" out of thin air is exactly the failure
    # mode this thesis exists to measure. The threshold still drives the WORDING.
    rv += "." if m.rv_ef_pct >= rv_ef_min else ", which is reduced."
    boundary = (
        " The left ventricular ejection fraction lies close to a category cut-point; "
        "the classification should be treated as provisional."
        if m.near_boundary
        else ""
    )
    provenance = (
        f" Measurements were derived from {'the ground-truth' if m.source == 'groundtruth' else 'the predicted'}"
        " segmentation by deterministic volumetry."
    )
    return lv + rv + boundary + provenance


def template_report(m: Measurements, cfg: Config) -> Report:
    """A fixed-template report from Agent 2's JSON. NO LLM AT ALL."""
    rv_ef_min = float(cfg.get_path("quantification.rv_ef_min", _RV_EF_MIN_DEFAULT))
    hf = m.hf_category

    diagnosis = (
        f"{hf}: left ventricular ejection fraction {m.lvef_pct:.1f}%"
        f"{' (near the category boundary)' if m.near_boundary else ''}."
    )

    return Report(
        subject_id=m.subject_id,
        diagnosis=diagnosis,
        differential=list(_DIFFERENTIAL[hf]),
        hf_category=hf,
        findings=_fmt(m, rv_ef_min),
        citations=[],  # no retriever. Vacuous, NOT perfect. See the module docstring.
        recommended_actions=list(_ACTIONS[hf]),
        conflicts=[],  # a template cannot detect a conflict, let alone attribute one.
    )
