"""The fast tier: no data, no network, no GPU. Runs in under a second.

These are the checks that would have caught the defects the proposal actually shipped
with — a heart-failure rule that misclassifies at both boundaries, and a label
convention that silently inverts LV and RV across datasets.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmr import config
from cmr.checks import LabelOrderError, assert_canonical, canonicalise, lv_label
from cmr.types import LV, MYO, RV, Citation, Fidelity, Report, RunState, Status


def ring_mask(lv_label_value: int = LV, rv_label_value: int = RV) -> np.ndarray:
    """A synthetic heart: an LV cavity with a myocardial ring around it, and a
    separate RV cavity well off to the side. Exactly the anatomy the concentricity
    assertion relies on."""
    m = np.zeros((64, 64, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:64, 0:64]
    r = np.sqrt((xx - 20) ** 2 + (yy - 32) ** 2)
    for z in range(3):
        m[:, :, z][r < 6] = lv_label_value  # cavity
        m[:, :, z][(r >= 6) & (r < 10)] = MYO  # ring AROUND the cavity
        rr = np.sqrt((xx - 45) ** 2 + (yy - 32) ** 2)
        m[:, :, z][rr < 8] = rv_label_value  # the other ventricle, far away
    return m


# ── the label-order guard ────────────────────────────────────────────────────
def test_canonical_mask_passes():
    assert_canonical(ring_mask(), "synthetic")


def test_inverted_mask_is_caught():
    """This is the bug that would otherwise be invisible: swap LV and RV and every
    ejection fraction in the thesis becomes wrong-but-plausible, with no stack trace."""
    inverted = ring_mask(lv_label_value=RV, rv_label_value=LV)
    with pytest.raises(LabelOrderError):
        assert_canonical(inverted, "inverted")


def test_lv_label_detection():
    assert lv_label(ring_mask()) == LV
    assert lv_label(ring_mask(lv_label_value=RV, rv_label_value=LV)) == RV


def test_acdc_remap_fixes_an_inverted_mask():
    acdc_native = ring_mask(lv_label_value=RV, rv_label_value=LV)  # ACDC: 1=RV, 3=LV
    assert_canonical(canonicalise(acdc_native, "acdc"), "remapped")


def test_mnms_remap_is_identity():
    m = ring_mask()
    assert np.array_equal(canonicalise(m, "mnms"), m)


# ── the heart-failure boundary bug ───────────────────────────────────────────
@pytest.mark.parametrize(
    "lvef,expected",
    [
        (25.0, "HFrEF"),
        (39.9, "HFrEF"),
        (40.0, "HFrEF"),  # the proposal's rule says HFmrEF here. It is WRONG.
        (40.1, "HFmrEF"),
        (45.0, "HFmrEF"),
        (49.9, "HFmrEF"),
        (50.0, "HFpEF"),  # the proposal's rule says HFmrEF here too. Also WRONG.
        (65.0, "HFpEF"),
    ],
)
def test_hf_category_boundaries(lvef, expected):
    """ESC 2021 and AHA/ACC/HFSA 2022 agree: HFrEF <= 40, HFmrEF 41-49, HFpEF >= 50.

    The proposal emits (<40 = reduced, 40-50 = mildly reduced, >50 = preserved),
    which misclassifies BOTH cut-points, in a thesis whose entire premise is
    guideline fidelity.
    """
    from cmr.quantify import hf_category

    assert hf_category(lvef, config.load()) == expected


# ── contracts ────────────────────────────────────────────────────────────────
def test_report_rejects_a_malformed_payload():
    """C10, at the software level: a bad payload cannot cross a node boundary."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Report(subject_id="x", diagnosis="y", hf_category="NOT_A_CATEGORY")


def test_run_state_refusal_flag():
    s = RunState(subject_id="x", status=Status.FAILED_SEGMENTATION)
    assert s.refused
    assert RunState(subject_id="x").refused is False


def test_config_id_is_deterministic():
    a = config.config_id("full", {"retriever": "hybrid", "gate": True})
    b = config.config_id("full", {"gate": True, "retriever": "hybrid"})  # key order differs
    c = config.config_id("full", {"retriever": "dense", "gate": True})
    assert a == b, "same spec must land in the same directory"
    assert a != c, "a changed spec must NOT overwrite an old result"


def test_experiment_grid_loads():
    for name in ("full", "no_rag", "no_gate", "no_feedback", "template"):
        cfg = config.load_experiment(name)
        assert cfg.config_id.startswith(name)


def test_hf_thresholds_come_from_config_not_code():
    cfg = config.load()
    assert cfg.quantification.hf_thresholds["hfref_max"] == 40.0
    assert cfg.quantification.hf_thresholds["hfpef_min"] == 50.0


# ── the metric's own proof ───────────────────────────────────────────────────
def test_factcheck_catches_a_corrupted_report():
    """A tampered report MUST score below 1.0 on both fidelities, or the headline
    metric is measuring nothing."""
    from cmr.factcheck import check
    from cmr.types import Chunk, Measurements

    m = Measurements(
        subject_id="p1", edv_ml=210.0, esv_ml=150.0, sv_ml=60.0, lvef_pct=28.6,
        lv_mass_g=180.0, rv_edv_ml=140.0, rv_esv_ml=90.0, rv_ef_pct=35.7,
        hf_category="HFrEF",
    )
    chunk = Chunk(chunk_id="abc123", text="HFrEF is defined as LVEF <= 40%.",
                  source="2021 ESC HF", source_id="esc_hf_2021", section="4.1", page=3612)
    index = {"abc123": chunk}

    honest = Report(
        subject_id="p1",
        diagnosis="Severely reduced systolic function.",
        findings="LVEF is 28.6%. EDV is 210.0 mL.",
        hf_category="HFrEF",
        citations=[Citation(chunk_id="abc123", source="2021 ESC HF", section="4.1", page=3612)],
    )
    corrupt = Report(
        subject_id="p1",
        diagnosis="Severely reduced systolic function.",
        findings="LVEF is 71.2%. EDV is 999.0 mL.",  # numbers Agent 2 never produced
        hf_category="HFrEF",
        citations=[Citation(chunk_id="INVENTED", source="Fake", section="1", page=1)],
    )
    good, bad = check(honest, m, index), check(corrupt, m, index)
    assert good.numeric_fidelity == 1.0 and good.citation_fidelity == 1.0
    assert bad.numeric_fidelity < 1.0, "hallucinated measurements went undetected"
    assert bad.citation_fidelity < 1.0, "a fabricated citation went undetected"


def test_fidelity_model():
    f = Fidelity(numeric_fidelity=1.0, citation_fidelity=0.5)
    assert 0.0 <= f.citation_fidelity <= 1.0
