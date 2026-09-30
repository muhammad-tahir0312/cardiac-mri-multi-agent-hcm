"""factcheck must catch a corrupted report. No LLM, no network — this runs in milliseconds.

If these ever go green on a hallucinated number, the headline metric is decorative.
"""

from __future__ import annotations

import pytest

from cmr.factcheck import check
from cmr.report import _link_supports
from cmr.types import Chunk, Citation, Measurements, Report

CHUNK = Chunk(
    chunk_id="a1b2c3",
    text="Heart failure with reduced ejection fraction (HFrEF) is defined as an LVEF of 40% "
    "or less. Guideline-directed medical therapy is indicated (Class I, Level A).",
    source="2021 ESC HF Guideline",
    source_id="esc_hf_2021",
    section="5.2 Definitions",
    page=3627,
    class_of_recommendation="I",
    level_of_evidence="A",
)
INDEX = {CHUNK.chunk_id: CHUNK}


@pytest.fixture
def m() -> Measurements:
    return Measurements(
        subject_id="patient101",
        edv_ml=210.0,
        esv_ml=150.0,
        sv_ml=60.0,
        lvef_pct=28.6,
        lv_mass_g=180.0,
        rv_edv_ml=140.0,
        rv_esv_ml=90.0,
        rv_ef_pct=35.7,
        hf_category="HFrEF",
        source="groundtruth",
    )


def _report(**kw) -> Report:
    base = dict(
        subject_id="patient101",
        diagnosis="Severely impaired LV systolic function with LVEF 28.6%, consistent with HFrEF.",
        findings="LV EDV 210.0 mL and ESV 150.0 mL give a stroke volume of 60.0 mL and an LVEF "
        "of 28.6%. LV mass is 180.0 g. The LVEF is at or below the 40% threshold.",
        hf_category="HFrEF",
        citations=[
            Citation(
                chunk_id="a1b2c3",
                source=CHUNK.source,
                section=CHUNK.section,
                page=CHUNK.page,
                class_of_recommendation="I",
                level_of_evidence="A",
                supports="Severely impaired LV systolic function with LVEF 28.6%, consistent "
                "with HFrEF.",
            )
        ],
    )
    return Report(**(base | kw))


def test_clean_report_is_perfect(m: Measurements) -> None:
    f = check(_report(), m, INDEX)
    assert f.numeric_fidelity == 1.0
    assert f.citation_fidelity == 1.0
    assert not f.unsupported_numbers and not f.dangling_citations
    assert f.n_numbers >= 5  # it really did scan the numbers, it didn't just find none


def test_corrupted_number_is_caught(m: Measurements) -> None:
    """One invented volume (210 -> 275) must drop numeric_fidelity below 1."""
    bad = _report(findings=_report().findings.replace("210.0 mL", "275.0 mL"))
    f = check(bad, m, INDEX)
    assert f.numeric_fidelity < 1.0
    assert 275.0 in f.unsupported_numbers
    assert f.citation_fidelity == 1.0  # the citation was untouched


def test_invented_chunk_id_is_caught(m: Measurements) -> None:
    bad = _report(citations=[_report().citations[0].model_copy(update={"chunk_id": "deadbeef"})])
    f = check(bad, m, INDEX)
    assert f.citation_fidelity == 0.0
    assert f.dangling_citations == ["deadbeef"]


def test_both_corruptions(m: Measurements) -> None:
    bad = _report(
        diagnosis="LVEF 31.2% consistent with HFrEF.",
        citations=[_report().citations[0].model_copy(update={"chunk_id": "nope"})],
    )
    f = check(bad, m, INDEX)
    assert f.numeric_fidelity < 1.0 and f.citation_fidelity < 1.0


def test_subject_id_digits_are_not_measurements(m: Measurements) -> None:
    """'patient101' must not be read as the number 101 — this would wreck every ACDC subject."""
    f = check(_report(findings="Subject patient101 shows an LVEF of 28.6%."), m, INDEX)
    assert f.unsupported_numbers == []


def test_year_and_page_are_not_measurements(m: Measurements) -> None:
    f = check(
        _report(findings="Per the 2021 ESC guideline (p. 3627, Section 5.2), LVEF 28.6% is low."),
        m,
        INDEX,
    )
    assert f.unsupported_numbers == []


def test_threshold_quoted_from_cited_passage_is_allowed(m: Measurements) -> None:
    """'40%' is in the prompt AND in the cited passage. It is not a hallucinated measurement."""
    f = check(_report(findings="LVEF 28.6% is below the 40% HFrEF threshold."), m, INDEX)
    assert f.numeric_fidelity == 1.0


def test_uncited_number_from_nowhere_is_unsupported(m: Measurements) -> None:
    """A number that is in no passage, no measurement and no threshold is a hallucination."""
    f = check(_report(findings="LVEF 28.6%. Septal wall thickness is 13.4 mm."), m, INDEX)
    assert 13.4 in f.unsupported_numbers


def test_tolerance(m: Measurements) -> None:
    """Rounding 28.6 -> 28.4 is inside tol=0.5; 28.6 -> 26.0 is not."""
    assert check(_report(findings="LVEF 28.4%."), m, INDEX).numeric_fidelity == 1.0
    assert check(_report(findings="LVEF 26.0%."), m, INDEX).numeric_fidelity < 1.0


def test_no_citations_scores_zero(m: Measurements) -> None:
    """The H1 (retriever: none) ablation: ungrounded means zero citation fidelity, by design."""
    f = check(_report(citations=[]), m, INDEX)
    assert f.citation_fidelity == 0.0
    assert f.n_citations == 0


# ── the Agent 4 join key ──────────────────────────────────────────────────────────────
# `supports` MUST be a verbatim sentence of the report. qwen2.5:14b instead returns the
# guideline passage, which validates, looks correct, and silently breaks the heatmap link.


def test_supports_recovered_from_inline_marker() -> None:
    """The model quoted the PASSAGE into supports. We must recover its own sentence instead."""
    r = Report(
        subject_id="patient101",
        diagnosis="Heart failure with reduced ejection fraction. [chunk_id: a1b2c3]",
        findings="The LVEF is 28.6%, which is severely reduced. [chunk_id: a1b2c3]",
        citations=[
            Citation(
                chunk_id="a1b2c3",
                source=CHUNK.source,
                section=CHUNK.section,
                page=CHUNK.page,
                supports="Heart failure with reduced ejection fraction (HFrEF) is defined as "
                "an LVEF of 40% or less.",  # <- the passage, not the report. The real failure.
            )
        ],
    )
    _link_supports(r)
    assert "[chunk_id:" not in r.findings and "[chunk_id:" not in r.diagnosis
    assert r.citations[0].supports == "Heart failure with reduced ejection fraction."
    assert r.citations[0].supports in f"{r.diagnosis}\n{r.findings}"


def test_supports_kept_when_model_quotes_itself() -> None:
    r = Report(
        subject_id="patient101",
        diagnosis="Severe LV systolic dysfunction.",
        findings="The LVEF is 28.6%.",
        citations=[
            Citation(
                chunk_id="a1b2c3", source="s", section="x", page=1, supports="The LVEF is 28.6%."
            ),
        ],
    )
    _link_supports(r)
    assert r.citations[0].supports == "The LVEF is 28.6%."


def test_unlinkable_citation_gets_empty_supports() -> None:
    """No marker, no self-quote -> we say 'no link' rather than invent a plausible one."""
    r = Report(
        subject_id="patient101",
        diagnosis="Severe LV systolic dysfunction.",
        findings="The LVEF is 28.6%.",
        citations=[
            Citation(
                chunk_id="zzz",
                source="s",
                section="x",
                page=1,
                supports="Some guideline sentence we never wrote.",
            )
        ],
    )
    _link_supports(r)
    assert r.citations[0].supports == ""
