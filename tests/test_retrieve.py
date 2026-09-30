"""Retrieval + corpus tests.

The slow ones load MedCPT and touch the real corpus, so they are marked and skipped unless
the corpus has actually been built (`scripts/fetch_guidelines.py` then `python -m cmr.corpus`).

NOTE: on this Mac `config.device()` resolves to "mps", and torch 2.13 SIGABRTs on any
`.to("mps")`. Run the slow tests with CMR_DEVICE=cpu until that is fixed.
"""

from __future__ import annotations

import pytest

from cmr import config
from cmr.corpus import extract_cor_loe
from cmr.retrieve import build_query, detect_conflicts
from cmr.types import Chunk


@pytest.fixture(scope="module")
def cfg():
    return config.load()


def _chunk(text: str, **kw) -> Chunk:
    base = dict(
        chunk_id="x", text=text, source="s", source_id="esc_hf_2021", section="Recommendations", page=1
    )
    return Chunk(**{**base, **kw})


# ─── CoR / LoE extraction ────────────────────────────────────────────────────


def test_esc_class_and_level():
    text = "An ACE-I is recommended for patients with HFrEF to reduce the risk of death. IA"
    assert extract_cor_loe(text, "esc") == ("I", "A")


def test_acc_aha_is_normalised_onto_the_esc_scale():
    # COR 2a -> IIa, LOE B-NR -> B. Without this, an ESC passage and an ACC/AHA passage
    # cannot be compared and detect_conflicts is meaningless.
    text = "2a B-NR 1. In patients with HFrEF, intravenous iron replacement is reasonable."
    assert extract_cor_loe(text, "acc_aha") == ("IIa", "B")


def test_prose_is_not_tagged():
    # "IA" appears, but this is not a recommendation. Tagging it would inflate coverage.
    assert extract_cor_loe("Figure 3 shows the trajectory of LVEF over time.", "esc") == (None, None)


def test_reference_values_document_carries_no_recommendations():
    assert extract_cor_loe("LV ejection fraction is recommended. IA", "none") == (None, None)


# ─── query construction ─────────────────────────────────────────────────────


def test_build_query_uses_concepts_not_just_numbers():
    q = build_query({"lvef_pct": 32.0, "hf_category": "HFrEF"}, pathology_hint="dilated cardiomyopathy")
    assert "reduced ejection fraction" in q
    assert "32%" in q
    assert "dilated cardiomyopathy" in q


# ─── conflict detection (C11) ───────────────────────────────────────────────


def test_conflict_is_detected_and_resolved_by_recency():
    a = _chunk("In HFrEF, LVEF <=40% defines reduced ejection fraction.",
               source_id="esc_hf_2021", source="2021 ESC HF", page=1,
               section="Recommendations for definition of heart failure",
               class_of_recommendation="I", level_of_evidence="A")
    b = _chunk("In HFrEF, LVEF <=45% defines reduced ejection fraction.",
               source_id="aha_hf_2022", source="2022 AHA HF", page=2,
               section="Recommendations for definition of heart failure",
               class_of_recommendation="I", level_of_evidence="B")
    out = detect_conflicts([a, b])
    assert len(out) == 1
    # BOTH sides are always named — the system never silently drops one.
    assert "40%" in out[0] and "45%" in out[0]
    assert "2021 ESC HF" in out[0] and "2022 AHA HF" in out[0]
    assert "recency" in out[0]


def test_different_clinical_questions_are_not_a_conflict():
    # <=35% for an ICD and <=40% for HFrEF are different questions, not a contradiction.
    icd = _chunk("An ICD is recommended when LVEF <=35%.", source_id="esc_hf_2021",
                 section="Recommendations for implantable cardioverter defibrillator")
    hf = _chunk("HFrEF is defined by LVEF <=40%.", source_id="aha_hf_2022",
                section="Recommendations for classification of heart failure")
    assert detect_conflicts([icd, hf]) == []


def test_same_document_quoting_two_numbers_is_not_a_conflict():
    a = _chunk("LVEF <=40% is HFrEF.", source_id="esc_hf_2021", section="Recommendations for HF")
    b = _chunk("LVEF <=35% indicates an ICD.", source_id="esc_hf_2021", section="Recommendations for HF")
    assert detect_conflicts([a, b]) == []


# ─── the acceptance test ────────────────────────────────────────────────────


@pytest.mark.slow
def test_hfref_threshold_query_retrieves_the_40_percent_cutpoint(cfg):
    """THE acceptance test: ask for the HFrEF cut-point, get a passage that states 40."""
    from cmr.corpus import load_chunks
    from cmr.retrieve import Retriever

    if not load_chunks(cfg):
        pytest.skip("corpus not built — run scripts/fetch_guidelines.py, then python -m cmr.corpus")

    r = Retriever(cfg)
    top = r.retrieve("LVEF threshold for HFrEF heart failure with reduced ejection fraction", k=3)

    assert top, "retriever returned nothing"
    assert any("40" in c.text for c in top), (
        "no top-3 passage states the 40% cut-point:\n"
        + "\n".join(f"  {c.source} p{c.page}: {c.text[:100]}" for c in top)
    )


@pytest.mark.slow
def test_every_retrieved_chunk_resolves_against_the_index(cfg):
    """Citation fidelity depends on this: a retrieved chunk_id MUST resolve."""
    from cmr.retrieve import Retriever

    r = Retriever(cfg)
    for c in r.retrieve("cardiac magnetic resonance in dilated cardiomyopathy", k=5):
        assert r.index[c.chunk_id].text == c.text
