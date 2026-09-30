"""Hits a real Ollama on localhost. Marked slow: `pytest -m slow`.

Guards the two things that break silently when Ollama is upgraded:
  - the strict json_schema path still constrains the grammar (a nested $defs schema included),
  - a chunk_id the model was given comes back unmodified, so citations resolve.
"""

from __future__ import annotations

import pytest

from cmr import config
from cmr.factcheck import check
from cmr.llm import LLM, LLMUnavailable
from cmr.report import generate
from cmr.types import Chunk, Measurements

pytestmark = pytest.mark.slow

CHUNKS = [
    Chunk(
        chunk_id="c1f3a90b",
        text="Heart failure with reduced ejection fraction (HFrEF) is "
        "defined as symptomatic heart failure with an LVEF of 40% or less.",
        source="2021 ESC HF Guideline",
        source_id="esc_hf_2021",
        section="3.2 Definitions",
        page=3607,
        class_of_recommendation="I",
        level_of_evidence="C",
    ),
    Chunk(
        chunk_id="7d2e44aa",
        text="In patients with HFrEF, an ACE inhibitor or ARNI, a "
        "beta-blocker, an MRA and an SGLT2 inhibitor are recommended to reduce the risk of "
        "heart failure hospitalisation and death.",
        source="2021 ESC HF Guideline",
        source_id="esc_hf_2021",
        section="5.2 Pharmacotherapy",
        page=3627,
        class_of_recommendation="I",
        level_of_evidence="A",
    ),
]

MEAS = Measurements(
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


def test_provider_none_refuses() -> None:
    cfg = config.load()
    cfg["llm"] = dict(cfg["llm"]) | {"provider": "none"}
    with pytest.raises(LLMUnavailable, match="template_report"):
        LLM(cfg)


def test_model_id() -> None:
    assert LLM(config.load()).model_id.startswith("ollama/")


def test_generate_is_grounded() -> None:
    cfg = config.load()
    r = generate(MEAS, CHUNKS, cfg)

    assert r.subject_id == "patient101"
    assert r.diagnosis
    assert r.citations, "no citations — the whole point of the architecture"
    given = {c.chunk_id for c in CHUNKS}
    assert all(c.chunk_id in given for c in r.citations), "invented a chunk_id"
    assert any(c.supports for c in r.citations), "no supports string -> Agent 4 has no link"

    # The Agent 4 invariant: `supports` is a VERBATIM sentence of this report, never the
    # guideline passage re-quoted. Without this, the heatmap<->citation join silently fails.
    body = f"{r.diagnosis}\n{r.findings}"
    for c in r.citations:
        if c.supports:
            assert c.supports in body, f"supports is not a report sentence: {c.supports!r}"
    assert "[chunk_id:" not in body, "inline markers should be stripped from the prose"

    f = check(r, MEAS, {c.chunk_id: c for c in CHUNKS})
    assert f.citation_fidelity == 1.0
    assert f.numeric_fidelity == 1.0, f"hallucinated numbers: {f.unsupported_numbers}"


def test_no_passages_yields_no_citations() -> None:
    """The H1 ablation: no passages -> the model must not invent one."""
    r = generate(MEAS, [], config.load())
    assert r.citations == [], f"fabricated citations with no corpus: {r.citations}"
