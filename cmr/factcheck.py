"""THE headline metric: is every number and every citation in the report actually real?

BAAI claims "zero hallucination" qualitatively. CardAIc does not measure it at all. This
file turns the claim into a number that can go in a table with a confidence interval.

    numeric_fidelity  = |numbers in the report that trace back to Agent 2| / |numbers|
    citation_fidelity = |citations whose chunk_id resolves in the index|   / |citations|

Extracting "the numbers in the report" is the entire difficulty, and a naive \\d+ regex makes
the metric a lie in both directions. A number in the report is legitimate if and only if we
handed it to the model. Three things were handed to it, so three whitelists:

  1. MEASUREMENTS      — m.numeric_facts(), matched within `tol`. The real target.
  2. The HF cut-points — 40 / 41 / 49 / 50 are printed in the system prompt, so quoting
                         "LVEF <= 40" is obedience, not hallucination.
  3. Cited passages    — any number appearing verbatim in a passage the report actually cites.
                         Quote a number from a guideline, and you must cite that guideline.
                         (Only CITED passages, not the whole corpus — whitelisting against
                         every chunk would let any number in any guideline pass unchecked.)

And three things are stripped before extraction, because they are not clinical claims:
  - the subject_id ("patient101" is not a measurement of 101 — this one silently wrecks ACDC),
  - citation apparatus: bracketed spans, "p. 3627", "Section 5.2", quoted chunk_ids,
  - years (1900-2100): "ESC 2021" is a reference, not a volume. Not counted either way.

Only `diagnosis` and `findings` are scanned — the fields that make factual assertions about
this patient. `recommended_actions` quote drug doses and trial thresholds from the guidelines;
scanning them would flag correct quotation as hallucination.
"""

from __future__ import annotations

import logging
import re

from .types import Chunk, Fidelity, Measurements, Report

log = logging.getLogger("cmr.factcheck")

# The HF cut-points are given to the model in the system prompt (ESC 2021 / AHA 2022).
GUIDELINE_CONSTANTS = {40.0, 41.0, 49.0, 50.0}

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3})")
_BRACKETED = re.compile(r"\[[^\]]*\]")  # [chunk_id: a3f...], [1], [ESC 2021]
_LOCATOR = re.compile(
    r"\b(?:pp?\.|pages?|sections?|tables?|figures?|chapters?|class|level)\s*[\w.]*\d[\w.\-]*",
    re.I,
)
_YEAR = re.compile(r"^(19|20)\d{2}$")


def _verbatim(tok: str, text: str) -> bool:
    """Does this exact numeral appear in the passage? Bounded so 40 does not match 1940."""
    return re.search(rf"(?<![\d.]){re.escape(tok)}(?![\d.])", text) is not None


def _scrub(text: str, m: Measurements, cited_ids: list[str]) -> str:
    """Remove everything that is digits-but-not-a-claim."""
    text = _THOUSANDS.sub("", text)
    for cid in cited_ids:
        text = text.replace(cid, " ")
    text = re.sub(re.escape(m.subject_id), " ", text, flags=re.I)
    text = _BRACKETED.sub(" ", text)
    return _LOCATOR.sub(" ", text)


def check(
    report: Report,
    m: Measurements,
    index: dict[str, Chunk],
    tol: float = 0.5,
) -> Fidelity:
    """Fidelity of one report against the measurements that produced it and the chunk index."""
    resolved = [c for c in report.citations if c.chunk_id in index]
    dangling = [c.chunk_id for c in report.citations if c.chunk_id not in index]

    facts = list(m.numeric_facts().values())
    cited_text = " ".join(index[c.chunk_id].text for c in resolved)

    text = _scrub(
        f"{report.diagnosis}\n{report.findings}", m, [c.chunk_id for c in report.citations]
    )

    counted = 0
    unsupported: list[float] = []
    for tok in _NUMBER.findall(text):
        if _YEAR.match(tok):  # a reference year is not a measurement; ignore it entirely
            continue
        val = float(tok)
        counted += 1
        if any(abs(val - f) <= tol for f in facts):
            continue
        if val in GUIDELINE_CONSTANTS:  # printed in the system prompt, so it was handed over
            continue
        if cited_text and _verbatim(tok, cited_text):  # quoted from a passage it actually cited
            continue
        unsupported.append(val)

    # No numbers at all -> nothing was fabricated. Vacuously perfect, and n_numbers=0 says so.
    numeric = 1.0 if counted == 0 else 1.0 - len(unsupported) / counted
    # No citations -> nothing is grounded -> 0.0. This is not a division-by-zero dodge: it is
    # what makes the H1 (retriever: none) ablation show the gap it exists to show.
    citation = len(resolved) / len(report.citations) if report.citations else 0.0

    if unsupported or dangling:
        log.warning(
            "%s: %d unsupported number(s) %s, %d dangling citation(s) %s",
            report.subject_id,
            len(unsupported),
            unsupported,
            len(dangling),
            dangling,
        )

    return Fidelity(
        numeric_fidelity=numeric,
        citation_fidelity=citation,
        n_numbers=counted,
        n_citations=len(report.citations),
        unsupported_numbers=unsupported,
        dangling_citations=dangling,
    )
