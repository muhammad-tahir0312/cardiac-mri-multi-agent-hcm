"""Agent 3b — grounded report generation.

The model never sees an image and is never asked to measure anything. Agent 2 has already
done the arithmetic; the LLM's only job is to turn a fixed set of numbers and a fixed set of
guideline passages into prose, and to say which passage licenses which sentence.

Two rules are enforced structurally rather than requested politely:

  1. Every number it may write is handed to it in the MEASUREMENTS block. It may not compute,
     round, or infer a new one. cmr/factcheck.py then re-derives every number in the output
     and fails the report if one is not traceable. The prompt asks; factcheck verifies.

  2. Every clinical claim carries a passage-level citation (chunk_id -> source + section +
     page), and `Citation.supports` holds the exact sentence that passage grounds. That string
     is the join key Agent 4 uses to tie a heatmap region to a guideline line. This is the
     differentiator: BAAI's structured report carries no citations at all — its RAG is a
     conversational sidecar, not part of the diagnostic path.

Citation metadata (source/section/page/class/level) is BACKFILLED from the index after
generation. The model is trusted to choose a chunk_id and nothing else; asking it to also
transcribe a page number is asking it to hallucinate one. A chunk_id that does not resolve is
left exactly as the model emitted it, so factcheck can count it as dangling.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from . import guardrails
from .llm import LLM
from .types import Chunk, Measurements, Report

log = logging.getLogger("cmr.report")

_SENT = re.compile(r"(?<=[.;])\s+")  # '28.6%' survives: a decimal point has no space after it
_MARKER = re.compile(r"\s*\[chunk_id:\s*([^\]\s]+)\s*\]")


def _strip(s: str) -> str:
    return _MARKER.sub("", s).strip()


def _link_supports(report: Report) -> None:
    """Guarantee the invariant Agent 4 depends on: a non-empty `Citation.supports` is a
    VERBATIM sentence of this report's diagnosis/findings.

    Asked for its own sentence, qwen2.5:14b hands back the guideline passage instead — every
    time, measured. That silently breaks the heatmap<->citation join, which is the whole
    contribution of Agent 4, and it breaks it in a way that still validates and still looks
    right in the JSON. So we do not rely on obedience: the model reliably DOES emit an inline
    [chunk_id: X] marker on the sentence it grounded, and we recover the link from that.

    Markers are then stripped from the prose — the citations list is the only citation
    carrier, and stripping keeps `supports` an exact substring of the text.
    A citation we cannot tie to a real sentence gets supports="" rather than a plausible lie.
    """
    # A marker TRAILS the sentence it grounds ("...HFrEF. [chunk_id: x]"), so it sits after the
    # full stop. Splitting into sentences first would hand every marker to the FOLLOWING
    # sentence — off by one, and wrong in a way that still looks plausible. Anchor each marker
    # to the text that precedes it instead.
    by_chunk: dict[str, str] = {}
    raw = f"{report.diagnosis}\n{report.findings}"
    for mt in _MARKER.finditer(raw):
        preceding = _SENT.split(raw[: mt.start()])[-1]
        sentence = _strip(preceding.rsplit("\n", 1)[-1])
        if sentence:
            by_chunk.setdefault(mt.group(1), sentence)

    report.diagnosis = _strip(report.diagnosis)
    report.findings = _strip(report.findings)
    report.recommended_actions = [_strip(a) for a in report.recommended_actions]
    body = f"{report.diagnosis}\n{report.findings}".lower()

    for cit in report.citations:
        claimed = _strip(cit.supports)
        if claimed and claimed.lower() in body:  # the model actually quoted itself
            cit.supports = claimed
            continue
        recovered = by_chunk.get(cit.chunk_id, "")
        if not recovered:
            log.warning(
                "%s: citation %s has no recoverable report sentence — supports left empty "
                "(Agent 4 cannot link this one)",
                report.subject_id,
                cit.chunk_id,
            )
        cit.supports = recovered


SYSTEM = """\
You are a cardiac MRI reporting assistant. You write the report; you do not measure anything.

ABSOLUTE RULES — a violation makes the report invalid:

1. NUMBERS. You have not seen an image. Every number you are allowed to state is given to you
   in the MEASUREMENTS block. You may only state numbers that appear in the MEASUREMENTS block.
   Do not compute, do not round, do not average, do not convert units, do not infer new
   numbers, do not estimate a range. If a number is not in the MEASUREMENTS block, do not
   write it. Copy values exactly as printed (28.6 stays 28.6; it does not become "about 29"
   or "<30").

2. CITATIONS. Every clinical claim — every diagnosis, every classification, every recommended
   action — must be supported by one of the GUIDELINE PASSAGES, cited by its exact chunk_id.
   Do not invent a chunk_id: use only the chunk_ids listed in the GUIDELINE PASSAGES block,
   exactly as printed.
   Mark the grounding INLINE: end every sentence in `findings` and `diagnosis` that a passage
   supports with its marker, like this:
       The LVEF is 28.6%, consistent with HFrEF. [chunk_id: c1f3a90b]
   Then, for each citation, set `supports` to that same sentence FROM YOUR OWN REPORT —
   the sentence you just wrote and marked, copied verbatim. `supports` is NOT the guideline
   text. Do not copy the passage into `supports`; copy your own sentence.

3. HEART FAILURE THRESHOLDS (ESC 2021 and AHA/ACC/HFSA 2022 agree):
       LVEF <= 40      -> HFrEF
       LVEF 41 to 49   -> HFmrEF
       LVEF >= 50      -> HFpEF
   Use these and only these cut-points.

4. CONFLICTS. If two passages contradict each other, do NOT silently pick one. Add an entry to
   `conflicts` stating BOTH positions, each attributed to its source and chunk_id, and say
   which you followed and why.

5. SCOPE. Report only what the measurements support. Do not describe wall motion, perfusion,
   late gadolinium enhancement, valves, or anything else you were not given. Uncertain is an
   acceptable answer; invented is not.

Write for a cardiologist: precise, terse, no hedging filler."""

_NO_PASSAGES = """\
GUIDELINE PASSAGES: NONE. Retrieval is disabled for this run.
You have no guideline text. You must therefore reason from the measurements alone, and you
MUST return an EMPTY `citations` list. Do NOT invent a chunk_id, a source, a section or a page
number — a fabricated citation is worse than no citation. Leave `citations` as [].
"""


def _fmt_measurements(m: Measurements) -> str:
    return "\n".join(
        [
            "MEASUREMENTS (the ONLY numbers you may state):",
            f"  subject_id      : {m.subject_id}",
            f"  LV EDV          : {m.edv_ml} mL",
            f"  LV ESV          : {m.esv_ml} mL",
            f"  LV stroke volume: {m.sv_ml} mL",
            f"  LVEF            : {m.lvef_pct} %",
            f"  LV mass         : {m.lv_mass_g} g",
            f"  RV EDV          : {m.rv_edv_ml} mL",
            f"  RV ESV          : {m.rv_esv_ml} mL",
            f"  RV EF           : {m.rv_ef_pct} %",
            f"  HF category     : {m.hf_category}  (computed from LVEF by the quantification "
            "agent — adopt it, do not recompute it)",
            f"  measurement source: {m.source}",
            f"  near a guideline cut-point: {m.near_boundary}",
        ]
    )


def _fmt_passages(passages: list[Chunk]) -> str:
    if not passages:
        return _NO_PASSAGES
    out = ["GUIDELINE PASSAGES (cite by chunk_id, exactly as printed):"]
    for c in passages:
        head = f"[chunk_id: {c.chunk_id}] {c.source} — {c.section}, p.{c.page}"
        if c.class_of_recommendation or c.level_of_evidence:
            head += f" (Class {c.class_of_recommendation}, Level {c.level_of_evidence})"
        out += [head, f"    {c.text.strip()}", ""]
    return "\n".join(out)


def generate(
    m: Measurements,
    passages: list[Chunk],
    cfg: Any,
    pathology_prior: str | None = None,
    _llm: LLM | None = None,
) -> Report:
    """One subject -> one schema-valid, citation-carrying Report.

    `passages` empty (retriever: none, the H1 ablation) is a supported state, not an error:
    the prompt says so and the model generates ungrounded. It will then have zero citations,
    which is precisely the gap H1 exists to measure.

    `_llm` is an internal reuse hatch for generate_all — constructing one client per subject
    would mean 830 of them per run. Callers pass cfg and ignore it.
    """
    llm = _llm or LLM(cfg)
    user = [_fmt_measurements(m), "", _fmt_passages(passages)]
    if pathology_prior:
        user += [
            "",
            f"DATASET PATHOLOGY PRIOR: {pathology_prior}",
            "This is a dataset label, not an observation. Treat it as a weak prior only. If the "
            "measurements do not support it, say so — do not restate it as a finding.",
        ]
    user += [
        "",
        f"Write the report for subject {m.subject_id}.",
        "Ground every clinical claim in a cited passage; state no number that is not above.",
    ]

    report: Report = llm.complete_json(SYSTEM, "\n".join(user), Report)  # type: ignore[assignment]

    # A schema-valid report can still be a USELESS one: Pydantic is happy with
    # diagnosis="" because the field is a str, and smaller local models routinely put
    # everything in `findings` and leave `diagnosis` blank. An empty diagnosis is a
    # non-answer, and an evaluation that scores it as a report would be measuring nothing.
    # So: one explicit retry that names the omission, then fail loudly.
    if not report.diagnosis.strip():
        log.warning("%s: empty diagnosis -> retrying once with an explicit instruction", m.subject_id)
        retry = user + [
            "",
            "YOUR PREVIOUS ATTEMPT LEFT `diagnosis` EMPTY. That is not an acceptable report.",
            "`diagnosis` MUST be one sentence naming the most likely diagnosis and the finding "
            "that supports it (e.g. 'Severely reduced left ventricular systolic function "
            "consistent with HFrEF, LVEF 23.7%.'). `findings` is the measurement narrative; "
            "`diagnosis` is the conclusion. Fill BOTH.",
        ]
        report = llm.complete_json(SYSTEM, "\n".join(retry), Report)  # type: ignore[assignment]
        if not report.diagnosis.strip():
            raise ValueError(
                f"{m.subject_id}: the model returned an empty diagnosis twice. The report is "
                f"schema-valid but empty of content; refusing to record it as a diagnosis."
            )

    # Identity is ours, not the model's.
    report.subject_id = m.subject_id

    # ── ENFORCEMENT ──────────────────────────────────────────────────────────
    # Up to here the report is only SCHEMA-valid: Pydantic guarantees the shape, not the
    # truth. cmr/guardrails.py checks the four invariants that actually matter (citations
    # resolve; numbers are traceable; the HF category is Agent 2's arithmetic, not the
    # model's opinion; no claims about modalities we never imaged) and runs a bounded repair
    # loop, naming each violation back to the model.
    #
    # If it still fails, this RAISES rather than downgrading to best-effort. An ungrounded
    # clinical report is worse than no report — the same refusal posture the plausibility
    # gate takes on a broken mask, applied to a broken report.
    if cfg.get_path("llm.enforce_invariants", True):
        index = {c.chunk_id: c for c in passages}
        report, _fid, repairs = guardrails.enforce(
            report, m, passages, index, llm, SYSTEM, "\n".join(user), cfg
        )
        if repairs:
            log.warning("%s: guardrails applied %s", m.subject_id, "; ".join(repairs))

    # Backfill citation metadata from the real chunk. After enforcement every chunk_id
    # resolves, so this cannot silently invent provenance.
    index = {c.chunk_id: c for c in passages}
    for cit in report.citations:
        ch = index.get(cit.chunk_id)
        if ch is None:
            log.warning("%s: citation to unknown chunk_id %r", m.subject_id, cit.chunk_id)
            continue
        cit.source, cit.section, cit.page = ch.source, ch.section, ch.page
        cit.class_of_recommendation = ch.class_of_recommendation
        cit.level_of_evidence = ch.level_of_evidence

    _link_supports(report)

    log.info(
        "%s: report ok (%d citations, %d conflicts, %d tok)",
        m.subject_id,
        len(report.citations),
        len(report.conflicts),
        llm.last_usage.get("completion_tokens", 0),
    )
    return report


def generate_all(
    cfg: Any,
    measurements: list[Measurements],
    passages_by_subject: dict[str, list[Chunk]],
) -> list[Report]:
    """The whole cohort. A subject that fails after all retries is logged and skipped, not
    fatal — one flaky completion must not cost an 830-subject run."""
    llm = LLM(cfg)  # one client for the cohort, not one per subject
    reports: list[Report] = []
    for i, m in enumerate(measurements, 1):
        try:
            reports.append(generate(m, passages_by_subject.get(m.subject_id, []), cfg, _llm=llm))
        except Exception:  # noqa: BLE001 — the run continues; the loss is recorded
            log.exception("%s: report generation failed, skipping", m.subject_id)
        if i % 25 == 0:
            log.info("progress: %d/%d subjects, %d reports", i, len(measurements), len(reports))
    if len(reports) < len(measurements):
        log.error(
            "%d/%d subjects produced no report", len(measurements) - len(reports), len(measurements)
        )
    return reports
