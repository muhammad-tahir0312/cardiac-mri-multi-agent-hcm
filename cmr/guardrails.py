"""Guardrails — hallucination enforcement and the security boundary.

The honest framing first, because the thesis has to survive cross-examination:

    A "100% hallucination-free" RAG system does not exist and cannot exist. Retrieval can
    always miss a relevant passage, and a language model can always write a sentence that is
    grounded, correctly cited, and still clinically wrong. Nobody has solved that -- BAAI
    included, which asserts "zero hallucination" qualitatively and never measures it.

What CAN be built, and what this module builds, are ENFORCEABLE INVARIANTS: properties that
are checked mechanically and whose violation prevents the report from being emitted at all.
Not "the model usually cites correctly" -- "a report with a fabricated citation cannot leave
this system."

    INVARIANT 1  Every citation resolves.
                 Each Citation.chunk_id must exist in the corpus index. A chunk_id the model
                 invented is not a weak citation, it is a fabricated one, and it is deleted.

    INVARIANT 2  Every number is traceable.
                 Every numeric token in the diagnosis/findings must either (a) match one of
                 Agent 2's measurements within tolerance, or (b) appear verbatim in a passage
                 the report actually cites. Numbers with no provenance are hallucinated
                 measurements -- the single most dangerous failure mode in a clinical report.

    INVARIANT 3  Arithmetic is not up for negotiation.
                 hf_category is computed deterministically by Agent 2 from LVEF and the
                 guideline cut-points. If the model disagrees, the model is wrong. It does not
                 get a vote on arithmetic.

    INVARIANT 4  No claims about unseen modalities.
                 The system sees a short-axis cine segmentation and nothing else. A report
                 that describes late gadolinium enhancement, perfusion, valves or wall motion
                 is describing something it never looked at, however plausible it sounds.

Enforcement is a bounded repair loop: generate -> validate -> if violations, regenerate with
the violations named explicitly -> at most `max_repairs` attempts -> if it still fails, the
report is NOT recorded as a grounded diagnosis. Failing loudly is the point.

SECURITY. The threat model is stated honestly rather than inflated (panel comment C15):

    The only text reaching the model is (a) guideline passages we ingested ourselves and
    (b) numeric JSON produced by our own code. There is NO user-supplied free text anywhere
    in the pipeline, so the prompt-injection surface is genuinely small. Saying so is more
    credible than pretending to a threat we do not have.

    It is not zero, though, and the non-zero part is the corpus: a PDF is an untrusted byte
    stream. Text can be white-on-white, in a form field, or in a footnote nobody reads, and a
    future corpus (a hospital's own protocol documents, say) is far less trustworthy than an
    ESC guideline. So passages are treated as DATA, never as instructions:

      - scan_injection()  quarantines chunks carrying imperative/role-play patterns at
                          INGEST time, before they can ever be retrieved.
      - Passages are delimited and the system prompt states they are reference text only.
      - The model has no tools, no network, no filesystem. It emits one JSON object against
        a fixed schema. There is no action for an injected instruction to take.
      - With provider=ollama nothing leaves the machine at all.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from .types import Chunk, Fidelity, Measurements, Report

if TYPE_CHECKING:
    from .llm import LLM

log = logging.getLogger("cmr.guardrails")

# ─────────────────────────────────────────────────────────────────────────────
# Security: the corpus is an untrusted byte stream
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that have no business appearing in a cardiology guideline and every business
# appearing in a prompt-injection payload. Deliberately conservative: a false positive
# quarantines one chunk out of a thousand, which costs nothing. A false negative lets an
# instruction into the prompt.
_INJECTION = [
    re.compile(p, re.I)
    for p in (
        r"\bignore (all |any |the )?(previous|prior|above|preceding) (instruction|prompt|direction)",
        r"\bdisregard (all |any |the )?(previous|prior|above)",
        r"\byou are now\b",
        r"\bact as\b.{0,30}\b(assistant|ai|model|system)\b",
        r"^\s*(system|assistant|user)\s*:",  # role markers
        r"<\s*/?\s*(system|assistant|user|im_start|im_end)\s*>",
        r"\bnew instructions?\b",
        r"\boverride\b.{0,20}\b(instruction|rule|constraint|guardrail)",
        r"```",  # code fences: no guideline uses them; injections love them
        r"\bprint\b.{0,20}\b(your|the)\b.{0,20}\b(prompt|instruction)",
    )
]


def scan_injection(text: str) -> list[str]:
    """Return the names of injection patterns present. Empty list == clean.

    Run at CORPUS BUILD time, not at retrieval time: a poisoned chunk should never reach
    the index in the first place, so that no query can ever surface it.
    """
    return [p.pattern for p in _INJECTION if p.search(text)]


def sanitise(text: str) -> str:
    """Neutralise structural tokens that could break out of the passage delimiter.

    We do not silently rewrite clinical content -- that would be worse than the disease.
    We only defang role markers and fences, and quarantine (see scan_injection) anything
    that looks like a genuine instruction.
    """
    text = re.sub(r"<\s*/?\s*(system|assistant|user|im_start|im_end)\s*>", " ", text, flags=re.I)
    return text.replace("```", "`​``")


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination: the invariants
# ─────────────────────────────────────────────────────────────────────────────

# Numbers that are structural, not clinical: years, page numbers, guideline class markers.
# Counting these as "unsupported measurements" would make the metric noise.
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_NUM = re.compile(r"\d+(?:\.\d+)?")

# The model marks grounding inline: "The LVEF is 23.7%. [chunk_id: ce297edabb7ec8c0]".
# chunk_ids are HEX, so they contain digits — 'ce297edab...' contains "297". Scanning for
# numbers before these markers are stripped makes the guardrail invent its own violations
# and refuse every report on earth. Strip first, then count. (report.py strips them for
# real in _link_supports, but that runs AFTER enforcement and enforcement may regenerate
# the report, so guardrails cannot depend on someone else having done it.)
_MARKER = re.compile(r"\[\s*chunk_id\s*:\s*[^\]]*\]", re.I)

# Modalities this system never observed. A short-axis cine segmentation cannot show them.
_UNSEEN = {
    "late gadolinium": "LGE was never acquired or analysed by this system",
    "gadolinium": "no contrast imaging was performed",
    "lge": "LGE was never acquired or analysed by this system",
    "perfusion": "no perfusion imaging was performed",
    "t1 mapping": "no parametric mapping was performed",
    "t2 mapping": "no parametric mapping was performed",
    "regurgitation": "no valvular assessment was performed",
    "stenosis": "no valvular or coronary assessment was performed",
    "wall motion abnormalit": "regional wall motion was not assessed",
}


def _numbers_in(text: str) -> list[float]:
    """Every number the report ASSERTS. Deliberately not every digit it contains.

    Excluded, because counting them would measure noise rather than hallucination:
      - inline [chunk_id: ...] markers  -- hex ids contain digits
      - years                           -- "the 2021 ESC guideline" is a reference, not a
                                           measurement
      - bare list ordinals ("1.", "2)") -- enumeration, not a claim
    """
    scrubbed = _MARKER.sub(" ", text)
    scrubbed = _YEAR.sub(" ", scrubbed)
    scrubbed = re.sub(r"(?m)^\s*\d+[.)]\s+", " ", scrubbed)  # "1. Consider..." list markers
    return [float(m.group()) for m in _NUM.finditer(scrubbed)]


def violations(
    report: Report,
    m: Measurements,
    index: dict[str, Chunk],
    tol: float = 0.5,
) -> list[str]:
    """Every way this report breaks an invariant. Empty list == emittable.

    The strings are written to be fed straight back to the model in the repair prompt, so
    they name the offence and the fix rather than just flagging it.
    """
    v: list[str] = []

    # ── INVARIANT 1: citations resolve ───────────────────────────────────────
    for c in report.citations:
        if c.chunk_id not in index:
            v.append(
                f"FABRICATED CITATION: chunk_id '{c.chunk_id}' does not exist in the corpus. "
                f"You may only cite chunk_ids printed in the GUIDELINE PASSAGES block."
            )

    # ── INVARIANT 2: numbers are traceable ───────────────────────────────────
    facts = list(m.numeric_facts().values())
    cited_text = " ".join(index[c.chunk_id].text for c in report.citations if c.chunk_id in index)
    for n in _numbers_in(f"{report.diagnosis} {report.findings}"):
        from_agent2 = any(abs(n - f) <= tol for f in facts)
        # A guideline threshold quoted from a passage the report actually cites is legitimate
        # provenance -- "LVEF <= 40%" is not a hallucinated measurement.
        from_passage = re.search(rf"\b{re.escape(f'{n:g}')}\b", cited_text) is not None
        if not (from_agent2 or from_passage):
            v.append(
                f"UNSUPPORTED NUMBER: {n:g} appears in your report but comes from neither the "
                f"MEASUREMENTS block nor any passage you cited. Remove it or cite its source. "
                f"Never compute, round, or infer a new number."
            )

    # ── INVARIANT 3: arithmetic is not negotiable ────────────────────────────
    if report.hf_category not in (m.hf_category, "not_applicable"):
        v.append(
            f"WRONG HF CATEGORY: you said {report.hf_category}. LVEF is {m.lvef_pct:.1f}%, which "
            f"is {m.hf_category} under ESC 2021 / AHA-ACC-HFSA 2022 (HFrEF <=40, HFmrEF 41-49, "
            f"HFpEF >=50). The category is arithmetic, not judgement. Use {m.hf_category}."
        )

    # ── INVARIANT 4: no claims about unseen modalities ───────────────────────
    body = f"{report.diagnosis} {report.findings} {' '.join(report.recommended_actions)}".lower()
    for term, why in _UNSEEN.items():
        # Recommending an LGE study is fine. ASSERTING an LGE finding is not.
        if term in body and not re.search(
            rf"(consider|recommend|suggest|obtain|refer|further|would|should be)\b[^.]{{0,60}}{re.escape(term)}",
            body,
        ):
            v.append(
                f"CLAIM ABOUT UNSEEN DATA: your report asserts something about '{term}', but {why}. "
                f"You may RECOMMEND further imaging; you may not report its findings."
            )

    return v


def fidelity(report: Report, m: Measurements, index: dict[str, Chunk], tol: float = 0.5) -> Fidelity:
    """The headline metric. Measured AFTER enforcement, so on an enforced pipeline it should
    read 1.0/1.0 -- and if it ever does not, the enforcement is broken and we want to know."""
    facts = list(m.numeric_facts().values())
    cited_text = " ".join(index[c.chunk_id].text for c in report.citations if c.chunk_id in index)
    nums = _numbers_in(f"{report.diagnosis} {report.findings}")

    unsupported = [
        n
        for n in nums
        if not any(abs(n - f) <= tol for f in facts)
        and not re.search(rf"\b{re.escape(f'{n:g}')}\b", cited_text)
    ]
    dangling = [c.chunk_id for c in report.citations if c.chunk_id not in index]

    return Fidelity(
        numeric_fidelity=1.0 - len(unsupported) / len(nums) if nums else 1.0,
        citation_fidelity=(
            1.0 - len(dangling) / len(report.citations) if report.citations else 0.0
        ),
        n_numbers=len(nums),
        n_citations=len(report.citations),
        unsupported_numbers=unsupported,
        dangling_citations=dangling,
    )


class UngroundedReport(RuntimeError):
    """The model could not produce a report satisfying the invariants. We do not downgrade
    to 'best effort' -- an ungrounded clinical report is worse than no report."""


def enforce(
    report: Report,
    m: Measurements,
    passages: list[Chunk],
    index: dict[str, Chunk],
    llm: LLM,
    system: str,
    user: str,
    cfg: Any,
) -> tuple[Report, Fidelity, list[str]]:
    """The bounded repair loop. Returns (report, fidelity, repairs_applied).

    Raises UngroundedReport if the invariants still fail after `max_repairs` attempts. The
    orchestrator turns that into Status.LLM_ERROR and emits NO diagnosis -- the same refusal
    posture the plausibility gate takes on a broken mask, applied to a broken report.
    """
    max_repairs = int(cfg.get_path("llm.max_repairs", 2))
    applied: list[str] = []

    for attempt in range(max_repairs + 1):
        v = violations(report, m, index)
        if not v:
            return report, fidelity(report, m, index), applied

        if attempt == max_repairs:
            # Last resort BEFORE giving up: delete what is provably fabricated. A citation
            # that resolves to nothing is not a weak citation, it is a fiction, and a report
            # is strictly better without it.
            n_before = len(report.citations)
            report.citations = [c for c in report.citations if c.chunk_id in index]
            if len(report.citations) < n_before:
                applied.append(f"stripped {n_before - len(report.citations)} fabricated citation(s)")
            remaining = violations(report, m, index)
            if not remaining:
                log.warning("%s: emitted after stripping fabricated citations", m.subject_id)
                return report, fidelity(report, m, index), applied
            raise UngroundedReport(
                f"{m.subject_id}: report still violates {len(remaining)} invariant(s) after "
                f"{max_repairs} repair attempt(s). Refusing to emit an ungrounded clinical "
                f"report. First violation: {remaining[0]}"
            )

        log.warning(
            "%s: %d invariant violation(s), repair attempt %d/%d",
            m.subject_id, len(v), attempt + 1, max_repairs,
        )
        applied.append(f"repair {attempt + 1}: {len(v)} violation(s)")
        repair = "\n".join(
            [
                user,
                "",
                "YOUR PREVIOUS REPORT VIOLATED THE RULES. Fix EVERY item below and re-emit the",
                "whole report. Do not argue, do not explain, do not keep any offending content:",
                *(f"  {i + 1}. {x}" for i, x in enumerate(v)),
            ]
        )
        report = llm.complete_json(system, repair, Report)  # type: ignore[assignment]
        report.subject_id = m.subject_id

    raise AssertionError("unreachable")
