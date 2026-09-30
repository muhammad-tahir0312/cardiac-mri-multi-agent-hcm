"""Orchestration — the LangGraph state machine.

    segment -> quantify -> [gate] -> retrieve -> report -> factcheck -> xai -> END
                             |
                             +-- fail, budget left -----> recover (TTA)  --> segment
                             +-- fail, budget spent ----> REFUSE          --> END
                             +-- pass, near boundary ---> recompute (TTA) --> segment

Two behaviours here are the actual research contribution. The graph itself is plumbing.

REFUSAL (panel comment C10, hypothesis H3)
    The panel asked: if the segmentation agent errs on a difficult apical slice, what
    stops that error cascading into quantification, reasoning and explanation? The
    answer is not "we hope it doesn't". On unrecoverable gate failure the pipeline
    emits status=FAILED_SEGMENTATION and **Agent 3 does not diagnose at all**. No
    silent downstream reasoning on a broken mask. That refusal IS the failure-isolation
    claim, made concrete, logged, and testable.

DECISION-BOUNDARY-AWARE RECOMPUTATION (panel comment C12, hypothesis H2)
    Slide 22 promised that "guideline evidence can refine segmentation masks". That is
    mechanically impossible: a text-retrieval agent has no imaging signal and cannot
    touch a voxel. The reframing: Agent 3 knows the guideline DECISION BOUNDARIES. When
    a computed value lands near one (LVEF 39.4% against the HFrEF cut-point at 40%), a
    re-measurement is requested, and Agent 1 re-runs with test-time augmentation and
    full ensembling to tighten the estimate. Triggered by guideline knowledge, executed
    in imaging space, BOUNDED at max_iterations, fully traced.

    Prior art, cited honestly: CardAIc-Agents' adaptive workflow refines the PLAN when
    evidence changes (0.80 -> 0.87). We refine the MEASUREMENT when it lands near a
    clinical decision boundary. Different trigger, different action, different failure mode.

Every node boundary is a Pydantic model (cmr/types.py). A malformed payload cannot cross
one, because it will not validate. That is the C10 answer at the software level.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from langgraph.graph import END, StateGraph

from . import quantify
from .config import Config, artifacts
from .types import GateResult, RunState, Status, Subject

log = logging.getLogger("cmr.graph")


@dataclass
class Deps:
    """Heavy objects built once and shared across all subjects — a Segmenter that
    reloads 3 checkpoints per patient would dominate the runtime."""

    cfg: Config
    segmenter: Any = None  # cmr.seg.Segmenter
    retriever: Any = None  # cmr.retrieve.Retriever
    llm: Any = None  # cmr.llm.LLM

    # per-subject scratch. Masks are numpy and do not belong in a serialised state object.
    subject: Subject | None = None
    mask_ed: np.ndarray | None = None
    mask_es: np.ndarray | None = None
    ent_ed: np.ndarray | None = None
    ent_es: np.ndarray | None = None
    tta: bool = False
    timings: dict[str, float] = field(default_factory=dict)

    # Optional live sink: called with the SAME dict s.log() records, the instant a node
    # finishes, from whatever thread run_subject() is running on. None (the default, every
    # existing batch run) costs nothing — this is additive, not a replacement for the JSONL
    # trace. cmr/app.py's live-run viewer is the only caller that sets it.
    on_event: Callable[[dict], None] | None = None


def _emit(d: Deps, s: RunState, node: str, **kw) -> None:
    s.log(node, **kw)
    if d.on_event is not None:
        d.on_event(s.events[-1])


def _timed(name: str, d: Deps):
    class _T:
        def __enter__(self_):
            self_.t = time.perf_counter()
            return self_

        def __exit__(self_, *a):
            d.timings[name] = d.timings.get(name, 0.0) + time.perf_counter() - self_.t

    return _T()


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────
def _node_segment(d: Deps):
    def run(s: RunState) -> dict:
        cfg, subj = d.cfg, d.subject
        backend = cfg.segmentation.backend

        with _timed("segment", d):
            if backend == "groundtruth":
                # Lets the entire downstream pipeline be built and tested before
                # segmentation exists. The thin vertical slice.
                d.mask_ed, d.mask_es = subj.gt_ed, subj.gt_es
                d.ent_ed = d.ent_es = None
            else:
                seg = d.segmenter
                seg.tta = d.tta  # the feedback loop turns this on to tighten an estimate
                d.mask_ed, d.ent_ed = seg.segment(subj.img_ed)
                d.mask_es, d.ent_es = seg.segment(subj.img_es)

        _emit(d, s, "segment", backend=backend, tta=d.tta)
        return {"events": s.events}

    return run


def _node_quantify(d: Deps):
    def run(s: RunState) -> dict:
        with _timed("quantify", d):
            m = quantify.measure(
                d.mask_ed,
                d.mask_es,
                d.subject.spacing_mm,
                s.subject_id,
                d.cfg,
                source="groundtruth"
                if d.cfg.segmentation.backend == "groundtruth"
                else "predicted",
            )
        _emit(
            d,
            s,
            "quantify",
            lvef=round(m.lvef_pct, 2),
            hf=m.hf_category,
            near_boundary=m.near_boundary,
        )
        return {"measurements": m, "events": s.events}

    return run


def _node_gate(d: Deps):
    def run(s: RunState) -> dict:
        if not d.cfg.orchestration.gate:
            # the no_gate ablation (H3) — pass everything through, including garbage
            g = GateResult(passed=True, violations=[])
            _emit(d, s, "gate", enabled=False)
            return {"gate": g, "events": s.events}
        with _timed("gate", d):
            g = quantify.gate(s.measurements, d.mask_ed, d.mask_es, d.cfg)
        _emit(d, s, "gate", passed=g.passed, violations=g.violations)
        return {"gate": g, "events": s.events}

    return run


def _node_recover(d: Deps):
    """Gate failed but we still have budget. Retry harder: TTA + full ensemble."""

    def run(s: RunState) -> dict:
        d.tta = True
        _emit(d, s, "recover", reason="gate_failed", violations=s.gate.violations)
        log.info("[%s] gate failed %s -> retry with TTA", s.subject_id, s.gate.violations)
        return {"iteration": s.iteration + 1, "status": Status.GATE_FAILED, "events": s.events}

    return run


def _node_recompute(d: Deps):
    """H2. The value landed near a guideline decision boundary. Re-measure precisely.

    This is the ONLY place guideline knowledge reaches back into imaging space, and it
    does so by requesting a better measurement — not by editing voxels.
    """

    def run(s: RunState) -> dict:
        d.tta = True
        m = s.measurements
        _emit(
            d,
            s,
            "recompute",
            reason="near_decision_boundary",
            lvef=round(m.lvef_pct, 2),
            hf_category=m.hf_category,
        )
        log.info(
            "[%s] LVEF %.1f%% is near a guideline cut-point -> re-measuring with TTA",
            s.subject_id,
            m.lvef_pct,
        )
        return {"iteration": s.iteration + 1, "events": s.events}

    return run


def _node_refuse(d: Deps):
    """The mask is unusable and the retry budget is spent. Agent 3 does NOT diagnose.

    This node emitting no report is hypothesis H3, executable.
    """

    def run(s: RunState) -> dict:
        _emit(d, s, "refuse", violations=s.gate.violations if s.gate else [])
        log.warning(
            "[%s] REFUSING to diagnose: segmentation failed the plausibility gate (%s)",
            s.subject_id,
            s.gate.violations if s.gate else "unknown",
        )
        return {"status": Status.FAILED_SEGMENTATION, "report": None, "events": s.events}

    return run


def _node_retrieve(d: Deps):
    def run(s: RunState) -> dict:
        if d.cfg.retrieval.mode == "none" or d.retriever is None:
            _emit(d, s, "retrieve", mode="none", n=0)  # the H1 (no-RAG) ablation
            return {"passages": [], "events": s.events}
        from .retrieve import build_query

        with _timed("retrieve", d):
            q = build_query(s.measurements.numeric_facts(), s.measurements.hf_category)
            passages = d.retriever.retrieve(q)
        _emit(
            d,
            s,
            "retrieve",
            mode=d.cfg.retrieval.mode,
            n=len(passages),
            chunks=[p.chunk_id for p in passages],
        )
        return {"passages": passages, "events": s.events}

    return run


def _node_report(d: Deps):
    def run(s: RunState) -> dict:
        from .guardrails import UngroundedReport

        with _timed("report", d):
            if d.cfg.llm.provider == "none":
                from baselines.template_report import template_report

                rep = template_report(s.measurements, d.cfg)  # the zero-LLM floor
            else:
                from . import report as report_mod

                # d.llm is built ONCE per run (cli.py). Passing it matters: constructing an
                # LLM per subject re-handshakes the client every patient and, on a local
                # backend, can force a model reload -- it was costing ~20 s of the ~134 s
                # per-subject wall-clock before this was wired through.
                try:
                    rep = report_mod.generate(s.measurements, s.passages, d.cfg, _llm=d.llm)
                except UngroundedReport as e:
                    # The model could not produce a report satisfying the invariants (every
                    # citation resolves; every number traceable). We do NOT fall back to a
                    # best-effort report: an ungrounded clinical report is worse than none.
                    # Same refusal posture the gate takes on a broken mask, applied to a
                    # broken report. This is what makes "cannot emit a hallucination" true.
                    _emit(d, s, "report", refused=True, reason=str(e)[:200])
                    log.error("[%s] REFUSING to emit an ungrounded report: %s", s.subject_id, e)
                    return {"status": Status.LLM_ERROR, "report": None, "events": s.events}
        _emit(
            d,
            s,
            "report",
            n_citations=len(rep.citations),
            hf=rep.hf_category,
            n_conflicts=len(rep.conflicts),
        )
        return {"report": rep, "events": s.events}

    return run


def _node_factcheck(d: Deps):
    def run(s: RunState) -> dict:
        if s.report is None:  # the report node refused; there is nothing to score
            return {"events": s.events}
        from .factcheck import check

        index = d.retriever.index if d.retriever is not None else {}
        with _timed("factcheck", d):
            f = check(s.report, s.measurements, index)
        _emit(
            d,
            s,
            "factcheck",
            numeric=round(f.numeric_fidelity, 3),
            citation=round(f.citation_fidelity, 3),
        )
        return {"fidelity": f, "events": s.events}

    return run


def _node_xai(d: Deps):
    def run(s: RunState) -> dict:
        if s.report is None:
            return {"events": s.events}
        if not d.cfg.xai.get("save_overlays", True) or d.cfg.segmentation.backend == "groundtruth":
            return {"events": s.events}
        try:
            from .xai import explain

            with _timed("xai", d):
                ex = explain(
                    d.subject,
                    d.mask_ed,
                    d.ent_ed,
                    s.report,
                    d.cfg,
                    model=getattr(d.segmenter, "model", None),
                )
            _emit(d, s, "xai", n=len(ex))
            return {"explanations": ex, "events": s.events}
        except Exception as e:  # XAI is a renderer; it must never sink a whole run
            log.warning("[%s] xai failed: %s", s.subject_id, e)
            _emit(d, s, "xai", error=str(e))
            return {"events": s.events}

    return run


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────
def _route_after_gate(d: Deps):
    max_iter = d.cfg.orchestration.max_iterations

    def route(s: RunState) -> str:
        if not s.gate.passed:
            if s.iteration < max_iter:
                return "recover"
            return "refuse" if d.cfg.orchestration.refuse_on_gate_failure else "retrieve"

        # gate passed. Is the value sitting on a clinical decision boundary? (H2)
        if (
            d.cfg.orchestration.feedback
            and s.measurements.near_boundary
            and s.iteration < max_iter
            and not d.tta  # TTA already applied — a second pass would change nothing
        ):
            return "recompute"
        return "retrieve"

    return route


# ─────────────────────────────────────────────────────────────────────────────
def build(d: Deps):
    g = StateGraph(RunState)
    for name, fn in [
        ("segment", _node_segment(d)),
        ("quantify", _node_quantify(d)),
        ("gate", _node_gate(d)),
        ("recover", _node_recover(d)),
        ("recompute", _node_recompute(d)),
        ("refuse", _node_refuse(d)),
        ("retrieve", _node_retrieve(d)),
        ("report", _node_report(d)),
        ("factcheck", _node_factcheck(d)),
        ("xai", _node_xai(d)),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("segment")
    g.add_edge("segment", "quantify")
    g.add_edge("quantify", "gate")
    g.add_conditional_edges(
        "gate",
        _route_after_gate(d),
        {
            "recover": "recover",
            "recompute": "recompute",
            "refuse": "refuse",
            "retrieve": "retrieve",
        },
    )
    g.add_edge("recover", "segment")  # the bounded loops
    g.add_edge("recompute", "segment")
    g.add_edge("refuse", END)  # <- H3: no report is emitted on this path
    g.add_edge("retrieve", "report")
    g.add_edge("report", "factcheck")
    g.add_edge("factcheck", "xai")
    g.add_edge("xai", END)
    return g.compile()


def run_subject(subject: Subject, d: Deps, config_id: str = "") -> RunState:
    d.subject = subject
    d.tta = d.cfg.segmentation.tta
    d.timings = {}
    app = build(d)
    init = RunState(subject_id=subject.subject_id, dataset=subject.dataset, config_id=config_id)
    # recursion_limit guards against a routing bug; the loops are already bounded by
    # max_iterations, so hitting this would itself be a defect worth crashing on.
    out = app.invoke(init, {"recursion_limit": 50})
    state = RunState(**out) if isinstance(out, dict) else out
    state.events.append({"node": "_timings", **{k: round(v, 3) for k, v in d.timings.items()}})
    return state


def write_trace(cfg: Config, config_id: str, s: RunState) -> Path:
    """One JSONL per subject. This trace is the evidence for H3 and the audit log
    for the whole system: which node ran, in what order, with what verdict."""
    out = artifacts(cfg, "runs", config_id, "traces") / f"{s.subject_id}.jsonl"
    with open(out, "w") as f:
        for e in s.events:
            f.write(json.dumps(e, default=str) + "\n")
    return out


def write_report(cfg: Config, config_id: str, s: RunState) -> Path | None:
    if s.report is None:
        return None
    out = artifacts(cfg, "runs", config_id, "reports") / f"{s.subject_id}.json"
    payload = {
        "report": s.report.model_dump(),
        "measurements": s.measurements.model_dump() if s.measurements else None,
        "fidelity": s.fidelity.model_dump() if s.fidelity else None,
        "status": s.status.value,
        "iterations": s.iteration,
        "explanations": [e.model_dump() for e in s.explanations],
        # Batch subjects are always ACDC/MnMs/MnMs2 and app.py's cohort() (built from
        # gt_measurements.parquet) already knows their dataset. A live-uploaded subject has
        # no row in that parquet, so it has no other way to tell the viewer which dataset
        # loader (or upload store) to render its images from.
        "dataset": s.dataset,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out
