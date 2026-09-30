"""Every contract between agents, in one place.

These Pydantic models are the software design AND the answer to panel comment C10
("how do you stop an error in the segmentation agent propagating downstream?").
The answer is: a malformed payload cannot cross a node boundary, because it will
not validate. That is enforcement, not intention.

Units are fixed and never renegotiated:
    volume -> mL          mass -> g          ejection fraction -> %
    spacing -> mm         distance -> mm
A function returning a bare float names its unit (`edv_ml`, not `edv`).

CANONICAL LABELS, everywhere in this codebase, without exception:
    0 = background   1 = LV cavity   2 = Myocardium   3 = RV cavity
ACDC natively uses 1=RV / 3=LV and is remapped at load. See cmr/checks.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

LV, MYO, RV = 1, 2, 3
LABEL_NAMES = {LV: "LV", MYO: "Myo", RV: "RV"}

Dataset = Literal["ACDC", "MnMs", "MnMs2"]
Phase = Literal["ED", "ES"]


class Status(StrEnum):
    """Terminal states of the pipeline. FAILED_SEGMENTATION is the refusal (H3)."""

    OK = "ok"
    GATE_FAILED = "gate_failed"
    FAILED_SEGMENTATION = "failed_segmentation"  # Agent 3 will NOT diagnose
    LLM_ERROR = "llm_error"


@dataclass(frozen=True)
class Subject:
    """One patient, canonicalised. The output of the data spine; the input to everything."""

    subject_id: str
    dataset: Dataset
    split: str  # train | val | test
    pathology: str
    ed_idx: int  # 0-indexed frame in the 4-D volume
    es_idx: int
    spacing_mm: tuple[float, float, float]
    img_ed: np.ndarray  # (X, Y, Z) float32
    img_es: np.ndarray
    gt_ed: np.ndarray | None  # (X, Y, Z) uint8, CANONICAL 1=LV 2=Myo 3=RV
    gt_es: np.ndarray | None
    vendor: str | None = None  # M&Ms / M&Ms-2
    centre: str | None = None  # M&Ms only (1..5)
    field_strength: float | None = None

    @property
    def has_gt(self) -> bool:
        return self.gt_ed is not None and self.gt_es is not None

    @property
    def voxel_ml(self) -> float:
        return float(np.prod(self.spacing_mm)) / 1000.0


class Measurements(BaseModel):
    """Agent 2's output. Deterministic arithmetic — no model, no training."""

    subject_id: str
    edv_ml: float
    esv_ml: float
    sv_ml: float
    lvef_pct: float
    lv_mass_g: float
    rv_edv_ml: float
    rv_esv_ml: float
    rv_ef_pct: float
    hf_category: Literal["HFrEF", "HFmrEF", "HFpEF"]
    near_boundary: bool = False  # within margin of a guideline cut-point -> triggers H2 loop
    source: Literal["groundtruth", "predicted"] = "predicted"

    def numeric_facts(self) -> dict[str, float]:
        """The only numbers the report is allowed to contain. factcheck.py enforces this."""
        return {
            "edv_ml": self.edv_ml,
            "esv_ml": self.esv_ml,
            "sv_ml": self.sv_ml,
            "lvef_pct": self.lvef_pct,
            "lv_mass_g": self.lv_mass_g,
            "rv_edv_ml": self.rv_edv_ml,
            "rv_esv_ml": self.rv_esv_ml,
            "rv_ef_pct": self.rv_ef_pct,
        }


class GateResult(BaseModel):
    """The plausibility gate. Rule-based; no training. Failure triggers refusal (C10/H3)."""

    passed: bool
    violations: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """One guideline passage.

    class_of_recommendation / level_of_evidence are the differentiator: NEITHER
    BAAI nor CardAIc carries this metadata on its chunks.

    Two societies grade recommendations on two different scales, and they are NOT
    interchangeable. class_of_recommendation / level_of_evidence are always on the ESC
    scale (I/IIa/IIb/III, A/B/C) because that is what every consumer — report.py,
    xai.py, retrieve._COR_RANK — already reads, and because detect_conflicts has to
    rank an ESC passage against an ACC/AHA one on a single axis.

    The normalisation is lossy in exactly one direction, so the original tokens are
    kept beside it rather than thrown away:

        ACC/AHA "3: Harm" B-NR  ->  cor=III  loe=B  cor_raw="3: Harm"  loe_raw="B-NR"
        ESC     IIa B           ->  cor=IIa  loe=B  cor_raw="IIa"      loe_raw="B"

    "3: No Benefit" and "3: Harm" both collapse to ESC III, and B-R/B-NR both collapse
    to B — a real clinical difference that cor_raw/loe_raw preserve. Cite the normalised
    field; quote the raw one.
    """

    chunk_id: str  # sha256 of text — stable across rebuilds, so citations are verifiable
    text: str
    source: str  # "2021 ESC HF Guideline"
    source_id: str  # "esc_hf_2021"
    section: str
    page: int
    class_of_recommendation: str | None = None  # I | IIa | IIb | III  (ESC scale, always)
    level_of_evidence: str | None = None  # A | B | C                  (ESC scale, always)
    cor_scheme: Literal["ESC", "ACC_AHA"] | None = None  # which scale it was graded on
    cor_raw: str | None = None  # verbatim class token, e.g. "3: Harm"
    loe_raw: str | None = None  # verbatim level token, e.g. "B-NR"
    tokens: int = 0


class Citation(BaseModel):
    """Passage-level, in the diagnostic path. BAAI's citations are document-level and
    live in a conversational sidecar; its structured report carries none at all."""

    chunk_id: str  # MUST resolve against the index. factcheck.py verifies every one.
    source: str
    section: str
    page: int
    class_of_recommendation: str | None = None
    level_of_evidence: str | None = None
    supports: str = ""  # the report sentence this passage grounds -> the XAI link


class Report(BaseModel):
    """Agent 3's output. Schema-constrained at generation time, so it cannot be malformed."""

    subject_id: str
    diagnosis: str
    differential: list[str] = Field(default_factory=list)
    hf_category: Literal["HFrEF", "HFmrEF", "HFpEF", "not_applicable"] = "not_applicable"
    findings: str = ""
    citations: list[Citation] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)  # C11 — both sides, attributed


class Fidelity(BaseModel):
    """The headline metric. BAAI claims 'zero hallucination' qualitatively;
    neither competitor quantifies it. This does."""

    numeric_fidelity: float  # fraction of numbers in the report traceable to Agent 2
    citation_fidelity: float  # fraction of citations resolving to a real chunk
    n_numbers: int = 0
    n_citations: int = 0
    unsupported_numbers: list[float] = Field(default_factory=list)
    dangling_citations: list[str] = Field(default_factory=list)


class Explanation(BaseModel):
    """THE contribution of Agent 4. Not 'a heatmap and a citation' — the LINK between them.

    BAAI has no visual XAI at all. CardAIc has visual panels but they are not tied to
    the textual rationale. This object is the tie.
    """

    sentence: str  # a claim in the report
    chunk_id: str | None  # the guideline passage that grounds it
    heatmap_path: str | None  # the Seg-Grad-CAM region that supports it
    entropy_path: str | None  # ensemble uncertainty over the same region
    structure: Literal["LV", "Myo", "RV"] | None = None
    mean_uncertainty: float | None = None


class RunState(BaseModel):
    """The orchestration state object. One per subject. Serialised to the JSONL trace."""

    subject_id: str
    dataset: str = ""
    config_id: str = ""
    status: Status = Status.OK
    iteration: int = 0  # bounded by orchestration.max_iterations
    measurements: Measurements | None = None
    gate: GateResult | None = None
    passages: list[Chunk] = Field(default_factory=list)
    report: Report | None = None
    fidelity: Fidelity | None = None
    explanations: list[Explanation] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)  # the audit trail

    def log(self, node: str, **kw) -> None:
        self.events.append({"node": node, "iteration": self.iteration, **kw})

    @property
    def refused(self) -> bool:
        return self.status is Status.FAILED_SEGMENTATION
