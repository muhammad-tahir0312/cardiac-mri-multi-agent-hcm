# BUILD.md — Engineering Plan

**Multi-Agent CMR Analysis & Decision Support · Muhammad Tahir · MCS by Research, Universiti Malaya**
Revision 2 · 2026-07-11 · supersedes the v1 sketch.

Scope: **code only.** Every line of software this thesis needs, what it does, what it depends on,
what proves it works, and in what order it gets written.

| Companion | Holds |
|---|---|
| [PROCESS.md](PROCESS.md) | Verified facts — datasets, hardware, competitor papers, defect register |
| [PLAN.md](PLAN.md) | Research rationale — why each component exists |
| [CD_TIMELINE.md](CD_TIMELINE.md) | Candidature-defence schedule, panel comments, thesis writing |
| **BUILD.md** (this) | **The software** |

---

## 0. Summary

**~1,450 lines of Python across 16 modules, one package, one CLI.** Fourteen work packages,
~45 working days of engineering, of which **only two require a GPU** and neither is on the critical
path. Two-thirds of the system runs on the M4 with no network and no API key.

The system is small. What makes it a thesis is not its size — it is three facts embedded in it:
the **canonical label remap** (without which every cross-dataset number is silently wrong), the
**CoR/LoE guideline chunk metadata** (which no competitor carries), and the **factual-consistency
check** (which no competitor quantifies). Everything else is glue, and glue should be thin.

**Guiding rule for the whole build:** get one patient end-to-end to a report — with ground-truth masks
and a stub LLM if necessary — *before* optimising any single agent. A thin vertical slice through all
four agents de-risks every interface. A perfectly tuned segmentation model with nothing downstream
de-risks nothing.

---

## 1. Repository layout

The datasets stay where they are and are **never written to**. All code and all outputs live in a new
sibling package.

```
cardio_dataset/
├── ACDC/  MnMs/  MnMs2/          read-only. Never touched by the code.
├── code/                          the four planning documents (this file included)
└── cmr/                           ← the package
    ├── pyproject.toml
    ├── configs/
    │   └── experiments.yaml       the ablation grid, as data (§7)
    ├── cmr/
    │   ├── __init__.py
    │   ├── config.py              paths, seeds, constants, model IDs
    │   ├── types.py               every inter-agent contract (Pydantic)      ~120
    │   ├── data.py                three adapters → one canonical Subject     ~180
    │   ├── checks.py              the regression guards                       ~50
    │   ├── quantify.py            Agent 2 — volumes, EF, mass, gate           ~90
    │   ├── seg.py                 Agent 1 — CineMA ensemble → masks+entropy  ~130
    │   ├── corpus.py              guideline PDFs → chunks w/ CoR-LoE         ~140
    │   ├── retrieve.py            Agent 3a — MedCPT + FAISS + BM25 hybrid    ~100
    │   ├── report.py              Agent 3b — schema-constrained generation   ~110
    │   ├── factcheck.py           the headline metric                         ~70
    │   ├── xai.py                 Agent 4 — Seg-Grad-CAM, entropy, the link  ~130
    │   ├── graph.py               orchestration, gate, refusal, recompute    ~160
    │   ├── eval.py                all metrics + significance                 ~170
    │   ├── figures.py             every thesis figure/table, from artifacts   ~90
    │   └── cli.py                 the single entrypoint                       ~80
    ├── baselines/
    │   ├── run_baai.py            BAAI Cardiac Agent on ACDC/M&Ms  (GPU)      ~90
    │   ├── nnunet_convert.py      ACDC → nnU-Net raw format        (GPU)      ~60
    │   └── template_report.py     zero-LLM floor                              ~40
    ├── tests/                     the runnable checks (§8)                   ~200
    └── artifacts/                 all outputs. gitignored. §3.
```

**One package, one CLI, no service layer.** There is no web app, no API server, and no daemon. This is
a batch research pipeline; it should look like one.

---

## 2. Tooling — and what we deliberately do not build

```bash
brew install python@3.11              # CineMA needs 3.11; system 3.9 is unusable
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`pyproject.toml` dependencies:

| Package | For | Note |
|---|---|---|
| `torch` | CineMA inference, Seg-Grad-CAM | MPS backend; no CUDA on this machine |
| `nibabel` | NIfTI I/O | M&Ms GT is 4-D; M&Ms-2 is uncompressed |
| `numpy`, `scipy`, `pandas` | everything | |
| `monai` | `DiceMetric`, `HausdorffDistanceMetric` | Do not hand-write HD95 |
| `transformers` | MedCPT / BioBERT encoders | inference only, frozen |
| `faiss-cpu` | `IndexFlatIP` | exact search; a few thousand chunks — IVF/HNSW is pointless |
| `rank_bm25` | keyword half of hybrid retrieval | 30 lines you don't write |
| `pypdf` | guideline PDF → text | |
| `anthropic` | Agent 3 generation | §6.7 |
| `pydantic` | node contracts + structured output | **this is the C10 *and* C15 answer** |
| `langgraph` | orchestration | only after the agents work standalone |
| `pyyaml` | the experiment grid | |
| `matplotlib` | figures | |
| **dev** | `pytest`, `ruff` | lint + format + test. That's all. |

**Deliberately skipped — stated so it's my decision to defend, not a silent omission:**

| Not building | Why |
|---|---|
| Docker | One machine, one venv, one user. A container adds a build step and solves nothing. |
| CI/CD pipeline | No collaborators, no deploys. `pytest` before each commit is the whole story. |
| MLflow / Weights & Biases | Eight experiment configs. A `run.json` + a parquet file is enough, and it's auditable by a panel member without an account. |
| `mypy --strict` | Pydantic already enforces every boundary that matters. Type-checking the numpy internals buys nothing. |
| A custom training loop | nnU-Net ships one. CineMA is inference-only. |
| ED/ES frame detector | All three datasets provide the indices (PROCESS.md §4). Building one injects unattributable error. **Ablation only, if at all.** |
| A web UI | The deliverable is a thesis, not a product. Figures are PNGs. |

---

## 3. Artifact layout — the reproducibility spine

Every output is keyed by a **config id**, so ablations never overwrite each other and every number in
the thesis traces to a directory.

```
artifacts/
├── manifest.parquet                    855 rows — the canonical subject table
├── gt_measurements.parquet             Stage-1 ground-truth EDV/ESV/LVEF/LVM (the denominator)
├── masks/{ckpt}/{dataset}/{sid}_{ed,es}.nii.gz
├── entropy/{ckpt}/{dataset}/{sid}_{ed,es}.npy
├── corpus/
│   ├── chunks.jsonl                    text + source + section + page + CoR + LoE + sha256
│   ├── embeddings.npy
│   └── index.faiss
├── runs/{config_id}/
│   ├── run.json                        provenance — see below
│   ├── reports/{sid}.json              validated Report objects
│   ├── traces/{sid}.jsonl              per-node LangGraph trace (the H3 evidence)
│   └── metrics.json                    every number for that config
└── figures/                            thesis-ready PNG/PDF
```

**`run.json` is written at the start of every run and is non-negotiable:**

```json
{
  "config_id": "full_a3f21c",
  "config": { "retriever": "hybrid", "embedder": "medcpt", "llm": "claude-opus-4-8", "gate": true, "feedback": true },
  "started_at": "2026-07-20T09:14:03+08:00",
  "git_sha": "…",
  "python": "3.11.9",
  "packages": { "torch": "…", "anthropic": "…", "monai": "…" },
  "llm_model_id": "claude-opus-4-8",
  "llm_access_date": "2026-07-20",
  "corpus_sha256": "…",
  "seeds": [0, 1, 2]
}
```

That `llm_model_id` + `llm_access_date` pair is not bureaucracy — PROCESS.md §8.2 (X6) flags that the
proposal names **GPT-4**, which is retired, and the slides say GPT-4o. "Name a current model and record
the exact API version and access date" is a panel-visible fix, and this file is where it lives.

`config_id = f"{name}_{sha256(json.dumps(config, sort_keys=True))[:6]}"` — deterministic, so re-running
the same config lands in the same directory and a changed config cannot silently contaminate old results.

---

## 4. Contracts — `cmr/types.py`

Every boundary between agents is a Pydantic model. **This is simultaneously the software design and the
answer to C10** ("how do you prevent error propagation between agents?"). The answer is: a malformed
payload cannot cross a node boundary, because it will not validate.

```python
class Status(str, Enum):
    OK                  = "ok"
    GATE_FAILED         = "gate_failed"
    FAILED_SEGMENTATION = "failed_segmentation"   # terminal — Agent 3 refuses
    REFUSED             = "refused"

@dataclass(frozen=True)                 # arrays → dataclass, not Pydantic
class Subject:
    subject_id: str
    dataset: Literal["ACDC", "MnMs", "MnMs2"]
    vendor: str | None                  # Philips | Siemens | GE | Canon
    centre: str | None                  # M&Ms: 1–5
    pathology: str
    ed_idx: int
    es_idx: int
    img_ed: np.ndarray                  # (X, Y, Z)
    img_es: np.ndarray
    gt_ed: np.ndarray                   # CANONICAL: 1=LV, 2=Myo, 3=RV
    gt_es: np.ndarray
    spacing_mm: tuple[float, float, float]

class Measurements(BaseModel):
    subject_id: str
    edv_ml: float
    esv_ml: float
    lvef_pct: float
    lv_mass_g: float
    hf_category: Literal["HFrEF", "HFmrEF", "HFpEF"]
    near_boundary: bool                 # |LVEF − 40| ≤ 2 or |LVEF − 50| ≤ 2 → triggers §6.9

class GateResult(BaseModel):
    passed: bool
    violations: list[str]               # "esv_ge_edv", "lvef_out_of_range", "multi_component", …

class Chunk(BaseModel):
    chunk_id: str                       # sha256 of the text — stable across rebuilds
    text: str
    source: str                         # "2021 ESC HF Guideline"
    section: str
    page: int
    class_of_recommendation: str | None # I | IIa | IIb | III
    level_of_evidence: str | None       # A | B | C

class Citation(BaseModel):
    chunk_id: str                       # MUST resolve in the index — factcheck verifies
    source: str
    section: str
    page: int
    class_of_recommendation: str | None
    level_of_evidence: str | None
    supports: str                       # the exact sentence in the report it grounds

class Report(BaseModel):
    subject_id: str
    diagnosis: str
    differential: list[str]
    hf_category: Literal["HFrEF", "HFmrEF", "HFpEF", "not_applicable"]
    citations: list[Citation]
    recommended_actions: list[str]
    conflicts: list[str] = []           # C11 — contradicting guideline thresholds, both attributed

class RunState(BaseModel):              # the LangGraph state object
    subject_id: str
    status: Status = Status.OK
    iteration: int = 0                  # bounded at 2 (§6.9)
    measurements: Measurements | None = None
    gate: GateResult | None = None
    passages: list[Chunk] = []
    report: Report | None = None
    fidelity: dict[str, float] | None = None
```

**Units are fixed and never negotiated:** volumes in **mL**, mass in **g**, ejection fraction in **%**,
spacing in **mm**. Any function that returns a bare float documents its unit in the name (`edv_ml`).

---

## 5. Global conventions

1. **Canonical labels are `1 = LV, 2 = Myo, 3 = RV`.** ACDC is remapped `{1↔3}` at load. This is the
   single most dangerous fact in the project (PROCESS.md §5); it is enforced by an assertion, not by
   discipline.
2. **Everything downstream reads masks from disk, never a network.** `seg.py` writes NIfTI; nothing
   else imports torch except `xai.py`.
3. **No `print`.** `logging`, one logger per module, level from `CMR_LOG`.
4. **Seeds fixed** (`0, 1, 2`) and recorded in `run.json`.
5. **The datasets are read-only.** Any code that opens a path under `ACDC/`, `MnMs/`, or `MnMs2/` opens
   it in `"rb"`.
6. **Every non-trivial module leaves one runnable check behind** (§8). No frameworks, no fixtures.

---

## 6. Work packages

Each has a goal, a public API, a dependency, a definition of done, and the check that proves it.

---

### WP-0 · Bootstrap — 0.5 day · no GPU

Python 3.11, venv, `pyproject.toml`, `ruff`, `pytest`, `artifacts/` skeleton, `.gitignore`.
Download **CineMA checkpoints** and **BAAI weights**; confirm each loads on MPS.

**DoD:** `pytest` runs (zero tests, green). `python -c "import torch; torch.zeros(1).to('mps')"` works.
🔴 **If BAAI's 7 B LMM will not fit in 16 GB, you must know it on day two, not in week five** — it
changes the GPU budget and it is now the largest technical unknown in the plan.

---

### WP-1 · `config.py` + `types.py` — 0.5 day · no GPU

Paths, seeds, the HF thresholds, the model IDs, and every contract from §4.

**DoD:** `from cmr.types import Report` works; `Report(**bad_dict)` raises `ValidationError`.

---

### WP-2 · `data.py` + `checks.py` — 3 days · no GPU · **the file that must be perfect**

Three adapters, one record. Every other module reads its output; get this wrong and every number in the
thesis is wrong-but-plausible, with no stack trace.

```python
def load_acdc(root: Path)  -> Iterator[Subject]: ...
def load_mnms(root: Path)  -> Iterator[Subject]: ...
def load_mnms2(root: Path) -> Iterator[Subject]: ...
def load_all()             -> Iterator[Subject]: ...      # 855
def build_manifest()       -> pd.DataFrame: ...           # → artifacts/manifest.parquet
```

Three ED/ES mechanisms — this is the entire reason the module exists:

| Dataset | ED/ES source | Trap |
|---|---|---|
| ACDC | `Info.cfg` → `ED:` / `ES:` | **Labels inverted** (`1=RV, 3=LV`). Remap on load. |
| M&Ms | metadata CSV → `ED` / `ES` | GT is a **4-D volume**, non-zero only at ED/ES. Indices are *not* in the file. `M4P7Q6` has **ED=2**, not 0 — a loader assuming frame 0 crashes with `KeyError`. |
| M&Ms-2 | pre-extracted `_ED.nii` / `_ES.nii` | Uncompressed. `dataset_information.csv` is padded to **1,048,575 rows**; only 360 carry data. Any naive `len(df)` is wrong. |

`checks.py` — 50 lines, and the cheapest insurance in the project:

```python
def assert_canonical(gt: np.ndarray) -> None:
    """The myocardium is a ring around the LV cavity, so the myo centroid is
    near-coincident with LV (≤4.2 px) and far from RV (25–42 px). Verified on
    9 subjects across all three datasets — the separation is unambiguous."""
    c = {l: np.argwhere(gt == l).mean(0) for l in (1, 2, 3) if (gt == l).any()}
    d_lv = np.linalg.norm(c[1][:2] - c[2][:2])
    d_rv = np.linalg.norm(c[3][:2] - c[2][:2])
    assert d_lv < d_rv, f"label order wrong: d(myo,LV)={d_lv:.1f}  d(myo,RV)={d_rv:.1f}"

def assert_esv_lt_edv(m: Measurements) -> None: ...
```

**DoD:** `manifest.parquet` has exactly **855** rows. `assert_canonical` passes on all 1,710 GT volumes
(ED + ES). `ESV < EDV` violations are *logged, not suppressed* — they are a real data-quality finding
and worth a paragraph in the thesis.

---

### WP-3 · `quantify.py` — 2 days · no GPU · **no AI at all**

Pure arithmetic on masks. Run it on **ground truth first**, for all 855 — that output is the denominator
of every MAE in the entire evaluation, and it is a full result before a single network has been touched.

```python
def measure(mask_ed, mask_es, spacing_mm) -> Measurements: ...
def gate(m: Measurements, mask_ed, mask_es) -> GateResult: ...
```

```python
voxel_ml = np.prod(spacing_mm) / 1000.0
edv = (mask_ed == 1).sum() * voxel_ml
esv = (mask_es == 1).sum() * voxel_ml
lvef = 100.0 * (edv - esv) / edv
lv_mass_g = (mask_ed == 2).sum() * voxel_ml * 1.05    # Kawel-Boehm (2025), NOT Leiner
```

**The heart-failure thresholds in the proposal are broken at both boundaries.** §3.4 emits
`(<40 = reduced, 40-50 = mildly reduced, >50 = preserved)`, which misclassifies LVEF = 40.0 **and**
LVEF = 50.0. ESC 2021 and AHA/ACC/HFSA 2022 agree:

```python
def hf_category(lvef: float) -> str:
    if lvef <= 40.0: return "HFrEF"      # ≤ 40
    if lvef <  50.0: return "HFmrEF"     # 41–49
    return "HFpEF"                        # ≥ 50
```

**Plausibility gate** — rule-based, no training: LVEF outside `[5, 90]`%; `ESV ≥ EDV`; more than one
connected component per structure per slice; base/apex discontinuity.

🔴 **Deliverable due in week 2: count how many subjects have ground-truth LVEF ∈ [35, 55]%.** That
single number sizes **H2**. If almost nobody sits near a guideline decision boundary, boundary-aware
recomputation has nothing to act on and H2 must be reframed as a *safety property* **now** — not in
October when the report is being written.

**DoD:** `gt_measurements.parquet`, 855 rows. **DCM cohort mean LVEF < NOR cohort mean LVEF.** If that
inequality fails, the loader is wrong and you have found it in week 2 instead of in the viva.

---

### WP-4 · `seg.py` — 5 days · GPU optional · **Agent 1**

CineMA's fine-tuned checkpoints are public — `acdc_sax`, `mnms_sax`, `mnms2_sax`, `mnms2_lax_4c`,
**3 seeds each**. Masks with zero training and zero GPU.

```python
def segment(img: np.ndarray, checkpoint: str, seeds=(0,1,2)) -> tuple[np.ndarray, np.ndarray]:
    """Returns (canonical_mask, entropy_map). The 3 released seeds are a free ensemble."""
    probs   = np.stack([_infer(img, checkpoint, s) for s in seeds])   # (3, C, X, Y, Z)
    mean    = probs.mean(0)
    entropy = -(mean * np.log(mean + 1e-9)).sum(0)                    # Agent 4's 2nd XAI channel
    return canonicalise(mean.argmax(0), checkpoint), entropy

def run_all(checkpoint: str, dataset: str) -> None:                   # writes masks/ + entropy/
```

⚠️ **Every checkpoint emits its own dataset's label order.** `acdc_sax` outputs `1=RV`; `mnms_sax`
outputs `1=LV`. Canonicalise predictions exactly as you canonicalise GT, then run `assert_canonical`
**on the prediction**. Cross-checkpoint work is precisely where the §5 bug resurfaces.

⚠️ **Label the runs honestly.** PLAN.md originally called CineMA a "zero-shot ablation." It is not:

| Run | What it actually is |
|---|---|
| `acdc_sax` → ACDC test | **Supervised baseline** |
| `acdc_sax` → M&Ms / M&Ms-2 | **True cross-dataset zero-shot. This is RQ3.** |
| `mnms_sax` → M&Ms | In-domain ceiling |

Reporting an ACDC-finetuned checkpoint's ACDC score as "zero-shot" is disprovable from the HuggingFace
page in four minutes, and it would poison everything else you claim.

**DoD:** masks + entropy maps on disk for all 855 × {ED, ES}. `assert_canonical` passes on every
prediction. Dice/HD95 against GT computed by `eval.py`.

---

### WP-5 · `corpus.py` — 5 days · no GPU · **the contribution nobody else has**

```python
def ingest(pdf: Path, spec: SourceSpec) -> list[Chunk]: ...
def build_corpus() -> None:            # → chunks.jsonl, with sha256 per chunk
```

Guideline-only. **The purity *is* contribution #2** — CardAIc mixes Mayo Clinic, the NHS website and
MedlinePlus with guidelines; BAAI mixes ChatCAD+, guidelines and PubMed. A guideline-only corpus does
not exist.

| Document | Role |
|---|---|
| Petersen et al., *CMR in the guidelines of the ESC* (PMC10364363) | Aggregated CMR recommendations |
| 2021 ESC HF guideline + 2023 focused update | HF classification, LVEF thresholds |
| 2022 AHA/ACC/HFSA HF guideline | US counterpart; the conflict testbed |
| 2021 AHA/ACC Chest Pain guideline | Where real ESC/ACC conflicts live |
| SCMR 2025 reference values (Kawel-Boehm) | Normal ranges; myocardial density 1.05 g/mL |

🚫 **The "ESC 2022 Guidelines on Cardiac Magnetic Resonance Imaging" does not exist.** It is cited in
three places in the proposal and repeated on the slides. It must never appear in code or text again.

Chunk at **200–400 tokens, 50-token overlap**, and carry on every chunk:
`{chunk_id, text, source, section, page, class_of_recommendation, level_of_evidence}`.

**The CoR/LoE extraction is the real work here.** Recommendation tables are formatted consistently
*within* a guideline but not *across* them, so this is one small parser per document — tedious, ~1 day
each, and it is the differentiator. Neither competitor carries this metadata.

⚠️ **Copyright.** These guidelines cannot be redistributed. Release the **pipeline**: acquisition script,
chunk boundaries as byte offsets, SHA-256 per chunk, embeddings, and the FAISS index — so anyone holding
legal copies reconstructs the corpus bit-for-bit. Say this in the thesis before a reviewer says it for you.

**DoD:** `chunks.jsonl` — every chunk has non-empty `source`, `section`, `page`; SHA-256 stable across
two independent builds; ≥ 60% of chunks in the two HF guidelines carry a CoR **and** an LoE.

---

### WP-6 · `retrieve.py` — 3 days · no GPU · **Agent 3a**

```python
def embed(texts: list[str], model: str = "medcpt") -> np.ndarray: ...
def build_index() -> None: ...                     # FAISS IndexFlatIP
def retrieve(query: str, k: int = 5, mode: str = "hybrid") -> list[Chunk]: ...
def detect_conflicts(passages: list[Chunk]) -> list[str]: ...
```

- **MedCPT, not BioBERT.** BioBERT is a masked LM; MedCPT is *trained* for biomedical retrieval and is
  also 768-dim, so nothing else changes. **Keep BioBERT as the ablation row** — it costs nothing and
  turns a weakness into a table entry.
- **Hybrid, not dense-only.** CardAIc explicitly argues dense passage retrieval *"often lacks semantic
  relevance"*, and its own ablation shows vector-only and keyword-only both underperform hybrid. Ship
  dense-only BioBERT and a reviewer who knows CardAIc will ask why you ignored a documented improvement
  in a paper you cite (PROCESS.md §8.6 M4).
- `IndexFlatIP` — exact search. A few thousand chunks; IVF/HNSW would be pure ceremony.
- **Conflict resolution (C11):** flag contradicting thresholds; resolve by recency, then class of
  recommendation; **if unresolved, present both with attribution** — never silently pick one.
  🔴 HF thresholds *agree* across ESC and ACC/AHA (PROCESS.md §6.1). If the corpus yields no real
  conflicts, **say so and reframe C11 as a safety property**, not a demonstrated capability. The list of
  conflicts you find — or the finding that there are none — is itself a contribution.

**DoD:** query *"LVEF threshold for HFrEF"* returns a chunk containing "40". Retrieval mode is a config
switch (`hybrid` | `dense` | `bm25`), because that switch **is** an ablation row.

---

### WP-7 · `report.py` — 4 days · no GPU · **Agent 3b**

```python
def generate(m: Measurements, passages: list[Chunk], cfg: LLMConfig) -> Report: ...
def generate_batch(jobs: list[tuple[Measurements, list[Chunk]]]) -> list[Report]: ...
```

**Model:** `claude-opus-4-8` — current, and decisively, it supports **schema-constrained structured
output**. Do not prompt for JSON and hope; constrain the schema and the API guarantees the response
validates. **That mechanism is what makes C15 real rather than aspirational.**

```python
resp = client.messages.parse(
    model="claude-opus-4-8",
    max_tokens=2000,
    system=SYSTEM_PROMPT,          # task + output rules + correct HF thresholds
    messages=[{"role": "user", "content": build_prompt(measurements, passages)}],
    output_format=Report,          # ← the guarantee
)
report = resp.parsed_output        # a validated Report, or it raised
```

**The LLM never sees the image and never invents a number.** Measurements arrive as structured fields.

**Threat model, stated honestly (C15):** the only text reaching the model is curated guideline chunks
(trusted) and numeric JSON from your own code (trusted). **There is no user-supplied free text, so the
prompt-injection surface is genuinely small — say that**, rather than claiming a threat you don't have.
The real risks are hallucinated measurements and fabricated citations. Both are caught by WP-8.

**Cost — smaller than you'd expect, and worth knowing before you plan around it.** Never estimate tokens
with `tiktoken` (OpenAI's tokenizer; it undercounts Claude). Use `client.messages.count_tokens()`.
Per subject ≈ 3,000 input + 600 output tokens.

| | Calls | Cost |
|---|---|---|
| One full pass, 855 subjects | 855 | **≈ $26** |
| Full ablation grid (~8 configs) | ~6,800 | **≈ $210** |
| Same via the **Batch API** (50% off) | ~6,800 | **≈ $105** |
| With dev iteration and dead ends | — | budget **$400–500 total** |

**Use the Batch API.** 855 offline reports with no latency requirement is exactly what it's for:
`client.messages.batches.create(...)`, poll until `processing_status == "ended"`, key by `custom_id`.
It supports structured outputs.

**Skip prompt caching.** The stable prefix here is likely under Opus 4.8's 4,096-token minimum cacheable
prefix, so it would silently not cache (`cache_read_input_tokens == 0`), and at 855 calls the saving is
a few dollars. Not worth the complexity.

The headline worth internalising: **the entire LLM budget for this thesis is a few hundred dollars.**
GPU rental will cost more than the API.

**DoD:** 855 validated `Report` objects on disk, each with ≥ 1 citation. Config switch for LLM
(`opus-4-8` | `sonnet-5` | open-weights | `none`) — because that switch is an ablation row.

---

### WP-8 · `factcheck.py` — 1 day · no GPU · **the headline metric**

Deterministic, cheap, and it measures hallucination *directly*. **BAAI claims "zero hallucination" only
qualitatively; neither competitor quantifies it.** This is the metric the whole architecture exists to
optimise — make it the headline result.

```python
def factual_consistency(report: Report, m: Measurements, index: dict[str, Chunk]) -> dict[str, float]:
    nums = extract_numbers(report)                       # from diagnosis + recommended_actions
    grounded = sum(any(abs(n - v) <= TOL for v in m.model_dump().values()) for n in nums)
    resolved = sum(c.chunk_id in index for c in report.citations)
    return {
        "numeric_fidelity":  grounded / len(nums) if nums else 1.0,
        "citation_fidelity": resolved / len(report.citations) if report.citations else 0.0,
    }
```

`citation_fidelity` is the number BAAI **structurally cannot report**, because its diagnostic report
carries no citations at all — its RAG is a conversational sidecar.

**DoD:** a deliberately corrupted report (a number changed, a `chunk_id` invented) scores
`numeric_fidelity < 1.0` and `citation_fidelity < 1.0`. That test is the metric's own proof.

---

### WP-9 · `xai.py` — 5 days · GPU optional · **Agent 4 — the strongest surviving differentiator**

BAAI has **no visual XAI at all**. CardAIc has visual panels but they are **not linked to the textual
rationale**. The *link* is unclaimed.

```python
def seg_grad_cam(model, img, target_class: int, target_region) -> np.ndarray: ...
def uncertainty_map(entropy: np.ndarray) -> np.ndarray: ...
def link(report: Report, cams: dict, entropy) -> list[Explanation]: ...
```

- **Seg-Grad-CAM** (Vinogradova et al., 2020), **not vanilla Grad-CAM.** Grad-CAM is defined for
  *classifiers*; segmentation needs the gradient of a region-summed class score. §3.6 currently cites
  only Selvaraju — that is defect M5.
- **Entropy uncertainty maps** come free from WP-4's 3-seed ensemble. A second, independent visual
  channel, arguably more clinically useful than Grad-CAM.
- **The link is the contribution.** Not "here is a heatmap and here is a citation" — but *"this heatmap
  region supports this cited guideline sentence."* Build the data structure that expresses that link
  (`Explanation{cited_sentence, chunk_id, heatmap_path, region_mask}`) and **evaluate it** — it becomes a
  rated axis in the reader study.

**DoD:** for one subject, a rendered figure showing the report sentence, its citation with page number,
and the heatmap region that supports it — all three connected by an object in the trace, not by a caption.

---

### WP-10 · `graph.py` — 5 days · no GPU · **orchestration**

LangGraph `StateGraph`, **Pydantic validation at every node boundary — that is the C10 answer.**

```
data → seg → quantify → [gate] → retrieve → report → factcheck → xai
                          │
                          ├─ pass ──────────────────────────────────────→ continue
                          └─ fail → retry w/ TTA → 2D fallback → status: FAILED_SEGMENTATION
                                                                  Agent 3 REFUSES to diagnose
```

Two behaviours that are the actual contribution — the graph itself is plumbing:

**Refusal (C10 / H3).** On unrecoverable gate failure, emit `status: FAILED_SEGMENTATION` and **Agent 3
does not diagnose.** No silent downstream reasoning on a broken mask. That refusal *is* the
failure-isolation claim made concrete and testable.

**Decision-boundary-aware recomputation (C12 / H2).** Slide 22's *"guideline evidence can refine
segmentation masks"* is mechanically impossible — a text retriever has no imaging signal. **Reframe:**
Agent 3 knows the guideline *decision boundaries*. When a computed value lands near one (LVEF 39.4%
against the HFrEF cut at 40%), Agent 3 emits a **re-measurement request**, and Agent 1 re-runs with
test-time augmentation and full ensembling to tighten the estimate. Triggered by guideline knowledge,
executed in imaging space, **bounded at 2 iterations**, fully logged.

Cite CardAIc's adaptive workflow honestly as prior art for the *mechanism class* (0.80 → 0.87). Your
difference: **CardAIc refines the plan when evidence changes; you refine the measurement when it lands
near a clinical decision boundary.** Different trigger, different action, different failure mode.

**Persist a JSONL trace per subject per node.** That trace is the evidence for H3 and the audit log for
the entire system.

**DoD:** feed the graph a zeroed mask → it reaches `FAILED_SEGMENTATION`, emits **no diagnosis**, and the
trace shows exactly which node refused and why. **That is the most important test in the repository.**

---

### WP-11 · `eval.py` — 5 days · no GPU

```python
def segmentation_metrics(pred, gt, spacing) -> dict:   # Dice, HD95 (MONAI)
def agreement(pred_m, gt_m) -> dict:                    # MAE, Bland-Altman, ICC
def diagnostic_metrics(reports, manifest) -> dict:      # pathology top-1, HF-category accuracy
def bootstrap_ci(a, b, metric, n=10_000) -> tuple:      # paired, 95% CI
def stratify(results, by: str) -> pd.DataFrame:         # by vendor, by centre
```

**RQ2 asks about "diagnostic accuracy," and Dice, MAE and BERTScore measure none of it.** The panel found
only its shadow (C13). Three metrics close the gap:

1. **Pathology top-1** — ACDC has 5 classes, M&Ms a `Pathology` column, M&Ms-2 eight disease classes.
   Score Agent 3's top differential against the label. Objective, no clinician needed, computable today.
   Also gives a comparison axis against BAAI (AUC 0.93 internal / 0.81 external).
2. **HF-category accuracy** — HFrEF/HFmrEF/HFpEF from predicted vs GT EF. Isolates how measurement error
   propagates into a *clinical decision*. **Restricted to GT LVEF ∈ [35, 55]%, it is the test for H2.**
3. **Factual consistency** (WP-8) — the headline.

Bland-Altman and ICC are now **table stakes, not differentiators** — BAAI reports Bland-Altman and
Pearson r. Report them because their absence would be noticed, not because they distinguish you.

**BERTScore needs care.** ACDC and M&Ms ship masks and pathology labels; **neither ships a radiology
report**, so BERTScore ≥ 0.85 is currently measured against nothing — and BAAI already achieves 0.898
against real reports. Fix: generate reference reports from GT measurements + pathology via a fixed
template (clinician-signed on a sample), report **baseline-rescaled** BERTScore, and make the numeric
factual-consistency score the headline instead.

**Stratify every segmentation result by vendor and centre.** M&Ms gives you Philips / Siemens / GE /
Canon and five centres for free. Report the **ACDC→M&Ms Dice drop** — that *is* RQ3.

**DoD:** `metrics.json` per config, and a paired bootstrap CI for every headline comparison.

---

### WP-12 · Baselines — 6 days · **GPU for the first one**

Five baselines you can actually run. Table 3.4's current list is not runnable (CardAIc is not a CMR
system; MARL-Rad has no public weights).

| # | Baseline | Where | GPU |
|---|---|---|---|
| 1 | nnU-Net standalone | WP-13 | yes |
| 2 | CineMA (supervised + cross-dataset) | WP-4 | no |
| 3 | **BAAI Cardiac Agent on ACDC / M&Ms** | `baselines/run_baai.py` | **yes** |
| 4 | Measurements → LLM, **no RAG** | config switch `retriever: none` | no |
| 5 | **Template report from Agent-2 JSON, no LLM at all** | `baselines/template_report.py` | no |

**Baseline 3 is the headline.** BAAI's weights are public (Apache 2.0). No agentic CMR system has ever
been benchmarked on a public cardiac MRI dataset — BAAI's cohort is private, CardAIc's is public but not
MRI. Running it is the first public-benchmark evaluation of the state of the art, it gives you the one
baseline that matters, and it converts *"they never validated publicly"* from a complaint into an
experiment you performed. It is also a submittable short paper, which feeds UM's publication requirement.

**Baseline 5 will score embarrassingly well on factual consistency**, because a template cannot
hallucinate. **Include it anyway.** Reporting the baseline that flatters you least is what makes the rest
credible — and the gap between it and your system is exactly the value the LLM adds.

Cite MARL-Rad and CardAIc as literature context; state plainly that they are not re-run, and why.

**DoD:** every baseline writes a `metrics.json` in the same schema as the full system, so the comparison
table in the thesis is generated, not typed.

---

### WP-13 · nnU-Net — parallel, GPU, **not on the critical path**

`baselines/nnunet_convert.py` — ACDC → nnU-Net raw format, **emitting the canonical label map into
`dataset.json`** so predictions come out canonical everywhere. Train `2d` + `3d_fullres`, 5-fold, on
rented GPU (Kaggle T4×2 / Colab / vast.ai / RunPod). Budget 40–80 GPU-hours.

**Do not hand-specify preprocessing.** §3.3.3 currently pins `1.5 × 1.5 × 8 mm` and `192 × 192` patches,
which contradicts §3.3.2's stated reason for choosing nnU-Net — that it *self-configures*. Let it derive
spacing and patch size, and **report them post-hoc as a result** (its own ACDC `3d_fullres` target is
≈ 1.56 × 1.56 × 5.0 mm).

**Report the GPU you actually used.** Do not write "A100" if you trained on a T4.

**Fallback, decided in advance:** if nnU-Net does not finish in time, **present CineMA as the interim
backbone and say so.** The panel will not fail you for an unfinished training run. They will fail you for
having no results. Do not let a GPU queue hold the defence hostage.

---

### WP-14 · `figures.py` + `cli.py` — 3 days · no GPU

Every thesis figure and table generated **from `artifacts/`**, never typed by hand:

- EF distribution by pathology (the DCM sanity check)
- Bland-Altman: predicted vs GT LVEF
- Dice by vendor and by centre (the RQ3 generalisation gap)
- The ablation table — **generated directly from `configs/experiments.yaml`**
- The head-to-head table: rows = BAAI, CardAIc, Proposed; columns = modality, orchestration,
  guideline-passage citation, linked dual XAI, boundary-aware recomputation, cohort size, cohort public,
  public-benchmark results. **Be scrupulously honest in the cohort-size row** — 495 public vs BAAI's
  2,413 private. Honesty there is what earns the reader's trust in every other column.
- One linked-XAI figure (WP-9)

```bash
cmr manifest                       # WP-2  → manifest.parquet
cmr quantify --gt                  # WP-3  → gt_measurements.parquet
cmr segment --ckpt acdc_sax --dataset MnMs
cmr corpus build
cmr run --config full              # the whole graph, 855 subjects
cmr run --config no_rag            # an ablation
cmr eval --config full
cmr figures
```

**DoD:** `cmr figures` regenerates every figure in the thesis from artifacts alone. If a number in the
thesis cannot be traced to a file under `artifacts/`, it does not go in the thesis.

---

## 7. The ablation grid is data, not code

`configs/experiments.yaml`. Each row is a config; each config is a directory; the ablation table in the
thesis is generated from this file. That is why every "which retriever / which embedder / which LLM /
gate on or off" decision above was a **config switch** rather than an `if` statement.

```yaml
full:        {retriever: hybrid, embedder: medcpt,  llm: claude-opus-4-8, gate: on,  feedback: on}
no_rag:      {retriever: none,   embedder: null,    llm: claude-opus-4-8, gate: on,  feedback: on}
dense_only:  {retriever: dense,  embedder: medcpt,  llm: claude-opus-4-8, gate: on,  feedback: on}
bm25_only:   {retriever: bm25,   embedder: null,    llm: claude-opus-4-8, gate: on,  feedback: on}
biobert:     {retriever: hybrid, embedder: biobert, llm: claude-opus-4-8, gate: on,  feedback: on}
cheap_llm:   {retriever: hybrid, embedder: medcpt,  llm: claude-sonnet-5, gate: on,  feedback: on}
open_llm:    {retriever: hybrid, embedder: medcpt,  llm: local-open,      gate: on,  feedback: on}
no_gate:     {retriever: hybrid, embedder: medcpt,  llm: claude-opus-4-8, gate: off, feedback: on}
no_feedback: {retriever: hybrid, embedder: medcpt,  llm: claude-opus-4-8, gate: on,  feedback: off}
template:    {retriever: none,   embedder: null,    llm: none,            gate: on,  feedback: off}
```

Ten configs, mapping directly onto the three hypotheses:

| Hypothesis | Compare |
|---|---|
| **H1** — grounding helps | `full` vs `no_rag` (pathology top-1, factual consistency, Likert) |
| **H2** — boundary-aware recomputation helps where it matters | `full` vs `no_feedback`, restricted to GT LVEF ∈ [35, 55]% |
| **H3** — decomposition makes failure visible | `full` vs `no_gate`, on corrupted and hard cases |

Plus the retriever ablation (`hybrid` vs `dense_only` vs `bm25_only`), the embedder ablation (`medcpt`
vs `biobert`), and the LLM ablation — all of which you already promised the panel.

---

## 8. Testing

`pytest`, ~200 lines, **no frameworks, no fixtures, no mocks** except one for the API call. Two tiers:

**Fast (< 10 s, no data, runs on every commit):**

| Test | Proves |
|---|---|
| `test_hf_thresholds` | LVEF 40.0 → HFrEF; 50.0 → HFpEF; 49.9 → HFmrEF (**the boundary bug**) |
| `test_canonical_synthetic` | a synthetic ring-around-cavity mask passes; a swapped one fails |
| `test_report_schema` | a malformed dict raises `ValidationError` |
| `test_factcheck_catches_corruption` | a tampered report scores < 1.0 on both fidelities |
| `test_config_id_deterministic` | same config → same id; changed config → different id |

**Slow (marked `@pytest.mark.slow`, runs over the real 855):**

| Test | Proves |
|---|---|
| `test_all_canonical` | `assert_canonical` on all 1,710 GT volumes **and every prediction** |
| `test_esv_lt_edv` | logs violations rather than hiding them |
| `test_dcm_ef_below_nor` | the loader is not silently corrupt |
| `test_retrieval_finds_hfref_threshold` | query "LVEF threshold for HFrEF" returns a chunk containing "40" |
| **`test_zeroed_mask_refuses`** | **the graph reaches `FAILED_SEGMENTATION` and emits no diagnosis** |

That last test is H3, executable. It is the most important test in the repository.

---

## 9. Schedule and dependencies

```
WP-0 ─ WP-1 ─ WP-2 ─ WP-3 ─┬─ WP-4 ─────────────┬─ WP-10 ─ WP-11 ─ WP-14
                            │                    │
                            ├─ WP-5 ─ WP-6 ─ WP-7 ─ WP-8
                            │                    │
                            └────────────────────┴─ WP-9

parallel, GPU, off the critical path:  WP-12 (BAAI) · WP-13 (nnU-Net)
```

| Phase | Work packages | Days | GPU |
|---|---|---|---|
| Foundations | WP-0, WP-1, WP-2, WP-3 | 6 | no |
| Perception | WP-4 | 5 | optional |
| Knowledge | WP-5, WP-6 | 8 | no |
| Generation | WP-7, WP-8 | 5 | no |
| Explanation | WP-9 | 5 | optional |
| Orchestration | WP-10 | 5 | no |
| Measurement | WP-11, WP-14 | 8 | no |
| Baselines | WP-12, WP-13 | 6 + train | **yes** |
| | **Total** | **~48 days** | |

Real results exist by **day 8** (WP-3: ground-truth quantification over all 855, the EF-by-pathology
plot, the H2 boundary count). That matters — the candidature-defence rubric awards 20% for "Results and
Discussions," and nothing in the first eight days needs a GPU, an API key, or the network.

---

## 10. Engineering risk register

| # | Risk | Detect | Mitigate |
|---|---|---|---|
| 1 | **Label-order corruption across checkpoints** — silent, plausible, catastrophic | `assert_canonical` on **predictions**, every checkpoint, every run | It is not a discipline problem. It is an assertion. WP-2. |
| 2 | **BAAI's 7 B LMM will not fit in 16 GB** — now the largest technical unknown | Load it in **WP-0, day two** | Rent GPU for BAAI *before* renting for nnU-Net |
| 3 | **H2 has no patients to act on** (too few near a decision boundary) | Count GT LVEF ∈ [35, 55]% in **WP-3, week 2** | Reframe H2 as a safety property in July, not October |
| 4 | **C11 has no conflicts to resolve** (ESC and ACC/AHA agree on HF) | `detect_conflicts` over the corpus, WP-6 | Say so, and reframe C11 as a safety property. The null finding is itself reportable. |
| 5 | **CoR/LoE parsing is slower than budgeted** (5 documents, 5 formats) | Track chunks-with-metadata per day | It is the differentiator — do not cut it. Cut a *guideline* before cutting the metadata. |
| 6 | **nnU-Net does not finish** | GPU queue length | **Decided in advance:** present CineMA as interim backbone. Do not let this block the defence. |
| 7 | **Reports drift from measurements** (hallucinated numbers) | `factcheck` on every report, every run | Schema-constrained output + numeric fidelity < 1.0 fails the build |

---

## 11. Definition of done — per work package, no exceptions

A work package is done when **all five** hold:

1. The code is written and `ruff` is clean.
2. **Its runnable check passes** (§8).
3. Its artifact exists at the documented path in `artifacts/` (§3).
4. Its numbers appear in a `metrics.json` or a figure — not in a notebook, not in a terminal scrollback.
5. **There is no `TODO`, no stub, and no placeholder left in it.**

If a shortcut was taken deliberately, it carries a `shortcut:` comment naming the ceiling and the upgrade
path. An undocumented shortcut is a defect.

---

## 12. First three days, concretely

1. **Day 1** — Python 3.11, venv, `pyproject.toml`, `artifacts/` skeleton. Download CineMA checkpoints
   **and BAAI weights**; load each on MPS. *If BAAI won't fit, you now know on day one.*
2. **Day 2** — `config.py`, `types.py`. Every contract in §4, with `ValidationError` on a bad payload.
3. **Day 3–5** — `data.py`, three adapters, canonical remap. `checks.py`. Run `assert_canonical` over all
   **1,710** ground-truth volumes. Fix whatever it surfaces.

Then WP-3, and you have real, defensible results in the thesis by the end of week two — before a single
network has been trained and before a single dollar has been spent.
