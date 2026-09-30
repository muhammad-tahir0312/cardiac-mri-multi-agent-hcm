# CMR — Multi-Agent Cardiac MRI Analysis & Decision Support

Muhammad Tahir · MCS by Research, Universiti Malaya · FSKTM

Four agents over three public cardiac-MRI datasets, orchestrated as a typed LangGraph
state machine that **refuses to diagnose a mask it cannot trust**.

| Agent | Module | Trained? |
|---|---|---|
| 1 · Segmentation | `cmr/seg.py` — CineMA 3-seed ensemble → masks + free uncertainty | no (nnU-Net, optional, is the only trained component) |
| 2 · Quantification | `cmr/quantify.py` — EDV/ESV/LVEF/mass + plausibility gate | **no** — deterministic arithmetic |
| 3 · Clinical reasoning | `cmr/corpus.py` + `cmr/retrieve.py` + `cmr/report.py` | **no** — frozen encoders, prompted LLM |
| 4 · Explainability | `cmr/xai.py` — Seg-Grad-CAM + ensemble entropy + **the link** | **no** — gradients on a frozen model |
| Orchestration | `cmr/graph.py` — gate, refusal, bounded recomputation | **no** |
| Fidelity | `cmr/factcheck.py` — the headline metric | **no** — numeric comparison |

**Exactly one component is trainable.** That is a feature, not an apology: every failure is
attributable, and it is the direct answer to panel comment C14.

---

## Quickstart

```bash
brew install python@3.11
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cmr doctor                       # what works on this machine right now
cmr manifest                     # build + VERIFY the 830-subject manifest
cmr quantify --gt                # ground-truth EDV/ESV/LVEF/mass — the denominator
cmr corpus                       # guideline PDFs -> chunks -> FAISS index
cmr run --config vertical_slice --limit 5   # prove the graph, no GPU, no LLM
cmr run --config full            # the whole system, 830 subjects
cmr eval --config full
cmr figures --config full

cmr serve                        # read-only viewer -> http://127.0.0.1:8000
```

`cmr serve` is the **viva demo**. It is a viewer and nothing else: it never starts a run,
never writes to `artifacts/`, binds to 127.0.0.1 only, and loads no CDN — it works with the
wi-fi off. It shows the four things the competition cannot: passage-level citations *inside*
the report, the **link** from a cited sentence to the heatmap that supports it, the 11
subjects the system **REFUSED** to diagnose (H3), and numeric/citation fidelity per report.

Everything is driven by `configs/default.yaml`. **No path, threshold, model name or device
is hard-coded anywhere in the source.** On UM's HPC cluster, use `configs/hpc.yaml` (CUDA)
or just `export CMR_DEVICE=cuda CMR_DATA_ROOT=/scratch/$USER/cardio_dataset`.

---

## The three facts this codebase exists to get right

**1 · The label convention silently inverts between datasets.**
ACDC is `1=RV, 2=Myo, 3=LV`. M&Ms and M&Ms-2 are `1=LV, 2=Myo, 3=RV`. All three use
`{0,1,2,3}` and are visually indistinguishable. Train on one, evaluate on another without
remapping, and every LV/RV Dice, every ejection fraction and every heart-failure
classification is **wrong but plausible** — with no stack trace. So there is an assertion
instead of a hope: the myocardium is a *ring around the LV cavity*, so its centroid must be
near-coincident with the LV and far from the RV. `cmr/checks.py::assert_canonical` runs on
every ground-truth volume at load **and on every predicted mask**, because each CineMA
checkpoint emits its own dataset's order and that is exactly where the bug comes back.

**2 · The guideline chunks carry class-of-recommendation and level-of-evidence.**
Neither BAAI Cardiac Agent nor CardAIc-Agents carries this metadata. The corpus is
**guideline-only** — no Mayo Clinic pages, no PubMed, no ChatCAD. That purity is the
contribution, and it is why citations can be resolved, ranked, and conflict-checked.

**3 · Hallucination is measured, not asserted.**
`cmr/factcheck.py` scores every report on **numeric fidelity** (is every number traceable to
Agent 2?) and **citation fidelity** (does every citation resolve to a real indexed chunk?).
BAAI claims "zero hallucination" qualitatively. Neither competitor quantifies it.

---

## Reproducibility

Every run writes `artifacts/runs/{config_id}/run.json` — git SHA, package versions, device,
seeds, the **exact LLM model id and access date**, and the corpus SHA-256. The `config_id` is
a hash of the experiment spec, so a changed configuration can never silently overwrite an
older result.

**If a number in the thesis cannot be traced to a file under `artifacts/`, it does not go in
the thesis.**

The ablation grid lives in `configs/experiments.yaml` — it is *data*, not code, and the
ablation table in the thesis is generated from it.

| Hypothesis | Comparison |
|---|---|
| **H1** grounding helps | `full` vs `no_rag` |
| **H2** boundary-aware recomputation helps where it matters | `full` vs `no_feedback`, restricted to GT LVEF ∈ [35, 55] % |
| **H3** decomposition makes failure visible | `full` vs `no_gate` |

---

## Verified facts about the data (measured, not assumed)

| | |
|---|---|
| Subjects with ground truth | **830** — ACDC 150 + M&Ms 320 + M&Ms-2 360 |
| (M&Ms' 25 *unlabeled* subjects bring the raw total to 855) | |
| M&Ms | **345 subjects, 5 centres** — not the 375 / 6 in the proposal |
| Vendors | Siemens, Philips, GE, Canon (4) |
| Ground-truth LVEF ∈ [35, 55] % — **the H2 test set** | **221 (26.6 %)** |
| Within 2 % of a guideline cut-point — the H2 *trigger* | **80 (9.6 %)** |
| DCM mean LVEF | **41.1 %** — versus NOR **60.2 %**. The loader is provably correct. |
| ESV ≥ EDV violations | **0** |

---

## Layout

```
configs/     default.yaml · hpc.yaml · experiments.yaml   ← change these, not the code
cmr/         the package (see the agent table above)
baselines/   nnU-Net · BAAI Cardiac Agent · template (zero-LLM floor)
tests/       fast tier (no data) + slow tier (all 830)
artifacts/   every output, keyed by config_id. Gitignored.
```

Run `pytest -m "not slow"` for the fast tier (< 1 s). It contains the two tests that would
have caught the two worst defects in the proposal: the heart-failure rule that misclassifies
at *both* cut-points, and the label inversion.
