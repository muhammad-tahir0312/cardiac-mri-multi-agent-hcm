# Security & Hallucination — the threat model, stated honestly

This is the answer to panel comment **C15** (*"agent-to-agent security must be a core part of the
multi-agent workflow design… malicious prompts, prompt injection between agents, hallucinated
outputs. Guardrails for hallucination mitigation must be explicitly incorporated"*).

The implementation is `cmr/guardrails.py`. The tests that prove it are `tests/test_guardrails.py`
(20 tests). Read those two files if you want the truth rather than the summary.

---

## 0. The claim I am NOT making

**There is no such thing as a 100% hallucination-free RAG system, and this is not one.**

Retrieval can always miss a relevant passage. A language model can always write a sentence that is
grounded in a real citation, contains only traceable numbers, and is *still clinically wrong*. No
architecture eliminates that — not this one, and not BAAI Cardiac Agent, which asserts "zero
hallucination" qualitatively and never measures it.

If you claim perfection in the viva, one counterexample destroys you. **Claim the thing that is
actually true and actually stronger:**

> *"I cannot prevent the model from being wrong. I can prevent it from being **unaccountable**.
> Every number in every report traces to the quantification agent or to a cited passage, every
> citation resolves to a real indexed chunk, and a report that fails either test is not emitted —
> it is refused. That is enforced in code, not requested in a prompt, and it is tested."*

---

## 1. Hallucination: four enforceable invariants

Pydantic guarantees the report's **shape**. It says nothing about its **truth** — `diagnosis=""`
and `chunk_id="I_made_this_up"` are both schema-valid. So four invariants are checked mechanically
after generation, and a violation prevents emission.

| # | Invariant | Why it is the right invariant |
|---|---|---|
| **1** | **Every citation resolves.** Each `chunk_id` must exist in the corpus index. | A `chunk_id` the model invented is not a weak citation — it is a fabricated one. It is deleted. |
| **2** | **Every number is traceable.** Each numeric token in `diagnosis`/`findings` must match one of Agent 2's measurements within tolerance, **or** appear verbatim in a passage the report actually cites. | A hallucinated *measurement* is the most dangerous failure a clinical report can contain. Note the second clause: "HFrEF is LVEF ≤ 40%" quotes 40 from a cited guideline — that is **provenance, not invention**. The same sentence with no citation is a hallucination. `tests/test_guardrails.py` pins both cases. |
| **3** | **Arithmetic is not negotiable.** `hf_category` is computed deterministically by Agent 2 from LVEF and the guideline cut-points. | If the model disagrees with arithmetic, the model is wrong. It does not get a vote. |
| **4** | **No claims about modalities never imaged.** The system sees a short-axis cine segmentation. A report asserting late gadolinium enhancement, perfusion, valve or wall-motion findings is describing something it never looked at. | *Recommending* an LGE study is good practice. *Reporting its results* without doing it is fabrication. The invariant distinguishes the two. |

### Enforcement: a bounded repair loop, ending in refusal

```
generate → validate → violations? → regenerate, naming each violation explicitly
                          ↓ (after max_repairs = 2)
                    strip provably fabricated citations
                          ↓ still violating?
                    raise UngroundedReport  →  Status.LLM_ERROR, NO report emitted
```

**The system does not downgrade to a best-effort report with a low score.** An ungrounded clinical
report is worse than no report. This is deliberately the *same refusal posture* the plausibility
gate takes on a broken segmentation mask (H3) — applied to a broken report. One principle, two
places.

Config: `llm.enforce_invariants: true`, `llm.max_repairs: 2`.

**Consequence to state plainly in the thesis:** on an enforced pipeline, `citation_fidelity` and
`numeric_fidelity` read **1.0 by construction**. That is not a result — it is the *definition* of
enforcement working. **The scientifically interesting number is how often the guardrail had to
fire**, which is logged per subject and counted in `metrics.json`. Report *that*, and report the
**unenforced** fidelity as the honest baseline of what the raw model does. Reporting 1.0 fidelity
without saying it was enforced would be the kind of quiet overclaim this thesis exists to avoid.

### Measured, 2026-07-13 (qwen2.5:14b, 2,592-chunk corpus, 6 ACDC subjects)

| | |
|---|---|
| Reports emitted | **6 / 6** |
| numeric fidelity (enforced) | **1.000** |
| citation fidelity (enforced) | **1.000** |
| dangling citations | **0** |
| unsupported numbers | **0** |
| **Subjects where the RAW model violated an invariant** | **3 / 6 (50 %)** — all repaired on the first retry |

**Read that last row, not the 1.000s.** Half the time, an unguarded 14-billion-parameter model
writes something it cannot support. The guardrail catches it. *That* is the finding; the 1.000 is
just arithmetic once you have the guardrail.

### A bug worth confessing in the methods section

The first version of this guardrail **refused 100 % of reports.** `chunk_id`s are hex
(`ce297edabb7ec8c0`), the model marks its grounding inline (`... [chunk_id: ce297edabb7ec8c0]`),
and the number scanner ran *before* those markers were stripped — so it pulled "297" out of the
middle of a hex id and called it a hallucinated measurement. The guard was fabricating its own
violations.

This is the same class of failure as the plausibility gate that initially rejected 65 % of the
expert ground truth: **a guard that rejects everything is exactly as broken as one that rejects
nothing — it just fails in the flattering direction**, because it *looks* rigorous. Both are now
pinned by regression tests (`test_inline_chunk_id_markers_are_not_read_as_measurements`,
`test_gate_accepts_expert_ground_truth`). If you present a guardrail in a viva, be ready to say
what it accepts, not only what it rejects.

---

## 2. Security: the injection surface, not inflated

**The honest position, and it is a strength — say it rather than pretending to a threat you don't
have:**

> The only text that ever reaches the language model is (a) guideline passages we ingested
> ourselves and (b) numeric JSON produced by our own code. **There is no user-supplied free text
> anywhere in the pipeline.** The prompt-injection surface is therefore genuinely small.

That is more credible than claiming a heroic defence against an attacker who cannot reach you.

**But it is not zero, and the non-zero part is the corpus.** A PDF is an untrusted byte stream:
text can be white-on-white, inside a form field, or in a footnote nobody reads. Today's corpus is
five ESC/AHA guidelines and the risk is negligible — but a *future* corpus (a hospital's own
protocol documents, a preprint, anything a collaborator hands you) is far less trustworthy. So the
defence is built now, while it costs nothing.

| Control | Where | What it does |
|---|---|---|
| **Ingest-time quarantine** | `guardrails.scan_injection`, called in `corpus.build_corpus` | Chunks carrying imperative/role-play patterns (`ignore previous instructions`, `system:`, `<im_start>`, code fences, `you are now`) are **quarantined before they enter the index**. A poisoned chunk that is never indexed can never be retrieved by any query. Ten patterns; deliberately conservative. |
| **Structural sanitisation** | `guardrails.sanitise` | Role markers and code fences are defanged so a passage cannot break out of its delimiter. Clinical text is **not** rewritten — a defence that silently edits guidelines would be worse than the disease. |
| **Passages are data, never instructions** | `report.SYSTEM` | Passages are delimited and the system prompt states they are reference text only. |
| **No action surface** | `llm.py` | The model has **no tools, no network, no filesystem**. It emits one JSON object against a fixed schema. Even a successful injection has nothing to *do* — there is no command for it to issue. This is the strongest control here, and it is architectural rather than a filter. |
| **Schema-constrained output** | `llm.complete_json` | The model cannot emit free-form text, only a validated `Report`. |
| **No data egress** | `llm.provider: ollama` | With the default local provider, **nothing leaves the machine**. Under any provider, images and identifiers are never transmitted — only numbers and guideline text. `allow_remote_llm` is an explicit, conscious opt-in. |

**Agent-to-agent security (the specific C15 wording).** Every inter-agent boundary is a Pydantic
model (`cmr/types.py`). A malformed or adversarial payload **cannot cross a node boundary, because
it will not validate** — that is enforcement, not intention. Agent 2 cannot be persuaded by text;
it is arithmetic. Agent 1 cannot be persuaded by text; it is a convolution. The only agent that
consumes language is Agent 3, and it is the one wrapped in the invariants above.

**Tested:** five real injection payloads are detected (`test_injection_payloads_are_detected`), and
— just as important — **real guideline text is not flagged** (`test_real_guideline_text_is_not_flagged`).
A defence that quarantines genuine clinical text would silently delete the corpus, which is a worse
failure than the attack.

---

## 3. What is still open, honestly

- **Semantic hallucination is not solved and cannot be.** A report can satisfy all four invariants
  and still reach the wrong conclusion. That is what the **expert reader study** is for, and it is
  why the study asks clinicians to rate *guideline fidelity* and *citation correctness* — the two
  axes BAAI's six-radiologist study structurally could not assess, because its reports carry no
  citations at all.
- **Retrieval recall is not measured.** We know every citation *resolves*; we do not know whether
  the *best* passage was retrieved. Measuring that needs a labelled query→passage relevance set,
  which does not exist for this corpus. Say so.
- **The injection blocklist is a blocklist.** It will not catch a novel phrasing. The architectural
  control (no tools, no action surface) is what actually holds; the pattern scan is defence in
  depth, not the defence.
