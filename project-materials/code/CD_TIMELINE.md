# Candidature Defence — 4-Month Timeline

Muhammad Tahir · MCS by Research, Universiti Malaya · FSKTM
Written 2026-07-11. Target CD: **week of 9–15 November 2026**.
Companion to [PLAN.md](PLAN.md) (what to do) and [PROCESS.md](PROCESS.md) (what is known).

---

## 1. What the panel actually marks — and why it is not the defence you just passed

`[WEB]` [UM Candidature Defence Evaluation form, Appendix C, effective 24.11.2023](https://ias.um.edu.my/Student%20Affairs/downloadable%20form/Candidature/Seminar%20Form%2011032025/Candidature%20Def/Evaluation%20Form%20for%20Panel.pdf) ·
[FSKTM Guidelines for PD/CD](https://fsktm.um.edu.my/fsktm/doc/postgraduate/5-%20Application-Guideline-PD-CD.pdf)

The CD rubric is a **different instrument** from the PD rubric. Compare:

| Criterion | Proposal Defence | **Candidature Defence** | Your PD score |
|---|---|---|---|
| Title & Abstract | 5% | — *(gone)* | 4.0/5 |
| Introduction | 25% | **10%** | 18.8/25 |
| Literature Review | 25% | **25%** | 20.0/20 |
| Conceptual Framework / Methods | 20% | **20%** | 15.0/20 |
| **Results and Discussions** | — | **20%** ← *new* | — |
| Summary → **Conclusion** | 5% | **10%** | 3.8/5 |
| Academic Style, Language, References | 10% | **10%** | 7.5/10 |
| Communication / Q&A | 10% | **5%** | 8.0/10 |

**Pass mark 65.00.** Below 65 = **fail → repeat the CD seminar, or terminated from study.** A Turnitin
similarity index is recorded on the form.

Read that table as a set of instructions:

1. **20% of your mark is for results that do not exist yet.** This is the whole game. The rubric's
   Excellent descriptor is *"The research objectives are fulfilled / research questions or hypotheses raised
   are addressed… Results are well analysed and interpreted… the research significance and contribution are
   shown."* Not "will be shown."
2. **Literature Review stays at 25% — your single largest block — and it is your biggest exposure, not your
   strength.** PD gave it 20/20. PROCESS.md §7.3 shows the chapter mischaracterises both competitor papers
   on eight separate counts. You got full marks for a chapter that is wrong. A CD panel that reads the BAAI
   paper collapses a quarter of your mark. **Rewriting Chapter 2 is the highest-expected-value writing task
   in this entire plan.**
3. **Introduction fell 25% → 10%.** Stop polishing it.
4. **Conclusion doubled to 10%, and the rubric explicitly requires a *"research continuation plan."*** Free
   marks. Write it deliberately.
5. **Style/Language/References is still 10%** and is pure mechanical grind — ~12 citation-year mismatches,
   3 phantom citations, 7 broken entries (PROCESS.md §8.5). You scored 7.5/10 with those errors present.
   This is the cheapest 2.5 marks on the board.

**The purpose of CD, in the form's own words:** *"to monitor the research progress of the candidate and to
give feedback on how to further improve the research."* It is a progress review, not a thesis defence. You
do not need a finished system. You need **credible, honestly-labelled partial results, a method that
survives questioning, and a plan that gets you to submission.**

### 1.1 The CD report

`[WEB]` FSKTM guideline. **5,000–10,000 words**, submitted to the Postgraduate Office **one week before the
presentation**, and it must be consistent with the slides. Required contents:

> a) Abstract (500 words in Bahasa Malaysia and English); b) Objectives and statement of problem; c) The
> importance and relevance of the study; d) Brief and concise literature review; e) **Solid research
> methods**; f) **Research results obtained so far**; g) Brief bibliography; h) **Research plan that leads to
> the submission of the dissertation on the designated date**; i) List of publications or conference papers
> presented during the period of candidature, if any.

Eligibility: Active (AK) status · Research Methodology course completed · Proposal Defence completed ·
supervisor's approval. **Master's candidates must complete CD no later than the 3rd semester** — for you,
Semester I 2026/2027. November is inside that window. It is not optional.

---

## 2. Two facts I resolved today that change the plan

These were PROCESS.md §12 open questions 1 and 2. Both are now answered, and both are good news.

### 2.1 BAAI Cardiac Agent's weights ARE public

`[WEB]` [github.com/plantain-herb/Cardiac-Agent](https://github.com/plantain-herb/Cardiac-Agent) — **Apache
2.0**. Released: agent weights (`TaipingQu/BAAI-Cardiac-Agent` on HuggingFace), all expert checkpoints
(SAX/2CH/4CH cine seg, LGE seg, CDS, NICMS) as `.pth` under `weights/`, training code, inference code,
FastAPI serving stack, plus `TaipingQu/CMRAgentEvalSet` (1,003 NIfTI) and `TaipingQu/CMR-MULTI`.

**So the highest-value experiment in the project is a two-week job, not a two-month one.** Run the
state-of-the-art agentic CMR system on ACDC and M&Ms — the first public-benchmark evaluation of it that
anyone has done. It is your headline result, your one baseline that matters, and it is a submittable paper
on its own.

### 2.2 CineMA is public — and it is a trap as well as a gift

`[WEB]` [github.com/mathpluscode/CineMA](https://github.com/mathpluscode/CineMA) ·
[huggingface.co/mathpluscode/CineMA](https://huggingface.co/mathpluscode/CineMA). Python 3.11. Released
fine-tuned checkpoints, **3 seeds each**:

| Checkpoint | Fine-tuned on |
|---|---|
| `finetuned/segmentation/acdc_sax/` | **ACDC** |
| `finetuned/segmentation/mnms_sax/` | **M&Ms** |
| `finetuned/segmentation/mnms2_sax/`, `mnms2_lax_4c/` | **M&Ms-2** |
| CVD classification | ACDC, M&Ms, M&Ms-2 |
| Direct EF regression | ACDC, M&Ms, M&Ms-2 |

**The gift:** you get segmentation masks on all three datasets with **zero training and zero GPU**, in days.
The entire downstream pipeline (Agents 2, 3, 4) is unblocked immediately. And the **3 released seeds give you
an ensemble for free** — which means Agent 4's entropy-uncertainty maps no longer depend on nnU-Net's 5-fold
ensemble existing.

**The trap:** PLAN.md §3.2 and §2.9 call CineMA a *"zero-shot ablation."* **That is now false and you must
stop saying it.** The released ACDC checkpoint was *fine-tuned on ACDC*. Running it on ACDC is a **fully
supervised baseline**, not zero-shot. If you report an ACDC-finetuned checkpoint's ACDC score and label it
"zero-shot," a panel member who spends four minutes on the HuggingFace page will find it, and everything
else you claim becomes suspect.

**Use it correctly and it is worth more, not less:**

- `acdc_sax` → ACDC test = supervised baseline (label it that way).
- `acdc_sax` → **M&Ms / M&Ms-2 = genuine cross-dataset zero-shot.** *This is RQ3.* It is the real
  generalisation experiment and the number the thesis is actually about.
- `mnms_sax` → M&Ms = in-domain upper bound, i.e. the ceiling the cross-dataset run is falling short of.

⚠️ **Before you believe a single number:** each checkpoint emits its own dataset's label order. The ACDC
checkpoint will output `1=RV, 2=Myo, 3=LV`. The M&Ms checkpoint will output `1=LV, 2=Myo, 3=RV`. Run the
concentricity assertion (PROCESS.md §11.3) on the *predictions*, not just the ground truth. This is exactly
the bug PROCESS.md §5 exists to prevent, and cross-checkpoint work is precisely where it will bite.

---

## 3. C9 — decided: public datasets only. Now build the rebuttal.

**Decision (candidate, 2026-07-11): no proprietary hospital cohort. ACDC + M&Ms + M&Ms-2 as per the
proposal.** This section is no longer a task. It is a *defence preparation* item, and it is the single most
likely question you will be asked.

I read `PD Comments by Panel.pdf` — it had never been read. The panel's actual words were stronger than the
knowledge base's transcription of them:

> **C9. Use of Private/Proprietary Dataset.** Since this is a Master by Research programme, the candidate is
> expected to collect and process their own data. The panel **strongly recommends incorporating a proprietary
> hospital dataset** to ensure the candidate undergoes the full data collection and preprocessing process,
> which is a core research requirement.

Note "**strongly recommends**," not "requires." That is the gap you are working in. But **Dr. Mohamad Hazim
issued C9 and will be sitting on the CD panel.** He will ask. Walking in without a prepared answer is the
one way this decision costs you marks; walking in with the answer below costs you nothing.

### 3.1 What you must NOT say

*"We decided secondary data was sufficient."* That is the answer they pre-emptively rejected. It concedes the
premise — that you skipped a core research requirement out of convenience.

### 3.2 The answer to give — three moves, in this order

**Move 1 — Reject the premise that you collected no data. You collect and curate a novel dataset; it just
isn't imaging.**

The guideline corpus **is** primary data collection and preprocessing, and it is the more novel half of the
thesis. You will personally: acquire the source guidelines, parse and segment them, chunk them at 200–400
tokens, and **hand-annotate every chunk with its class-of-recommendation and level-of-evidence** — metadata
that, per PROCESS.md §7.4, **neither BAAI nor CardAIc carries.** That is acquisition → cleaning →
structuring → annotation → validation. It is the full pipeline the panel asked you to undergo. It is also
the artefact your novelty claim rests on. Say this first, and say it as a contribution, not as a substitute.

Add the imaging-side preprocessing you genuinely do: three heterogeneous datasets, three *different* ED/ES
index mechanisms (PROCESS.md §4), spacing from 1.37–1.79 mm in-plane and 5–10 mm slice, and a
**label-convention inversion between ACDC and M&Ms that is documented nowhere and silently corrupts every
cross-dataset metric** (PROCESS.md §5). You found that yourself, by decoding the volumes. Bring the
concentricity-test table. That is what "undergoing the full preprocessing process" looks like, and it is far
more convincing than having FTP'd 25 DICOMs off a hospital PACS.

**Move 2 — Argue that a small single-centre cohort would have made the science *worse*, not better.**

This is the strong move, and it is true. Your research question (**RQ3**) is *cross-vendor, cross-centre
generalisation*. Your public data gives you **855 subjects, 4 vendors (Philips, Siemens, GE, Canon), and 5+
centres**, with ground truth. A realistic retrospective UMMC cohort inside this timeline is **20–30 subjects,
one centre, one or two scanners, and unannotated** — you would have to segment it yourself to have any ground
truth at all. It cannot support a generalisation claim; it can only dilute one. Adding it would trade a
statistically powered multi-vendor experiment for an anecdote, and consume the months you need to run BAAI.

**Move 3 — Convert the constraint into the contribution.** (PLAN.md §0.3 — this is already your Key Claim.)

BAAI's 2,413-patient cohort is **private**. Nobody can reproduce, audit, or independently compare its result.
Its Data Availability statement reads *"No other publicly available datasets were used."* **No agentic CMR
system has ever been benchmarked on a public cardiac MRI dataset.** Your rivals do not lack scale — they lack
**verifiability**. Using public benchmarks is not the weakness in your design; it is the *thesis*. And since
BAAI's weights turned out to be public (§2.1), you can prove the point by running their system on ACDC and
M&Ms yourself — the experiment their own paper made impossible to check.

That is not a defensive answer. It is a better research programme than the one C9 proposed, and it should be
delivered that way.

### 3.3 What still has to happen

- **Rewrite §3.11.** It currently states outright that *"ethics clearance from the UMMC Ethics Committee
  would not be necessary"* — a flat disclaimer, which is precisely what *triggered* C9. Replace it with:
  the datasets' own ethics provenance (ACDC/M&Ms/M&Ms-2 were each collected under approved protocols with
  consent, and are released for research), your data-governance position, and the annotation/curation work
  you *do* perform. Same conclusion, argued instead of asserted.
- **Put the C9 answer on a slide.** Do not improvise it.
- **Reader study — keep it, and note it is a *separate* decision from C9.** 2–3 clinicians rating ~50
  generated reports on citation correctness and guideline fidelity is (a) primary data you collect, and
  (b) the only thing that answers **C13** (clinical validity) and feeds the 20% Results block. It needs only
  a **low-risk** ethics application, not the 3–6 month MREC route the hospital cohort needed. **If you want
  it in the CD, file that application by ~mid-August.** Flagging it explicitly so it is your call, not a
  silent omission — if you drop this too, C13 has no human-validation answer and you lean entirely on
  pathology top-1 + factual consistency, which is thinner but not fatal.

---

## 4. The timeline

Working backwards from a **CD in the week of 9–15 Nov 2026**:

- Report to Postgraduate Office **one week before** → hard deadline **~2 Nov**. Target **30 Oct** with buffer.
- Turnitin + supervisor sign-off needs ~2 weeks before that → **writing freeze ~16 Oct**.
- Therefore **every experiment must produce a number by ~9 Oct**. That gives **13 experimental weeks and 4
  writing weeks.** Everything below is sized to fit that, not to fill four months.

### Week 1 · 13–19 Jul — Environment + data spine. No modelling.

With the hospital cohort dropped (§3), nothing in this plan has a 3-month external clock any more. Week 1 is
now purely technical, and the *one* remaining external item is the low-risk reader-study ethics application,
which is due by mid-August, not now.

| | Task |
|---|---|
| Day 1 | Python 3.11 + venv. `torch` (MPS), `nibabel`, `SimpleITK`, `numpy`, `pandas`. Not system 3.9. |
| Day 1–3 | Download BAAI weights **and** CineMA checkpoints. **Confirm each loads and runs on MPS.** If BAAI's 7B LMM won't fit in 16 GB, you need to know in Week 1, not Week 5 — it changes your GPU budget and it is now the biggest technical unknown in the plan. |
| Day 3–5 | `cmr/data.py`: three adapters → one canonical record. **Canonical labels `1=LV, 2=Myo, 3=RV`;** remap ACDC on load. |
| Day 5 | Run the concentricity assertion + `ESV < EDV` check over **all 855 subjects**. Log violations — they are a real data-quality finding, worth a paragraph in the report **and part of your C9 answer** (§3.2, Move 1). |
| 5 min | Check the arrhythmia paper's submission portal: **BMJ or BMC** Medical Informatics and Decision Making? (PROCESS.md §1.2, F6.) One of those journals does not exist. |
| Also | Tell Dr. Uzair the C9 decision and the reasoning in §3.2, so he is not surprised by it in the room and can back you. |

### Week 2 · 20–26 Jul — Ground-truth quantification. Your first real result.

- EDV, ESV, LVEF, LV mass for **all 855 subjects** from ground-truth masks. No network, no AI.
- EF distribution per pathology. **DCM must sit low, HCM preserved.** If it doesn't, your loader is wrong and
  you have found it in Week 2 instead of October.
- HF categories at the *correct* cut-points (≤40 / 41–49 / ≥50 — PROCESS.md §6.1, not the proposal's broken rule).
- 🔴 **Count how many subjects fall in GT LVEF ∈ [35, 55]%.** This sizes **H2**. If only a handful of patients
  sit near a decision boundary, boundary-aware recomputation has almost nothing to act on, and H2 must be
  reframed **now** — as a *safety property* rather than an accuracy claim — not discovered in October when
  the report is written. This single count is the cheapest de-risking in the plan.
- **Deliverable:** Results §1 — dataset characterisation table. Real numbers in the CD report by 26 July.

### Weeks 3–4 · 27 Jul – 9 Aug — Segmentation results, still without training anything.

- CineMA `acdc_sax` → ACDC test. **Supervised baseline.** DSC, HD95.
- CineMA `acdc_sax` → **M&Ms + M&Ms-2. Cross-dataset zero-shot. This is RQ3.** Report the DSC drop,
  stratified by **vendor** and **centre** — M&Ms gives you Philips/Siemens/GE/Canon and 5 centres for free.
- CineMA `mnms_sax` → M&Ms. In-domain ceiling.
- ⚠️ Assert the label order of every prediction before computing any metric (§2.2).
- Write every predicted mask to disk. Everything downstream reads masks, never a network.
- **Deliverable:** Results §2 — segmentation + the generalisation gap. Two of the largest tables in the CD
  report exist by **9 August**, with no GPU rented and no model trained.

### Weeks 4–6 · 3–23 Aug — **BAAI Cardiac Agent on ACDC and M&Ms. The headline.**

- Stand it up from the public Apache-2.0 release; run it on ACDC and M&Ms.
- Report its DSC, its EF agreement, its diagnostic accuracy — on public data, for the first time.
- Compare against your CineMA numbers and (later) nnU-Net.
- **Deliverable:** Results §3, *and* a short paper. This experiment alone is publishable, and UM requires
  ≥1 Category A/B publication for graduation. Two birds.
- **Risk:** a 7B LMM plus nine expert models will likely not fit a 16 GB M4. Budget rented GPU here first,
  ahead of nnU-Net.

### Weeks 5–9 · 10 Aug – 13 Sep — Agent 3, guideline RAG. The novelty core. *(parallel; no GPU)*

- Corpus (guideline-only — that purity **is** contribution #2): Petersen *CMR in ESC guidelines*
  (PMC10364363) · 2021 ESC HF + 2023 focused update · 2022 AHA/ACC/HFSA HF · 2021 AHA/ACC Chest Pain ·
  SCMR 2025 reference values. **Never cite the non-existent "ESC 2022 CMR Guidelines" again (F2).**
- Chunk 200–400 tokens, 50 overlap, carrying `{source, section, page, class_of_recommendation,
  level_of_evidence}`. **CoR/LoE metadata is genuinely unclaimed — neither competitor carries it.**
- **MedCPT** embeddings + FAISS `IndexFlatIP` + **BM25 hybrid retrieval.** Keep BioBERT and dense-only as
  ablation rows — CardAIc's own ablation shows hybrid beats both, and shipping dense-only BioBERT means
  shipping a retriever weaker than a paper you cite (PROCESS.md §8.6 M4).
- **Factual-consistency checker:** every numeric claim in the report matches Agent 2's JSON within tolerance;
  every citation resolves to a real chunk. Deterministic, cheap, and it is the metric the whole architecture
  exists to optimise.
- Conflict detection between ESC and ACC/AHA (**C11**). 🔴 HF thresholds *agree* (PROCESS.md §6.1). If you find
  no real conflicts in the corpus, **say so and reframe C11 as a safety property** rather than inventing one.
- **Deliverable:** Results §4 — retrieval ablation + factual-consistency scores.

### Weeks 8–11 · 31 Aug – 27 Sep — Agent 4 (XAI) + orchestration.

- **Seg-Grad-CAM** (Vinogradova et al., 2020) — *not* vanilla Grad-CAM, which is defined for classifiers (M5).
- Entropy uncertainty maps **from CineMA's 3 released seeds** — free, no nnU-Net dependency.
- **Build the link structure**: *this heatmap region supports this cited sentence.* Not a heatmap beside a
  citation — the **link**. It is your strongest surviving differentiator; BAAI has no visual XAI at all.
- LangGraph `StateGraph`, Pydantic contracts at every node boundary (**C10**), plausibility gate, typed
  `FAILED_SEGMENTATION` refusal state, boundary-aware recomputation bounded at 2 iterations (**C12**), JSONL
  run trace per subject (evidence for **H3**).
- **Deliverable:** Results §5 + C10/C12/C15 answered as *implemented code*, not as promises. That distinction
  is exactly what C16 was asking for.

### Weeks 10–13 · 14 Sep – 11 Oct — nnU-Net + full evaluation.

- nnU-Net `2d` + `3d_fullres`, 5-fold, on rented GPU, running in the background while you write.
- **Be honest about the fallback:** if nnU-Net does not finish, present **CineMA as the interim backbone** and
  nnU-Net as in-progress. The panel will not fail you for an unfinished training run. They will fail you for
  having no results. Do not let a GPU queue hold the CD hostage.
- Full ablation grid; pathology top-1; HF-category accuracy; factual consistency; **baseline-rescaled**
  BERTScore; paired bootstrap CIs.
- Reader study (Weeks 11–14, ethics permitting): 2–3 clinicians × ~50 reports, Likert on **citation
  correctness and guideline fidelity** — the axes BAAI's six-reader study *could not* assess, because its
  reports carry no citations. Krippendorff's α.

### Weeks 12–16 · 28 Sep – 1 Nov — Writing. Start before the results are final.

Ordered by marks-per-hour, which is not the order you will feel like doing them in:

1. **Chapter 2 rewrite — 25% of the mark, and currently wrong.** Recharacterise BAAI and CardAIc from
   PROCESS.md §7. Move CardAIc out of "closest CMR systems" (it has zero mentions of MRI). Retire the three
   dead novelty claims. Build the honest head-to-head table. Replace the Experimental Gap with the
   **Reproducibility & Verifiability Gap**.
2. **Results & Discussion — 20%, new.** Everything from Weeks 2–13. Label every number honestly: supervised
   vs zero-shot, in-domain vs cross-dataset.
3. **Methods — 20%. Write §3.9 (Orchestration) and §3.10 (Evaluation Metrics). They do not exist** (C5 is not
   a renumbering problem; two sections are simply absent). Add the C14 no-training table.
4. **Conclusion — 10%.** Restate objectives, summarise findings, and give the **research continuation plan**
   the rubric explicitly demands.
5. **Style/References — 10%.** The C1/C6/C7/C8 grind: ~12 year mismatches, 3 phantom citations, 7 broken
   entries, alphabetise, fix the `M&Ms;` escape artifact throughout. Mechanical, boring, and the cheapest
   marks available.
6. **Introduction — 10%.** Fix C2 (855/495 counts) and C3 (one cell edit in Table 1.2). Do not gold-plate it;
   it is worth less than half what it was worth at PD.
7. Abstract, 500 words, **EN + BM**.

### Weeks 15–16 · 19 Oct – 1 Nov — Freeze, Turnitin, sign-off, submit.
Report to the Postgraduate Office by **30 Oct** (hard limit: one week before the presentation).

### Week 17 · 2–8 Nov — Slides + dry run.
Slides must be **consistent with the report** — the FSKTM guideline says so explicitly, and PD found three
places where they weren't (backbone, schedule, LLM version). Rehearse answers to all sixteen corrections.

### Week 18 · 9–15 Nov — **Candidature Defence.**

---

## 5. Risks, ranked

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Chapter 2 (25%) is built on false claims about both competitors.** PD awarded it 20/20 anyway. It is the largest single block of marks in the rubric. | Rewrite it first, Week 12 at the latest. |
| 2 | **Panel 2 asks why C9 was not acted on** — and it will be asked, by the man who wrote it. | Deliver §3.2 as a prepared, slide-backed argument, not an apology. Brief your supervisor in Week 1. |
| 3 | **Mislabelling CineMA's ACDC checkpoint as "zero-shot."** | It is a supervised baseline on ACDC. Cross-dataset (→ M&Ms) is the zero-shot run. Label both correctly. |
| 4 | **Label-order corruption across checkpoints.** ACDC ckpt emits `1=RV`; M&Ms ckpt emits `1=LV`. | Run the concentricity assertion on **predictions**, every time, not just on GT. |
| 5 | **BAAI's 7B LMM won't fit 16 GB** — now the biggest technical unknown. | Find out in Week 1. Rent GPU for BAAI *before* renting for nnU-Net. |
| 6 | **H2 has no patients to act on** (too few near the LVEF decision boundary). | Count them in Week 2. Reframe early if needed. |
| 7 | **nnU-Net doesn't finish training in time.** | Present CineMA as interim backbone. Do not let this block the CD. |
| 8 | **C13 has no human-validation answer** if the reader study is dropped along with the hospital cohort. | Decide by mid-August. If dropped, lean on pathology top-1 + HF-category accuracy + factual consistency, and say plainly that expert validation is deferred to the thesis. |

---

## 6. What "done" looks like on 9 November

Not a finished system. A **defensible progress position**:

- Results on public benchmarks from **three** segmentation sources (CineMA supervised, CineMA cross-dataset,
  BAAI), with the vendor/centre generalisation gap quantified.
- **The first public-benchmark evaluation of BAAI Cardiac Agent that anyone has run** — submitted as a paper.
- A working guideline-RAG agent with passage-level citations and a deterministic factual-consistency score.
- An orchestrated pipeline that **refuses to diagnose** on a failed segmentation, with the trace to prove it.
- A Chapter 2 that describes the competition accurately, and a novelty claim that survives someone reading
  the papers.
- A **curated, CoR/LoE-annotated guideline corpus you built yourself** — the data-collection-and-preprocessing
  answer to C9, and the artefact the novelty rests on.
- A rewritten §3.11 that *argues* the ethics position instead of disclaiming it, and a prepared C9 rebuttal
  on a slide.
- A continuation plan to submission.

That clears 65 comfortably. Chasing a finished system instead, and arriving with an unrewritten Chapter 2 and
no results, does not.
