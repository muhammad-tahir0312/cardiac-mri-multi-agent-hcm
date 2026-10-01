# Implementation Plan — Multi-Agent CMR Analysis & Decision Support

Muhammad Tahir · MCS by Research, Universiti Malaya
Revised 2026-07-10 after reading the proposal report, the defence slides, **the full BAAI Cardiac Agent
and CardAIc-Agents papers**, and inspecting the data on disk.

---

## 0. The novelty claim, rebuilt on what the papers actually say

I read both competitor papers end to end. Chapter 2 characterises them incorrectly in several places,
always in the direction that inflates your novelty. Here is the verified position.

### 0.1 BAAI Cardiac Agent (Qu et al., 2026 — arXiv:2604.04078)

| Report §2.7 claims | The paper actually says |
|---|---|
| *"evaluation is done only using the small dataset"* | 2,413 patients, two hospitals, IRB-approved |
| *"no cross-validation on multi-centre/multi-vendor generalization set"* | External validation cohort; scanners are **Philips Achieva, GE Discovery MR750w, and Siemens** — it *is* multi-vendor |
| *"no information about knowledge retrieval and output alignment with ACC/AHA, ESC clinical guidelines"* | **It has a RAG tool.** Corpus = ChatCAD+ cardiovascular section + CVD clinical guidelines + PubMed. Its worked example outputs *"Reference: 2023 ACC/AHA/SCAI Guideline for the Management of Patients With Aortic Stenosis"* |
| *"no explainability methods are employed"* | **True.** Zero mentions of Grad-CAM, saliency, attention maps, or uncertainty. "Transparency" means the user can query reasoning steps |

It also calls itself *"the first end-to-end agent framework specifically designed for CMR imaging
analysis."* It beats nnU-Net on its own data (SAX cine DSC 90.21 vs 87.42). It reports Bland-Altman and
Pearson r for LVEDV/LVESV/LVEF/SV/LVM (r = 0.968 / 0.979 / 0.925 / 0.906 / 0.937). It reports
**BERTScore F1 = 0.898**. It ran a **six-radiologist reader study** across three experience levels.

Read that list again. Bland-Altman, BERTScore, and a multi-reader Likert study are three of the things
this plan previously proposed as your contributions. They are not contributions. They are table stakes.

### 0.2 CardAIc-Agents (Zhang et al., 2025 — arXiv:2508.13256)

| Report §2.7 claims | The paper actually says |
|---|---|
| *"lack of implementation studies"* | Code public at github.com/ytz300/CardAIc-Agents |
| *"absence of guideline-based reasoning"* | **CardiacRAG agent.** Bio_ClinicalBERT embeddings + FAISS + chunking with overlap + hybrid dense/keyword retrieval + `Cite` returning original chunks. Knowledge base includes the 2021 ESC HF guideline |
| *"lack of explainability"* | **Visual review panels** — patient profile, labelled ECG waveforms, echo view identification, LV segmentation overlays |
| *"limited to one institution only"* | **Three public datasets** — MIMIC-IV (1,524 patients), PTB-XL (10,147), PTB Diagnostic (268) |

And the fact that reframes everything: **CardAIc-Agents never mentions MRI, CMR, or magnetic resonance.
Not once, in the entire paper.** It is an ECG / echocardiography / EHR system.

So CardAIc-Agents is not your closest competitor and should not be presented as one. BAAI is. But
CardAIc is dangerous to you in a different way — see §0.4.

### 0.3 The good news, and it is genuinely good

BAAI's Data Availability statement reads: *"No other publicly available datasets were used."* It has
**never been evaluated on ACDC or M&Ms** — zero mentions of either. Its 2,413-patient cohort is private.
Its code, however, is public (github.com/plantain-herb/Cardiac-Agent), and a test benchmark subset is on
HuggingFace (`TaipingQu/CMR-MULTI`).

**No agentic CMR system has ever been benchmarked on a public cardiac MRI dataset.** BAAI's are private;
CardAIc's are public but are not MRI.

That is a real, checkable, unclaimed gap, and it is *yours to take*. It also converts your weakest number
— 495 public subjects against BAAI's 2,413 private ones — into your argument. Your rivals do not lack
scale. They lack verifiability. Nobody can reproduce, audit, or independently compare BAAI's result.

**And it suggests the single highest-value experiment available to you:** check whether BAAI released
model weights alongside its code. If it did, **run BAAI Cardiac Agent on ACDC and M&Ms.** That would be
the first public-benchmark evaluation of the state-of-the-art agentic CMR system, it gives you the one
baseline that actually matters, and it turns "they never validated publicly" from a complaint into an
experiment you performed. If only the architecture is released and not the weights, retrain it on ACDC —
it was trained on just 150 annotated samples, which is within your reach.

Do this. It is a strong, cheap, defensible contribution, and it is exactly the kind of thing an external
examiner respects.

### 0.4 What actually survives as novelty

Stated precisely, so it holds under cross-examination:

1. **Guideline-passage grounding inside the diagnostic report.** BAAI's RAG is a *conversational sidecar*
   — a tool the user invokes to ask a knowledge question. Its structured diagnostic report carries **no
   citations at all** (see its Figure 1: *"Diagnostic Result: High probability of Hypertrophic
   Cardiomyopathy (HCM) is considered."* — no source). Its citations, when they appear, are
   *document-level*. Yours are passage-level, in the diagnostic path, with document + section + page.
2. **Corpus provenance purity.** CardAIc's knowledge base mixes Mayo Clinic, the NHS website, and
   MedlinePlus with guidelines. BAAI's mixes ChatCAD+, guidelines, and PubMed. **A guideline-only corpus,
   with class-of-recommendation and level-of-evidence carried as chunk metadata, does not exist.**
3. **Linked dual explainability.** BAAI has no visual XAI whatsoever. CardAIc has visual panels but they
   are not linked to the textual rationale. The *link* — this heatmap supports this cited sentence — is
   unclaimed.
4. **Decision-boundary-aware recomputation** (§2.6). Nobody has it. But see the warning in §5, H2.
5. **A quantitative guideline-fidelity metric** (§4.3). Neither paper reports anything like it.
6. **Public-benchmark reproducibility** (§0.3).

**Retire these claims immediately.** They are dead:
- *"no prior cardiac MRI system jointly integrates agentic orchestration, clinical reasoning, and explainability"* — BAAI does orchestration + reasoning + RAG; only explainability is missing.
- *"first end-to-end agentic CMR pipeline"* — BAAI claims this in print, before you.
- Any Experimental Gap phrased as *"rivals lack multi-centre or multi-vendor validation."* BAAI has both.

**Replace the Key Claim with something like:** *"No prior CMR system grounds its diagnostic statements in
citable clinical-guideline passages, links those citations to visual explanations, or reports its results
on public benchmarks."* Every clause of that is true and defensible.

**Concrete actions, in order:**
- Rewrite §2.7, Table 1.1, Table 2.1, Table 2.2, §1.2, §1.3, §1.7 around what these papers actually do.
- Move CardAIc-Agents out of "closest cardiac MRI systems." It is not a CMR system. Say so — it is a point
  in your favour, honestly made.
- Replace the Experimental Gap with a **Reproducibility & Verifiability Gap**.
- Build a head-to-head table: rows = BAAI, CardAIc, Proposed; columns = modality, orchestration,
  guideline-passage citation, linked dual XAI, boundary-aware recomputation, cohort size, cohort public,
  public-benchmark results. Be scrupulously honest in the cohort-size row. Honesty there is what earns
  the reader's trust in the other columns.
- Check whether BAAI's weights are public. If so, plan the ACDC/M&Ms run (§2.9).

Nothing below this section matters if this section is not done first.

---

## 1. What I verified in the data

All three datasets are downloaded and intact (46 GB). I read the NIfTI headers and decoded the label
volumes directly.

| Check | Result |
|---|---|
| ACDC | 100 train + 50 test patients, **both with ground truth**. `Info.cfg` gives ED/ES frame indices + pathology group (20 each: DCM, HCM, MINF, NOR, RV). |
| M&Ms | **345 subjects, 5 centres** — not the 375 / 6 centres in Table 3.1. Split: 150 labeled train, 25 unlabeled, 34 val, 136 test. |
| M&Ms-2 | 360 subjects, short-axis **and** long-axis, ED/ES pre-extracted, GT provided, 8 disease classes. |
| **Label convention** | **ACDC is `1=RV, 2=Myo, 3=LV`. M&Ms and M&Ms-2 are `1=LV, 2=Myo, 3=RV`.** Verified on 9 subjects (2 ACDC, 4 M&Ms, 3 M&Ms-2) by testing which cavity the myocardium ring is concentric with (myo↔LV centroid distance ≤ 4.2 px; myo↔RV distance 25–42 px). See `PROCESS.md` §5. |
| M&Ms GT layout | GT is a **4-D volume the same shape as the image**, non-zero only at ED and ES. Frame indices live in the metadata CSV, not the file. |
| Spacing | ACDC in-plane 1.37–1.79 mm, slice thickness 5–10 mm. Heterogeneous — do not hard-code. |
| Hardware | Apple M4, 16 GB unified memory, **no CUDA**. No Python packages installed; system Python is 3.9. |

The label mismatch is the single most dangerous fact here. Train on ACDC, evaluate on M&Ms without
remapping, and every LV/RV Dice, every ejection fraction, and every heart-failure classification is
silently wrong but plausible-looking. Stage 0 exists to close it.

---

## 2. Defects in the written proposal

The panel found sixteen. Reading the report against the slides, the competitor papers, and the data, I
found the sixteen plus roughly twenty more.

### 2.1 The report contradicts itself about the segmentation backbone

- §3.3.2: *"The segmentation backbone follows nnU-Net… The cineMA approach will be tested as a zero-shot approach… in the ablation study."*
- Chapter 6 Gantt, Phase 2: *"**CineMA Backbone Implementation**"*
- Slide 10, Objective O1: *"automated segmentation (**CineMA**)"*

Your primary objective names a different backbone in three places. **Decide: nnU-Net primary, CineMA as
ablation.** Then fix slide 10 and the Gantt. §3.2 explains why this decision also has a happy practical
consequence.

### 2.2 Chapter 3 is missing two whole sections, not just numbering

Sections run 3.1 … 3.8, then jump to **3.11 Ethical Considerations**. Tables run 3.1 then jump to **Table
3.4**. The panel called this "renumbering" (C5). It is worse: the two absent sections are the two that
matter most.

- **§3.9 Orchestration Layer** — LangGraph, agent state, inter-agent contracts, error handling, feedback loop. Exists only on slide 22.
- **§3.10 Evaluation Metrics & Protocol** — Dice, HD95, MAE, BERTScore, all targets. Exists only on slide 23. **The Methodology chapter never defines how the system is evaluated.**
- Tables 3.2 and 3.3 should be the metrics table and the ablation table.

A methodology chapter with no evaluation section is a structural failure, not a typo.

### 2.3 The heart-failure thresholds are broken at both boundaries

§3.4 emits `(<40%=reduced, 40-50%=mildly reduced, >50%=preserved)`.

- LVEF exactly **40%** → matched by `40-50` → classified *mildly reduced*.
- LVEF exactly **50%** → matched by `40-50`, not by `>50` → classified *mildly reduced*.

ESC 2021 and AHA/ACC/HFSA 2022 both define HFrEF **≤ 40%**, HFmrEF **41–49%**, HFpEF **≥ 50%**. Your rule
misclassifies at *both* cut-points, in a thesis whose premise is guideline fidelity. It is also the
concrete motivation for §2.6.

### 2.4 "ESC 2022 Guidelines on Cardiac Magnetic Resonance Imaging" does not exist

Cited in Table 1.2, §1.6, and §3.5.2 as corpus document (1). There is no standalone 2022 ESC CMR
guideline; CMR recommendations are spread across ~19 disease-specific ESC guidelines. Replace with:

- Petersen et al., *Cardiovascular magnetic resonance in the guidelines of the European Society of Cardiology* (PMC10364363) — an aggregated CMR-recommendation document, i.e. exactly the corpus you want
- 2021 ESC HF guideline (+ 2023 focused update)
- 2022 AHA/ACC/HFSA HF guideline
- 2021 AHA/ACC Chest Pain guideline
- SCMR 2025 reference values (Kawel-Boehm) — the source for normal ranges

### 2.5 BERTScore: no reference reports, and the target is below the competition

ACDC and M&Ms ship masks and pathology labels. **Neither ships a radiology report.** BERTScore F1 ≥ 0.85
is currently measured against nothing.

Worse: **BAAI already reports BERTScore F1 = 0.898**, against real clinical reference reports it owns. Your
target is not just unmeasurable — it is *lower than the state of the art it is meant to beat*. And
un-rescaled BERTScore sits near 0.85 for unrelated sentence pairs anyway, so the target is satisfied by
noise.

Fix: (a) generate reference reports from GT measurements + pathology label via a fixed template, clinician-
signed on a sample; (b) report **baseline-rescaled** BERTScore; (c) add the **numeric factual-consistency
score** of §4.3 and make *that* the headline. Deterministic, cheap, measures hallucination directly, and
neither competitor reports anything comparable.

### 2.6 The feedback loop as specified is mechanically impossible

Slide 22: *"Iterative Feedback Loop — guideline evidence can refine segmentation masks."* A text-retrieval
agent has no imaging signal. It cannot refine a voxel mask.

**Reframe.** Agent 3 knows the guideline *decision boundaries*. When a computed value lands near one
(LVEF 39.4% against the HFrEF cut at 40%), Agent 3 emits a **re-measurement request**, and Agent 1 re-runs
with test-time augmentation and full 5-fold ensembling to tighten the estimate. Triggered by guideline
knowledge, executed in imaging space, bounded at 2 iterations, fully logged. **Decision-boundary-aware
recomputation.** It answers C12 and it is testable (H2).

### 2.7 The preprocessing spec contradicts the choice of nnU-Net

§3.3.3 specifies *"1.5 × 1.5 mm and 8 mm slice thickness"* and *"192 × 192 patches."* But §3.3.2 chose
nnU-Net precisely because it *"dynamically reconfigures itself."* **nnU-Net derives spacing and patch size
itself.** (Its own ACDC `3d_fullres` target spacing is ≈ 1.56 × 1.56 × 5.0 mm — not 1.5 × 1.5 × 8.)

Rewrite as: nnU-Net's automatically configured preprocessing, with derived spacing and patch size
**reported post-hoc as a result**.

### 2.8 ED/ES frame detection is unnecessary work

§3.3.3(iv): *"selection of ED and ES frames based on time gradient analysis."* You don't need it. **ACDC
gives ED/ES in `Info.cfg`. M&Ms gives them in the metadata CSV. M&Ms-2 ships them pre-extracted.** An
automatic detector injects error you cannot attribute. Use the provided indices; make the detector an
ablation if you want it.

### 2.9 The baselines need rebuilding

Table 3.4 lists nnU-Net Standalone, CineMA, MARL-Rad, CardAIc-Agents. Problems:

- **CardAIc-Agents is not a CMR system.** It cannot be a cardiac-MRI baseline. Remove it.
- **MARL-Rad** has no public weights and was evaluated on data you don't have.
- §3.8 says *"five systems"*; the table lists four (C4).
- The metric column says *"Report Quality (BERTScore, ROUGE)"* — the exact lexical/semantic conflation of C8.
- It names a metric "CAS" that is never defined.

**Replace with five baselines you can actually run:**

1. **nnU-Net standalone** — segmentation floor (DSC, HD95)
2. **CineMA** — foundation-model comparison (DSC, EF MAE). ⚠️ **Corrected 2026-07-11:** CineMA's released
   checkpoints are **fine-tuned per dataset** (`acdc_sax`, `mnms_sax`, `mnms2_sax`; 3 seeds each). Running
   `acdc_sax` on ACDC is a **supervised baseline, NOT zero-shot.** The genuine zero-shot run is
   `acdc_sax` → **M&Ms / M&Ms-2** (cross-dataset), and that is RQ3. Label both correctly. See PROCESS.md §12 Q2.
3. **BAAI Cardiac Agent on ACDC/M&Ms** — *if weights are public* (§0.3). The closest system, never publicly benchmarked. This is your headline baseline.
4. **Measurements → LLM, no RAG** — ungrounded generation. The H1 ablation.
5. **Template report from Agent 2's JSON, no LLM at all** — the zero-AI floor.

Baseline 5 will score *embarrassingly well* on factual consistency, because a template cannot hallucinate.
Include it anyway. Reporting the baseline that flatters you least is what makes the rest credible, and the
gap between it and your system is exactly the value the LLM adds.

Cite MARL-Rad and CardAIc as literature context, state plainly that they are not re-run, and say why.

### 2.10 The citation audit is much larger than C1

Panel-identified and confirmed: Salih (2025 / 2023), Baba (2025 / 2026), Sahoo (2025 / 2024), Zeng (2025 / 2024).

**Additional mismatches the panel did not list:**

| In-text | Reference list | Where |
|---|---|---|
| Bennai et al., 2025 | Bennai (2023) | §2.5 |
| Xia et al., 2025 | Xia (2026) | §1.3, §2.4.1 |
| Choi et al. (2025) | Choi & Yoo (2026) | §2.4.1 |
| Selvaraju et al., 2017 | Selvaraju (2020) | §3.6 |
| Xu et al., 2025 **and** 2026 — same survey, both years, sometimes one paragraph apart | Xu (2026) | §1.1, §2.3.1, §1.3, §1.7 |
| "AgentsEval (Fu et al., 2025)" | Fu, S. (2026) | §2.4.2 |

**Phantom citations on the slides** — cited in text, in no reference list anywhere: `Atrey et al., 2025`,
`Barros et al., 2025` (slide 8), and `[Frontiers in Radiology, 2025]` (slide 6), which cites a *journal* as
an author; it should be Ganz et al. (2025).

**A wrong attribution that propagated into your knowledge base:** slide 8 credits the ACR RAG system to
*"Patel et al. (2025)"*. The report and reference list say **Pambudi & Menolascina (2025)**. One of those
authors is invented.

**Citations that don't support the claim:**
- §3.4 cites **Leiner (2019)** for the 1.05 g/mL myocardial density constant. That paper doesn't establish it. The slides cite Kawel-Boehm (2025) for the same fact. Use Kawel-Boehm in both.
- §1.1 cites **Kilner (2010)** — a paper on CMR in *adult congenital heart disease* — for the general gold-standard claim.

**Broken reference entries:** Wind et al. — journal field reads *"In Lisa Adams (Vol. 1, Issue 2)"*.
Kristijan CINCAR & Todor IVAŞCU — given names as surnames, ALL CAPS, truncated URL. Zhang
(MARL-MambaContour) — truncated EBSCO URL. Bennai — truncated DOI. Selvaraju — mixes ICCV proceedings with
IJCV volume/pages (it is IJCV 128(2), 2020). Ma et al. — DOI says 2020, volume says 2021, text says 2023.

**Report and slides cite different evidence for the same claims.** The report's reference list omits Bello,
Kawel-Boehm, Wenzel, Ganz, and Leiner (2026), all of which the slides rely on. For the 10% inter-observer
variability, the report cites Leiner (2019); the slides cite Bello (2025). Pick one evidence base.

**Also needing disambiguation:** Leiner appears as two different papers (2019, 2026), exactly like Fu and
Zhang in C7.

**Cosmetic but pervasive:** every "M&Ms" in the report renders as **"M&Ms;"** — an `&amp;` escape artifact.
Both abstracts, Table 1.2, §1.6, §3.7, references.

### 2.11 Counting errors

- Both abstracts claim the work solves *"four crucial problems"*. §1.2 states **three** problems; Table 1.1 states **four** gaps. Problems ≠ gaps.
- Both abstracts and Table 1.2 commit to **two** datasets (525 subjects). §3.7 says *"three sets of data"*. That is C2; real counts are **855** total, **495** for the ACDC→M&Ms headline.
- §3.1 describes **four** research stages; Chapter 6's Gantt has **seven** phases.
- **Three mutually inconsistent schedules.** Report: `14/06/26 – 01/02/28`. Slide 25: `07/03/2026 – 31/12/2027`. Knowledge base: four phases, different months. Neither range is 24 months — the report's is ≈19.5, the slides' ≈22.
- §3.5.2 names **GPT-4**; slides say **GPT-4o**. GPT-4 is retired by 2026. Name a current model, record the exact API version and access date.

### 2.12 RV: in scope or out (C3), resolved

Table 1.2's in-scope cell says *"**Left ventricular** segmentation"*; out-of-scope says *"Right ventricular
**strain**."* But §3.3.1 produces RV masks, Chapter 4 expects RV structures, slide 11 lists LV/RV/Myo, and
all three datasets label the RV.

**Fix:** change the in-scope cell to *"LV, RV and myocardial segmentation; computation of LVEF, EDV, ESV,
LV mass."* Leave RV strain out. One cell edit resolves C3, and you gain RV Dice — which is where
cross-vendor generalisation breaks first.

---

## 3. Build order

The governing principle: **get one patient end-to-end to a report before optimising any single agent.**
A thin vertical slice through all four agents — ground-truth masks, stub LLM — is worth more than a
perfectly tuned segmentation model with nothing downstream. It de-risks every interface, and a GPU outage
never blocks you.

### 3.1 Compute reality

The proposal specifies nnU-Net, 1000 epochs, 5-fold CV, **NVIDIA A100**. You have a 16 GB M4 Mac Mini with
no CUDA. nnU-Net's default schedule is 250,000 iterations *per fold*; on MPS that is roughly an order of
magnitude slower, and `3d_fullres` will be memory-tight at 16 GB shared between CPU and GPU.

Develop locally with a short-schedule trainer; train for real on rented GPU (Kaggle's free T4×2, Colab
Pro+, vast.ai, RunPod) or UM HPC. Budget ~40–80 GPU-hours for the full sweep. **Report the GPU you actually
used.** Do not write "A100" if you trained on a T4.

### 3.2 The sequencing trick that unblocks everything

CineMA's fine-tuned checkpoints are **public** (`huggingface.co/mathpluscode/CineMA`), so you can produce
usable masks on the M4 with *no training at all* in about three days. Those masks unblock Agents 2, 3 and 4
immediately. Build, debug and evaluate the entire downstream pipeline before renting a single GPU hour.
Swap nnU-Net's masks in when they exist.

⚠️ **Corrected 2026-07-11.** This section previously said "run CineMA zero-shot." **The released ACDC
checkpoint was fine-tuned on ACDC** — using it on ACDC is a *supervised* run, and calling it zero-shot in
the thesis is a claim a reviewer can disprove from the HuggingFace page in four minutes. Use it as:
`acdc_sax` → ACDC = supervised baseline; `acdc_sax` → M&Ms/M&Ms-2 = **true cross-dataset zero-shot (RQ3)**;
`mnms_sax` → M&Ms = in-domain ceiling. Bonus: the **3 released seeds are a free ensemble**, so Agent 4's
entropy uncertainty maps no longer depend on nnU-Net's 5-fold ensemble existing. PROCESS.md §12 Q2.

This is the modularity your architecture claims, actually paying off — and it means an unlucky GPU budget
never threatens the parts of the thesis that are genuinely novel.

### Stage 0 — Environment + data spine (week 1) · no GPU
- Python 3.11 (`brew install python@3.11` or `uv`), venv. Not system 3.9.
- `torch` (MPS), `nibabel`, `SimpleITK`, `numpy`, `pandas`.
- `cmr/data.py`: one adapter per dataset → one canonical record:
  `{subject_id, dataset, vendor, centre, pathology, ed_idx, es_idx, img_ed, img_es, gt_ed, gt_es, spacing_mm}`
- **Canonical labels: `1=LV, 2=Myo, 3=RV`** (M&Ms order — two of three datasets use it, and LV is the
  clinical primary). Remap ACDC on load. Write it once, never think about it again.
- **The one runnable check:** for every subject in all three datasets, assert after canonicalisation that
  the myocardium centroid is nearer the label-1 centroid than the label-3 centroid. Exactly the test that
  caught the mismatch. Keep it as the regression guard.
- Second check: assert `ESV < EDV` on ground truth for every subject. Log violations — real data-quality
  findings, worth a paragraph in the thesis.

### Stage 1 — Quantification on ground truth (week 1–2) · no GPU, no AI
Compute EDV, ESV, LVEF and LV mass from **ground-truth masks** for all 855 subjects.

Do this before touching a neural network. It yields the reference values every later MAE is measured
against, the GT EF distribution per pathology, and an immediate sanity check (DCM must show low EF). One
day of work on the M4, and it is the denominator of your entire evaluation. It also answers C14: **zero
trained components.**

### Stage 2 — Agent 1, segmentation (weeks 2–8)
- **Week 2, no GPU:** CineMA `acdc_sax` → ACDC → masks on disk → unblocks Stages 3–6. This is a *supervised*
  run, not zero-shot (§3.2). **Assert the predicted label order** before trusting a metric — the ACDC
  checkpoint emits `1=RV`, the M&Ms checkpoint emits `1=LV`.
- Convert ACDC to nnU-Net raw format, emitting the canonical label map in `dataset.json`.
- Train `2d` and `3d_fullres`, 5-fold, on rented GPU. Start with a 250-epoch trainer for signal.
- Evaluate on the ACDC test set (Dice, HD95).
- Run **the same trained model, zero-shot, on M&Ms and M&Ms-2** → that *is* the RQ3 generalisation
  experiment. Note explicitly that M&Ms ships 150 labeled training subjects you deliberately do not use;
  that strengthens the domain-shift claim.
- **If BAAI's weights are public, run BAAI here too** (§0.3, §2.9 baseline 3).
- **Write every predicted mask to disk.** Everything downstream reads masks, never the network.

### Stage 3 — Agent 2, quantification (weeks 8–9) · no GPU
Same code as Stage 1, on predicted masks. MAE, **Bland-Altman and ICC** against Stage-1 GT values — BAAI
reports Bland-Altman and Pearson r, so this is now the expected standard, not a differentiator. Then the
**plausibility gate**: LVEF outside [5, 90]%; ESV ≥ EDV; >1 connected component per structure per slice;
base/apex discontinuity. Gate failure triggers §2.6.

### Stage 4 — Agent 3, guideline RAG (weeks 6–12) · parallel with Stage 2, no GPU needed

**This agent must be redesigned in light of CardiacRAG.** As specified — BioBERT mean-pooled embeddings +
plain FAISS dense retrieval — your Agent 3 is *strictly weaker* than the retriever in a paper you cite.
CardAIc explicitly argues that dense passage retrieval *"often lacks semantic relevance"*, and its ablation
shows vector-only and keyword-only both underperform hybrid retrieval. Ship plain BioBERT dense retrieval
and a reviewer who knows CardAIc will ask why you ignored a documented improvement in your own citation.

- **Hybrid retrieval, not dense-only.** Dense vector search **plus** keyword/BM25 filtering over a clinical
  vocabulary. This is now table stakes.
- **Embed with MedCPT**, not BioBERT. BioBERT is a masked language model; MedCPT is *trained* for
  biomedical retrieval and is also 768-dim, so nothing else changes. **Keep BioBERT as the ablation you
  promised the panel** — it costs nothing and turns a weakness into a table row.
- FAISS `IndexFlatIP`. A few thousand chunks; exact search is instant, IVF/HNSW is pointless here.
- Chunk at 200–400 tokens, 50-token overlap, carrying `{source, section, page, class_of_recommendation,
  level_of_evidence}` on every chunk. **The CoR/LoE metadata is a genuine differentiator** — neither
  competitor carries it.
- Prompt: Agent-2 measurements JSON + top-5 passages → structured JSON (`diagnosis`, `differential`,
  `citations[]`, `recommended_actions`).
- **The LLM never sees the image and never invents a number.** Measurements arrive as structured fields;
  the factual-consistency checker enforces it.
- **Conflict resolution (C11):** flag contradicting thresholds among retrieved passages. Resolve by
  recency, then class of recommendation. If unresolved, **present both with attribution** — never silently
  pick one. Most ESC/ACC heart-failure thresholds agree; the interesting conflicts are in chest-pain
  pathways. The list you find is itself a contribution.

### Stage 5 — Agent 4, explainability (weeks 10–14)
This is your **strongest surviving differentiator** — BAAI has no visual XAI at all.

- **Seg-Grad-CAM** (Vinogradova et al., 2020), not vanilla Grad-CAM. Grad-CAM is defined for classifiers;
  segmentation needs the gradient of a region-summed class score. §3.6 currently cites only Selvaraju.
- **Softmax-entropy uncertainty maps from the 5-fold ensemble.** Free with nnU-Net, arguably more clinically
  useful than Grad-CAM, and a second independent visual channel.
- Citation-annotated report renderer: every clinical sentence carries source + section + page.
- **The link is the contribution.** Not "here is a heatmap and here is a citation" (CardAIc has panels;
  BAAI has citations) but "this heatmap region supports this cited guideline statement." Build the data
  structure that expresses that link, and evaluate it.

### Stage 6 — LangGraph orchestration (weeks 12–16) → becomes §3.9
- Typed `StateGraph`; Pydantic schema validation at every node boundary. **That is the C10 answer.**
- Error containment: gate failure → retry with TTA → fall back to the 2D model → if still failing, emit
  `status: FAILED_SEGMENTATION` and **Agent 3 refuses to diagnose.** No silent downstream reasoning on a
  broken mask. That refusal is the failure-isolation claim made concrete.
- Bounded feedback loop (§2.6): max 2 iterations, stop on gate pass or budget exhaustion.
- Persist a JSONL run trace per subject. That trace is the evidence for H3.

### Stage 7 — Evaluation (weeks 14–20) → becomes §3.10
Dice, HD95, MAE + Bland-Altman + ICC, HF-category accuracy, pathology top-1, **report factual consistency**,
baseline-rescaled BERTScore, expert Likert. Stratify by vendor and centre. Report the ACDC→M&Ms Dice drop.
Paired bootstrap or Wilcoxon with confidence intervals.

Ablations (Table 3.3): no-RAG · hybrid vs dense-only retrieval · MedCPT vs BioBERT · GPT-4o vs open model ·
no feedback loop · no gate · nnU-Net vs CineMA · provided ED/ES indices vs automatic detection.

### Stage 8 — Reader study (weeks 18–22) — see §6
### Stage 9 — Write-up and the corrections in §2

---

## 4. The missing metric: diagnostic accuracy

RQ2 asks about *"diagnostic accuracy."* Dice, MAE and BERTScore measure segmentation, measurement and text
similarity. **None measures whether the diagnosis is right.** The panel found only its shadow (C13).

### 4.1 Pathology top-1 accuracy
ACDC labels 5 pathologies; M&Ms has a `Pathology` column; M&Ms-2 has 8 disease classes. Score Agent 3's
top-1 differential against the dataset label. Objective, no clinician required, computable today. It also
gives a comparison axis to BAAI, which reports AUC 0.93 internal / 0.81 external.

### 4.2 Heart-failure category accuracy
HFrEF / HFmrEF / HFpEF from predicted EF versus GT EF. This isolates how measurement error propagates into
a *clinical decision*. Restricted to GT LVEF ∈ [35, 55]%, it is the test for H2.

### 4.3 Guideline-fidelity / factual-consistency score
Fraction of numeric claims in the report matching Agent 2's JSON within tolerance, plus fraction of
citations that resolve to a real chunk in the index. Deterministic, cheap, measures hallucination directly.
**BAAI claims "zero hallucination" only qualitatively; neither paper quantifies it.** Make this a headline
contribution — it is the metric your whole architecture exists to optimise.

---

## 5. C16 — three testable hypotheses

BAAI already built an end-to-end agentic CMR pipeline and said so in print. "I built a pipeline" is no
longer available to you. The hypothesis framing is not a nicety now; it is the survival path.

- **H1 — Grounding helps.** Guideline-grounded RAG produces more clinically correct diagnostic statements
  than an ungrounded LLM given identical measurements.
  *Measured by:* pathology top-1, factual consistency, expert Likert. *Ablation:* baseline 4.

- **H2 — Boundary-aware recomputation helps where it matters.** Triggering re-measurement when a value lands
  near a guideline threshold reduces HF-category misclassification for near-boundary patients.
  *Measured by:* HF-category accuracy restricted to GT LVEF ∈ [35, 55]%. *Ablation:* no feedback loop.
  **Still your most novel claim — but cite CardAIc honestly.** Its adaptive workflow (stepwise plan
  refinement on intermediate evidence) improved accuracy 0.80 → 0.87. That is prior art for *adaptive
  re-planning*. Your difference: CardAIc refines the **plan** when evidence changes; you refine the
  **measurement** when it lands near a clinical decision boundary. Different trigger, different action,
  different failure mode. Cite their ablation as evidence the mechanism class works, then claim the
  boundary-triggered variant.

- **H3 — Decomposition makes failure visible.** Per-agent gating catches bad segmentations before a report
  is emitted, at a rate a monolithic pipeline cannot match.
  *Measured by:* on deliberately corrupted and hard cases, fraction of failures caught before emission.
  *Ablation:* no gate.

Each maps to an ablation you were already going to run. Nothing extra to build — you only have to say what
you are testing.

---

## 6. C9 — "Master by Research requires own data collection"

§3.11 currently states outright that *"ethics clearance from the UMMC Ethics Committee would not be
necessary."* ACDC, M&Ms and M&Ms-2 are all secondary data.

**Route A (recommended — kills C13 at the same time): a structured expert reader study.** 2–3 cardiologists
or radiologists each rate N ≈ 50 generated reports on a 5-point Likert scale across measurement
plausibility, guideline fidelity, citation correctness, explanation usefulness, and clinical actionability.
Report inter-rater agreement (Krippendorff's α). Primary data you collected, answers C13, needs only a
low-risk ethics application.

Note honestly: **BAAI ran a six-radiologist study across three experience levels.** So a reader study is not
methodologically novel. What is novel is *what you ask them to rate* — citation correctness and guideline
fidelity, which BAAI's rubric could not assess because its reports carry no citations. Design the rubric
around that, and it becomes a contribution rather than a formality.

**Route B (stretch): a small retrospective UMMC cohort.** A prospective cohort is unrealistic inside the
timeline; a retrospective de-identified set of 20–30 studies through UMMC's MREC is not. Lead time is 3–6
months, so **start the ethics application now if you want it** — the only item here with an unavoidable
external delay.

Do Route A regardless. Either way, §3.11 must be rewritten, not amended.

---

## 7. Panel corrections — triage

| # | Type | Action |
|---|---|---|
| C1 | Writing | §2.10 — ~12 mismatches, not 4, plus 3 phantom citations and 6 broken entries. |
| C4 | Writing | §2.9 — five *runnable* baselines; drop CardAIc (not a CMR system). |
| C5 | Structural | §2.2 — **write the two missing sections**, then renumber. |
| C6 | Writing | Alphabetise references. |
| C7 | Writing | Disambiguate Fu 2025a/b, Zhang 2025a/b, **and Leiner 2019/2026**. Fix the Kristijan entry. |
| C8 | Writing | BERTScore is embedding-based and semantic. Fix §2.4.2 **and Table 3.4**. |
| C2 | Design | §2.11 — three datasets, distinct roles, correct counts (855 / 495). |
| C3 | Design | §2.12 — one cell edit in Table 1.2. |
| C9 | Design | §6 — reader study, with a citation-fidelity rubric. |
| C10 | Design | Stage 6 — Pydantic node contracts + gate + typed failure state + refusal. |
| C11 | Design | Stage 4 — recency → class of recommendation → present both with attribution. |
| C12 | Design | §2.6 — decision-boundary-aware recomputation, bounded at 2 iterations. |
| C13 | Design | §4 — pathology top-1 + HF-category accuracy + Likert. |
| C14 | Design | Stages 1, 3, 4 (retrieval) and 6 involve **no training**. Only nnU-Net is trained; MedCPT/BioBERT/GPT-4o are inference-only. State it in a table. |
| C15 | Design | §2.5 factuality checker + schema-constrained output + Guardrails. |
| C16 | Framing | §5 — three testable hypotheses. And §0, without which C16 cannot be answered. |

**C15, honestly stated:** the only text reaching the LLM is curated guideline chunks (trusted) and numeric
JSON from your own code (trusted). There is no user-supplied free text, so the prompt-injection surface is
genuinely small — say that, rather than pretending to a threat you don't have. The real risks are
hallucinated measurements and fabricated citations, both caught by the factual-consistency checker and by
validating every citation against the chunk index.

**One thing the panel did not raise, and should have:** contribution #2 claims a *"curated ESC/ACC/AHA/HFSA
corpus … [that] does not currently exist as a public resource."* It does not exist because **these
guidelines are copyrighted.** You cannot redistribute chunked full text. Release the *pipeline* — acquisition
script, chunk boundaries (offsets + SHA-256 per chunk), embeddings, index — so anyone with legal copies
reproduces your corpus bit-for-bit. Still a real, releasable contribution. Say so before a reviewer says it
for you.

---

## 8. Immediate next actions

**This week, before any code:**
1. Check whether BAAI released segmentation/diagnostic **weights** at github.com/plantain-herb/Cardiac-Agent. This determines whether baseline 3 is a two-week experiment or a two-month one.
2. Rewrite §2.7, Table 1.1, Table 2.1, Table 2.2 and the Experimental Gap around §0. Build the head-to-head table.
3. Move CardAIc-Agents out of the "closest CMR systems" discussion. It is an ECG/echo system.
4. Retire the three dead claims in §0.4. Write the replacement Key Claim.
5. Start C1/C4/C5/C6/C7/C8 — no code needed. C5 means *writing* §3.9 and §3.10.
6. Start the ethics paperwork if Route B is wanted.

**Then, no GPU and no cloud account needed:**
7. Python 3.11 + venv, six packages.
8. `cmr/data.py` with the canonical label remap and the two assertions.
9. Run the assertions across all 855 subjects. Fix whatever they surface.
10. Compute GT EDV/ESV/LVEF/mass for all 855; plot EF by pathology; confirm DCM sits low.
11. CineMA zero-shot on ACDC → predicted masks on disk → the whole downstream pipeline is unblocked.

**Only then:** nnU-Net conversion and the first rented GPU run.
