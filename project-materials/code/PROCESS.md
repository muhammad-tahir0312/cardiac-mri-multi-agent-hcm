# PROCESS.md — Verified Knowledge Base

**Project:** Multi-Agent System for Automated Cardiac MRI Analysis and Decision Support
**Candidate:** Muhammad Tahir · MCS by Research, Universiti Malaya
**Compiled:** 2026-07-10
**Companion:** [PLAN.md](PLAN.md) is what to *do*. This file is what is *known*, and how we know it.

---

## 0. Purpose and how to read this

Every factual claim below carries a provenance tag. Nothing here is remembered or assumed; it was read off
the disk, extracted from a PDF, or fetched from a source cited inline. If you are about to write a sentence
in the thesis that depends on a fact, find it here first.

| Tag | Meaning |
|---|---|
| `[MEASURED]` | Read directly off the data on this machine. Reproducible with the scripts in §11. |
| `[PAPER]` | Quoted or extracted from the full text of a PDF in `code/`. Page/line locatable. |
| `[WEB]` | Fetched from a cited external source on 2026-07-10. |
| `[DERIVED]` | Arithmetic or logic applied to `[MEASURED]` / `[PAPER]` facts. |
| `[UNVERIFIED]` | Believed true, not checked. **Do not cite without checking.** |

---

## 1. Thesis metadata, and where the documents disagree with each other

`[PAPER]` from `Report - Muhammad Tahir.pdf`, `Slide - Muhammad Tahir.pptx.pdf`, and
`Muhammad_Tahir_Thesis_Knowledge.md`.

| Field | Value |
|---|---|
| Title | Multi-Agent System for Automated Cardiac MRI Analysis and Decision Support |
| Matric | 25087075 (report cover) / 25087075/1 (knowledge base) |
| Supervisor | Dr. Uzair Iqbal |
| Co-supervisor | Dr. Nor Liyana Shuib (report) / **Prof.** Dr. Nor Liyana **Mohd** Shuib (knowledge base) |
| Report submitted | 14 June 2026 |
| Proposal defence | 26 June 2026, online (Google Meet) |
| Result | PASSED, 77.0 average (Panel 1: 79.5, Panel 2: 74.5; UM pass mark 65.0) |
| Candidature defence due | Semester I, 2026/2027 |
| Max study period | End of Semester I, 2029/2030 |

**Panel:** Chair Ts. Dr. Siti Nurliana Jamalai @ Jamali; Panel 1 Dr. Burhan Ul Islam Khan (issued C1–C8);
Panel 2 Dr. Mohamad Hazim Md Hanif (issued C9–C16).

**Score breakdown** (Panel 1 / Panel 2 / avg): Title & Abstract 5% → 4.5/3.5/4.0 · Introduction 25% →
20.0/17.5/18.8 · Literature Review 25% → 20.0/20.0/20.0 · Conceptual Framework & Methods 20% →
16.0/14.0/15.0 · Summary 5% → 4.0/3.5/3.8 · Academic Style & References 10% → 7.0/8.0/7.5 ·
Communication 10% → 8.0/8.0/8.0.

Note that **Methods scored lowest as a proportion (75%)**, and Literature Review scored full marks — which
is the opposite of what §7 of this document shows to be true.

### 1.1 Internal inconsistencies in the metadata itself

`[DERIVED]`

- Report cover says *"Programme: Master Of Computer Science"*; it is Master of Computer Science **by
  Research**. The distinction is what triggered panel comment C9.
- Report cover says *"No. Of Semesters Registered: 1st Semester"*; knowledge base says *"Sem 2,
  2025/2026"*.
- Co-supervisor's name and title differ between report and knowledge base.

### 1.2 The preliminary paper does not support the claim made for it

`[PAPER]` Report Chapter 5 + `[UNVERIFIED]` journal name.

- Title: *Transparent Cardiac Intelligence: A Post-Hoc Explainability Framework for Ensemble-Based
  Arrhythmia Classification*. Submitted 23 May 2026, under review.
- Journal stated as **"BMJ Medical Informatics and Decision Making"**. `[UNVERIFIED]` — there is no such
  journal. There *is* **BMC** Medical Informatics and Decision Making. **Check the submission portal
  before this appears in a thesis.**
- Report §5.2 claims the paper *"lays out the theoretical groundwork for the Explainability Agent
  introduced in Chapter 3."* `[DERIVED]` It does not. That paper applies **SHAP and LIME** to a **tabular
  ECG ensemble classifier**. Agent 4 applies **Grad-CAM** to a **segmentation CNN on images**. Different
  XAI family, different modality, different model class. Worse, Table 1.2 lists SHAP/LIME as explicitly
  **out of scope**. The bridge claimed between the two does not exist. Either build a real one, or state
  the paper's contribution as "demonstrates XAI competence" rather than "underpins Agent 4."

---

## 2. Machine and environment

`[MEASURED]` 2026-07-10.

| Property | Value |
|---|---|
| CPU | Apple M4, arm64 |
| RAM | 16 GB unified |
| OS | macOS, Darwin 25.3.0 |
| Disk | 460 GiB total, 246 GiB used, **173 GiB free** |
| CUDA | **None.** `nvidia-smi` absent (expected on macOS) |
| System Python | `/usr/bin/python3` → 3.9.6 |
| Installed packages | **None of**: torch, numpy, nibabel, SimpleITK, monai, nnunetv2, transformers, faiss, langgraph, scikit-learn, scipy, matplotlib, pandas |
| Package managers | Homebrew at `/opt/homebrew/bin/brew`. No conda, uv, or pipx |
| Installed this session | `poppler` 26.07.0 → `pdftotext`, `pdftoppm` |

**Consequence.** The proposal specifies nnU-Net, 1000 epochs, 5-fold CV on an **NVIDIA A100**. nnU-Net's
default schedule is 250,000 iterations *per fold* (1000 epochs × 250 iterations). On MPS this is roughly an
order of magnitude slower than an A100, and `3d_fullres` is memory-tight at 16 GB shared between CPU and
GPU. Real training must be rented (Kaggle T4×2 free tier, Colab Pro+, vast.ai, RunPod) or run on UM HPC.
Estimated budget ~40–80 GPU-hours for a full 5-fold × 2-config sweep. **Report the GPU actually used.**

---

## 3. Datasets on disk

All three are present, intact, and total **46 GB**. `[MEASURED]`

### 3.1 ACDC — 2.3 GB

| Property | Value |
|---|---|
| Training subjects | **100** (`patient001`–`patient100`) |
| Testing subjects | **50** (`patient101`–`patient150`) |
| Ground truth in test set | **Yes** — this is the full public release |
| Files per subject | `Info.cfg`, `patientXXX_4d.nii.gz`, `patientXXX_frameNN.nii.gz` ×2 (ED, ES), `patientXXX_frameNN_gt.nii.gz` ×2 |
| `Info.cfg` fields | `ED`, `ES`, `Group`, `Height`, `NbFrame`, `Weight` |
| Pathology groups | DCM, HCM, MINF, NOR, RV — **20 each** in training |
| Labels | **`1 = RV cavity, 2 = Myocardium, 3 = LV cavity`** |

Geometry is heterogeneous — do not hard-code:

| Subject | Shape (frame) | dtype | Spacing (mm) |
|---|---|---|---|
| patient001 | (216, 256, 10) | int16 | 1.562 × 1.562 × 10.0 |
| patient002 | (232, 256, 10) | int16 | 1.367 × 1.367 × 10.0 |
| patient020 | (208, 256, 8) | float32 | 1.758 × 1.758 × 10.0 |
| patient050 | (248, 256, 10) | int16 | 1.367 × 1.367 × 10.0 |
| patient099 | (154, 224, 16) | float32 | 1.786 × 1.786 × **5.0** |

In-plane spacing ranges **1.367–1.786 mm**; slice thickness **5–10 mm**. The 4-D volume for patient001 is
(216, 256, 10, 30). Image dtype varies between `int16` and `float32` across subjects; GT is `uint8`.

**ED/ES frame indices come from `Info.cfg`. No detection needed.**

### 3.2 M&Ms — 15 GB

| Property | Value |
|---|---|
| Training / Labeled | **150** |
| Training / Unlabeled | **25** |
| Validation | **34** |
| Testing | **136** |
| **Total** | **345** |
| Metadata | `211230_M&Ms_Dataset_information_diagnosis_opendataset.csv`, 345 rows + header |
| CSV columns | `(index), External code, VendorName, Vendor, Centre, ED, ES, Age, Pathology, Sex, Height, Weight` |
| Files per subject | `SID_sa.nii.gz`, `SID_sa_gt.nii.gz` |
| Labels | **`1 = LV cavity, 2 = Myocardium, 3 = RV cavity`** |

**Vendors:** Philips 125 · Siemens 95 · GE 75 · Canon 50.
**Centres:** 1→95 · 2→74 · 3→51 · 4→75 · 5→50. **Five centres, not six.**
**Pathology:** DCM 97 · NOR 89 · HCM 85 · Other 25 · HHD 24 · ARV 15 · IHD 5 · AHS 3 · LVNC 2.

**The proposal's Table 3.1 says 375 subjects across 6 clinical centres. The open release is 345 across 5.**
The 375/6 figure describes the full challenge cohort, part of which was never released.

**Critical layout gotcha.** The ground truth is a **4-D volume with the same shape as the image**, non-zero
only at the ED and ES frames. Example: `E9H1U4_sa_gt.nii.gz` has shape (196, 240, 11, 25), spacing
1.333 × 1.333 × 9.6, `pixdim[4] = 0.0`; the only non-zero frames are **0 and 10**, matching the CSV's
`ED=0, ES=10` for that subject. **Frame indices come from the CSV, not from the file.**

### 3.3 M&Ms-2 — 29 GB

| Property | Value |
|---|---|
| Subjects | **360** (`001`–`360`) |
| Files per subject | `SID_{LA,SA}_CINE.nii`, `SID_{LA,SA}_{ED,ES}.nii`, `SID_{LA,SA}_{ED,ES}_gt.nii` |
| Compression | **Uncompressed `.nii`** (unlike ACDC/M&Ms) |
| Views | Short-axis **and long-axis** — the only dataset here with LA |
| ED/ES | **Pre-extracted as separate files.** No detection, no CSV lookup |
| Labels | **`1 = LV cavity, 2 = Myocardium, 3 = RV cavity`** |

**Disease:** NOR 75 · LV 60 · HCM 60 · FALL 35 · CIA 35 · ARR 35 · TRI 30 · RV 30 = 360.
**Vendor:** SIEMENS 219 · Philips Medical Systems 88 · GE MEDICAL SYSTEMS 53.

Geometry: `001_SA_ED.nii` is (256, 256, 12) `float64`, spacing 1.484 × 1.484 × 10.0. `001_LA_ED.nii` is
(256, 256, 1), spacing 1.719 × 1.719 × 1.0.

**Data-quality note.** `dataset_information.csv` is an Excel export padded to **1,048,575 rows** with
trailing commas. Only 360 rows carry data. Any naive `wc -l` or `len(df)` will be wrong.

### 3.4 Corrected totals

`[DERIVED]`

| Claim in proposal | Reality |
|---|---|
| M&Ms: 375 subjects, 6 centres | **345 subjects, 5 centres** |
| "525 subjects" (ACDC + M&Ms) | **495** |
| Three datasets total | **855** (150 + 345 + 360) |

Abstract (EN and BM), Table 1.2 and §1.6 commit to **two** datasets; §3.7 and Table 3.1 use **three**.
That is panel comment C2.

---

## 4. Frame-index sources — three datasets, three mechanisms

`[MEASURED]` This is why a single loader adapter is required.

| Dataset | ED/ES source |
|---|---|
| ACDC | `Info.cfg` → `ED:`, `ES:` |
| M&Ms | metadata CSV → `ED`, `ES` columns |
| M&Ms-2 | pre-extracted `_ED.nii` / `_ES.nii` files |

Report §3.3.3(iv) proposes *"selection of ED and ES frames based on time gradient analysis."* This is
**unnecessary work that injects unattributable error**. All three datasets hand you the indices. If you
want the detector, make it an ablation and quantify its cost.

---

## 5. The label-convention finding

**The single most dangerous fact in this project.**

### 5.1 The finding

`[MEASURED]`

- **ACDC:** `1 = RV`, `2 = Myocardium`, `3 = LV`
- **M&Ms:** `1 = LV`, `2 = Myocardium`, `3 = RV`
- **M&Ms-2:** `1 = LV`, `2 = Myocardium`, `3 = RV`

ACDC's LV/RV order is **inverted** relative to the other two. All three use `{0,1,2,3}` and are visually
indistinguishable without checking. Train on ACDC, evaluate on M&Ms without remapping, and every LV/RV
Dice, every ejection fraction, and every heart-failure classification is **wrong but plausible-looking**.
There is no stack trace for this class of bug.

### 5.2 How it was established

Voxel counts alone are ambiguous — for a normal heart, LV and RV cavity volumes are comparable at ED. The
decisive test exploits anatomy: **the myocardium is a ring around the LV cavity**, so the myocardium
centroid must be nearly coincident with the LV centroid, and far from the RV centroid.

For each labelled volume, compute the in-plane centroid of labels 1, 2, 3. Whichever of label 1 or label 3
is nearer the label-2 (myocardium) centroid **is the LV**.

### 5.3 Results — 9 subjects, unanimous

| Subject | Dataset | Note | n(1) | n(2) | n(3) | d(myo,1) | d(myo,3) | LV is |
|---|---|---|---|---|---|---|---|---|
| patient001 | ACDC | DCM, ED | 5723 | 6728 | 12104 | 33.6 | **0.7** | label 3 |
| patient070 | ACDC | ED | 3973 | 2683 | 3667 | 25.9 | **1.1** | label 3 |
| E9H1U4 | M&Ms | NOR, ED=0 | 6395 | 4487 | 6015 | **0.3** | 25.3 | label 1 |
| M4P7Q6 | M&Ms | HCM, ED=2 | 12790 | 12436 | 7445 | **4.2** | 37.8 | label 1 |
| E9V4Z8 | M&Ms | ED=0 | 9224 | 6063 | 11373 | **0.3** | 30.5 | label 1 |
| D0R0R9 | M&Ms | ED=0 | 6132 | 8823 | 5501 | **2.8** | 32.2 | label 1 |
| 002 | M&Ms-2 | ED | 5121 | 3440 | 4347 | **0.8** | 27.5 | label 1 |
| 150 | M&Ms-2 | ED | 10630 | 6344 | 21481 | **2.0** | 41.6 | label 1 |
| 300 | M&Ms-2 | ED | 11238 | 6680 | 16131 | **1.7** | 37.6 | label 1 |

Distances in pixels. The separation is three orders of magnitude in signal-to-noise: **≤ 4.2 px** for the
correct cavity versus **25–42 px** for the wrong one. There is no ambiguity.

Note `E9H1U4`: raw voxel counts are 6395 / 4487 / 6015 — almost balanced between labels 1 and 3. Counting
voxels would have given no answer. Geometry gives an unambiguous one.

Note `M4P7Q6`: its ED frame is **2**, not 0. A loader that assumes frame 0 crashes (the GT frame is empty,
so label 2 is absent and the centroid lookup raises `KeyError`). This is how the CSV dependency was found.

### 5.4 What to do about it

- Canonicalise **at load time** to `1 = LV, 2 = Myo, 3 = RV` (M&Ms order — two of three datasets already
  use it, and LV is the clinical primary). Remap ACDC on read.
- Emit the canonical mapping into nnU-Net's `dataset.json` so predictions are canonical everywhere.
- **Keep the concentricity test as a permanent regression assertion** over all 855 subjects. It is the
  cheapest possible guard against the most expensive possible bug, and it is the "one runnable check" the
  data spine needs.

---

## 6. Clinical and guideline facts

### 6.1 Heart-failure LVEF thresholds

`[WEB]` [2021 ESC HF Guidelines, Eur Heart J 42(36):3599](https://academic.oup.com/eurheartj/article/42/36/3599/6358045) ·
[ACC/AHA/HFSA 2022 vs ESC 2021 comparison, PMC10192289](https://pmc.ncbi.nlm.nih.gov/articles/PMC10192289/)

| Category | LVEF |
|---|---|
| HFrEF | **≤ 40 %** |
| HFmrEF | **41 – 49 %** |
| HFpEF | **≥ 50 %** |

ESC 2021 and AHA/ACC/HFSA 2022 **agree** on these cut-points.

**The proposal is wrong at both boundaries.** Report §3.4 emits
`(<40% = reduced, 40-50% = mildly reduced, >50% = preserved)`:

- LVEF exactly **40 %** → matched by `40-50` → labelled *mildly reduced*. Guidelines say **HFrEF**.
- LVEF exactly **50 %** → matched by `40-50`, excluded by `>50` → labelled *mildly reduced*. Guidelines say
  **HFpEF**.

The knowledge base's `<40 / 40–49 / ≥50` is better but still mis-assigns 40.0 %.

This is an off-by-one in a thesis whose entire premise is guideline fidelity. It is also the concrete,
citable motivation for decision-boundary-aware recomputation.

### 6.2 "ESC 2022 Guidelines on Cardiac Magnetic Resonance Imaging" does not exist

`[WEB]` [ESC Clinical Practice Guidelines index](https://www.escardio.org/guidelines/clinical-practice-guidelines/all-esc-practice-guidelines/) ·
[Petersen et al., CMR in ESC guidelines, PMC10364363](https://pmc.ncbi.nlm.nih.gov/articles/PMC10364363/)

There is no standalone 2022 ESC guideline on CMR. CMR recommendations are distributed across ~19
disease-specific ESC guidelines (40 Class-I and 28 Class-IIa recommendations across them). ESC's 2022
guideline output covers ventricular arrhythmias/SCD, pulmonary hypertension, cardio-oncology, and
non-cardiac surgery — no CMR guideline.

The phantom citation appears in **three places**: Table 1.2 (*"ESC CMR 2022"*), §1.6 prose, and §3.5.2
corpus document (1). The slides repeat it.

**Replacement corpus:**

| Document | Role |
|---|---|
| Petersen et al., *CMR in the guidelines of the ESC* (PMC10364363) | Aggregated CMR recommendations — precisely the corpus wanted |
| 2021 ESC HF guideline (+ 2023 focused update) | HF classification, LVEF thresholds |
| 2022 AHA/ACC/HFSA HF guideline | US counterpart; conflict-resolution testbed |
| 2021 AHA/ACC Chest Pain guideline | Where real ESC/ACC conflicts live |
| SCMR 2025 reference values (Kawel-Boehm) | Normal ranges, myocardial density constant |

### 6.3 Myocardial density constant

`[PAPER]` Report §3.4 cites **Leiner et al. (2019)** for `1.05 g/mL`. That paper is *"Machine learning in
CMR: basic concepts and applications"* and does not establish the constant. The slides cite **Kawel-Boehm
(2025)** for the same fact. Use Kawel-Boehm in both.

### 6.4 Copyright constraint on the corpus

`[DERIVED]` Expected outcome #2 claims a *"curated ESC/ACC/AHA/HFSA corpus … [that] does not currently
exist as a public resource."* It does not exist because **the guidelines are copyrighted**. Chunked full
text cannot be redistributed.

Releasable alternative: the acquisition script, chunk boundaries (byte offsets + SHA-256 per chunk),
embeddings, and FAISS index — so anyone holding legal copies reconstructs the corpus bit-for-bit. State
this explicitly before a reviewer states it for you.

---

## 7. Competitor systems — read in full, 2026-07-10

Both papers are in `code/`. Everything below is `[PAPER]`.

### 7.1 BAAI Cardiac Agent

**Qu, T., Zhang, H., et al. (2026).** arXiv:2604.04078v1 [eess.IV], 5 Apr 2026.
Beijing Academy of Artificial Intelligence + Beijing Anzhen Hospital (Capital Medical University) + First
Affiliated Hospital, Henan Medical University.

**Cohort.** 2,413 patients, retrospective, **two hospitals**. IRB: Beijing Anzhen 2025216x; Xinxiang
Medical University 2019164. Consent waived. Three **3.0 T** whole-body scanners: **Philips Achieva, GE
Discovery MR750w, Siemens Healthineers** — i.e. **it is multi-vendor**. Population is entirely East Asian
(they name this as a limitation).

**Architecture.** LMM engine (LLaVA-Med 60K-IM init, LoRA r=128 / α=256 / dropout 0.05, 10 epochs, bs 8,
lr 1e-4, DeepSpeed ZeRO-3; 7 B params) orchestrating **9 tools**: SAX-cine / 2CH-cine / 4CH-cine / SAX-LGE
segmentation, Cardiac Disease Screening (CDS), Non-Ischemic Cardiomyopathy Subclassification (NICMS),
**Retrieval-Augmented Generation**, Medical Report Generation, and native VQA. 32 k instruction-tuning
samples (8 K VQA + 3 K each × 8 tasks). Segmentation is a two-stage coarse-to-fine ROI→fine architecture,
**not** nnU-Net. Diagnosis model: 60 epochs, Adam, lr 5e-4, wd 5e-4.

**Segmentation annotation budget: only 150 samples**, 7:1:2 split, annotated by two radiologists with ≥10
years' experience using ITK-SNAP.

**Segmentation results (DSC %, mean ± sd):**

| Method | SAX cine | 2CH cine | 4CH cine | SAX LGE |
|---|---|---|---|---|
| ResUNet++ | 84.69 | 86.97 | 84.41 | 58.08 |
| **nnU-Net** | 87.42 | 88.14 | 81.61 | 73.89 |
| DiffUNet | 77.19 | 83.05 | 77.38 | 47.28 |
| MedSAM2 | 80.86 | 77.89 | 82.07 | 69.70 |
| **BAAI (Ours)** | **90.21** | **88.75** | **86.92** | **75.07** |

**It beats nnU-Net on its own data.** Their HD (ours): 8.24 / 7.87 / 7.52 / 14.05. ASD: 0.64 / 0.27 / 0.26
/ 1.14.

**Diagnosis.** CDS internal test n=365: NH AUC 0.980 (0.956–0.995); IHD AUC 0.938; NICM AUC 0.960.
External n=275: AUC > 0.810 across all three. NICMS internal n=155: class-weighted AUC 0.9650, F1 0.834;
HCM AUC 0.975, DCM AUC 0.971, myocarditis AUC 0.938. External n=159: AUC > 0.860 across all five.
Frequency-weighted F1: internal 0.858 → external 0.728 (CDS); 0.834 → 0.754 (NICMS).

**Quantification.** Bland-Altman plots + Pearson r against clinical reports: LVEDV 0.968, LVESV 0.979,
LVEF 0.925, SV 0.906, LVM 0.937, LVEDD 0.874. AHA 17-segment wall-thickness polar maps, error < 1 mm/segment.

**Reports.** **BERTScore P/R/F1 = 0.903 / 0.894 / 0.898.** (Their Discussion restates this as 0.901 /
0.896 / 0.898 — an internal inconsistency in their own paper, worth noting if you cite the figure.)
Tool-invocation success 99.82 % internal, 99.46 % external. CMR sequence recognition 0.980. Reports
generated in ~60–90 s versus ~1800 s for the traditional workflow.

**Reader study.** **Two radiologists at each of three experience levels (junior <5 y, mid 5–10 y, senior
>10 y) = six readers.** 100-point scale. BAAI 87.93 / 87.52 / 86.53 vs Qwen-VL-30B 58.52 / 57.18 / 51.67.
Also measures *increase in report-writing confidence* after reading model output.

**Its RAG.** *"we extract the cardiovascular section from ChatCAD+, and combine it with clinical guidelines
related to CVDs as well as heart-related literature retrieved from PubMed."* Its worked example outputs
*"Reference: 2023 ACC/AHA/SCAI Guideline for the Management of Patients With Aortic Stenosis. Journal of
the American College of Cardiology, 2023."*

**Crucially, its RAG is a conversational sidecar.** It is invoked when a *user asks a knowledge question*
(*"What are the primary treatment options for severe symptomatic aortic stenosis? Please provide the
evidence."*). Its **structured diagnostic report carries no citations at all** — Figure 1's report ends
*"Diagnostic Result: High probability of Hypertrophic Cardiomyopathy (HCM) is considered."* with no source.
Citations, when present, are **document-level**, never passage-level with section and page.

**Explainability: none.** Zero occurrences of Grad-CAM, saliency, attention map, uncertainty, or heatmap.
"Transparent" means the user can query the reasoning steps.

**It claims:** *"the first end-to-end agent framework specifically designed for CMR imaging analysis."*

**Availability.** Code: `github.com/plantain-herb/Cardiac-Agent`. Benchmark test subset:
`huggingface.co/datasets/TaipingQu/CMR-MULTI`. Data Availability states: **"No other publicly available
datasets were used."**

**Zero mentions of ACDC or M&Ms in the entire paper.**

**Stated limitations.** Rare subtypes (ACM, RCM) under-sampled. Population entirely East Asian; validation
in Europe/Americas/Africa "will be necessary." CMR-only; no CT, echo, ECG, or laboratory data.

### 7.2 CardAIc-Agents

**Zhang, Y., Bunting, K. V., et al. (2025).** arXiv:2508.13256v2 [cs.AI], 23 Dec 2025.
University of Birmingham + University Hospitals Birmingham NHS + Manchester Metropolitan + Royal
Wolverhampton NHS + Northeast Forestry University + Utrecht + Manchester.

**It contains zero mentions of MRI, CMR, or magnetic resonance.** Not one, anywhere in the paper. Its own
keywords list *"echocardiographic imaging."* It is an **ECG / echocardiography / EHR** system.

**Datasets — all public.** MIMIC-IV (1,524 patients; labs + 12-lead ECG + echo) for HF diagnosis; PTB-XL
(10,147 patients) for MI diagnosis; PTB Diagnostic ECG (268 cases) for HF prediction.

**Results** (ACC / AUC): MIMIC-IV 0.87 / 0.89 · PTB-XL 0.96 / 0.96 · PTB Diagnostic 0.77 / 0.88.
Baselines beaten: LLaVA-Med (0.35 on MIMIC-IV), MedGemma, MedGemma+CoT, MedGemma+ReAct, MedAgents,
ReConcile (0.49 / 0.43 / 0.55), MDAgents.

**CardiacRAG — read this carefully, it is nearly your Agent 3.**

- Knowledge base: Mayo Clinic (2025), UK NHS (2025), MedlinePlus (2000), *"recently published official
  guidelines"* — the reference list includes the **2021 ESC HF guideline**.
- Ingestion: BeautifulSoup for HTML, Docling for PDF.
- Chunking with size `ds` and overlap `do`.
- Embedding: **Bio_ClinicalBERT** (Alsentzer et al., 2019).
- Index: **FAISS**, cosine similarity, retrieve top `3n` then filter to `n`.
- **Hybrid retrieval**: dense vector search **plus** TF-IDF keyword filtering with a clinical vocabulary
  weight `ω_medical` and a position bonus (×1.2 if the keyword appears in the first 30 % of a chunk).
- `Cite` = *"optional return of original chunks for transparency and reference."* Chunk-level.
- Planner LLM: DeepSeek-R1-Distill-Qwen-32B.

The paper **explicitly argues against plain dense retrieval**: *"Dense Passage Retrieval encodes queries
and documents into embeddings for similarity based retrieval yet often lacks semantic relevance."* Its
intra-module ablation confirms **vector-only and keyword-only both underperform hybrid.**

**Adaptive workflow.** Complexity assessment → CardiacRAG generates a stepwise plan → Chief agent executes
tools → plan is refined stepwise as evidence emerges → multidisciplinary discussion team (MDT) auto-invoked
for hard cases → visual review panels on clinician request.

**Ablation (MIMIC-IV accuracy):** no adaptive workflow 0.80 · no CardiacRAG 0.77 · no MDT 0.84 · **full
0.87**.

**Visual review panels.** Patient profile, ECG waveform with labelled P and T waves, echocardiographic view
identification (11 standard views; 100 % on key views A3C/A4C/PLAX/PSAX/SC, >80 % elsewhere, n=10 sampled),
LV segmentation on A4C (Dice 0.922 on EchoNet-Dynamic, from prior work). Assessed by two cardiologists.

**Availability.** Code: `github.com/ytz300/CardAIc-Agents`.

### 7.3 What the report says versus what is true

| Report §2.7 assertion | Verdict |
|---|---|
| BAAI: *"evaluation is done only using the small dataset"* | **False.** 2,413 patients |
| BAAI: *"no cross-validation on multi-centre/multi-vendor generalization set"* | **False.** Two hospitals, external cohort, three vendors |
| BAAI: *"no information about knowledge retrieval and output alignment with ACC/AHA, ESC clinical guidelines"* | **False.** It has a RAG tool that cites ACC/AHA guidelines |
| BAAI: *"no explainability methods are employed"* | **True.** No visual XAI of any kind |
| CardAIc: *"lack of implementation studies"* | **False.** Code public |
| CardAIc: *"absence of guideline-based reasoning"* | **False.** CardiacRAG cites chunks; KB includes ESC 2021 HF |
| CardAIc: *"lack of explainability"* | **False.** Visual review panels |
| CardAIc: *"limited to one institution only"* | **False.** Three public datasets |
| CardAIc presented as a cardiac-MRI competitor | **Category error.** It has no MRI at all |

### 7.4 Consequences for the novelty claim

`[DERIVED]`

**Dead. Retire immediately:**
- *"No prior cardiac MRI system jointly integrates agentic orchestration, clinical reasoning, and explainability."* BAAI does orchestration + reasoning + RAG. Only explainability is missing.
- *"First end-to-end agentic CMR pipeline."* BAAI claims this in print, first.
- Any Experimental Gap phrased as *"rivals lack multi-centre or multi-vendor validation."*

**Now table stakes, not contributions:** Bland-Altman + ICC. BERTScore. A multi-reader Likert study.

**Your BERTScore target (≥ 0.85) is below BAAI's achieved 0.898** — and BAAI measured it against real
clinical reference reports, which ACDC and M&Ms do not provide. The target is both unmeasurable as stated
and lower than the state of the art.

**Your Agent 3, as specified** (BioBERT mean-pooled + plain dense FAISS), is **strictly weaker** than
CardiacRAG's retriever, in a paper you cite. Hybrid retrieval is now the floor.

**What survives, stated so it holds under cross-examination:**

1. **Guideline-passage grounding *inside the diagnostic report*.** BAAI's RAG is a conversational sidecar;
   its report has no citations. Its citations are document-level. Yours are passage-level with document +
   section + page, in the diagnostic path.
2. **Corpus provenance purity + CoR/LoE metadata.** CardAIc mixes Mayo Clinic, NHS and MedlinePlus with
   guidelines; BAAI mixes ChatCAD+, guidelines and PubMed. A guideline-only corpus carrying class of
   recommendation and level of evidence as chunk metadata does not exist.
3. **Linked dual explainability.** BAAI has no visual XAI. CardAIc has panels but no link to text. The
   *link* — this heatmap region supports this cited sentence — is unclaimed.
4. **Decision-boundary-aware recomputation.** Unclaimed. **But cite CardAIc's adaptive workflow as prior
   art for the mechanism class** (0.80 → 0.87). Your difference: CardAIc refines the *plan* when evidence
   changes; you refine the *measurement* when it lands near a clinical decision boundary.
5. **A quantitative guideline-fidelity metric.** BAAI claims "zero hallucination" only qualitatively.
   Neither quantifies it.
6. **Public-benchmark reproducibility.** No agentic CMR system has ever been evaluated on a public cardiac
   MRI dataset. BAAI's cohort is private; CardAIc's is public but not MRI.

**Defensible replacement Key Claim:** *"No prior CMR system grounds its diagnostic statements in citable
clinical-guideline passages, links those citations to visual explanations, or reports its results on public
benchmarks."*

**The highest-value experiment available.** BAAI's **code is public**. Check whether **weights** are
released. If so, **run BAAI Cardiac Agent on ACDC and M&Ms** — the first public-benchmark evaluation of the
state of the art, the one baseline that matters, and it converts "they never validated publicly" from a
complaint into an experiment you performed. If only the architecture is released, retrain it: it was
trained on 150 annotated samples, which is within reach.

---

## 8. Defect register — proposal report and slides

Locatable, so each can be fixed and ticked off. Panel comment IDs in brackets.

### 8.1 Structural

| ID | Location | Defect |
|---|---|---|
| S1 | Chapter 3 | Sections run 3.1–3.8 then jump to **3.11**. **§3.9 (Orchestration) and §3.10 (Evaluation Metrics) do not exist.** The methodology chapter never defines how the system is evaluated. Metrics appear only on slide 23; orchestration only on slide 22. [C5] |
| S2 | Chapter 3 | Tables run 3.1 → **3.4**. Tables 3.2 and 3.3 missing (should be metrics and ablations). [C5] |
| S3 | TOC | §2.5, §2.7, §2.8 indented as children of §2.4/§2.6. |
| S4 | §3.1 vs Ch. 6 | §3.1 describes **four** research stages; the Gantt has **seven** phases. |
| S5 | Ch. 6 vs slide 25 vs KB | **Three mutually inconsistent schedules.** Report `14/06/26 – 01/02/28` (≈19.5 months); slide `07/03/2026 – 31/12/2027` (≈22 months); knowledge base four phases with different months. Both are labelled "24 Months". |

### 8.2 Contradictions within the proposal

| ID | Location | Defect |
|---|---|---|
| X1 | §3.3.2 vs Ch. 6 Gantt vs slide 10 | Backbone is **nnU-Net** in §3.3.2, **CineMA** in the Gantt (*"CineMA Backbone Implementation"*) and in slide 10's Objective **O1**. The primary objective names a different model in three places. |
| X2 | §3.3.2 vs §3.3.3 | §3.3.2 chooses nnU-Net *because it self-configures*; §3.3.3 then hand-specifies `1.5 × 1.5 × 8 mm` resampling and `192 × 192` patches. nnU-Net derives both. (Its own ACDC `3d_fullres` spacing is ≈1.56 × 1.56 × 5.0 mm.) |
| X3 | Table 1.2 vs §3.3.1 / Ch. 4 / slide 11 | Table 1.2 in-scope says *"**Left ventricular** segmentation"*; §3.3.1 produces RV masks, Ch. 4 expects RV structures, slide 11 lists LV/RV/Myo. Out-of-scope excludes only RV *strain*. [C3] |
| X4 | Abstracts vs §3.7 | Abstracts + Table 1.2 commit to **two** datasets (525 subjects); §3.7 says *"three sets of data"*. [C2] |
| X5 | Abstracts vs §1.2 vs Table 1.1 | Abstracts claim *"four crucial problems"*; §1.2 lists **three** problems; Table 1.1 lists **four** gaps. Problems ≠ gaps. |
| X6 | §3.5.2 vs slides/KB | Primary LLM is **GPT-4** in §3.5.2, **GPT-4o** on slides. GPT-4 is retired by 2026. |
| X7 | §1.2 vs slide 5 | Inter-observer variability (10 %) cited to **Leiner (2019)** in the report, **Bello (2025)** on the slides. |
| X8 | §3.4 vs slide 19 | Myocardial density 1.05 g/mL cited to **Leiner (2019)** in the report, **Kawel-Boehm (2025)** on the slides. Only the latter supports it. |

### 8.3 Factual errors

| ID | Location | Defect |
|---|---|---|
| F1 | §3.4 | HF flag `(<40 / 40-50 / >50)` **misclassifies LVEF = 40 % and LVEF = 50 %**. Correct: ≤40 / 41–49 / ≥50. |
| F2 | Table 1.2, §1.6, §3.5.2 | *"ESC 2022 Guidelines on Cardiac Magnetic Resonance Imaging"* **does not exist**. |
| F3 | Table 3.1, slide 17, abstracts | M&Ms is **345 subjects / 5 centres**, not 375 / 6. Totals 495 (two datasets) or 855 (three), not 525. |
| F4 | §2.7, §1.2, §1.3, §1.7, Tables 1.1 / 2.1 / 2.2 | BAAI and CardAIc mischaracterised. See §7.3. |
| F5 | §3.3.3(iv) | ED/ES "time gradient analysis" — unnecessary; all three datasets provide the indices. |
| F6 | Ch. 5 | Journal named *"BMJ Medical Informatics and Decision Making"*. `[UNVERIFIED]` — no such journal; likely **BMC**. |
| F7 | §5.2 | Claims the SHAP/LIME arrhythmia paper *"lays out the theoretical groundwork for the Explainability Agent."* It does not — different XAI family, different modality, and SHAP/LIME are out of scope per Table 1.2. |

### 8.4 Baselines

| ID | Location | Defect |
|---|---|---|
| B1 | §3.8 vs Table 3.4 | Text says *"five systems"*; table lists **four**. [C4] |
| B2 | Table 3.4 | **CardAIc-Agents is not a CMR system** and cannot be a cardiac-MRI baseline. |
| B3 | Table 3.4 | **MARL-Rad** has no public weights and was evaluated on unavailable data. Not runnable. |
| B4 | Table 3.4 | Metric column reads *"Report Quality (BERTScore, ROUGE)"* — repeats the C8 lexical/semantic conflation. |
| B5 | Table 3.4 | Metric **"CAS"** is never defined. |
| B6 | Table 2.1 | "Gap Addressed" column mixes gap *types* (Knowledge, Methodological…) with *limitations* ("Segmentation only", "Single task"). |

### 8.5 Citations

**Panel-identified [C1], all confirmed:** Salih (text 2025 / refs 2023) · Baba (2025 / 2026) · Sahoo (2025
/ 2024) · Zeng (2025 / 2024).

**Additional year mismatches the panel did not list:**

| In text | Reference list | Location |
|---|---|---|
| Bennai et al., 2025 | Bennai (2023) | §2.5 |
| Xia et al., 2025 | Xia (2026) | §1.3, §2.4.1 |
| Choi et al. (2025) | Choi & Yoo (2026) | §2.4.1 |
| Selvaraju et al., 2017 | Selvaraju (2020) | §3.6 |
| Xu et al., **2025** *and* Xu et al., **2026** — the same survey, both years, sometimes a paragraph apart | Xu (2026) | §1.1, §2.3.1, §1.3, §1.7 |
| "AgentsEval (Fu et al., 2025)" | Fu, S. (2026) | §2.4.2 |

**Phantom citations (slides) — appear in no reference list anywhere:**
`Atrey et al., 2025` and `Barros et al., 2025` (slide 8); `[Frontiers in Radiology, 2025]` (slide 6), which
cites a **journal as an author** — it should be Ganz et al. (2025).

**Wrong attribution, propagated into the knowledge base:** slide 8 and the knowledge base credit the ACR
RAG system to **"Patel et al. (2025)"**. The report and its reference list say **Pambudi & Menolascina
(2025)**. One of those authors is invented.

**Broken reference entries:**
- Wind et al. (2025) — journal field reads *"In Lisa Adams (Vol. 1, Issue 2)"* (Zotero import error).
- Kristijan CINCAR & Todor IVAŞCU — given names as surnames, ALL CAPS, truncated EBSCO URL. [C7]
- Zhang (MARL-MambaContour) — truncated EBSCO URL.
- Bennai — DOI `10.1016/j.cmpb.2023.1074` truncated.
- Selvaraju — mixes ICCV proceedings with IJCV volume/pages. It is IJCV 128(2), 2020.
- Ma et al. — text 2023, DOI `TMI.2020.…`, volume 40(10) is 2021.
- Campello (M&Ms) — EBSCO plink instead of the IEEE TMI DOI.

**Same-surname collisions needing a/b suffixes:** Fu (Y. 2025 CineMA vs S. 2026 AgentsEval) · Zhang (Y.
2025 CardAIc vs R. 2025 MARL-MambaContour) [C7] · **Leiner (2019 vs 2026)** — not flagged by the panel.

**Divergent evidence bases.** The report's reference list **omits** Bello, Kawel-Boehm, Wenzel, Ganz, and
Leiner (2026) — all of which the slides and knowledge base rely on.

**Ordering.** Reference list is not alphabetical (begins Sahoo, Wang, Haupt, Salih, Bernard, Hevner…). [C6]

**Encoding.** Every *"M&Ms"* in the report renders as **"M&Ms;"** — an `&amp;` escape artifact. Present in
both abstracts, Table 1.2, §1.6, §3.7, and the references.

### 8.6 Methodological gaps the panel only partly saw

| ID | Defect |
|---|---|
| M1 | **No metric measures diagnostic correctness.** Dice measures segmentation, MAE measures measurement, BERTScore measures text similarity. RQ2 asks about "diagnostic accuracy" and nothing answers it. [partially C13] |
| M2 | **BERTScore has no reference reports.** ACDC and M&Ms ship masks and labels, not radiology reports. The ≥0.85 target is measured against nothing, is below BAAI's 0.898, and un-rescaled BERTScore ≈0.85 for unrelated sentences anyway. |
| M3 | **The feedback loop is mechanically impossible.** *"Guideline evidence can refine segmentation masks"* — a text-retrieval agent has no imaging signal. [C12] |
| M4 | **Agent 3's retriever is weaker than a cited competitor's.** BioBERT mean-pooling + plain dense FAISS vs CardiacRAG's hybrid dense + keyword retrieval. |
| M5 | **§3.6 cites vanilla Grad-CAM** (defined for classifiers) for a segmentation network. Needs Seg-Grad-CAM (Vinogradova et al., 2020). |
| M6 | **§3.11 disclaims all data collection** — *"ethics clearance from the UMMC Ethics Committee would not be necessary"* — which is exactly what triggered C9. |
| M7 | **The guideline corpus cannot be released** (copyright). See §6.4. |

---

## 9. The sixteen panel corrections

| # | Panel | Type | Where it is addressed |
|---|---|---|---|
| C1 | 1 | Writing | §8.5 — ~12 year mismatches, not 4, plus 3 phantom citations and 7 broken entries |
| C2 | 1 | Design | §3.4 — three datasets, correct counts |
| C3 | 1 | Design | §8.2 X3 — one cell edit in Table 1.2 |
| C4 | 1 | Writing | §8.4 — five *runnable* baselines |
| C5 | 1 | Structural | §8.1 S1–S2 — **two missing sections**, then renumber |
| C6 | 1 | Writing | §8.5 — alphabetise |
| C7 | 1 | Writing | §8.5 — Fu/Zhang **and Leiner**; fix Kristijan |
| C8 | 1 | Writing | §2.4.2 **and Table 3.4** — BERTScore is embedding-based, semantic |
| C9 | 2 | Design | PLAN §6 — reader study as primary data collection |
| C10 | 2 | Design | PLAN Stage 6 — Pydantic node contracts, gate, typed failure, refusal |
| C11 | 2 | Design | PLAN Stage 4 — recency → class of recommendation → present both |
| C12 | 2 | Design | §8.6 M3 — decision-boundary-aware recomputation |
| C13 | 2 | Design | §8.6 M1 — pathology top-1 + HF-category accuracy + Likert |
| C14 | 2 | Design | §10 below |
| C15 | 2 | Design | PLAN §7 — threat model + factuality checker |
| C16 | 2 | Framing | PLAN §5 — three testable hypotheses, and §7.4 |

**C1 and C5 are far larger than the panel realised.** C5 in particular is not renumbering; two sections do
not exist.

---

## 10. C14 — which components involve no training

`[DERIVED]` The panel asked for this explicitly. State it as a table in the thesis.

| Component | Trained? |
|---|---|
| Agent 1 — nnU-Net segmentation | **Yes.** The only component trained in this work |
| CineMA (ablation) | No — zero-shot, pretrained weights |
| Agent 2 — voxel summation, EF/mass formulae | **No.** Deterministic arithmetic |
| Agent 2 — plausibility gate | **No.** Rule-based |
| Agent 3 — MedCPT / BioBERT embedding | **No.** Inference only, frozen |
| Agent 3 — FAISS retrieval | **No.** No learned parameters |
| Agent 3 — GPT-4o / MedLLaMA-3 generation | **No.** Inference only (prompted, not fine-tuned) |
| Agent 3 — conflict resolver | **No.** Rule-based |
| Agent 4 — Seg-Grad-CAM | **No.** Gradient computation on a frozen model |
| Agent 4 — entropy uncertainty maps | **No.** Derived from the 5-fold ensemble |
| Agent 4 — citation renderer | **No** |
| Orchestration — LangGraph | **No** |
| Factual-consistency checker | **No.** Numeric comparison |

**Exactly one trained component.** This is a feature — it means every failure is attributable, and it is a
direct answer to C14 *and* supporting evidence for H3.

---

## 11. Reproducing the measurements

Everything in §3 and §5 was produced with the standard library only — no numpy, no nibabel. This matters:
the machine has no packages installed, and these scripts run on system Python 3.9 as-is.

### 11.1 NIfTI header reader

```python
import gzip, struct, os
def hdr(p):
    op = gzip.open if p.endswith('.gz') else open
    with op(p, 'rb') as f: b = f.read(348)
    e = '<' if struct.unpack('<i', b[0:4])[0] == 348 else '>'
    dim   = struct.unpack(e + '8h', b[40:56])     # dim[0]=ndim, dim[1..]=shape
    dtype = struct.unpack(e + 'h',  b[70:72])[0]
    pix   = struct.unpack(e + '8f', b[76:108])    # pixdim[1..]=spacing
    return dim, dtype, pix
DT = {2:'uint8', 4:'int16', 8:'int32', 16:'float32', 64:'float64', 512:'uint16'}
```

### 11.2 Label-volume loader (pure stdlib)

```python
import gzip, struct, array
def load(p):
    op = gzip.open if p.endswith('.gz') else open
    with op(p, 'rb') as f: raw = f.read()
    b = raw[:348]
    e = '<' if struct.unpack('<i', b[0:4])[0] == 348 else '>'
    dim = struct.unpack(e + '8h', b[40:56])
    dt  = struct.unpack(e + 'h',  b[70:72])[0]
    vox = int(struct.unpack(e + 'f', b[108:112])[0])   # vox_offset
    n   = dim[0]; shape = dim[1:n+1]
    tc  = {2:('B',1), 4:('h',2), 8:('i',4), 16:('f',4), 64:('d',8), 512:('H',2)}[dt]
    cnt = 1
    for s in shape: cnt *= s
    a = array.array(tc[0]); a.frombytes(raw[vox : vox + tc[1]*cnt])
    if e == '>': a.byteswap()
    return shape, a
```

### 11.3 The label-convention test — keep this as a permanent assertion

```python
import math
def centroids(shape, a, frame=None):
    X, Y, Z = shape[0], shape[1], shape[2]
    off = 0 if frame is None else frame * X * Y * Z
    acc = {l: [0.0, 0.0, 0] for l in (1, 2, 3)}
    for idx in range(X * Y * Z):
        v = int(a[off + idx])
        if v in acc:
            acc[v][0] += idx % X
            acc[v][1] += (idx // X) % Y
            acc[v][2] += 1
    return {l: (c[0]/c[2], c[1]/c[2], c[2]) for l, c in acc.items() if c[2]}

def lv_label(shape, a, frame=None):
    """The myocardium ring is concentric with the LV cavity."""
    c = centroids(shape, a, frame)
    myo = c[2][:2]
    d = {l: math.dist(c[l][:2], myo) for l in (1, 3)}
    return min(d, key=d.get)          # ACDC -> 3, M&Ms/M&Ms-2 -> 1
```

**Assertion to run over all 855 subjects after canonicalisation:**
`assert lv_label(shape, canonical_gt, ed_frame) == 1`

### 11.4 M&Ms ED/ES lookup

```python
import csv
ed, es = {}, {}
with open('MnMs/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv') as f:
    for r in csv.DictReader(f):
        ed[r['External code']] = int(r['ED'])
        es[r['External code']] = int(r['ES'])
```

### 11.5 PDF text extraction

```bash
brew install poppler
pdftotext -layout "Report - Muhammad Tahir.pdf" report.txt
```

---

## 12. Open questions

### RESOLVED 2026-07-11

**Q1. Are BAAI Cardiac Agent's model weights public?** **YES.** `[WEB]`
[github.com/plantain-herb/Cardiac-Agent](https://github.com/plantain-herb/Cardiac-Agent), **Apache 2.0**.
Released: agent weights (`TaipingQu/BAAI-Cardiac-Agent` on HuggingFace), all expert `.pth` checkpoints
(SAX/2CH/4CH cine seg, LGE seg, CDS, NICMS), training code, inference code, FastAPI serving stack, plus
`TaipingQu/CMRAgentEvalSet` (1,003 NIfTI) and `TaipingQu/CMR-MULTI`.

⇒ **The highest-value experiment in the project (§7.4) is a two-week job, not a two-month one.** Run BAAI on
ACDC and M&Ms. It is the first public-benchmark evaluation of the state-of-the-art agentic CMR system.
Caveat: a 7 B LMM + nine expert models will likely not fit 16 GB of unified memory. Verify early; budget
rented GPU for this ahead of nnU-Net.

**Q2. Is CineMA downloadable and runnable without CUDA?** **YES, but it is not "zero-shot."** `[WEB]`
[github.com/mathpluscode/CineMA](https://github.com/mathpluscode/CineMA) ·
[huggingface.co/mathpluscode/CineMA](https://huggingface.co/mathpluscode/CineMA). Python 3.11. Released
fine-tuned checkpoints, **3 seeds each**: segmentation on **ACDC** (`acdc_sax`), **M&Ms** (`mnms_sax`),
**M&Ms-2** (`mnms2_sax`, `mnms2_lax_4c`); plus CVD classification and direct EF regression fine-tuned on
ACDC / M&Ms / M&Ms-2, and landmark localisation.

⇒ **PLAN.md §2.9 and §3.2 call CineMA a "zero-shot ablation." That is now FALSE and must be corrected.** The
released ACDC checkpoint was *fine-tuned on ACDC*; running it on ACDC is a **fully supervised baseline**.
Correct usage:

| Run | What it actually is |
|---|---|
| `acdc_sax` → ACDC test | Supervised baseline |
| `acdc_sax` → **M&Ms / M&Ms-2** | **Genuine cross-dataset zero-shot. This is RQ3.** |
| `mnms_sax` → M&Ms | In-domain upper bound |

⇒ **Two consequences.** (a) The 3 released seeds give an **ensemble for free**, so Agent 4's entropy
uncertainty maps no longer depend on nnU-Net's 5-fold ensemble existing. (b) **Each checkpoint emits its own
dataset's label order** — the ACDC checkpoint outputs `1=RV`, the M&Ms checkpoint outputs `1=LV`. **Run the
§11.3 concentricity assertion on the PREDICTIONS, not only on the ground truth.** Cross-checkpoint work is
precisely where the §5 bug will bite.

### Still open

3. **What journal was the arrhythmia paper actually submitted to** — BMJ or BMC? (§1.2)
4. **Does MedCPT outperform Bio_ClinicalBERT on a guideline-only corpus?** CardAIc used Bio_ClinicalBERT;
   the ablation is yours to run.
5. **RESOLVED 2026-07-13 — there are NO conflicts. C11 must be reframed.** `[MEASURED]` The corpus was
   built (992 chunks; ESC HF 2021, AHA/ACC/HFSA HF 2022, AHA Chest Pain 2021, CMR-in-ESC 2023, SCMR
   reference values) and `detect_conflicts` was run over **8 deliberately cross-society queries** —
   HFrEF/HFmrEF/HFpEF cut-points, natriuretic-peptide thresholds, ICD and CRT indications, chest-pain
   pathways, SGLT2 inhibitors. **Zero conflicts, on every query**, including the six where ESC *and*
   AHA passages were both retrieved.

   ⇒ **ESC and AHA/ACC agree wherever this corpus can see them.** The conflict-resolution mechanism
   (recency → class of recommendation → present both with attribution) is **implemented and tested, but
   it has nothing to resolve.** C11 must therefore be presented as a **safety property** — *"the system
   cannot silently pick a side, and here is the mechanism that guarantees it"* — and **NOT** as a
   demonstrated capability. Claiming a demonstrated conflict-resolution capability on this corpus would
   be unsupportable, and the null result is itself the honest, reportable finding.
6. **Does ACDC's 50-patient test set have official EF/volume ground truth**, or only masks? Stage 1 derives
   volumes from masks; if official values exist, agreement between the two is a free validity check.
7. **`MnMs/Training/Unlabeled` — 25 subjects.** Unused by this plan. Semi-supervised extension, or ignore?
   Say which, explicitly, rather than leaving it unmentioned.

---

## 13. Session change log

| Date | Action |
|---|---|
| 2026-07-10 | Read `Muhammad_Tahir_Thesis_Knowledge.md` and `CLAUDE.md`. |
| 2026-07-10 | Enumerated all three datasets; counted subjects, vendors, centres, pathologies. |
| 2026-07-10 | Wrote pure-stdlib NIfTI reader; extracted shapes, spacings, dtypes. |
| 2026-07-10 | **Discovered ACDC/M&Ms label inversion** via myocardium-concentricity test on 9 subjects. |
| 2026-07-10 | Verified ESC/AHA HF LVEF thresholds and the non-existence of an ESC 2022 CMR guideline. |
| 2026-07-10 | Profiled hardware: M4, 16 GB, no CUDA, no packages. |
| 2026-07-10 | Wrote `PLAN.md` (v1). |
| 2026-07-10 | Extracted and read `Report` (32 pp) and `Slide` (31 pp) in full; built the defect register. |
| 2026-07-10 | Fetched BAAI and CardAIc abstracts; rewrote `PLAN.md` (v2) around the novelty risk. |
| 2026-07-10 | Read **both competitor papers in full**; corrected and sharpened the novelty analysis; `PLAN.md` (v3). |
| 2026-07-10 | Compiled this document. |
| 2026-07-11 | **Read `PD Comments by Panel.pdf` in full.** Reconciled against the knowledge-base transcription — see below. |
| 2026-07-11 | Resolved §12 Q1 (**BAAI weights are public, Apache 2.0**) and Q2 (**CineMA checkpoints are public — and ACDC-finetuned, so not "zero-shot"**). |
| 2026-07-11 | Fetched the **UM Candidature Defence marking rubric** and FSKTM PD/CD guideline; wrote [CD_TIMELINE.md](CD_TIMELINE.md). |

### 13.1 Reconciliation of `PD Comments by Panel.pdf` against the knowledge base

`[PAPER]` The source PDF has now been read. The C1–C16 list used throughout this document is **substantively
accurate** — every comment ID maps to the right issue, and no comment was missing or invented. Two points
worth recording:

- **C9 is stronger in the source than in the transcription.** The panel's actual words: *"the candidate is
  expected to collect and process their own data. The panel **strongly recommends incorporating a proprietary
  hospital dataset** to ensure the candidate undergoes the full data collection and preprocessing process,
  which is a core research requirement."* Note **"strongly recommends," not "requires."**
  **DECIDED 2026-07-11 (candidate): no hospital cohort. Public datasets only — ACDC + M&Ms + M&Ms-2, as per
  the proposal.** C9 is therefore handled as a *prepared rebuttal*, not a task: see
  [CD_TIMELINE.md](CD_TIMELINE.md) §3.2. §3.11 of the report must still be rewritten — it currently
  *disclaims* the need for ethics clearance outright, which is what triggered C9 in the first place.
- **C1 names Bennai explicitly** ("...Zeng (cited 2025, listed 2024), **and Bennai**"), so the Bennai
  mismatch is panel-identified, not an additional finding. §8.5 currently lists it under "additional
  mismatches the panel did not list." Cosmetic, but correct it when revising.

**Correction list is now complete and verified against source.**
