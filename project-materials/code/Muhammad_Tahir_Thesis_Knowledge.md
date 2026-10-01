# Thesis Knowledge Base
## Multi-Agent System for Automated Cardiac MRI Analysis and Decision Support

**Candidate:** Muhammad Tahir
**Programme:** Master of Computer Science by Research
**Institution:** Faculty of Computer Science & Information Technology, Universiti Malaya (UM)
**Matric Number:** 25087075/1
**Semester:** Sem 2, 2025/2026
**Supervisor:** Dr. Uzair Iqbal
**Co-Supervisor:** Prof. Dr. Nor Liyana Mohd Shuib
**Proposal Defence Date:** 26th June 2026
**Venue:** Online — Google Meet
**Result:** PASSED (77.0 average)
**Candidature Defence Due:** Semester I, 2026/2027
**Maximum Study Period:** End of Semester I, 2029/2030 (8th semester)

---

## 1. RESEARCH DOMAIN

- **Primary Domain:** Artificial Intelligence / Computer Science
- **Applied Domain:** Medical Imaging / Biomedical Informatics
- **Broader Field:** Clinical AI / Healthcare Informatics
- **Sub-domains:**
  - Multi-Agent Systems & Agentic AI
  - Deep Learning for Medical Image Segmentation
  - Natural Language Processing (RAG, LLMs)
  - Explainable AI (XAI)
  - Clinical Decision Support Systems
  - Information Retrieval (Dense Vector Search)

---

## 2. RESEARCH OVERVIEW

### Title
**Multi-Agent System for Automated Cardiac MRI Analysis and Decision Support**

### Core Proposition
A four-agent AI pipeline for automated interpretation of Cardiovascular Magnetic Resonance (CMR) imaging, integrating segmentation, quantification, guideline-grounded clinical reasoning, and explainability into a single unified workflow orchestrated by LangGraph.

### Key Claim (Novelty)
> To the best of our knowledge, no prior cardiac MRI system jointly integrates agentic orchestration, clinical reasoning, and explainability.

### Broader Field Position
Situated within Clinical AI and Healthcare Informatics, targeting high-impact venues including *Computers in Biology and Medicine*, *AI in Medicine*, and *BMJ Medical Informatics and Decision Making*.

---

## 2. BACKGROUND & MOTIVATION

### Cardiovascular Disease Burden
- CVD causes approximately **17.9 million deaths annually** (~32% of global mortality) [WHO, 2023]
- Rising prevalence of hypertension, diabetes, and obesity increases diagnostic demand [Wang et al., 2024]
- CMR is the **gold standard** for non-invasive cardiac morphology and function assessment [Kawel-Boehm et al., 2025]
- Superior reproducibility for LVEF, EDV, ESV vs. echocardiography [Wenzel et al., 2025 — MATCH Study]
- ESC, ACC, and AHA affirm CMR for DCM, HCM, and ischaemic heart disease

### CMR & AI Landscape
- CMR studies generate hundreds of frames requiring expert multi-sequence interpretation
- **Manual analysis:** time-consuming, expertise-dependent, inter-observer variability up to 10% [Bello et al., 2025]
- **Deep learning:** near-expert segmentation (CineMA, nnU-Net) and scar quantification (ScarNet)
- Systems operate in isolation — no integrated clinical workflow [Bello et al., 2025]
- Agentic AI enables end-to-end reasoning that is accurate, traceable, and interpretable

### Why Multi-Agent Architecture?
Multi-agent is chosen not because it is novel, but because the clinical workflow is inherently multi-step, multi-specialist, and requires verifiable handoffs — properties a single model cannot provide.

| Reason | Justification |
|--------|---------------|
| 01 Inherently Decomposable Task | 4 distinct expert subtasks: segmentation, quantification, reasoning, explanation [Gorenshtein et al., 2025] |
| 02 Mirrors Clinical Specialisation | Radiographer → Cardiologist → Reporting Physician → Auditor [Chen et al., 2025] |
| 03 Failure Isolation & Safety | Single model fails silently; multi-agent makes each step observable and verifiable [Xu et al., 2026] |
| 04 RAG Requires Dedicated Orchestration | ESC/ACC/AHA reasoning cannot be embedded in a segmentation model [Kohandel Gargari & Habibi, 2025] |
| 05 Explainability as First-Class Output | XAI is not a post-hoc add-on — it is a dedicated agent [Ganz et al., 2025] |

> A monolithic end-to-end model cannot simultaneously segment, apply clinical guidelines, and produce traceable explanations — these are cognitively distinct tasks requiring architectural separation.

---

## 3. RESEARCH GAPS

Based on a systematic review of 40+ peer-reviewed studies (2022–2026):

### Knowledge Gap
No dedicated agentic system exists for cardiac MRI. Healthcare agents are domain-generic, lacking imaging-specific orchestration workflows.
- **Opportunity:** Design a dedicated cardiac MRI agentic framework integrating segmentation, quantitative analysis & guideline-grounded clinical reasoning.

### Methodological Gap
Multimodal agents (AURA, MedChat, MMedAgent-RL) operate on isolated tasks with no unified end-to-end pipeline for complex clinical diagnosis.
- **Opportunity:** Integrate segmentation, RAG retrieval & clinical reasoning into a single orchestrated multi-agent pipeline.

### Experimental Gap
BAAI Cardiac Agent & CardAIc-Agents lack validation on multi-centre or heterogeneous datasets; many systems are conceptual or insufficiently validated.
- **Opportunity:** Validate on ACDC & M&Ms datasets using standardised metrics for diagnostic accuracy and workflow alignment.

### Clinical Gap
XAI reviews confirm most deployed cardiac AI systems produce opaque predictions misaligned with ACC/AHA & ESC CMR guidelines.
- **Opportunity:** Embed guideline-grounded outputs & attention-based explanations for transparent, traceable clinical decision support.

---

## 4. PROBLEM STATEMENT

Three core problems identified:

### 4.1 Fragmented Task Automation
- CineMA (2025): Segmentation only — no clinical reasoning
- MARL-Rad (2025): Structured reports not grounded in imaging findings
- AURA & MedChat (2025): Multimodal dialogue lacking cardiac pipelines
- No unified end-to-end CMR workflow integrating segmentation, reporting, and reasoning

### 4.2 Absent Guideline Grounding
- BAAI Cardiac Agent & CardAIc-Agents: Outputs not aligned with ESC/ACC/AHA/HFSA guidelines
- Patel et al. (2025): ACR RAG system limited to general radiology text only
- No CMR AI retrieves from ESC 2022 CMR or ACC/AHA cardiac imaging guidelines
- Outputs lacking traceability to guideline criteria carry limited clinical utility

### 4.3 Insufficient Explainability
- Majority of cardiac AI systems are functionally opaque
- XAI mechanisms proposed but seldom validated against clinical utility criteria
- Black-box predictions carry significant patient safety risk [Rudin, 2019]
- Misclassified cardiomyopathy or undetected scar can alter life-altering treatment decisions

---

## 5. RESEARCH QUESTIONS & OBJECTIVES

### Research Questions

| ID | Question |
|----|----------|
| RQ1 | How can a multi-agent framework be developed to integrate automated segmentation, quantitative analysis, and guideline-grounded reasoning into a unified workflow that ensures clinical consistency in CMR interpretation? |
| RQ2 | How does the proposed framework perform in terms of diagnostic accuracy and explainability when benchmarked against established single-task AI baselines using standardized metrics (Dice, MAE, and BERTScore)? |
| RQ3 | To what extent does the framework maintain diagnostic robustness and generalizability when evaluated across heterogeneous, multi-vendor datasets (ACDC and M&Ms) from diverse clinical centers? |

### Research Objectives

| ID | Objective |
|----|-----------|
| O1 | To design and develop a multi-agent system that integrates automated segmentation (CineMA/nnU-Net), quantitative analysis, and guideline-grounded reasoning into a unified cardiac MRI analysis pipeline |
| O2 | To create a benchmark by evaluating the framework's clinical accuracy and explainability against single-task baselines, utilizing Dice Similarity Coefficient, Mean Absolute Error (MAE), and BERTScore metrics |
| O3 | To validate the generalizability of the proposed system across multi-center and multi-vendor heterogeneous datasets (ACDC and M&Ms) to ensure reliable performance across diverse clinical acquisition protocols |

---

## 6. RESEARCH SCOPE

### In Scope
- Cine Cardiovascular Magnetic Resonance Imaging (CMR)
- Left Ventricle/Right Ventricle/Myocardium (LV/RV/Myo) segmentation
- Left Ventricular Ejection Fraction (LVEF), End-Diastolic Volume (EDV), End-Systolic Volume (ESV)
- Automated Cardiac Diagnosis Challenge (ACDC) & Multi-Centre, Multi-Vendor (M&Ms) datasets
- Gradient-weighted Class Activation Mapping (Grad-CAM) + guideline citations
- Structured clinical report generation

### Out of Scope
- Late Gadolinium Enhancement (LGE) / T1 mapping
- Right Ventricular (RV) strain
- SHapley Additive exPlanations (SHAP) / Local Interpretable Model-agnostic Explanations (LIME) on raw MRI
- Patient-facing interfaces
- Proprietary hospital datasets *(noted as panel recommendation for revision)*
- PACS deployment

> Research prototype for clinical decision support — adjunct to, not replacement for, specialist judgement.

---

## 7. RESEARCH SIGNIFICANCE

### Scientific
- First joint agentic orchestration + clinical reasoning + explainability for CMR
- Novel architecture, replicable evaluation methodology, and benchmark results

### Clinical
- Guideline-cited outputs reduce cognitive burden and improve reporting consistency

### Societal
- Supports equitable cardiac imaging access in resource-limited settings [Leiner et al., 2026]

---

## 8. LITERATURE REVIEW SUMMARY

| Thematic Domain | Key Systems | Collective Limitations | Research Gap |
|----------------|-------------|----------------------|--------------|
| Cardiac MRI AI | CineMA (Fu et al., 2025), ScarNet (Tavakoli et al., 2025) | Task isolation: fragmented models focused on segmentation without clinical reasoning | Knowledge, Methodological |
| Cardiac AI Agents | BAAI Cardiac Agent (Qu et al., 2026), CardAIc-Agents (Zhang et al., 2025) | No grounding in ESC/ACC guidelines; limited multi-center validation | Experimental, Clinical |
| Multimodal Medical Agents | AURA (Fathi et al., 2025), MedChat (Liu et al., 2025) | Domain gap: generic healthcare frameworks; lack specialized 4D cardiac imaging pipelines | Methodological |
| RAG in Clinical AI | ACR RAG System (Pambudi et al., 2025) | Modality gap: focus exclusively on text; no extension to image-based diagnostic reasoning | Knowledge, Clinical |
| Explainable AI (XAI) | Haupt et al. (2025), Salih et al. (2023) | Opacity: saliency maps exist but not linked to specific clinical guideline passages | Clinical |

### Critical Synthesis
No existing framework simultaneously integrates:
1. Agentic orchestration of the CMR pipeline
2. Guideline-grounded reasoning (ESC/ACC)
3. Traceable XAI

The proposed system fills this gap by uniting **nnU-Net, FAISS, and BioBERT** into a novel unified architecture.

---

## 9. METHODOLOGY

### 9.1 Research Paradigm
- **Design Science Research (DSR)** — constructing a novel artefact as a solution to a defined domain problem [Hevner et al., 2004]
- Sequential mixed-methods with emphasis on quantitative metrics

### 9.2 Research Phases

| Phase | Description | Maps To |
|-------|-------------|---------|
| Phase 1 — Foundation | Literature review, dataset acquisition, segmentation backbone training | Slide 3: Agent 1 |
| Phase 2 — Construction | LangGraph orchestration, XAI integration, all 4 agents built | Slides 4,5,6,7: Agents 2-4 + Orchestration |
| Phase 3 — Evaluation | ACDC/M&Ms benchmarking, ablation, baseline comparisons | Slides 3,4: DSC, MAE targets |
| Phase 4 — Validation | Cross-dataset generalisation testing | Slide 3: M&Ms generalisation |

---

## 10. SYSTEM ARCHITECTURE

### Overview
Four-agent pipeline orchestrated by LangGraph:

```
CMR Input (NIfTI/DICOM)
        ↓
  Preprocessing
  (Normalisation, Resampling, Stacking)
        ↓
[Agent 1: Segmentation]
        ↓
[Agent 2: Quantitative Analysis]
        ↓
[Agent 3: Clinical Reasoning] ←→ Guideline Corpus (ESC/ACC/AHA/HFSA)
        ↓
[Agent 4: Explainability]
        ↓
  Structured Clinical Report
  + Visual Validation Panels
```

### Orchestration Layer (LangGraph)
- **Manage Agent State:** Coordinate status across all four agents
- **Inter-agent Data Transfer:** Structured JSON between pipeline stages
- **Error Handling:** Retry logic and graceful failure recovery
- **Iterative Feedback Loop:** Guideline evidence from Agent 3 can refine segmentation masks from Agent 1
- **Modular:** Individual agents upgradeable without full architecture redesign

---

## 11. FOUR-AGENT PIPELINE — DETAILED

### Agent 1: Segmentation (O1) — Phase 1 & 3
- **Task:** Delineates LV cavity, RV cavity, and LV myocardium from cine CMR stacks
- **Backbone:** nnU-Net — SOTA on ACDC [Isensee et al., 2021]; CineMA as zero-shot ablation alternative [Fu et al., 2025]
- **Preprocessing:** 1.5×1.5 mm resampling, z-score normalisation, ED/ES frame extraction
- **Training:** ACDC subjects, 5-fold cross-validation, 1000 epochs, NVIDIA A100
- **Target:** Dice ≥ 0.88 (LV, RV, myocardium) on ACDC test set
- **Output:** Segmentation masks passed downstream as JSON

### Agent 2: Quantitative Analysis (O1) — Phase 2 & 3
- **EDV/ESV:** Voxel summation across short-axis slices at ED/ES frames
- **LVEF Formula:** LVEF (%) = (EDV − ESV) / EDV × 100
- **LV Mass:** Myocardium volume × 1.05 g/mL [Kawel-Boehm et al., 2025]
- **Output:** Structured JSON with ESC heart failure classification flags
  - HFrEF: LVEF < 40%
  - HFmrEF: LVEF 40–49%
  - HFpEF: LVEF ≥ 50%
- **Target:** LVEF MAE ≤ 5% against expert ground truth

### Agent 3: Clinical Reasoning (O2) — Phase 2
- **Core Contribution:** Guideline-grounded diagnostic reasoning using RAG
- **Corpus:**
  - ESC 2022 CMR Guidelines
  - ACC/AHA HF 2022
  - AHA/ACC Chest Pain 2020
  - HFSA 2023
- **Embedding:** BioBERT (768-dim) [Lee et al., 2020]
- **Index:** FAISS for approximate nearest-neighbour search
- **Chunking:** 200–400 token chunks, 50-token overlap
- **Retrieval:** Top-k=5 most relevant guideline passages
- **Generation:** GPT-4o prompt → cited rationale, differential diagnosis, recommended clinical actions
- **Ablation:** MedLLaMA-3 for open-source reproducibility
- **Hallucination Mitigation:** Guardrails AI (panel recommendation)

### Agent 4: Explainability (O2) — Phase 2
- **Visual:** Grad-CAM heatmaps on segmentation feature maps (ED frame overlays) — for radiologists
- **Textual:** Guideline citations formatted into citation-annotated clinical report — for cardiologists
- **Traceability:** Each statement linked to source document, section, and page number
- **Dual Mechanism:** Interpretable visually (imaging) AND clinically (guideline language)
- **Gap Addressed:** No existing CMR system provides both visual and textual explainability

---

## 12. DATASETS

| Dataset | Year | Subjects | Classes | Modality | Purpose |
|---------|------|----------|---------|----------|---------|
| ACDC | 2018 | 150 | 5 pathologies (DCM, HCM, MINF, ARV, Normal) | Cine CMR | Primary training & validation; segmentation + EF benchmarking |
| M&Ms | 2021 | 375 | 4 pathologies | Cine CMR | Generalisation testing across 4 vendors and 6 clinical centres |
| M&Ms-2 | 2022 | 360 | Multi-label | Cine CMR | Extended multi-label generalisation and stress-test evaluation |

- **Total subjects evaluated:** 525 (ACDC 150 + M&Ms 375)
- **M&Ms held out entirely from training** — conservative domain-shift robustness test
- All datasets are publicly available and de-identified — no ethics approval required

---

## 13. EVALUATION METRICS

| Category | Metric | Target |
|----------|--------|--------|
| Segmentation | Dice Similarity Coefficient (DSC) | ≥ 0.88 |
| Segmentation | HD95 boundary accuracy | Minimise |
| Quantification | LVEF MAE | ≤ 5% |
| Quantification | EDV/ESV MAE | Minimise |
| RAG/Report Quality | BERTScore F1 | ≥ 0.85 |
| RAG/Report Quality | Clinical Alignment Score (5-point Likert) | Maximise |
| Generalisation | Cross-dataset DSC drop (ACDC → M&Ms) | ≤ 0.05 |
| Generalisation | Vendor/centre stratified analysis | Document |

> **Note:** BERTScore is embedding-based and semantic — NOT lexical. It must NOT be grouped with BLEU/ROUGE. (Panel correction required.)

---

## 14. KEY NUMBERS TO REMEMBER

| Parameter | Value |
|-----------|-------|
| CVD deaths annually | ~17.9 million |
| Global mortality share | ~32% |
| Inter-observer variability (manual CMR) | Up to 10% |
| ACDC subjects | 150 |
| M&Ms subjects | 375 |
| M&Ms-2 subjects | 360 |
| Total subjects (ACDC + M&Ms) | 525 |
| Segmentation target (Dice) | ≥ 0.88 |
| LVEF MAE target | ≤ 5% |
| BERTScore F1 target | ≥ 0.85 |
| DSC generalisation drop limit | ≤ 0.05 |
| BioBERT embedding dimensions | 768 |
| RAG chunk size | 200–400 tokens |
| Chunk overlap | 50 tokens |
| Top-k retrieval | 5 |
| Training epochs | 1000 |
| Cross-validation folds | 5 |
| HFrEF threshold (ESC 2022) | LVEF < 40% |
| HFmrEF range | LVEF 40–49% |
| HFpEF threshold | LVEF ≥ 50% |
| LV mass density constant | 1.05 g/mL |
| Guideline corpus documents | 4 (ESC, ACC/AHA HF, AHA/ACC CP, HFSA) |
| Panel average score | 77.0 / 100 |
| Passing mark (UM) | 65.0 |

---

## 15. TECHNOLOGY STACK

| Component | Technology |
|-----------|-----------|
| Segmentation backbone | nnU-Net |
| Foundation model ablation | CineMA |
| Orchestration framework | LangGraph |
| Embedding model | BioBERT (768-dim) |
| Vector index | FAISS |
| Primary LLM | GPT-4o |
| Open-source LLM ablation | MedLLaMA-3 |
| Visual XAI | Grad-CAM |
| Safety/Guardrails | Guardrails AI (panel recommendation) |
| Data format | NIfTI / DICOM → JSON |
| GPU | NVIDIA A100 |

---

## 16. EXPECTED OUTCOMES & CONTRIBUTIONS

1. **Novel four-agent framework** for end-to-end CMR interpretation — first to jointly address agentic orchestration, guideline-grounded clinical decision support, and explainability for CMR
2. **Curated ESC/ACC/AHA/HFSA corpus** specifically for cardiac imaging RAG — does not currently exist as a public resource
3. **Benchmark results** on ACDC and M&Ms with component-level ablation evidence
4. **Target Q1 publication:** *Computers in Biology and Medicine* or *AI in Medicine*

### Research Novelty
- First to jointly address agentic orchestration, clinical decision support, and explainability for CMR
- Curated ESC/ACC/AHA/HFSA corpus specifically for cardiac imaging reasoning
- LangGraph pipeline mirroring multi-disciplinary clinical imaging workflow

---

## 17. PRELIMINARY FINDINGS

### Journal Paper Submitted
- **Title:** Transparent Cardiac Intelligence: A Post-Hoc Explainability Framework for Ensemble-Based Arrhythmia Classification
- **Journal:** BMJ Medical Informatics & Decision Making
- **Submitted:** 23 May 2026
- **Status:** Under Review
- **Significance:**
  - Integrates post-hoc XAI (SHAP/LIME) for clinical transparency
  - Demonstrates model performance on real-world cardiac dataset
  - Contributes to trustworthy AI adoption in healthcare informatics
  - Validates XAI methodology that underpins Agent 4 of the thesis

---

## 18. WORK SCHEDULE

**Duration:** 24 months (07/03/2026 – 31/12/2027)

| Activity | Months |
|----------|--------|
| Phase 1: Literature Review & Segmentation Backbone Training | 1–6 |
| Phase 2: LangGraph Construction & XAI Integration (Agents 2–4) | 4–8 |
| Phase 3: ACDC/M&Ms Benchmarking & Ablation | 7–12 |
| Phase 4: Generalisation Validation (M&Ms) | 9–18 |
| Thesis Writing, Analysis & Final Submission | 19–24 |

---

## 19. ETHICAL CONSIDERATIONS

- Public de-identified datasets (ACDC, M&Ms) — no institutional ethics approval required
- Research prototype only — not for clinical deployment or patient-facing use
- *(Panel recommendation: explore proprietary hospital dataset acquisition with appropriate ethics approval)*

---

## 20. PANEL EVALUATION — PROPOSAL DEFENCE

**Date:** 26th June 2026 | **Result:** PASSED

### Panel Members
- **Chairperson:** Ts. Dr. Siti Nurliana Jamalai @ Jamali
- **Panel Member 1:** Dr. Burhan Ul Islam Khan
- **Panel Member 2:** Dr. Mohamad Hazim Md Hanif

### Score Breakdown

| Criteria | Panel 1 | Panel 2 | Average |
|----------|---------|---------|---------|
| Title & Abstract (5%) | 4.5 | 3.5 | 4.0 |
| Introduction (25%) | 20.0 | 17.5 | 18.8 |
| Literature Review (25%) | 20.0 | 20.0 | 20.0 |
| Conceptual Framework/Methods (20%) | 16.0 | 14.0 | 15.0 |
| Summary/Conclusion (5%) | 4.0 | 3.5 | 3.8 |
| Academic Style & References (10%) | 7.0 | 8.0 | 7.5 |
| Communication/Q&A (10%) | 8.0 | 8.0 | 8.0 |
| **Total** | **79.5** | **74.5** | **77.0** |

---

## 21. PANEL COMMENTS — REQUIRED CORRECTIONS

### From Panel 1 (Dr. Burhan Ul Islam Khan)

| # | Issue | Action Required |
|---|-------|----------------|
| C1 | Citation years systematically off (Salih 2025/2023, Baba 2025/2026, Sahoo 2025/2024, Zeng 2025/2024) | Verify and correct ALL in-text citation years to match reference list |
| C2 | Dataset inconsistency — abstract/scope commit to 2 datasets but methodology introduces M&Ms-2 as 3rd | Make dataset scope consistent throughout all sections |
| C3 | RV analysis listed as out of scope in Table 1.2 but Agent 1 produces RV masks | Resolve contradiction — either include RV in scope or remove from methodology |
| C4 | Section 3.8 says "five" baselines but Table 3.4 lists four | Reconcile baseline count and name all explicitly |
| C5 | Chapter 3 numbering broken — sections skip 3.9, 3.10; tables skip 3.2, 3.3 | Full structural review and renumbering of Chapter 3 |
| C6 | Reference list not alphabetically ordered | Sort entire reference list alphabetically by first author surname |
| C7 | "Fu et al., 2025" and "Zhang et al., 2025" each point to two different papers | Disambiguate as 2025a and 2025b; fix Kristijan citation |
| C8 | BERTScore mislabelled as "lexical" in Section 2.4.2 | Correctly classify BERTScore as embedding-based and semantic |

### From Panel 2 (Dr. Mohamad Hazim Md Hanif)

| # | Issue | Action Required |
|---|-------|----------------|
| C9 | Master by Research requires own data collection | Explore and incorporate proprietary hospital dataset |
| C10 | No error propagation strategy when Agent 1 fails | Define concrete error containment mechanism across pipeline |
| C11 | Conflicting ESC vs ACC/AHA guidelines not addressed | Specify conflict resolution mechanism in Agent 3 |
| C12 | Iterative feedback loop trigger not specified | Define exact programmatic trigger, execution, and stopping criterion |
| C13 | No metric for clinical correctness of diagnostic outputs | Add human-in-the-loop or quantitative clinical accuracy metric |
| C14 | Non-AI-trained components not explicitly stated | Clearly mark which components involve no AI training |
| C15 | No agent-to-agent security mechanism | Integrate Guardrails AI; address prompt injection and hallucination |
| C16 | Work framed as engineering not research | Reframe to highlight novel research contribution and theoretical advancement |

---

## 22. KEY CITATIONS — COMPLETE APA REFERENCE LIST

> Listed alphabetically as required by APA 7th edition and panel correction C6.

- Bello, G. A., et al. (2025). Bridging the gap in cardiovascular magnetic resonance imaging artificial intelligence implementations: From ambitious goals to real-world progress using foundation models. *Journal of Cardiovascular Magnetic Resonance*. PMC12766593.

- Bernard, O., Lalande, A., Zotti, C., Cervenansky, F., Yang, X., Heng, P.-A., Cetin, I., Lekadir, K., Camara, O., Ballester, M. A. G., Sanfilippo, F., Amic, A., Petitjean, C., & Jodoin, P.-M. (2018). Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: Is the problem solved? *IEEE Transactions on Medical Imaging*, *37*(11), 2514–2525.

- Chen, Y., Zhou, H., & Tan, W. (2025). Agentic systems in radiology: Design, applications, evaluation, and challenges. *Radiology: Artificial Intelligence*, *7*(2), e240089. https://doi.org/10.1148/ryai.240089

- Fotaki, A., Ferreira, V. M., Khalique, Z., Swoboda, P., McDiarmid, A. K., Kardos, A., Westwood, M., & Prasad, S. K. (2023). Cardiovascular magnetic resonance imaging: When to use it and what to look for. *European Heart Journal*, *44*(44), 4678–4692. https://doi.org/10.1093/eurheartj/ehad500

- Fu, S., Dong, J., Ding, X., Sun, R., Yang, Y., Cui, S., & Li, Z. (2026). AgentsEval: Clinically faithful evaluation of medical imaging reports via multi-agent reasoning. *arXiv*. http://arxiv.org/abs/2601.16685

- Fu, Y., et al. (2025). CineMA: A foundation model for cardiac cine MRI. *arXiv*. https://arxiv.org/abs/2506.00679

- Ganz, M., Scholten, E. T., & Bobowicz, M. (2025). Explainable AI in medicine: Challenges of integrating XAI into the future clinical routine. *Frontiers in Radiology*, *5*, 1627169. https://doi.org/10.3389/fradi.2025.1627169

- Gorenshtein, A., Omar, M., Glicksberg, B. S., Nadkarni, G. N., & Klang, E. (2025). AI agents in clinical medicine: A systematic review. *medRxiv*. https://doi.org/10.1101/2025.08.22.25334232

- Haupt, M., Maurer, M. H., & Thomas, R. P. (2025). Explainable artificial intelligence in radiological cardiovascular imaging: A systematic review. *Diagnostics*, *15*(11). https://doi.org/10.3390/diagnostics15111399

- Hevner, A. R., March, S. T., Park, J., & Ram, S. (2004). Design science in information systems research. *MIS Quarterly*, *28*(1).

- Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, *18*(2), 203–211. https://doi.org/10.1038/s41592-020-01008-z

- Kawel-Boehm, N., Hetzel, S. J., Ambale-Venkatesh, B., Captur, G., Chin, C. W. L., François, C. J., Jerosch-Herold, M., Luu, J. M., Raisi-Estabragh, Z., Starekova, J., Taylor, M., van Hout, M., & Bluemke, D. A. (2025). Society for Cardiovascular Magnetic Resonance reference values in cardiovascular magnetic resonance: 2025 update. *Journal of Cardiovascular Magnetic Resonance*. https://doi.org/10.1016/j.jocmr.2025.101853

- Kohandel Gargari, O., & Habibi, G. (2025). Enhancing medical AI with retrieval-augmented generation: A mini narrative review. *SAGE Open Medicine*, *13*. https://doi.org/10.1177/20552076251337177

- Kolk, M. Z. H., Ruipérez-Campillo, S., Allaart, C. P., Wilde, A. A. M., Knops, R. E., Narayan, S. M., Tjong, F. V. Y., Raijmakers, F. D., van der Lingen, A. L. C. J., Götte, M. J. W., Selder, J. L., Alvarez-Florez, L., Išgum, I., & Bekkers, E. J. (2024). Multimodal explainable artificial intelligence identifies patients with non-ischaemic cardiomyopathy at risk of lethal ventricular arrhythmias. *Scientific Reports*, *14*(1). https://doi.org/10.1038/s41598-024-65357-x

- Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2020). BioBERT: A pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, *36*(4), 1234–1240. https://doi.org/10.1093/bioinformatics/btz682

- Leiner, T., et al. (2026). The global roadmap for cardiovascular imaging: Bridging the diagnostic divide. *European Heart Journal — Imaging Methods and Practice*. https://doi.org/10.1093/ehjimp/qyag031

- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, *33*, 9459–9474.

- Sahoo, P., Tripathi, A., Saha, S., & Mondal, S. (2024). FedMRL: Data heterogeneity aware federated multi-agent deep reinforcement learning for medical imaging. *arXiv*. http://arxiv.org/abs/2407.05800

- Salih, A., Boscolo Galazzo, I., Gkontra, P., Lee, A. M., Lekadir, K., Raisi-Estabragh, Z., & Petersen, S. E. (2023). Explainable artificial intelligence and cardiac imaging: Toward more interpretable models. *Circulation: Cardiovascular Imaging*, *16*(4). https://doi.org/10.1161/CIRCIMAGING.122.014519

- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2020). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *International Journal of Computer Vision*, *128*(2), 336–359. https://doi.org/10.1007/s11263-019-01228-7

- Tavakoli, A., Nakamori, S., Menze, B., & Nezafat, R. (2025). ScarNet: A novel foundation model for automated myocardial scar quantification from late gadolinium enhancement cardiac MRI. *arXiv*. https://arxiv.org/abs/2501.01066

- Wang, Y. R., Yang, K., Wen, Y., Wang, P., Hu, Y., Lai, Y., Wang, Y., Zhao, K., Tang, S., Zhang, A., Zhan, H., Lu, M., Chen, X., Yang, S., Dong, Z., Wang, Y., Liu, H., Zhao, L., Huang, L., … Zhao, S. (2024). Screening and diagnosis of cardiovascular disease using artificial intelligence-enabled cardiac magnetic resonance imaging. *Nature Medicine*, *30*(5), 1471–1480. https://doi.org/10.1038/s41591-024-02971-2

- Wenzel, J. P., Albrecht, J. N., Toprak, B., et al. (2025). Head-to-head comparison of cardiac magnetic resonance imaging and transthoracic echocardiography in the general population (MATCH). *Clinical Research in Cardiology*. https://doi.org/10.1007/s00392-025-02660-1

- World Health Organization. (2023). *Cardiovascular diseases (CVDs) fact sheet*. https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

- Xu, G., Li, X., Chen, Y., Duan, Y., Wu, S., Yu, H., Chiu, C.-H., Ni, J., Tang, N., Li, T. J.-J., Yuille, A., Jin, W., & Shi, Y. (2026). A comprehensive survey of AI agents in healthcare. *Journal of Biomedical Informatics*, *179*, 105045. https://doi.org/10.1016/j.jbi.2026.105045

---

*This document was compiled from all presentation slides, panel evaluation forms, and research discussions as of July 2026. It represents the complete known state of the thesis at proposal defence stage.*
