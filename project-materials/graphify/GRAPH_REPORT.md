# Graph Report - /Users/it-macmini-1-tahir/Documents/Archive  (2026-09-28)

## Corpus Check
- 59 files · ~106,477 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 848 nodes · 1798 edges · 48 communities (43 shown, 5 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 194 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Explainability and XAI
- Templates and Configuration
- nnU-Net Data Conversion
- BAAI Baseline Integration
- Quantification and Gate
- Thesis System Landscape
- Guardrail Tests
- Evaluation Metrics
- Factuality Scoring
- CineMA Segmentation
- Guideline Corpus Parsing
- LLM Client and Schema
- Report Generation
- Command-Line Workflow
- Live Run Management
- Viewer and Evidence UI
- Streaming Run Events
- Guideline Acquisition
- Run and Subject Artifacts
- Recommendation Grade Tests
- Hybrid Retrieval Logic
- Grounding Enforcement
- UI Configuration
- Embedding Index Build
- Citation Claim Linking
- Corpus Coverage Checks
- Viewer Tests
- NIfTI Upload Pipeline
- Retriever and Fusion
- Panel Requirements
- Defence Report Claims
- Default Pipeline Design
- Engineering and Reproducibility
- nnU-Net Training
- Injection Detection
- Ablation Grid
- Security Threat Model
- Project Status Claims
- Original Thesis Claims
- Proposal Methodology
- Viva Dashboard Views
- Defence Timeline
- Ollama Startup
- Gate Calibration Test
- Human Writing Guidance
- HPC Configuration
- End-to-End Run Script
- Agent Interaction Rules

## God Nodes (most connected - your core abstractions)
1. `Config` - 73 edges
2. `Measurements` - 34 edges
3. `Chunk` - 33 edges
4. `Subject` - 29 edges
5. `Report` - 29 edges
6. `LLM` - 26 edges
7. `Deps` - 24 edges
8. `run()` - 22 edges
9. `check()` - 22 edges
10. `artifacts()` - 21 edges

## Surprising Connections (you probably didn't know these)
- `Measured Dataset Label Inversion` --semantically_similar_to--> `Cross-Dataset Label Convention Inversion`  [INFERRED] [semantically similar]
  code/PROCESS.md → cmr/README.md
- `Guideline-Passage Grounding in Diagnostic Reports` --semantically_similar_to--> `BAAI Retrieval-Augmented Generation Tool`  [INFERRED] [semantically similar]
  code/PLAN.md → code/BAAI Cardiac Agent An intelligent multimodal agent for automated reasoning and diagnosis of cardiovascular diseases from cardiac magnetic resonance imaging.pdf
- `Decision-Boundary-Aware Recomputation` --semantically_similar_to--> `Stepwise Plan Update`  [INFERRED] [semantically similar]
  code/PLAN.md → code/CardAIc-Agents A Multimodal Framework.pdf
- `Original First Integrated CMR System Claim` --conceptually_related_to--> `Rebuilt Defensible Novelty Claim`  [INFERRED]
  code/Report - Muhammad Tahir.pdf → code/PLAN.md
- `test_model_id()` --calls--> `LLM`  [INFERRED]
  cmr/tests/test_llm.py → cmr/cmr/llm.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **CMR Four-Agent Pipeline** — cmr_readme_segmentation_agent, cmr_readme_quantification_agent, cmr_readme_clinical_reasoning_agent, cmr_readme_explainability_agent [EXTRACTED 1.00]
- **Three Testable Thesis Hypotheses** — code_cd_report_grounding_hypothesis, code_cd_report_boundary_recomputation_hypothesis, code_cd_report_failure_visibility_hypothesis [EXTRACTED 1.00]
- **CardAIc Hierarchical Adaptation** — code_cardaic_agents_a_multimodal_framework_cardiacrag_agent, code_cardaic_agents_a_multimodal_framework_chief_agent, code_cardaic_agents_a_multimodal_framework_stepwise_plan_update, code_cardaic_agents_a_multimodal_framework_multidisciplinary_discussion_team [EXTRACTED 1.00]

## Communities (48 total, 5 thin omitted)

### Community 0 - "Explainability and XAI"
Cohesion: 0.05
Nodes (60): Explanation, THE contribution of Agent 4. Not 'a heatmap and a citation' — the LINK between t, _as_logits(), auto_layer(), _best_slice(), _cam_to_disk(), _canon(), _ed() (+52 more)

### Community 1 - "Templates and Configuration"
Cohesion: 0.08
Nodes (58): _fmt(), The zero-AI floor: a fixed-template report. NO LLM, NO RETRIEVAL, NO MODEL.  Thi, The findings paragraph. Every number here is a verbatim field of `m`., A fixed-template report from Agent 2's JSON. NO LLM AT ALL., template_report(), api_compare(), _apply_env(), Config (+50 more)

### Community 2 - "nnU-Net Data Conversion"
Cohesion: 0.06
Nodes (54): AssertionError, _affine(), convert(), main(), _phases(), ndarray, Path, ACDC / M&Ms -> nnU-Net v2 raw format, in CANONICAL label order.  THE POINT OF TH (+46 more)

### Community 3 - "BAAI Baseline Integration"
Cohesion: 0.08
Nodes (49): canonicalise_baai(), detect_native_order(), _device(), main(), _predict(), _predictor(), ndarray, Path (+41 more)

### Community 4 - "Quantification and Gate"
Cohesion: 0.08
Nodes (48): n_components(), Max connected components of `label` on any slice, ignoring specks.      `min_are, Fraction of a structure's voxels NOT in its largest 3-D connected component., stray_fraction(), artifacts(), _ef_pct(), gate(), hf_category() (+40 more)

### Community 5 - "Thesis System Landscape"
Cohesion: 0.05
Nodes (41): Clinical Reasoning Agent, Explainability Agent, Guideline-Only Corpus with CoR and LoE Metadata, Cross-Dataset Label Convention Inversion, Measured Numeric and Citation Fidelity, Multi-Agent CMR Analysis and Decision Support System, Quantification Agent, Segmentation Agent (+33 more)

### Community 6 - "Guardrail Tests"
Cohesion: 0.12
Nodes (29): Every way this report breaks an invariant. Empty list == emittable.      The str, violations(), _FakeLLM, The invariants, proven.  These tests are the evidence for the claim "this system, Returns a scripted sequence of reports. Stands in for the model so the loop can, The headline guarantee. A model that keeps hallucinating does not get its report, A citation resolving to nothing is not a weak citation, it is a fiction. If the, After enforcement the metric MUST read 1.0/1.0. If it ever does not, enforcement (+21 more)

### Community 7 - "Evaluation Metrics"
Cohesion: 0.11
Nodes (31): agreement(), bootstrap_ci(), canonical_pathology(), evaluate_run(), fidelity_summary(), hf_category_accuracy(), icc21(), _json_safe() (+23 more)

### Community 8 - "Factuality Scoring"
Cohesion: 0.13
Nodes (29): check(), THE headline metric: is every number and every citation in the report actually r, Does this exact numeral appear in the passage? Bounded so 40 does not match 1940, Remove everything that is digits-but-not-a-claim., Fidelity of one report against the measurements that produced it and the chunk i, _scrub(), _verbatim(), Measurements (+21 more)

### Community 9 - "CineMA Segmentation"
Cohesion: 0.11
Nodes (22): _Canonical, _fingerprint(), load_entropy(), load_mask(), _paths(), _postprocess(), _preprocess(), Module (+14 more)

### Community 10 - "Guideline Corpus Parsing"
Cohesion: 0.12
Nodes (25): _aha_grade(), build_corpus(), _clean(), _esc_grade(), extract_grade(), Grade, _is_heading(), _is_references() (+17 more)

### Community 11 - "LLM Client and Schema"
Cohesion: 0.12
Nodes (17): _inline_defs(), LLM, LLMError, LLMUnavailable, Any, BaseModel, RuntimeError, Schema-constrained completion. The only LLM entry point in the codebase. (+9 more)

### Community 12 - "Report Generation"
Cohesion: 0.15
Nodes (17): One provider-agnostic LLM client. One code path.  Ollama, Gemini, Groq and OpenR, _fmt_measurements(), _fmt_passages(), generate(), generate_all(), Any, Agent 3b — grounded report generation.  The model never sees an image and is nev, One subject -> one schema-valid, citation-carrying Report.      `passages` empty (+9 more)

### Community 13 - "Command-Line Workflow"
Cohesion: 0.18
Nodes (17): serve(), cmd_doctor(), cmd_eval(), cmd_figures(), cmd_manifest(), cmd_quantify(), cmd_segment(), cmd_serve() (+9 more)

### Community 14 - "Live Run Management"
Cohesion: 0.18
Nodes (10): Field, Runs live, single-subject pipeline requests one at a time.      Segmenter/Retrie, Write Agent 1's mask + entropy to the SAME checkpoint-keyed pool `cmr segment`, RunManager, LabelOrderError, Raised when a mask is not in canonical 1=LV / 2=Myo / 3=RV order., Three seeds, one mask, one entropy map. Reused across subjects — loading 127M, Segmenter (+2 more)

### Community 15 - "Viewer and Evidence UI"
Cohesion: 0.19
Nodes (16): api_chunk(), api_chunks(), api_config_presets(), api_img(), _cam(), corpus(), links(), _png() (+8 more)

### Community 16 - "Streaming Run Events"
Cohesion: 0.14
Nodes (11): api_live_stream(), api_runs(), _LiveRun, _provider_models(), Each provider's default model — 'qwen2.5:14b' is meaningless to Gemini, so the U, One upload's event log, not a mailbox.      It keeps the full event HISTORY plus, Block (in a worker thread) until there is an event past index `i`, or the run en, Replay everything this run has already emitted, then stream the rest.      Repla (+3 more)

### Community 17 - "Guideline Acquisition"
Cohesion: 0.23
Nodes (16): Source, _download(), fetch(), _fetch_one(), _pages(), Path, Download the guideline corpus into cfg.paths.guidelines. Re-runnable; verifies c, Page count, or 0 if the bytes are not a readable PDF. (+8 more)

### Community 18 - "Run and Subject Artifacts"
Cohesion: 0.17
Nodes (16): api_subject(), api_subjects(), _clean(), cohort(), corpus_sha256(), index(), Path, Hash of the corpus THIS SERVER is resolving citations against.      A chunk_id i (+8 more)

### Community 19 - "Recommendation Grade Tests"
Cohesion: 0.21
Nodes (13): extract_cor_loe(), (class_of_recommendation, level_of_evidence), normalised onto the ESC scale., detect_conflicts(), Contradicting thresholds among retrieved passages (C11).      Resolved by recenc, _chunk(), Retrieval + corpus tests.  The slow ones load MedCPT and touch the real corpus,, test_acc_aha_is_normalised_onto_the_esc_scale(), test_conflict_is_detected_and_resolved_by_recency() (+5 more)

### Community 20 - "Hybrid Retrieval Logic"
Cohesion: 0.15
Nodes (13): _attrib(), build_query(), Agent 3a — retrieval over the guideline corpus.  HYBRID, NOT DENSE-ONLY, and tha, Agent 2's numbers -> a retrieval query.      Deliberately verbose and clinical:, The (direction, value) cut-points a passage actually asserts., A passage's clinical topic — two cut-points only compete if they answer one ques, The shared topic of two passages, or empty if they are not answering the same qu, _same_question() (+5 more)

### Community 21 - "Grounding Enforcement"
Cohesion: 0.18
Nodes (13): enforce(), fidelity(), _numbers_in(), Any, RuntimeError, Guardrails — hallucination enforcement and the security boundary.  The honest fr, Neutralise structural tokens that could break out of the passage delimiter., Every number the report ASSERTS. Deliberately not every digit it contains. (+5 more)

### Community 22 - "UI Configuration"
Cohesion: 0.23
Nodes (13): api_config(), api_config_reset(), api_config_set(), api_live_checkpoints(), _coerce(), _config_payload(), effective_cfg(), load_overrides() (+5 more)

### Community 23 - "Embedding Index Build"
Cohesion: 0.24
Nodes (8): cmd_corpus(), build_index(), _Encoder, _paths(), ndarray, Path, Embed every chunk; write artifacts/corpus/{embeddings,index}_{embedder}.{npy,fai, MedCPT is a two-tower model: queries and articles get DIFFERENT encoders.      U

### Community 24 - "Citation Claim Linking"
Cohesion: 0.23
Nodes (13): _link_supports(), Guarantee the invariant Agent 4 depends on: a non-empty `Citation.supports` is a, Citation, Passage-level, in the diagnostic path. BAAI's citations are document-level and, Agent 3's output. Schema-constrained at generation time, so it cannot be malform, Report, A tampered report MUST score below 1.0 on both fidelities, or the headline     m, test_factcheck_catches_a_corrupted_report() (+5 more)

### Community 25 - "Corpus Coverage Checks"
Cohesion: 0.20
Nodes (10): chunk_index(), coverage(), load_chunks(), chunk_id -> Chunk. factcheck.py resolves every Citation against this., Per-document CoR/LoE coverage. Reported as-is — this number is not to be inflate, _grade(), main(), Corpus acceptance report: coverage per document + the acceptance queries.      . (+2 more)

### Community 26 - "Viewer Tests"
Cohesion: 0.24
Nodes (6): _biggest(), Smoke tests for the viewer. Skipped whole if no run exists on this machine — the, H3, asserted: the refusal must survive the round trip to the browser., test_refused_subject_shows_refused_and_has_no_report(), test_subject_detail_returns_a_report(), test_subject_list_returns_rows()

### Community 27 - "NIfTI Upload Pipeline"
Cohesion: 0.27
Nodes (9): api_live_upload(), _best_slice(), ndarray, Bytes from an upload -> (volume, spacing_mm). Rejects anything implausible befor, Optional ground truth for side-by-side comparison. Checked with the SAME     con, _read_upload_mask(), _read_upload_nifti(), _save_upload_volume() (+1 more)

### Community 28 - "Retriever and Fusion"
Cohesion: 0.24
Nodes (6): chunk_id -> Chunk. factcheck.py resolves every citation against this., Reciprocal Rank Fusion: score(d) = sum_r 1 / (k + rank_r(d)).      Score-free —, Retriever, _rrf(), Citation fidelity depends on this: a retrieved chunk_id MUST resolve., test_every_retrieved_chunk_resolves_against_the_index()

### Community 29 - "Panel Requirements"
Cohesion: 0.20
Nodes (10): Agent-to-Agent Security Requirement, Clinical Validity Evaluation Requirement, Document Consistency and Citation Corrections, Error Containment Requirement, Feedback Trigger and Stopping Requirement, Guideline Conflict Resolution Requirement, Proposal Defence Panel Comments, Research Novelty and Scientific Knowledge Requirement (+2 more)

### Community 30 - "Defence Report Claims"
Cohesion: 0.29
Nodes (7): Candidature Defence Report, H1 Grounding Improves Clinical Correctness, Linked Dual Explainability Gap, Passage-Level Grounding Gap, Reported Interim Results, Public-Benchmark Reproducibility Gap, Unresolved Validation Gaps

### Community 31 - "Default Pipeline Design"
Cohesion: 0.33
Nodes (6): Boundary-Aware Orchestration, CineMA Three-Seed Ensemble, Default Pipeline Configuration, Hybrid MedCPT Retrieval, Calibrated Plausibility Gate, Qwen Grounded Report Generation

### Community 32 - "Engineering and Reproducibility"
Cohesion: 0.33
Nodes (6): Artifact Reproducibility Spine, CMR Engineering Plan, Fourteen Engineering Work Packages, Thin Vertical Slice First, Typed Agent Contracts, H3 Failure Visibility Through Decomposition

### Community 33 - "nnU-Net Training"
Cohesion: 0.40
Nodes (4): nnUNet_preprocessed, nnUNet_raw, nnUNet_results, train_nnunet.sh script

### Community 34 - "Injection Detection"
Cohesion: 0.40
Nodes (5): Return the names of injection patterns present. Empty list == clean.      Run at, scan_injection(), A defence that quarantines real clinical text is worse than no defence: it would, test_injection_payloads_are_detected(), test_real_guideline_text_is_not_flagged()

### Community 35 - "Ablation Grid"
Cohesion: 0.40
Nodes (5): Executable Ablation Grid, H1 Grounding Ablation, H2 Feedback Ablation, H3 Gate Ablation, Zero-LLM Template Baseline

### Community 36 - "Security Threat Model"
Cohesion: 0.40
Nodes (5): Bounded Repair and Refusal, Four Grounding Invariants, Hallucination and Security Threat Model, Ingest-Time Injection Quarantine, Semantic Hallucination Remains Open

### Community 37 - "Project Status Claims"
Cohesion: 0.40
Nodes (5): Compute and BAAI Licence Blockers, End-to-End System Complete, CMR Project Status 2026-07-13, Real-Data Results, Remaining Evaluation Gaps

### Community 38 - "Original Thesis Claims"
Cohesion: 0.40
Nodes (5): Four-Agent Thesis Architecture, Original First Integrated CMR Agent Novelty Claim, Proposal Dataset and Metric Claims, Sixteen Panel Corrections, Proposal-Stage Thesis Knowledge Base

### Community 39 - "Proposal Methodology"
Cohesion: 0.40
Nodes (5): CineMA Segmentation Objective, Guideline-to-Segmentation Feedback Loop Claim, Four-Agent System Architecture Slide, Original Proposal Defence Slides, Three-Dataset Methodology Table

### Community 40 - "Viva Dashboard Views"
Cohesion: 0.50
Nodes (4): Linked Evidence View, Live Pipeline View, Run Comparison View, CMR Viva Dashboard

### Community 41 - "Defence Timeline"
Cohesion: 0.50
Nodes (4): BAAI Public-Benchmark Experiment, Candidature Defence Timeline, Candidature Defence Delivery Schedule, Public-Datasets-Only Strategy

### Community 44 - "Human Writing Guidance"
Cohesion: 1.00
Nodes (3): AI Writing Patterns, Draft Audit Final Loop, Humanizer

## Knowledge Gaps
- **73 isolated node(s):** `train_nnunet.sh script`, `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`, `run_all.sh script` (+68 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Templates and Configuration` to `Explainability and XAI`, `nnU-Net Data Conversion`, `BAAI Baseline Integration`, `Quantification and Gate`, `Evaluation Metrics`, `CineMA Segmentation`, `Guideline Corpus Parsing`, `Live Run Management`, `Guideline Acquisition`, `UI Configuration`, `Embedding Index Build`, `Corpus Coverage Checks`?**
  _High betweenness centrality (0.152) - this node is a cross-community bridge._
- **Why does `Chunk` connect `Report Generation` to `BAAI Baseline Integration`, `Guardrail Tests`, `Factuality Scoring`, `Guideline Corpus Parsing`, `Guideline Acquisition`, `Recommendation Grade Tests`, `Hybrid Retrieval Logic`, `Grounding Enforcement`, `Embedding Index Build`, `Citation Claim Linking`, `Corpus Coverage Checks`, `Retriever and Fusion`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Why does `LLM` connect `LLM Client and Schema` to `BAAI Baseline Integration`, `Report Generation`, `Command-Line Workflow`, `Live Run Management`, `Viewer and Evidence UI`, `Streaming Run Events`, `Grounding Enforcement`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `Config` (e.g. with `Deps` and `_Canonical`) actually correct?**
  _`Config` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Measurements` (e.g. with `UngroundedReport` and `test_factcheck_catches_a_corrupted_report()`) actually correct?**
  _`Measurements` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `Chunk` (e.g. with `Grade` and `Source`) actually correct?**
  _`Chunk` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `Subject` (e.g. with `Field` and `_LiveRun`) actually correct?**
  _`Subject` has 6 INFERRED edges - model-reasoned connections that need verification._