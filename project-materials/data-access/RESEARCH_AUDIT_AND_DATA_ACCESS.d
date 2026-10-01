# Research Audit and Private CMR Data-Access Plan

Prepared for Muhammad Tahir  
Last verified: 28 September 2026 (Asia/Kuala Lumpur)  
Scope: all Markdown, PDF and text research documents in this workspace, the implemented CMR prototype, and current official routes for obtaining controlled/private CMR data.

## 1. Executive decision

The project is a promising working research prototype, but the present evidence does not yet support the strongest novelty, clinical-performance, or completion claims in the documents.

The defensible contribution is not that segmentation, multi-agent orchestration, RAG, refusal, or explainability is individually new. It is the CMR-specific integration of:

1. segmentation and deterministic ventricular quantification;
2. guideline-only retrieval with recommendation class and level of evidence;
3. an enforceable refusal path after failed segmentation;
4. boundary-triggered remeasurement;
5. claim/citation/image evidence packaging; and
6. evaluation on auditable public CMR benchmarks.

A private cohort would materially improve external validity and answer the panel's data-collection concern, but only if it is acquired under a proper protocol and evaluated correctly.

### Confirmed supervisor decisions (28 September 2026)

1. Access to an existing external controlled/private dataset is acceptable for the degree requirement.
2. The study will not add or engage a medical doctor as a collaborator or investigator.

These decisions remove the proposed UMMC patient-cohort route. Under the published UMMC procedure, a student outside the Faculty of Medicine may need UMMC/FOM staff to act as PI; without such involvement, UMMC patient data should not be pursued. The viable route is therefore secondary use of an external controlled dataset through its formal custodian or repository. Contact with an author or administrator solely to request data access does not make that person a study collaborator.

Recommended strategy:

- Primary route: request secondary-data access from the BAAI Cardiac Agent authors, whose paper explicitly states that de-identified data may be shared for non-commercial academic purposes under an MTA and DUA. Request access, not co-authorship or clinical participation.
- Structured backup routes: SCMR Registry, MESA/BioLINCC and NAKO. HCMR is appropriate only if the thesis narrows to HCM and its custodian accepts a non-clinical applicant.
- Institutional route: the supervisor remains the academic lead; obtain a written UM ethics/jurisdiction determination and any required institutional signature without adding a doctor to the research team.
- Do not ask anyone to email raw patient scans. Access must be through an approved institutional or secure platform.

No one can honestly guarantee that a custodian will release data. What can be made accurate is the request, governance route, contact source, and scientific plan. Every contact below was verified against an official page or the corresponding paper as of the date above.

## 2. What was audited

The audit covered 44 research documents present at the time of review:

- 11 Markdown documents;
- 27 PDFs; and
- 6 text documents.

The audit also inspected the implementation, experiment outputs, tests, configuration and repository state. The generated knowledge-graph artefacts are in `graphify-out/` and are not source evidence by themselves.

## 3. Critical scientific and implementation findings

### 3.1 Findings that must be fixed before strong claims are made

1. **Voxel spacing can silently default to 1 mm.** The graph calls the segmentation stage without passing spacing, so downstream volumes and mass can be numerically wrong even when masks look correct.

2. **The nnU-Net folds leak patients.** ED and ES samples from the same patient are placed into different folds. The observed overlap is approximately 30–36 patients per fold. All phases belonging to one patient must remain in the same split.

3. **The 830-study “full-cohort” execution uses ground-truth masks.** Perfect or near-perfect quantification and validation metrics there are pipeline checks, not evidence of real model performance. The 11 refusals indicate rejection of provided annotations, not predicted segmentation failures.

4. **There is no current LLM evaluation for the 6,577-chunk corpus.** The six generated reports belong to an older corpus/configuration and cannot validate the present retrieval system.

5. **The experimental matrix is incomplete.** Only 2 of 12 intended configuration rows have usable artefacts; the full segmentation runs, BAAI comparison, XAI artefacts and clinician reader study are incomplete. The nnU-Net run is only a one-epoch smoke test.

6. **Heart-failure categories are clinically overclaimed.** Ejection fraction can categorize EF ranges, but heart failure is a clinical syndrome. EF alone cannot diagnose HFrEF, HFmrEF or HFpEF.

7. **The pathology top-1 evaluator does not reliably evaluate the model's top diagnosis.** It uses a fixed phrase-priority rule. For example, “DCM most likely; HCM less likely” is scored as HCM. Parse the structured top diagnosis directly and test it.

8. **Experiment identifiers are unsafe.** `config_id` hashes only the experiment-row fields, not the inherited base configuration, so materially different experiments can share an ID and overwrite one another.

### 3.2 Methodological gaps

- The segmentation gate is calibrated and evaluated on the same 830 cases; it needs independent calibration/development/test partitions.
- There is no independent annotation-quality assessment or adjudication protocol.
- Recommendation class and level of evidence extraction have no manually labelled gold set.
- Retrieval has no query-level relevance benchmark with measures such as Recall@k, MRR or nDCG.
- Numeric and citation checks establish string-level consistency but not clinical entailment or whether a citation actually supports the claim.
- The XAI proposal is heuristic and lacks localization or clinician validation.
- Entropy is used as uncertainty without calibration or a demonstrated relationship to error.
- H3 lacks a controlled corruption/failure-injection benchmark.
- H2 lacks paired predicted-mask experiments around genuine clinical decision boundaries.
- Cine CMR alone is insufficient for many intended disease diagnoses. Scar, myocarditis, infiltration, perfusion defects and several cardiomyopathies may require LGE, T1/T2 mapping, perfusion, clinical history, ECG or laboratory data.
- Between-configuration statistical comparisons are not yet implemented or reported.
- The reader study is absent, so claims of clinical usefulness, fidelity or expert acceptability are premature.

### 3.3 Reproducibility and software gaps

- Fast test suite result: 108 passed, 4 skipped, 10 deselected.
- Full suite result during audit: 110 passed, 4 skipped and 8 failed; the failures were tied to slow/legacy data and unavailable local Ollama services, so they require environment-specific resolution rather than being ignored.
- Ruff passed.
- LangGraph was missing from the audited environment even though orchestration depends on it.
- The nested code repository had no commits and all files were untracked; provenance was therefore not auditable.
- Dataset, checkpoint, corpus and model hashes are not systematically recorded.
- The HPC script contains a literal `$USER` path problem.
- `allow_remote` is configured but apparently unused.
- The report viewer is not genuinely read-only.
- `subject_id` can be used in filesystem paths without adequate traversal protection.
- Dense-index staleness is checked only by chunk count, not by content/configuration hash.
- Chunks use word counts even though the encoder has token limits; some chunks can exceed the intended model input.

### 3.4 Documentary inconsistencies

The documents contain mutually inconsistent statements, including:

- corpus size: 2,592 vs 3,397 vs 6,577 chunks;
- source count: 5 vs 12 vs 17 guideline documents;
- completed configurations: 10 vs 12;
- gate calibration set: 180 vs 830;
- success rate: 99% vs 98.7%;
- claim that every chunk has recommendation class/level of evidence, while the observed coverage is roughly 40%;
- SCMR “2025” wording despite inclusion of 2020 material;
- approximately 7,000 vs approximately 9,671 lines of code;
- “122 tests” versus the actual selected/excluded test counts; and
- “complete” language while core predicted-mask, baseline, XAI and reader-study evidence remains pending.

These numbers must be generated from one versioned manifest rather than manually copied into several documents.

## 4. Defensible wording for the current state

Use wording close to the following until the missing experiments are complete:

> A functioning research prototype and reproducible evaluation framework was developed for integrating CMR segmentation, deterministic quantification, guideline retrieval, constrained reporting, refusal, and evidence packaging. Ground-truth vertical-slice execution was demonstrated on 830 public studies, while predicted-mask evaluation, hypothesis testing, baselines, XAI validation, and expert clinical assessment remain incomplete.

Do not call ground-truth-mask execution “end-to-end model validation.” Do not claim clinical diagnostic accuracy from Dice, MAE, BERTScore, string matching, or EF categories.

## 5. Why private/custom data could help—and what it will not fix by itself

A properly designed independent cohort can provide:

- true external testing on a new site/vendor/population;
- evidence about domain shift and refusal behaviour;
- clinical reports and richer labels unavailable in ACDC/M&Ms;
- an opportunity for blinded clinician adjudication;
- a stronger answer to the panel's proprietary-data recommendation; and
- potentially novel Malaysian/Asian external validation.

It will not fix:

- leakage in the training splits;
- invalid evaluators;
- missing predicted-mask experiments;
- unsupported novelty language;
- the absence of a retrieval gold set;
- the absence of statistical analysis; or
- diagnosis claims made from sequences that do not contain the required evidence.

## 6. Exact data specification to agree before contacting custodians

The request should be a one-page data specification, not “please give me your dataset.” At minimum define:

### 6.1 Core cine/segmentation package

- de-identified short-axis cine CMR covering the ventricles from base to apex;
- full temporal sequence, with ED and ES frames identified;
- DICOM preferred if scanner metadata are required, otherwise NIfTI plus a metadata table;
- pixel spacing, slice thickness/gap, temporal resolution and orientation;
- scanner vendor, model, field strength, site and acquisition date/year band;
- patient-level pseudonymous ID with no direct identifiers;
- age band or age, sex and body-surface area if permitted;
- final clinical diagnosis and the method by which it was established;
- clinical report, de-identified, if report generation is to be evaluated;
- LV cavity, LV myocardium and RV cavity contours at ED/ES for at least a prespecified annotated subset;
- annotator expertise, contouring guideline, software and adjudication method; and
- explicit licence/DUA permission for model development, evaluation, publication of aggregate results and—only if permitted—release of derived code/models.

### 6.2 If diagnostic claims go beyond ventricular function

Request the modality actually needed:

- LGE for scar/infarction/fibrosis;
- T1/T2 maps for tissue characterization;
- first-pass perfusion for ischemia/perfusion claims;
- long-axis cine and valve sequences for relevant structural/valvular findings;
- ECG, laboratory and clinical phenotype variables where the diagnosis requires them.

If these are unavailable, restrict the research question to segmentation, quantification, quality control and report grounding. Do not infer disease labels that cine images cannot support.

### 6.3 Sample and split design

- Obtain a feasibility count first; do not promise an arbitrary sample size.
- Calculate the final sample using the primary endpoint and expected effect/precision.
- Split by patient, never by frame or slice.
- Preserve one locked external test set that is never used for threshold tuning.
- Stratify or report by centre, vendor, diagnosis and key demographics where sample size permits.
- Predefine exclusions and report the full inclusion flow.

## 7. The approval chain: who approves what

External researchers cannot grant Universiti Malaya ethics approval. The roles are different:

1. **Supervisor/programme:** confirms that the cohort answers the degree requirement and approves the study direction.
2. **Appropriate UM ethics/governance office:** gives ethics review, exemption/waiver determination, a not-human-subjects determination, or routes the application to the correct committee.
3. **Dataset applicant/academic PI:** normally the supervisor or another eligible non-clinical UM academic under the repository's rules; this role need not be a medical doctor unless the custodian specifically requires one.
4. **Data custodian/hospital/registry:** decides whether the requested data may be released or accessed.
5. **Institutional legal/authorized signatories:** execute the DUA, MTA, collaboration agreement or licence.
6. **Information-security/privacy officers:** approve storage, transfer, access controls and any cross-border processing.

For UMMC patient data, the official UMMC Medical Research Ethics Unit states that all medical research involving humans requires UMMC-MREC approval. The published eligibility procedure also states that a student outside the Faculty of Medicine should appoint UMMC/FOM staff to apply as PI; confirm that this remains the current rule before submission.

Required sequence:

1. Freeze the research question and primary endpoint.
2. Record that the supervisor accepts an existing external controlled/private dataset and does not want a doctor added to the study.
3. Prepare the protocol, statistical plan, data-management plan and requested-variable list.
4. Ask the relevant UM office, in writing, which committee has jurisdiction over secondary analysis of externally sourced, de-identified medical images with no recruitment and no UMMC data. Do not assume either UMREC-NM or UMMC-MREC will accept it.
5. Seek the resulting ethics approval, exemption or formal determination before receiving or analysing controlled data.
6. Obtain custodian approval and execute the DUA/MTA through authorized institutional signatories.
7. Set up the approved secure environment and access log.
8. Receive only the approved fields through the approved transfer route.
9. Lock the analysis plan and external test set before looking at outcomes.

No pilot extraction of patient data should begin before the required approval. Do not put raw scans in personal cloud storage, personal email, public Git, an unapproved LLM service or a consumer file-sharing account.

## 8. Local contacts and official routes — reference only, not the selected data route

The supervisor has decided not to engage a medical doctor in the study. Therefore, the following clinical contacts must not be approached as collaborators under the current plan. They are retained only as an audit trail showing why the UMMC route was considered and rejected.

### Tier L1 — best local clinical/data-feasibility route

1. **Dr Nor Ashikin Md Sari — Department of Medicine, Faculty of Medicine, UM**  
   Email: `ashikin77@um.edu.my`  
   Profile: https://umexpert.um.edu.my/ashikin77.html  
   Why: formal subspecialty training in cardiovascular magnetic resonance; UM profile lists MRI expertise, SCMR Level 3 competency, CMR research and supervision of multiple UMMC CMR studies. This is the strongest first clinical introduction for feasibility and potential PI/co-investigator routing.

2. **Professor Dato' Dr Yang Faridah Abdul Aziz — Department of Biomedical Imaging, UM**  
   Email: `yangf@um.edu.my`  
   Profile: https://umexpert.um.edu.my/yangf  
   Why: consultant radiologist whose official profile states that she helped establish cardiac CT/MRI services at UMMC/UMSC and has cardiac imaging as a research niche. Appropriate for imaging-department collaboration and custodian routing.

3. **Associate Professor Dr Alexander Loch — Department of Medicine, Faculty of Medicine, UM**  
   Email: `aloch@um.edu.my`  
   Profile: https://umexpert.um.edu.my/aloch.html  
   Why: cardiologist with stated cardiac-MRI and heart-failure expertise. Useful if the intended endpoint includes ventricular function or heart-failure phenotyping.

4. **Professor Dr Khairul Azmi Abd Kadir — Head, Department of Biomedical Imaging**  
   Email: `khriazmi@um.edu.my`  
   Department page: https://medicine.um.edu.my/biomedical-imaging-department  
   Why: departmental authorization/feasibility route. The department page also describes UMRIC as facilitating research requiring imaging services.

Current decision: do not send collaboration requests to these four clinicians. Reconsider the UMMC route only if the supervisor later changes the no-doctor constraint.

### Tier L2 — ethics, institutional and custodian routes

5. **UMMC Medical Research Ethics Unit / UMMC-MREC — not the default route without UMMC data**  
   Emails: `ummc-mrec@ummc.edu.my`, `ku_uepp@ummc.edu.my`  
   Phone: +60 3-7949 3209 / 8473 / 4656  
   Official page: https://www.ummc.edu.my/department/department_sub.asp?kodjabatan=2h9G7o7c  
   Purpose: confirm eligibility, PI requirement, review category, current submission portal and required documents. This committee reviews ethics; it is not the dataset owner.

6. **Faculty of Medicine Research Office**  
   Email: `resfom@um.edu.my`  
   Official page: https://resfom.um.edu.my/about  
   Purpose: help locate the appropriate FOM/UMMC collaborator and institutional process.

7. **Faculty of Medicine Research Management Unit**  
   Email: `grants.fom@um.edu.my`  
   Official page: https://medicine.um.edu.my/research-management-unit  
   Purpose: research-management and collaboration support.

8. **UMMC Data Protection Officer / Patient Information Department**  
   Email: `dpo@ummc.edu.my`  
   Phone: +60 3-7949 2816  
   Official notice: https://www.ummc.edu.my/pesakit/PDPA.asp?kodBM=2G9q3V  
   Purpose: privacy/security questions. Do not use this as a substitute for MREC or the data-custodian application.

9. **UMREC–Non-Medical — jurisdiction enquiry, not assumed approval route**  
   Email: `umrec@um.edu.my`  
   Official page: https://umresearch.um.edu.my/umrec/  
   Purpose: ask where externally sourced, already de-identified secondary medical-imaging research should be reviewed. Its official page says it does not accept patient studies, so obtain a written routing answer rather than claiming that UMREC-NM is automatically appropriate.

The UMMC Research, Development and Innovation department also processes requests for use of hospital clinical/non-clinical data. This route is not applicable unless the decision changes and UMMC data are pursued.

## 9. Ranked international data custodians and researchers

### Tier I1 — highest scientific fit

1. **BAAI Cardiac Agent corresponding authors**  
   Paper: https://arxiv.org/abs/2604.04078  
   Repository: https://github.com/plantain-herb/Cardiac-Agent  
   Public evaluation data: https://huggingface.co/datasets/TaipingQu/CMRAgentEvalSet  
   Public multi-sequence segmentation data: https://huggingface.co/datasets/TaipingQu/CMR-MULTI  
   Corresponding contacts, verified from the paper PDF:
   - Ruifang Yan — `yrf718@163.com`
   - Zhongyuan Wang — `zhongyuan@baai.ac.cn`
   - Tiejun Huang — `tjhuang@baai.ac.cn`
   - Lei Xu — `leixu2001@hotmail.com`
   - Henggui Zhang — `henggui.zhang@gmail.com`

   Why: their private two-hospital cohort of 2,413 patients and their multimodal CMR-agent work are the closest direct comparator to this thesis. Their data-availability statement says de-identified data can be shared for non-commercial academic purposes, subject to institutional policy, privacy/IP review, a formal MTA and DUA, by contacting the corresponding authors.

   Contact rule: send one message to the corresponding-author group, preferably from the supervisor's institutional email, naming the smallest useful subset rather than requesting all 2,413 cases. Ask first for ordinary controlled access or analysis within their secure environment. If they require a clinical collaboration or co-investigator and the supervisor maintains the no-doctor decision, decline that condition and use another repository.

2. **SCMR Registry**  
   Registry Chair: Dr Dipan Shah  
   Program Manager: Chelsea Smart  
   Official access page and contact links: https://scmr.org/scmr-registry/  
   Why: a purpose-built global registry containing de-identified CMR images, clinical variables and outcomes, with a formal research-access process. The process is: read the policy, submit a Search Request, undergo review, then submit a Data Access Application. Listed Search Request deadlines are 1 January, 1 April, 1 July and 1 October.

   Use the page's official email links/forms rather than guessing addresses; the public crawler redacts the email strings.

3. **HCMR (Hypertrophic Cardiomyopathy Registry)**  
   Overall PI: Professor Christopher M. Kramer — `ckramer@virginia.edu`  
   Study site: https://hcmregistry.org/  
   Protocol record: https://clinicaltrials.gov/study/NCT07054073  
   Why: international HCM registry with cine CMR, LGE, T1 mapping, biomarkers and outcomes. A published data-availability statement says de-identified imaging may be shared on reasonable request subject to ethics, consent, data-protection governance, agreements and funding. This is suitable for a disease-specific external-validation study, not a balanced five-class benchmark.

4. **MESA / NHLBI BioLINCC**  
   Coordinating Center: `chsccweb@u.washington.edu`  
   Request page: https://biolincc.nhlbi.nih.gov/studies/mesa/  
   Why: multi-ethnic, six-centre cohort with baseline CMR ventricular mass/function and an MRI RV-function ancillary study. BioLINCC has a formal Request Data route. Verify whether the required raw image files—not just derived study tables—are available before writing a proposal.

### Tier I2 — strong controlled-access backups

5. **NAKO German National Cohort**  
   Scientific project management: `wpm@nako.de`  
   Research data management: `fdm@nako.de`  
   TransferHub: https://transfer.nako.de/transfer/index  
   Research page: https://nako.de/forschung/  
   Why: more than 205,000 participants overall and a substantial MRI subset. The official route is the TransferHub plus Use & Access review. Ask specifically which cardiac cine sequences, labels and export modes are available to an institution in Malaysia.

6. **Cardiac Atlas Project (CAP)**  
   General email: `cardiacatlasproject@gmail.com`  
   Team: Professor Alistair Young and Dr Avan Suinesiaputra  
   Contact: https://www.cardiacatlas.org/contact-us/  
   Why: multiple CMR collections and formal dataset-request routes. Dataset-specific cautions:
   - DETERMINE has 450 cine plus LGE cases but is currently marked temporarily unavailable while its sharing agreement is renewed: https://www.cardiacatlas.org/determine/
   - SCMR Consensus data provide DICOM and metadata but do not release gold consensus contours; it is an external-validation service: https://www.cardiacatlas.org/scmr-consensus-contours/request-scmr-consensus-data/
   - biv-me rescans are ten healthy repeat-scan subjects, useful for reproducibility but not diagnosis. Contact Joshua Dillon at `joshua.dillon@auckland.ac.nz`: https://www.cardiacatlas.org/biv-me-rescans/

7. **Oxford Centre for Clinical Magnetic Resonance Research (OCMR)**  
   Operations contact: Marcin Grzegorczyk — `marcin.grzegorczyk@cardiov.ox.ac.uk`  
   Official contact: https://www.rdm.ox.ac.uk/about/our-facilities-and-units/oxford-centre-for-clinical-magnetic-resonance-research/how-to-find-ocmr  
   Why: relevant CMR research centre. A published data-availability statement says de-identified clinical imaging may be shared subject to ethics, consent, governance, DUA and available funding. Ask for the correct scientific data custodian rather than assuming the operations manager can approve access.

### Tier I3 — useful but not the immediate private-data answer

8. **UK Biobank**  
   Official access page: https://www.ukbiobank.ac.uk/use-our-data/apply-for-access/  
   Researcher contact: use the Community “Submit a request” link on that page.  
   Why: large cardiac imaging resource and strong population-scale external validation. Current limitation: the official page says new applications are paused and intended to reopen in late 2026. It is also a controlled-access resource, not “custom data collected by the candidate.”

9. **M&Ms challenge organizers / public benchmark**  
   Santi Seguí — `santi.segui@ub.edu`  
   Sergio Escalera — `sergio.escalera.guerrero@gmail.com`  
   Paper: https://doi.org/10.1109/TMI.2021.3090082  
   Why: appropriate only for questions about the existing public multi-centre/multi-vendor benchmark and permitted access. It will not satisfy a proprietary-cohort requirement by itself.

Use professional institutional routes whenever available. Personal addresses are included only where they are the authors' published correspondence addresses. Do not use LinkedIn as the first approach when an official email or portal exists.

## 10. Contact priority for this thesis

The recommended order is:

1. document the two confirmed supervisor decisions in the research log;
2. ask the supervisor to act as or nominate the eligible academic applicant for repository and institutional paperwork;
3. obtain a written UM ethics/jurisdiction determination for external de-identified secondary CMR data;
4. send one formal BAAI corresponding-author request for data access without requesting collaboration;
5. submit an SCMR Registry Search Request if its applicant rules permit this non-clinical team;
6. submit MESA/BioLINCC and NAKO feasibility enquiries;
7. use HCMR only if the thesis narrows to HCM and its access rules fit the team;
8. check UK Biobank when applications reopen.

Do not send all nine approaches at once. Track each request, wait 10–14 business days, send one polite follow-up, and then move to the next route. Custodians may require months, not days.

## 11. Email templates

### 11.1 Supervisor confirmation for the research record

**Subject:** Confirmation of agreed private-data route for my thesis

Dear Dr [Supervisor surname],

Thank you for confirming that an existing external controlled/private dataset is acceptable for the panel's data requirement and that we will not add a medical doctor as a collaborator or investigator.

I will therefore not pursue a UMMC patient cohort. I will prepare a controlled secondary-data request, beginning with the BAAI Cardiac Agent custodians, followed by formal registry routes if needed. Before receiving any data, I will obtain the appropriate UM ethics/jurisdiction determination and ensure that any DUA/MTA is signed through the authorized university process.

Please confirm that you are willing to be the academic applicant/PI for these repository and institutional submissions, or nominate the appropriate eligible UM academic if the repository requires a staff applicant.

The proposed primary use remains a locked external evaluation of the CMR pipeline, not model tuning. I will send you the final two-page protocol, variable list, statistical analysis plan and data-management plan for approval before contacting custodians.

Kind regards,  
Muhammad Tahir  
[programme, department, student ID, institutional email]

### 11.2 Archived local UMMC clinical-collaborator email — do not send under the current decision

This template is retained only in case the supervisor later changes the no-doctor decision. It is not part of the active plan.

**Subject:** Supervisor-supported feasibility enquiry: de-identified UMMC cardiac MRI cohort for external AI validation

Dear Dr [Surname],

My name is Muhammad Tahir, a [degree/programme] student at Universiti Malaya supervised by Dr [name]. I am developing a research prototype that integrates CMR segmentation, deterministic ventricular quantification, guideline-grounded reporting and explicit refusal when image analysis fails.

I am writing to ask whether you would be willing to discuss a small, formally approved retrospective external-validation study using de-identified UMMC CMR data. I am not requesting data by email and would not access any patient information before the required ethics, hospital and custodian approvals.

The proposed primary endpoint is [state one endpoint, e.g. patient-level LV/RV/myocardium segmentation accuracy and ventricular-volume error on a locked external test cohort]. The minimum feasibility query is:

- number of eligible studies in [date range];
- availability of complete short-axis cine CMR and ED/ES identification;
- vendor/field-strength distribution;
- availability and provenance of clinical diagnosis and reports; and
- availability of expert contours for a subset, or the feasibility of blinded annotation/adjudication.

If broader disease classification is considered, I would restrict it to diagnoses supported by the available CMR sequences and clinical reference standard.

Would you be open to a 20-minute meeting with my supervisor to advise on feasibility, the appropriate UMMC/FOM PI, and the current UMMC-MREC/JPPI-RDI route? I can send a one-page protocol, variable list, security plan and proposed authorship/contribution discussion in advance.

Thank you for considering this request.

Kind regards,  
Muhammad Tahir  
[programme, department, student ID]  
[UM email and phone]  
Supervisor: [name, title, email]

### 11.3 External author/custodian email

**Subject:** Non-commercial academic CMR data-access enquiry for independent external validation

Dear Professor/Dr [Surname],

I am Muhammad Tahir, a [degree] researcher at Universiti Malaya, supervised by Dr [name]. I am studying an auditable CMR workflow combining segmentation, ventricular quantification, guideline-grounded reporting and failure-aware refusal.

Your [paper/dataset name] is particularly relevant because [one precise sentence showing that the work was read]. I would like to ask whether a de-identified subset could be accessed for a non-commercial, locked external-validation study, or whether the analysis could be performed within your secure environment.

I am requesting only the minimum variables needed for the prespecified endpoint: [sequence], [ED/ES or labels], [metadata], [reference standard] and [requested sample strata]. The data would not be redistributed, uploaded to public services, or used beyond the approved protocol. We are prepared to obtain the required UM ethics determination, execute your DUA/MTA, and follow your security, acknowledgement and publication rules. At present, we are seeking controlled data access rather than asking your clinical investigators to join our study.

Could you please advise:

1. whether these data and labels are available;
2. the formal proposal and review route;
3. whether international access from Malaysia is permitted;
4. whether analysis must remain on-site or in a secure platform;
5. required ethics/IRB documents and agreements; and
6. expected fees and review timeline?

I can provide a two-page protocol, data-variable table, analysis plan, ethics status and supervisor letter immediately.

Kind regards,  
Muhammad Tahir  
[institutional details]  
Supervisor: [name and institutional contact]

### 11.4 UM ethics-jurisdiction enquiry for external secondary data

**Subject:** Ethics-jurisdiction enquiry: externally sourced de-identified CMR secondary data

Dear Research Ethics Secretariat,

I am a [degree/programme] student from [faculty] at Universiti Malaya. With my supervisor, I am planning secondary analysis of an existing controlled dataset containing de-identified cardiac MRI images and associated labels supplied by an external international custodian. The study will involve no recruitment, no contact with patients, no UMMC data and no attempt to re-identify individuals.

No data have been extracted or accessed. Before preparing the full submission, I would be grateful for confirmation of:

1. which UM committee or office has jurisdiction over this project;
2. whether a full review, exemption, waiver or formal non-human-subjects determination is required;
3. whether my UM academic supervisor may act as the applicant/PI;
4. the correct submission platform and document set; and
5. required privacy, international-transfer and storage documents for analysis on approved UM infrastructure.

I can provide the protocol synopsis, variable list and data-management plan if helpful.

Kind regards,  
Muhammad Tahir  
[institutional details and supervisor]

## 12. Documents to prepare before any data request

Prepare one consistent package:

1. two-page protocol synopsis;
2. precise research question and primary endpoint;
3. inclusion/exclusion criteria and feasibility variables;
4. statistical analysis plan and sample-size justification;
5. data dictionary with direct identifiers explicitly excluded;
6. data-flow diagram from custodian to analysis to destruction/archival;
7. storage, encryption, access-control, backup and breach-response plan;
8. model/data governance statement, including prohibition on unapproved cloud/LLM uploads;
9. publication, authorship and derived-output plan;
10. supervisor support letter;
11. CVs and training certificates required by the reviewing body;
12. ethics approval/determination when available; and
13. draft DUA/MTA points: purpose, users, location, duration, onward sharing, IP, publication review, incident reporting, destruction and audit.

## 13. Scientifically valid private-cohort design

Use the private cohort primarily as a locked external test, not as another tuning set.

Suggested hierarchy of endpoints:

1. **Primary:** patient-level segmentation/quantification performance using predicted masks, with confidence intervals.
2. **Secondary:** gate sensitivity/specificity for detecting unacceptable segmentations against blinded expert QC.
3. **Secondary:** numeric report fidelity against deterministic measurements.
4. **Secondary:** retrieval/citation correctness against a manually labelled query/claim set.
5. **Exploratory:** disease classification only where the available modalities and reference standard genuinely support it.

Minimum safeguards:

- patient-level split and locked analysis;
- blinded expert adjudication;
- no threshold selection on the external test cohort;
- predeclared handling of failed/unreadable studies;
- report confidence intervals, calibration and subgroup results;
- compare against a simple baseline and, where feasible, a strong published baseline;
- disclose missing data and selection bias;
- retain failure cases rather than silently excluding them; and
- document every deviation from the protocol.

Under the confirmed no-doctor decision, remove the clinician reader study from the active protocol and do not claim expert clinical validation. If the decision changes later, obtain a separate ethics determination and predefine the rubric, sampling, blinding, washout, conflicts and inter-rater agreement.

## 14. Private-data risk register

| Risk | Consequence | Mitigation |
|---|---|---|
| External private data does not meet panel's “own collection” requirement | Months lost without answering C9 | Obtain written supervisor/programme clarification first |
| No doctor/clinical investigator by supervisor decision | UMMC cohort and clinician reader study are unavailable; some custodians may also reject the team | Use external repositories that accept a non-clinical academic PI; abandon any route that makes clinical collaboration mandatory |
| Cine-only cohort used for broad diagnosis | Invalid clinical claims | Restrict outcomes or request LGE/T1/T2/perfusion/clinical reference data |
| No manual contours | Cannot evaluate segmentation objectively | Fund or negotiate a blinded annotated subset and adjudication |
| Small or imbalanced cohort | Unstable disease-performance estimates | Feasibility count, power/precision calculation, narrower endpoint |
| Data transfer prohibited | No local copy | Offer federated/on-site/secure-platform analysis |
| DUA prohibits model release | Reproducibility constrained | Negotiate release of code and aggregate results; disclose limits |
| Labels derived only from reports | Circular/noisy reference standard | Define independent diagnosis adjudication and provenance |
| Ethics and custodian approval confused | Unauthorized research | Obtain both approvals and institutional signatures |
| Long approval timeline | Degree delay | Run public-data fixes and controlled-access applications in parallel |

## 15. Immediate action plan

### In the next 48 hours

1. Save the supervisor's two decisions in writing using Section 11.1.
2. Confirm that the supervisor will be the academic applicant/PI or will nominate an eligible UM academic; no doctor is required unless a particular custodian makes that a condition.
3. Reduce the private-cohort purpose to one primary endpoint.
4. Prepare the one-page variable list from Section 6.
5. Fix the patient leakage, spacing propagation, evaluator and experiment-ID defects before new-data analysis.

### In the next two weeks

1. Obtain a written UM ethics/jurisdiction determination for externally sourced, de-identified secondary CMR data with no UMMC data and no recruitment.
2. Submit one BAAI data-access enquiry without asking the authors to join the study.
3. Submit one SCMR feasibility/Search Request only if its eligibility terms fit the team.
4. Build the ethics-ready protocol, analysis plan and data-management plan.
5. Freeze one current corpus/configuration and rerun the LLM/retrieval evaluation on it.

### Before receiving any controlled data

1. Written ethics approval, exemption or formal determination as required.
2. Written data-custodian approval.
3. Fully executed DUA/MTA/collaboration agreement by authorized institutional signatories.
4. Approved storage and transfer mechanism.
5. Named-user access list and audit log.
6. Locked protocol, outcome definitions and test plan.

## 16. Priority remediation order for the thesis

1. Correct spacing, patient leakage, pathology parsing and experiment identity.
2. Freeze dataset/corpus/model manifests and reconcile every reported count.
3. Separate calibration, development and locked evaluation data.
4. Create manual gold sets for retrieval, recommendation class/level of evidence and citation entailment.
5. Complete predicted-mask segmentation and downstream evaluation.
6. Redesign H2 and H3 as paired, controlled experiments.
7. Run the declared baselines and statistical comparisons.
8. Validate uncertainty and XAI rather than treating them as self-evident explanations.
9. Complete an ethics-approved expert study or clearly defer clinical validation.
10. Rewrite novelty, results and completion claims to match the evidence.

## 17. Source register

Primary/official sources used for the data-access section:

- UMMC Medical Research Ethics Unit: https://www.ummc.edu.my/department/department_sub.asp?kodjabatan=2h9G7o7c
- UMMC published initial-review eligibility procedure: https://www.ummc.edu.my/files/ethic/2.%20INITIAL%20REVIEW%20PROCEDURE_for%20web.pdf
- UMREC–Non-Medical: https://umresearch.um.edu.my/umrec/
- UMMC privacy/data-protection notice: https://www.ummc.edu.my/pesakit/PDPA.asp?kodBM=2G9q3V
- UM Faculty of Medicine Research Office: https://resfom.um.edu.my/about
- UM Biomedical Imaging Department: https://medicine.um.edu.my/biomedical-imaging-department
- Nor Ashikin Md Sari profile: https://umexpert.um.edu.my/ashikin77.html
- Alexander Loch profile: https://umexpert.um.edu.my/aloch.html
- Yang Faridah Abdul Aziz profile: https://umexpert.um.edu.my/yangf
- BAAI Cardiac Agent: https://arxiv.org/abs/2604.04078
- SCMR Registry: https://scmr.org/scmr-registry/
- HCMR: https://hcmregistry.org/
- HCMR protocol record: https://clinicaltrials.gov/study/NCT07054073
- MESA/BioLINCC: https://biolincc.nhlbi.nih.gov/studies/mesa/
- NAKO research access: https://nako.de/forschung/
- Cardiac Atlas Project: https://www.cardiacatlas.org/contact-us/
- OCMR official contact: https://www.rdm.ox.ac.uk/about/our-facilities-and-units/oxford-centre-for-clinical-magnetic-resonance-research/how-to-find-ocmr
- UK Biobank access: https://www.ukbiobank.ac.uk/use-our-data/apply-for-access/

Source-status note: websites, staff roles, access pauses, deadlines and contact details can change. Re-open the official link immediately before sending or submitting anything. The scientific audit reflects the workspace state at the audit date and must be rerun after material code/data changes.
