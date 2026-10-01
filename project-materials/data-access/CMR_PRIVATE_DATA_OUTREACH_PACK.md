# Controlled/Private CMR Data Outreach Pack

Prepared for Muhammad Tahir, Master of Computer Science by Research, Universiti Malaya  
Supervisor: Dr Uzair Iqbal, Faculty of Computer Science & Information Technology, Universiti Malaya  
Prepared: 29 September 2026 (Asia/Kuala Lumpur); source routes reverified the same day  
Status: Wave 1 enquiries were confirmed sent on 30 September 2026; Wave 2 and later entries remain drafts only

## 1. Executive decision

The most credible strategy is a staged campaign to custodians that explicitly accept research applications or authors whose papers explicitly offer data under a data-use agreement. Sending a generic request to many hospitals is unlikely to work and could damage credibility.

The first wave should contain eight individually addressed enquiries:

1. University of Pennsylvania CMR cohort;
2. BSCMR-AS multi-centre cohort;
3. KORA-MRI;
4. NAKO Health Study;
5. SHIP;
6. CAHHM;
7. Qatar Precision Health Institute/Qatar Biobank; and
8. Jackson Heart Study.

MESA/BioLINCC should be investigated in parallel, but the official catalogue currently lists **study datasets only**, so the initial question must ask whether source CMR images and any reference contours are actually distributable. Its application should use a Universiti Malaya institutional address and normally be led by the supervisor, not a Gmail account.

No custodian can be guaranteed to release data. “Available on reasonable request” is an invitation to apply, not a promise. The safest wording is therefore a short feasibility enquiry followed by the custodian's formal process.

## 2. Non-negotiable boundaries

- The study seeks access to an existing controlled dataset. It is **not** asking the custodian, author, radiologist or cardiologist to join the research team.
- An administrative email to a doctor who happens to be the corresponding author does not itself make that person a collaborator. If the custodian requires co-authorship, a local clinical co-investigator or formal collaboration, that route must be referred back to the supervisor before proceeding.
- Do not claim that ethics approval has already been granted. Say that required UM and custodian approvals will be completed before access.
- Do not request patient-identifiable data, free-text identifiers, exact dates, raw data by ordinary email, or transfer to personal cloud storage.
- Do not send one mass email or expose recipients in CC. Send a separate, personalized message to each target and copy only Dr Uzair Iqbal.
- Do not promise clinical diagnosis from cine CMR alone. The primary use is external validation of segmentation, ventricular quantification and failure-aware quality control. Report/disease endpoints are conditional on permitted data.

## 3. Ranked contact list

### Tier A — send a feasibility enquiry first

| Priority | Target and evidence | Public contact/route | What may be useful | Fit and exact first question |
|---:|---|---|---|---|
| 1 | **University of Pennsylvania CMR cohort**. The 2026 paper reports 2,070 scans from 2,033 unique clinical patients, with 2CH, 3CH, 4CH and short-axis views. Its data-availability statement says Penn data are available under an appropriate DUA. | Rohan Shad, `rohan.shad@pennmedicine.upenn.edu`; [paper and data-availability statement](https://www.nature.com/articles/s41551-026-01637-3) | Multi-vendor clinical CMR and report-derived disease labels; potentially an excellent locked external test cohort. Manual masks are not promised. | **Excellent for external generalization.** Ask whether a de-identified subset, source cine series, reports/labels and any expert contours can be licensed to a UM academic project without adding Penn staff as investigators. |
| 2 | **BSCMR-AS**. The paper reports 599 severe-aortic-stenosis subjects from six UK hospitals and nine scanner types; LV and myocardium have manual labels, and the data statement says the dataset is available on reasonable request. | Chen Chen, `chen.chen15@imperial.ac.uk`; [paper and explicit availability statement](https://arxiv.org/html/1907.01268v2) | Multi-centre, multi-scanner cine CMR; LV at ED/ES and myocardium at ED according to the paper. | **Best segmentation/domain-shift match.** Ask for the application conditions and whether use as a locked external test set is permitted. Do not imply that RV labels exist. |
| 3 | **KORA-MRI, Helmholtz Munich**. KORA data are available by project agreement and KORA Board approval through KORA.PASST. | `KORA.PASST@helmholtz-munich.de`; [official KORA use-and-access page](https://www.helmholtz-munich.de/en/epi/cohort/kora) | KORA-MRI includes a 3T whole-body MRI subset with short-axis cine cardiac imaging and cardiometabolic phenotypes; confirm exact image/label availability in the catalogue. | **Good independent cohort.** Ask which image objects, derived ventricular measures and manual corrections are requestable by an international university team. |
| 4 | **NAKO Health Study, Germany**. Registered researchers can apply through TransferHub; NAKO documentation specifically describes access to MRI image data. | `transfer@nako.de`; [TransferHub](https://transfer.nako.de/transfer/index); [MRI application information, v1.1](https://transfer.nako.de/transfer/media/TFS-Info-02c_EN_Information%20on%20applications%20for%20MRI%20image%20data_v1.1.pdf) | Large population cohort; cardiac module includes cine SSFP and mapping. DICOM may be available; confirm annotations and derived measures. | **Strong scale, formal route.** The official instructions describe postal delivery on an applicant-supplied encrypted hard drive, not remote access. Ask whether a Malaysian university is eligible and whether international shipment/cross-border transfer is permitted, as well as the available cardiac objects, labels, costs and security conditions. |
| 5 | **SHIP, University Medicine Greifswald**. A formal transfer application exists. | `transferstelle@med.uni-greifswald.de`; [SHIP application portal](https://transfer.ship-med.uni-greifswald.de/FAIRequest/) | Population CMR with cine long/short axis and some tissue-characterization data; exact counts and annotations require catalogue confirmation. | **Good external population cohort.** Ask for English-language eligibility guidance and whether original cardiac image data, derived measures and any contours may be requested. |
| 6 | **Canadian Alliance for Healthy Hearts and Minds (CAHHM)**. Its official page states that data are available for approved secondary-research projects. | `Alliance@phri.ca`; secondary programme contact Dipika Desai, `Dipika.Desai@phri.ca`; [official data-access page](https://cahhm.mcmaster.ca/data-access-process/) | Large Canadian cohort with cardiac cine imaging; confirm image-level availability, annotations and cross-border access. | **Good structured route.** The official page asks applicants to email their name, institute and research interest. Ask whether UM-led image-method research is eligible as a governed data-access arrangement rather than a scientific collaboration. |
| 7 | **Qatar Precision Health Institute/Qatar Biobank**. Institution-bound researchers may apply through the Research Portal; a pre-application query is provided. | `qphi-ro@qf.org.qa`; [official application page](https://www.qphi.org.qa/research/how-to-apply) | Middle Eastern cohort with imaging and phenotypes; the exact distributable cardiac sequences and labels must be confirmed in the portal catalogue. | **Regionally valuable but fees/IRB likely.** Ask whether a UM computer-science supervisor and student are eligible and whether cardiac cine images plus ventricular measures are available through remote access. |
| 8 | **Jackson Heart Study (JHS)**. Existing data require an approved proposal or ancillary study and an executed DMDA. | `JHSCC@wfusm.edu`; [official access page](https://www.jacksonheartstudy.org/public/dspDataAccess.cfm) | A well-characterized cohort with a CMR subset; confirm whether source images rather than only measurements may be requested. | **High scientific value.** Ask which proposal path covers existing de-identified CMR image records and whether an overseas academic PI is eligible without a JHS co-investigator. |
| 9 | **MESA via NHLBI BioLINCC**. The catalogue describes baseline cardiac MRI and an open request workflow, but currently labels the resource “Study Datasets Only.” | `biolincc@imsweb.com`; MESA coordinating centre `chsccweb@u.washington.edu`; [official MESA catalogue](https://biolincc.nhlbi.nih.gov/studies/mesa/) | Approximately 5,000 baseline CMR examinations are described in the literature, with rich phenotypes. The availability of actual images/contours through BioLINCC is uncertain. | **Do not submit until confirmed.** Ask BioLINCC whether source images or only tabular variables are distributable. If eligible, Dr Uzair should normally serve as faculty PI and use a UM address. |
| 10 | **Framingham Heart Study (FHS)**. All data/material requests must use the FHS ResApp and may attract service fees. | `FHSapp@bu.edu`; general study contact `fhs@bu.edu`; [official ResApp page](https://www.framinghamheartstudy.org/fhs-for-researchers/research-application/) | CMR subset with cine long- and short-axis acquisitions; image and contour availability must be confirmed. | **Formal but potentially costly.** Ask whether international academic access to source CMR images is permitted before creating a full application. |

### Tier B — useful formal routes, but send after Wave 1

| Target | Contact/route | Value | Constraint or question |
|---|---|---|---|
| **AGES-Reykjavik** | `afgreidsla@hjarta.is`; [AGES research page](https://hjarta.is/en/research/ages-phase-1/) | Older-adult cohort with cardiac cine/LGE in a subset. | Confirm whether external image-level projects are accepted and whether contours exist. |
| **Generation R** | `generationr@erasmusmc.nl`; [official researcher page](https://generationr.nl/researchers/) | Large paediatric/young-person imaging cohort. | Different population from the current adult thesis; use only if paediatric external validation is scientifically justified. |
| **PESA-CNIC** | `pesa-h@cnic.es`; [PESA site](https://estudiopesa.org/) | Subclinical atherosclerosis cohort with CMR in a subset, including cine and advanced sequences. | Ask for the formal external-data route and whether image-level access requires collaboration. |
| **Hamburg City Health Study (HCHS) — exceptional/uncertain route** | `dm-hchs@uke.de`; [published access statement](https://pmc.ncbi.nlm.nih.gov/articles/PMC11003678/) | Large German cohort with CMR acquisition in a subset. | Published availability statements are not uniform: one says third-party sharing is restricted for ethical/legal reasons and routes enquiries to the Steering Committee, while some project-specific papers mention reasonable-request access with HCHS permission. Treat this as low probability and potentially incompatible with the no-external-investigator condition. |
| **Courtois Cardiovascular Signature, Montreal** | `ccvs@muhc.mcgill.ca`; [programme site](https://cvsignature.ca/) | Deep cardiovascular phenotyping and CMR. | Access route and independence from a local collaborator are unclear; send only a routing enquiry. |
| **SingHEART, Singapore** | Siew Ching Kong, `kong.siew.ching@singhealth.com.sg`; [official programme page](https://www.nhcs.com.sg/research-innovation/research-cores/data-digital-technology-ai/singheart) | Asian cohort and therefore attractive for population-shift validation. | Site emphasizes research opportunities/collaboration. Ask whether a data-only external application is possible; stop if formal collaboration is mandatory. |
| **ECCO-GEN — routing enquiry only** | Yasmine Aguib, `y.aguib@imperial.ac.uk`; [current EGA DAC record](https://ega-archive.org/dacs/EGAC00001001680) | A 2026 review reports a 391-participant Egyptian CMR cohort, which would add useful population diversity. | The linked EGA DAC currently controls a valve-tissue methylation dataset, not CMR images. Do **not** submit an EGA request for CMR; ask only whether a separate CMR governance route exists. |
| **DZHK Heart Bank / research platform** | Alexandra Klatt / Use and Access Office, `use.access@dzhk.de`; [official application page](https://dzhk.de/en/dzhk-heart-bank/submitting-applications) | Multiple German cardiovascular studies may include relevant CMR collections. | First ask which requestable collections contain cine CMR and expert ventricular labels. |
| **ANZACS-QI CMR Registry** | `anzacsqi@nihi.auckland.ac.nz`; [registry site](https://www.anzacs-qi.nz/) | Real-world New Zealand CMR registry. | Likely governance-heavy and may hold registry variables rather than transferable image pixels; ask before preparing a proposal. |
| **MyoFit46** | `mrclha.enquiries@ucl.ac.uk`; [cohort site](https://myofit46.com/) | Older-adult cohort with CMR-related phenotyping. | Small and population-specific; confirm image access and annotations. |

### Tier C — investigator-held, disease-specific datasets

These are genuine leads, but they should be contacted only if their disease scope supports a pre-specified thesis endpoint.

| Dataset | Contact | Reported data and value | Important limitation |
|---|---|---|---|
| **Charité multi-disease vendor-AI comparison** | Thomas Hadler, `thomas.hadler@charite.de`; Charité data-protection office `datenschutzbeauftragte@charite.de`; [paper](https://www.nature.com/articles/s41598-026-54182-z) | 346 clinical CMR cases including DCM, LVH, healthy and other disease; paper routes data enquiries through institutional protection controls. | Access is not guaranteed; ask Thomas Hadler first, not the privacy office alone. |
| **West China Hospital LVH cohort** | Kang Li, `likang@wchscu.cn`; Yong He, `heyong_huaxi@163.com`; [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10126185/) | 302 primary plus 53 external patients; cine CMR and CA/HCM/HHD labels, with automated myocardium work. | Disease-classification focus; verify whether source images and manual masks can be shared. |
| **KU Leuven repaired Tetralogy of Fallot** | Alexander Van De Bruaene, `alexander.vandebruaene@uzleuven.be`; [full paper](https://www.journalofcmr.com/article/S1097-6647%2824%2901119-0/fulltext) | Reported training/test/external cohorts with ED/ES short-axis images and manual labels. | Congenital-disease population; useful only as a pathology stress test, not the main general cohort. |
| **CHUV free-running 4D whole-heart CMR** | Augustin Ogier, `augustin.ogier@chuv.ch`; [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13112478/) | 35 healthy/HFpEF/HFrEF subjects; novel 4D acquisition and model resources may be requestable. | Small and acquisition-specific; not a substitute for standard cine validation. |
| **HCMR Registry** | Christopher Kramer, `ckramer@virginia.edu`; [official contacts](https://hcmregistry.org/contacts/) | Approximately 2,750 HCM participants with CMR and outcomes. | HCM-specific and likely to require detailed governance or collaboration. Proceed only if data-only access is allowed. |
| **OxAMI** | `oxami@cardiov.ox.ac.uk`; [official contact page](https://oxami.org.uk/contact-us/) | Acute myocardial-infarction cohort with CMR and linked clinical data. | Disease-specific; image-level secondary access is not established by the contact page and must first be confirmed. |
| **BIO FOr CARE** | Use the corresponding-author route in the [cohort publication](https://pure.eur.nl/ws/portalfiles/portal/52953097/Jansen2021_Article_BIOFOrCAREBiomarkersOfHypertro.pdf) | HCM/founder-mutation cohort; external requests are reviewed by a Data Access Board. | It may provide phenotypes rather than complete image series; low priority until confirmed. |

### Commercial/consortium possibility

- **Bunkerhill Health**: the 2026 generalizable-CMR paper reports that 17,088 Stanford/UCSF/MedStar patients were governed through Bunkerhill agreements and directs applications to [Bunkerhill Health](https://www.bunkerhillhealth.com). This is worth a web-form enquiry, but it may be commercial, costly or restricted. Do not guess an employee email.

## 4. Do not spend time on these now

| Route | Reason |
|---|---|
| **UK Biobank** | Its official page says new applications are paused and are intended to reopen in late 2026. Monitor the [official application page](https://www.ukbiobank.ac.uk/use-our-data/apply-for-access/) rather than emailing investigators. Imaging access also carries fees and cloud-compute costs. |
| **Cardiac Atlas Project MESA** | Its MESA page reports the resource as temporarily unavailable while its data-sharing agreement is renewed; it also does not provide the complete contour/clinical package needed here. Monitor rather than request now. |
| **CMR-CLIP / Cleveland Clinic** | The paper's data-availability statement says the clinical images and reports are not available for public use. Do not email the authors asking them to override this statement. |
| **NHS/Duke large clinical cine datasets** | The relevant publication states that the source clinical datasets cannot be made publicly available. Do not treat a publication author list as a lead list. |
| **Dallas Heart Study** | Outside proposals require an internal DHS Executive Committee partner. This conflicts with the present no-collaborator decision unless the supervisor changes that constraint. |
| **SCMR Registry** | A Search Request was already submitted on 29 September 2026. Wait for the committee response; do not duplicate it. A contributing-site partnership may ultimately be recommended or required. |
| **BAAI Cardiac Agent authors** | Initial feasibility enquiry was sent 28 September 2026. Do not follow up before 12 October 2026. |

## 5. Three-wave execution plan

### Wave 1 — sent 30 September 2026

Separate messages were sent to Penn, BSCMR-AS, KORA, NAKO, SHIP, CAHHM, QPHI and JHS. For MESA, only the short image-availability question was sent to BioLINCC; no MESA application should be started unless source-image access is confirmed.

The authoritative operational status and follow-up dates are recorded in `DATA_ACCESS_OUTREACH_TRACKER.md`.

### Wave 2 — after supervisor review, staggered 5–7 days after Wave 1

If Wave 1 yields fewer than three viable pathways, use the individually tailored messages in `CMR_WAVE2_EMAIL_DRAFTS.md` for FHS, PESA, AGES, Generation R, MyoFit46, SingHEART, Courtois and DZHK. Keep HCHS and ECCO-GEN on hold unless the supervisor specifically approves their low-probability routing enquiries.

### Wave 3 — after scope decision

Use the disease-specific contacts only after the supervisor agrees which pathology is methodologically relevant. A disease-specific cohort should be a deliberate stress test, not opportunistic data collection.

### Follow-up cadence

- Wait 10 business days after each first email.
- Send one reply in the same thread, no new attachment unless requested.
- If there is no response after another 7–10 business days, mark “no response” and move on.
- If told to use a portal, stop emailing and use the portal instructions.
- If told collaboration or a clinician is mandatory, do not negotiate around it; ask the supervisor whether to close that route.

## 6. Ready-to-send initial emails

Replace only the bracketed fields. Keep Dr Uzair Iqbal in CC. Use a UM institutional email if available; this matters especially for repository applications.

### Template A — author-held dataset

**Subject:** Academic data-access enquiry — [DATASET NAME] cardiac MRI

Dear [TITLE AND SURNAME],

I am Muhammad Tahir, a Master of Computer Science by Research student at Universiti Malaya, supervised by Dr Uzair Iqbal (copied). I read your paper, “[PAPER TITLE],” and noted that [EXACT DATA-AVAILABILITY WORDING IN PARAPHRASE]. We are seeking an independent, de-identified CMR test cohort to evaluate segmentation, ventricular quantification and failure-aware quality control; the data would not be redistributed or used commercially. Would access to [EXACT IMAGES/LABELS] potentially be possible under your institution's formal approval and data-use process? Our present request is for governed data access rather than adding external investigators; please let us know whether that is compatible with your access model. If so, please advise the appropriate application route and available data elements. We will complete all required institutional, ethics and security approvals before access.

Kind regards,  
Muhammad Tahir  
Master of Computer Science by Research  
Faculty of Computer Science & Information Technology, Universiti Malaya  
mtahirkhanyousufzai@gmail.com

### Template B — repository/custodian

**Subject:** Eligibility and CMR image-access enquiry — Universiti Malaya research project

Dear [TEAM/OFFICE],

I am Muhammad Tahir, a Master of Computer Science by Research student at Universiti Malaya, supervised by Dr Uzair Iqbal (copied). We are planning a non-commercial study using an independent, de-identified cardiac MRI cohort to evaluate segmentation, ventricular quantification and failure-aware quality control. Could you please confirm whether an international university team may apply for [DATASET], and whether source short-axis cine images, ED/ES indicators, expert contours or derived ventricular measurements are available? If potentially feasible, please direct us to the correct application, fee and secure-access requirements. We will complete all required institutional and ethics approvals before access.

Kind regards,  
Muhammad Tahir  
Master of Computer Science by Research  
Faculty of Computer Science & Information Technology, Universiti Malaya  
mtahirkhanyousufzai@gmail.com

### Template C — very short portal-routing enquiry

**Subject:** Cardiac MRI data-access route

Dear [TEAM],

Could you please confirm whether your external research process permits a Universiti Malaya team to request de-identified short-axis cine CMR images and any associated ED/ES contours or ventricular measurements for non-commercial external validation, and direct us to the correct application route?

Kind regards,  
Muhammad Tahir

### Template D — one follow-up only

**Subject:** Re: [ORIGINAL SUBJECT]

Dear [TITLE AND SURNAME/TEAM],

I am following up on my enquiry below in case it was missed. We would be grateful simply to know whether the requested CMR data are potentially accessible and, if so, which formal route we should use. Thank you for your time.

Kind regards,  
Muhammad Tahir

## 7. Tailoring lines for Wave 1

Copy one sentence into Template A or B; do not copy the whole table into an email.

| Target | Tailored sentence |
|---|---|
| Penn | “Your paper states that the University of Pennsylvania CMR data may be available under an appropriate data-use agreement; we are specifically interested in using a permitted subset as a locked external test cohort.” |
| BSCMR-AS | “The six-site, nine-scanner BSCMR-AS cohort is especially relevant to our pre-specified evaluation of segmentation under domain shift.” |
| KORA | “Before preparing a KORA.PASST proposal, we would like to confirm whether source cardiac cine images, ED/ES information and any manual corrections or derived ventricular measures are requestable.” |
| NAKO | “The NAKO MRI documentation appears to support image-data applications; we would like to confirm the available cardiac DICOM objects, annotation status and international access conditions.” |
| SHIP | “We are interested in the source cardiac cine component of SHIP and would appreciate guidance on image-level variables, annotations and the external application procedure.” |
| CAHHM | “Your access page invites enquiries for approved secondary-research projects; our requested use is a locked, non-commercial external validation rather than participant recruitment.” |
| QPHI | “Before creating a full Research Portal application, we would like to confirm applicant eligibility and whether cardiac cine image objects and ventricular reference measures are available.” |
| JHS | “Please advise whether an existing de-identified CMR image request belongs under the standard Proposal or Ancillary Study path, and whether an international university PI is eligible.” |
| MESA/BioLINCC | “The catalogue currently says ‘Study Datasets Only’; could you confirm whether source MESA CMR images or only derived/tabular variables are distributable through BioLINCC?” |

## 8. One-page protocol synopsis to attach only when requested

**Working title**  
External validation of an auditable cardiac magnetic resonance analysis pipeline integrating automated segmentation, deterministic ventricular quantification and failure-aware quality control

**Investigators**  
Academic lead/supervisor: Dr Uzair Iqbal, Universiti Malaya  
Student researcher: Muhammad Tahir, Master of Computer Science by Research, Universiti Malaya

**Rationale**  
Public cardiac MRI benchmarks are valuable but may not represent the scanner, site, acquisition and disease variation encountered in routine practice. The proposed study will test a frozen analysis pipeline on an independent controlled cohort and quantify performance degradation, measurement error and failure detection under domain shift.

**Primary objective**  
Evaluate patient-level LV cavity, LV myocardium and RV cavity segmentation at ED and ES, together with errors in ventricular volumes, ejection fraction and myocardial mass where corresponding reference standards exist.

**Secondary objectives**  
Evaluate performance by scanner vendor, field strength, centre and diagnosis; test whether a pre-specified quality-control gate identifies unreliable segmentations; and, only where de-identified reports and appropriate sequences are available, evaluate structured report grounding as a separate endpoint.

**Design**  
Retrospective secondary analysis of de-identified data. The external cohort will remain locked: model weights, thresholds and primary analysis rules will be frozen before outcome evaluation. All splits and analyses will be at patient level. The external data will not be used to tune the evaluated model.

**Minimum data**  
Short-axis cine CMR covering base to apex; full temporal series or clearly identified ED/ES frames; image geometry; pseudonymous patient ID; and, if available, LV cavity, LV myocardium and RV cavity contours or adjudicated ventricular measurements. Scanner/site metadata and diagnosis are requested only where permitted.

**Analysis**  
Segmentation metrics will include Dice and surface-distance measures. Quantification will include bias, MAE, confidence intervals and Bland–Altman limits for EDV, ESV, EF and mass as applicable. Pre-specified subgroup analyses will assess vendor, field strength, centre and disease. Missing reference labels will not be treated as negative findings.

**Privacy and governance**  
No direct identifiers are requested. Access will occur only after the applicable Universiti Malaya determination, custodian approval and institutional agreement. Data will be stored and analysed only in the approved environment, with access limited to named investigators. No patient-level data will be redistributed, uploaded to public repositories or submitted to unapproved external AI services. Publications will report aggregate results and follow custodian review/acknowledgement rules.

**Expected output**  
A thesis chapter and academic publication describing external performance, failure modes and domain-shift limitations. The study is non-commercial.

## 9. Exact variable list

### Essential image package

- de-identified short-axis cine CMR from base to apex;
- full cardiac cycle, or ED and ES image volumes with phase identifiers;
- DICOM preferred where acquisition metadata are part of the approved release; otherwise NIfTI plus a metadata table;
- pixel spacing, slice thickness, slice gap, orientation and temporal resolution;
- pseudonymous patient/study identifier that is stable across slices and phases; and
- scanner vendor, scanner model, field strength and anonymized centre identifier.

### Reference standard, if available

- LV blood-pool contour/mask at ED and ES;
- LV myocardium contour/mask at ED and ES;
- RV blood-pool contour/mask at ED and ES;
- EDV, ESV, EF and LV mass from the clinical/reference analysis;
- contouring convention, software, annotator expertise and adjudication method; and
- flags for excluded slices, papillary-muscle convention and basal/apical handling.

### Optional covariates

- age or age band, sex and body-surface area;
- clinical diagnosis and reference method;
- acquisition year band, not an exact identifying date;
- image-quality or artefact labels;
- de-identified report, only if specifically approved; and
- LGE or mapping sequences only for a separately pre-specified endpoint.

### Explicitly not requested

- name, address, telephone number, email or government identifier;
- medical-record number or institution-specific patient identifier;
- facial images or burned-in identifiers;
- exact admission, scan or birth dates unless essential and specifically approved;
- contact with participants; or
- any permission to redistribute the original data.

## 10. Document checklist

Do not attach everything to the first email. Have these ready:

1. this one-page protocol synopsis;
2. the exact variable list;
3. a one-page supervisor support letter on UM letterhead;
4. short CVs for Dr Uzair and Muhammad Tahir;
5. a data-management/security plan;
6. the UM ethics approval, waiver or jurisdiction determination when obtained;
7. proof of institutional appointment/enrolment;
8. proposed publication and derived-data plan;
9. conflict-of-interest and funding statement; and
10. the custodian's DUA/MTA signed only by authorized institutional signatories.

### Draft supervisor support letter

**To whom it may concern**

I confirm that Muhammad Tahir is undertaking a Master of Computer Science by Research at Universiti Malaya under my supervision. His project concerns external validation of an auditable cardiac MRI analysis pipeline, with a primary focus on segmentation, ventricular quantification and failure-aware quality control. I support a formal application for access to a suitable de-identified controlled dataset for non-commercial academic research. Universiti Malaya and the research team will complete all ethics, governance, security and contractual requirements specified by the data custodian before access, and will not redistribute patient-level data.

Sincerely,  
Dr Uzair Iqbal  
Faculty of Computer Science & Information Technology  
Universiti Malaya

## 11. How to judge replies

### Green — continue

- formal application is open to international academic researchers;
- source cine images are included;
- the supervisor may act as PI without a clinical co-investigator;
- access can be granted under a DUA/MTA and institutional approval;
- cost and timeline are feasible; and
- the data include a meaningful reference standard or can support a clearly narrowed endpoint.

### Amber — clarify before applying

- only derived measurements are listed;
- contours exist only for a subset;
- reports cannot be released;
- secure remote analysis is mandatory;
- fees are unspecified;
- the access page uses “collaboration” ambiguously; or
- publication review/co-authorship language is unclear.

### Red — stop and tell the supervisor

- a clinician/local investigator must formally join the study;
- a custodian offers to send files through personal email/cloud storage;
- only identifiable data are available;
- consent does not permit the proposed AI/image analysis;
- data cannot be used in a thesis or publication;
- an individual asks for personal payment outside the institution; or
- the source cannot document its authority to release the data.

## 12. Source and accuracy notes

- Dataset sizes and sequences may differ between publications, current catalogues and the subset actually releasable. The custodian's current data dictionary and approval letter control.
- The broad candidate search was cross-checked against a 2026 systematic review of global CMR imaging biobanks: [Journal of Cardiovascular Magnetic Resonance article](https://www.jacc.org/doi/10.1016/j.jcmg.2026.01.007). The linked official study pages remain the operational source for current applications.
- Penn availability and contact are taken directly from the 2026 paper's data-availability and correspondence sections.
- BSCMR-AS size, scanner diversity, label scope and “reasonable request” statement are taken directly from the linked paper.
- KORA, NAKO, FHS, MESA, JHS, CAHHM, QPHI and UK Biobank conditions are taken from their official access pages linked above.
- Public professional emails are included only where published by an institution or corresponding paper. No guessed email address is included.
- LinkedIn should be a last resort only when an official email or portal is unavailable; none of the first-wave targets requires it.

### Reverification corrections made on 29 September 2026

- The obsolete NAKO v1.3 document link was replaced with the accessible v1.1 instruction document, and the transfer description was corrected from possible remote access to encrypted-hard-drive delivery by post.
- Generation R's contact was updated to the address on its current official researcher page.
- HCHS was downgraded to an uncertain, low-probability route because published availability statements differ; its Steering Committee/data-management contact replaced the general participant address.
- ECCO-GEN was downgraded to a routing enquiry because the cited EGA DAC does not currently govern CMR images.
- The Penn, BSCMR-AS, KORA, NAKO, SHIP, CAHHM, QPHI, JHS and BioLINCC first-wave addresses and access claims were checked again against institutional pages or the corresponding publications.
