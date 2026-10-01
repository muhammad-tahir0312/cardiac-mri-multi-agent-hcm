# A Multi-Agent System for Automated Cardiac MRI Analysis and Guideline-Grounded Decision Support

**Candidature Defence Report**

Muhammad Tahir (25087075)
Master of Computer Science (Research)
Faculty of Computer Science and Information Technology, Universiti Malaya

Supervisor: Dr. Uzair Iqbal
Co-supervisor: Prof. Dr. Nor Liyana Mohd Shuib

Draft of 13 July 2026

> **Formatting for submission.** UM requires Times New Roman 12, double spacing, A4, and APA
> referencing. This draft is in Markdown so the content can be reviewed first. Convert to Word,
> apply the styles, and renumber the figures and tables last.
>
> **Word count of the body (Sections 1 to 6): approximately 8,900.** The FSKTM requirement is
> 5,000 to 10,000 words excluding references.

---

## Abstract

Cardiac magnetic resonance (CMR) imaging is the reference standard for measuring ventricular
volume and ejection fraction, but the reporting workflow that surrounds it is slow and variable.
Contours are drawn or corrected by hand, volumes are derived from those contours, and the clinician
must then connect the resulting numbers to the relevant clinical practice guideline from memory.
Recent work has begun to automate the whole chain with large language model agents. Two systems
define the current state of the art: the BAAI Cardiac Agent, which orchestrates segmentation,
diagnosis and report generation over a private cohort of 2,413 patients, and CardAIc-Agents, which
performs guideline-informed reasoning over electrocardiography and echocardiography. Neither system
solves the problem this project addresses.

Three gaps remain. First, no agentic CMR system has been evaluated on a public cardiac MRI dataset.
The BAAI cohort is private and cannot be independently reproduced or audited, and the CardAIc
datasets, while public, contain no MRI at all. Second, guideline citations are absent from the
diagnostic path. The BAAI retrieval component is a conversational side channel that answers user
questions; its structured diagnostic report carries no citations. Third, no published system links
a visual explanation to the specific guideline statement it supports. BAAI reports no visual
explainability of any kind, and CardAIc provides visual panels that are not tied to the textual
rationale.

This project builds a four-agent system that addresses those gaps and evaluates it on 830 publicly
available cardiac MRI studies drawn from ACDC, M&Ms and M&Ms-2, covering four scanner vendors and
five clinical centres. A segmentation agent produces left ventricular, myocardial and right
ventricular masks together with an ensemble uncertainty map. A quantification agent converts those
masks into end-diastolic volume, end-systolic volume, ejection fraction and myocardial mass using
deterministic arithmetic, and applies a rule-based plausibility gate. A clinical reasoning agent
retrieves passages from a corpus built exclusively from clinical practice guidelines, where every
chunk carries its class of recommendation and level of evidence, and generates a report in which
each clinical claim is bound to a cited passage. An explainability agent produces gradient-based
saliency maps and links each map region to the cited sentence it supports.

Two design decisions distinguish the system. When the plausibility gate rejects a mask, the
reasoning agent does not produce a diagnosis at all; the pipeline records a refusal rather than a
plausible but unfounded report. When a measured ejection fraction lands near a guideline decision
boundary, the reasoning agent requests a re-measurement rather than attempting to edit the mask
itself, which a text retrieval component cannot do.

Work completed to date establishes the data spine, the quantification agent, the guideline corpus,
the retrieval and generation stack, the orchestration layer and the evaluation harness. Several
findings are already reportable. The label conventions of ACDC and M&Ms are inverted with respect
to one another, a discrepancy that silently corrupts every cross-dataset measurement if it is not
corrected. The plausibility gate, once calibrated against expert annotation, accepts 98.7 per cent
of ground truth and rejects eleven studies whose expert contours are anatomically disconnected.
Segmentation on the ACDC test set reaches a Dice coefficient of 0.93 to 0.97 for the left
ventricular cavity. A guideline corpus of 2,592 passages was constructed from twelve documents,
with 50.2 per cent carrying class of recommendation and level of evidence. Finally, an unguarded
14-billion-parameter language model violated at least one grounding constraint in half of the
reports it generated, and every violation was detected and repaired by the enforcement layer.

**Keywords:** cardiac magnetic resonance, multi-agent systems, retrieval-augmented generation,
clinical decision support, explainable artificial intelligence

---

## Abstrak

Pengimejan resonans magnetik jantung (CMR) merupakan piawaian rujukan bagi pengukuran isi padu
ventrikel dan fraksi ejeksi, tetapi aliran kerja pelaporan yang mengelilinginya adalah perlahan dan
tidak konsisten. Kontur dilukis atau diperbetulkan secara manual, isi padu diterbitkan daripada
kontur tersebut, dan pakar perubatan kemudiannya perlu menghubungkan nombor yang terhasil dengan
garis panduan amalan klinikal yang berkaitan berdasarkan ingatan.

Dua sistem menakrifkan keadaan semasa bidang ini. BAAI Cardiac Agent menyelaraskan segmentasi,
diagnosis dan penjanaan laporan ke atas kohort persendirian seramai 2,413 pesakit. CardAIc-Agents
melaksanakan penaakulan berpandukan garis panduan ke atas elektrokardiografi dan ekokardiografi.
Kedua-duanya tidak menyelesaikan masalah yang ditangani oleh projek ini.

Tiga jurang kekal. Pertama, tiada sistem agen CMR pernah dinilai ke atas set data pengimejan
jantung awam. Kohort BAAI adalah persendirian dan tidak boleh diaudit secara bebas, manakala set
data CardAIc, walaupun awam, langsung tidak mengandungi MRI. Kedua, petikan garis panduan tidak
wujud di dalam laluan diagnostik. Komponen perolehan BAAI hanyalah saluran perbualan sampingan;
laporan diagnostik berstrukturnya tidak membawa sebarang petikan. Ketiga, tiada sistem yang
diterbitkan menghubungkan penjelasan visual kepada kenyataan garis panduan yang menyokongnya.

Projek ini membina sistem empat agen dan menilainya ke atas 830 kajian CMR awam daripada ACDC, M&Ms
dan M&Ms-2, merangkumi empat vendor pengimbas dan lima pusat klinikal. Agen segmentasi menghasilkan
topeng ventrikel kiri, miokardium dan ventrikel kanan berserta peta ketidakpastian. Agen
pengkuantitian menukarkan topeng tersebut kepada isi padu akhir diastol, isi padu akhir sistol,
fraksi ejeksi dan jisim miokardium menggunakan aritmetik berketentuan, serta menerapkan get
kemunasabahan berasaskan peraturan. Agen penaakulan klinikal memperoleh petikan daripada korpus
yang dibina secara eksklusif daripada garis panduan amalan klinikal, di mana setiap petikan membawa
kelas cadangan dan tahap buktinya, dan menjana laporan di mana setiap dakwaan klinikal terikat
kepada petikan yang dirujuk. Agen kebolehjelasan menghasilkan peta salience berasaskan kecerunan
dan menghubungkan setiap kawasan peta kepada ayat yang disokongnya.

Apabila get kemunasabahan menolak sesuatu topeng, agen penaakulan tidak menghasilkan diagnosis
langsung; sistem merekodkan penolakan dan bukannya laporan yang kelihatan munasabah tetapi tidak
berasas. Apabila fraksi ejeksi yang diukur berada berhampiran sempadan keputusan garis panduan,
agen penaakulan memohon pengukuran semula, dan bukannya cuba menyunting topeng, kerana komponen
perolehan teks tidak mempunyai isyarat pengimejan.

Beberapa penemuan sudah boleh dilaporkan. Konvensyen label ACDC dan M&Ms adalah songsang antara satu
sama lain, satu percanggahan yang secara senyap merosakkan setiap pengukuran merentas set data
sekiranya tidak diperbetulkan. Get kemunasabahan, setelah ditentukur terhadap anotasi pakar,
menerima 98.7 peratus kebenaran asas. Segmentasi pada set ujian ACDC mencapai pekali Dice 0.93
hingga 0.97 bagi kaviti ventrikel kiri. Korpus garis panduan sebanyak 2,592 petikan telah dibina
daripada dua belas dokumen, dengan 50.2 peratus membawa kelas cadangan dan tahap bukti. Akhirnya,
model bahasa 14 bilion parameter yang tidak dikawal melanggar sekurang-kurangnya satu kekangan
dalam separuh daripada laporan yang dijananya, dan setiap pelanggaran dikesan serta dibaiki oleh
lapisan penguatkuasaan.

**Kata kunci:** resonans magnetik jantung, sistem pelbagai agen, penjanaan tambahan perolehan,
sokongan keputusan klinikal, kecerdasan buatan yang boleh dijelaskan

---

## 1. Introduction

### 1.1 Background

Cardiovascular disease remains the leading cause of death worldwide. Within cardiac imaging,
cardiac magnetic resonance is regarded as the reference standard for quantifying ventricular volume,
ejection fraction and myocardial mass, because it does not depend on geometric assumptions and does
not suffer from the acoustic window limitations of echocardiography.

The measurement itself is only part of the clinical task. A CMR study produces a stack of
short-axis cine images across the cardiac cycle. Someone must delineate the blood pool and the
myocardium at end diastole and end systole, sum the resulting areas to obtain volumes, compute the
ejection fraction, and then interpret that number against the relevant clinical practice guideline.
In current practice the delineation is performed or corrected manually, and the interpretation is
performed from memory or by consulting the guideline document separately.

Two consequences follow. Reporting is slow, and it is variable. Manual contouring introduces
inter-observer variability in the derived volumes, and the mapping from a measured ejection fraction
to a guideline category is applied inconsistently, particularly for values that sit close to a
threshold.

Deep learning has largely solved the segmentation step in isolation. Methods such as nnU-Net
(Isensee et al., 2021) reach expert-level agreement on public benchmarks. What has not been solved
is the rest of the chain: converting masks into clinically meaningful measurements, grounding the
resulting interpretation in a citable guideline, and explaining the decision in a form a clinician
can check.

### 1.2 Problem statement

Three problems motivate this work.

**Problem 1: automated CMR analysis stops at segmentation.** Published segmentation models produce
masks. They do not produce measurements, they do not produce an interpretation, and they do not
produce a report. The clinician still performs the reasoning step, which is where the variability
and the time cost actually reside.

**Problem 2: where clinical reasoning has been automated, it is not verifiably grounded.** Systems
that generate diagnostic text from imaging either do not cite their sources, or cite them at the
level of the whole document rather than the specific passage. A citation to "the 2021 ESC Heart
Failure Guideline" does not permit a reader to check whether the statement in the report is
actually supported by that guideline. It is not falsifiable, and therefore it is not useful as
evidence.

**Problem 3: the reported results of the closest systems cannot be independently verified.** The
BAAI Cardiac Agent (Qu et al., 2026) reports strong results on a cohort of 2,413 patients from two
hospitals. That cohort is private. Its data availability statement records that no publicly
available dataset was used. No third party can reproduce the result, audit it, or compare a new
method against it on the same data.

### 1.3 Research questions

**RQ1.** Can a modular multi-agent architecture perform the full CMR reporting chain, from image to
measurement to guideline-grounded diagnostic report, on public benchmark data?

**RQ2.** Does grounding the diagnostic report in retrieved clinical guideline passages improve its
clinical correctness relative to an ungrounded language model given identical measurements?

**RQ3.** How well does the pipeline generalise across scanner vendors and clinical centres, and
where does that generalisation fail?

### 1.4 Research objectives

**O1.** To develop a segmentation agent producing left ventricular, myocardial and right ventricular
masks together with a voxel-wise uncertainty estimate.

**O2.** To develop a quantification agent that derives end-diastolic volume, end-systolic volume,
ejection fraction and myocardial mass by deterministic arithmetic, and that applies a rule-based
plausibility gate to its own output.

**O3.** To develop a clinical reasoning agent that retrieves passages from a guideline-only corpus
carrying class of recommendation and level of evidence, and generates a report in which every
clinical claim is bound to a cited passage.

**O4.** To develop an explainability agent that produces gradient-based saliency and ensemble
uncertainty maps, and that links each map region to the cited guideline statement it supports.

**O5.** To evaluate the assembled system on 830 public CMR studies, stratified by vendor and centre,
against five runnable baselines.

### 1.5 Scope

**In scope.** Short-axis cine CMR. Segmentation of the left ventricular cavity, the myocardium and
the right ventricular cavity. Computation of left and right ventricular volumes, ejection fraction
and left ventricular mass. Heart failure categorisation from ejection fraction. Retrieval over a
corpus of cardiovascular clinical practice guidelines. Generation of a structured diagnostic report
with passage-level citations. Gradient-based visual explanation.

**Out of scope.** Late gadolinium enhancement, perfusion imaging and parametric mapping, none of
which are present in the datasets used. Strain analysis. Valvular assessment. Coronary anatomy.
Prospective clinical deployment.

Right ventricular segmentation is explicitly in scope. This resolves a contradiction in the
proposal, where Table 1.2 listed right ventricular analysis as out of scope while the methodology
and expected outcomes both required it. Right ventricular strain remains out of scope.

### 1.6 Significance

The system is designed so that its claims can be checked. All three datasets are public, the model
weights used are public, the language model runs locally with open weights, and every reported
number is written to a file that accompanies the code. A reader with the same data can rerun the
pipeline and obtain the same result. This is a deliberate response to the position of the closest
comparable system, whose results rest on a cohort nobody else can see.

The guideline corpus carries class of recommendation and level of evidence on every passage. This
allows a citation to convey not merely that a guideline mentions something, but how strongly it
recommends it, which is the information a clinician actually needs when reading a decision support
output.

---

## 2. Literature Review

### 2.1 Cardiac MRI segmentation

Automatic segmentation of the cardiac chambers from short-axis cine MRI has been driven by public
challenge datasets. The Automated Cardiac Diagnosis Challenge (Bernard et al., 2018) provided 150
annotated studies across five diagnostic groups and established that convolutional architectures
approach inter-observer agreement on this task. The Multi-Centre, Multi-Vendor and Multi-Disease
Cardiac Segmentation challenge (Campello et al., 2021) extended the problem to four scanner vendors
and multiple centres, and showed that performance degrades when a model trained on one vendor is
applied to another. That degradation, rather than in-domain accuracy, is the open problem.

nnU-Net (Isensee et al., 2021) is the standard reference method. Its contribution is not a novel
architecture but an automated configuration procedure: it derives its own preprocessing, patch size,
spacing and training schedule from the dataset fingerprint. This matters for the present work in a
practical way. Because nnU-Net configures itself, hand-specifying its resampling spacing or patch
size, as the original proposal did, contradicts the reason for choosing it.

More recently, foundation models pretrained on large unlabelled cine collections have appeared.
CineMA (Fu et al., 2025) is pretrained on UK Biobank data with a masked autoencoder objective and
released with checkpoints fine-tuned separately on ACDC, M&Ms and M&Ms-2, three random seeds each.
This is used as the segmentation backbone in the present work, with a caveat developed in Section
3.3: a checkpoint fine-tuned on ACDC and applied to ACDC is a supervised model, not a zero-shot one,
and must not be described as the latter.

### 2.2 Automated quantification and reference ranges

Once masks exist, volumes follow by summing labelled voxels and multiplying by the voxel volume.
For a contiguous short-axis stack this is exactly the disc summation integral used clinically, so no
approximation is introduced. Myocardial mass is obtained by multiplying myocardial volume by a
density constant of 1.05 g/mL, which is the value given in the Society for Cardiovascular Magnetic
Resonance reference ranges paper (Kawel-Boehm et al., 2020). The proposal attributed this constant
to a different source that does not establish it; the attribution is corrected here.

The heart failure categories are defined by ejection fraction. The 2021 European Society of
Cardiology heart failure guideline (McDonagh et al., 2021) and the 2022 AHA/ACC/HFSA guideline
(Heidenreich et al., 2022) agree on the cut-points: heart failure with reduced ejection fraction at
40 per cent or below, mildly reduced from 41 to 49 per cent, and preserved at 50 per cent or above.
Both boundaries are closed. A rule that treats the interval as "40 to 50" is undefined at exactly 40
and at exactly 50, which are the two values a boundary-aware system encounters most often. The
proposal contained such a rule, and it is corrected in Section 3.4.

### 2.3 Retrieval-augmented generation for clinical text

Retrieval-augmented generation (Lewis et al., 2020) conditions a language model on passages
retrieved from an external corpus, which allows the model's factual claims to be traced to a source
and allows the corpus to be updated without retraining the model. In the clinical setting this is
attractive because guidelines change and because an unsourced clinical assertion is not actionable.

Retrieval quality depends on the retriever. Dense retrieval encodes query and passage into a shared
embedding space, but embeddings trained with a masked language modelling objective are not optimised
for retrieval. MedCPT (Jin et al., 2023) is trained contrastively on PubMed search logs specifically
for biomedical retrieval, and is used here in preference to BioBERT, which is retained as an
ablation. Sparse lexical retrieval, of which BM25 (Robertson and Zaragoza, 2009) is the standard
form, complements dense retrieval by matching exact clinical terms and numeric thresholds that an
embedding may blur. Hybrid retrieval combining both is used in the present work, a choice supported
by the ablation reported in CardAIc-Agents (Zhang et al., 2025), which found that vector-only and
keyword-only retrieval each underperformed their combination.

### 2.4 Explainability in medical image analysis

Grad-CAM (Selvaraju et al., 2020) produces a class-discriminative saliency map by weighting the
activations of a convolutional layer by the gradient of a class score. It is defined for
classifiers, where a single scalar logit exists per class. Segmentation networks produce a spatial
map of logits and therefore have no single scalar to differentiate. Seg-Grad-CAM (Vinogradova et
al., 2020) resolves this by summing the class logit over a region of interest before taking the
gradient. The distinction is not cosmetic: applying the classifier formulation to a segmentation
network is not well defined, and the original proposal cited only the classifier paper.

A second and independent explanation channel is available at no additional cost. Because the CineMA
checkpoints are released with three seeds, the softmax entropy of the ensemble mean gives a
voxel-wise uncertainty map without any additional training or inference machinery.

### 2.5 Agentic systems for cardiac imaging

Two systems are close enough to this work to require careful positioning. Both were read in full.

#### 2.5.1 BAAI Cardiac Agent

Qu et al. (2026) present what they describe as the first end-to-end agent framework designed
specifically for CMR image analysis. A large multimodal model orchestrates nine tools covering
segmentation of several cine and late-enhancement views, disease screening, cardiomyopathy
subclassification, retrieval-augmented generation, report generation and visual question answering.
The system is trained and evaluated on 2,413 patients from two hospitals, scanned on Philips, GE and
Siemens equipment. It reports a short-axis cine Dice of 90.21, exceeding the nnU-Net baseline of
87.42 on the same data; Pearson correlations against clinical reports of 0.968 for end-diastolic
volume and 0.925 for ejection fraction; a BERTScore F1 of 0.898 for generated reports; and a reader
study with six radiologists across three experience levels.

The proposal characterised this system as evaluated only on a small dataset, without multi-vendor
validation, and without guideline-based retrieval. All three characterisations are incorrect. The
cohort is large, it is multi-vendor and multi-centre with an external validation set, and the system
does contain a retrieval component that cites ACC/AHA guidelines. The literature review is corrected
accordingly. The one negative claim that does hold is explainability: the paper contains no mention
of saliency, attention maps, uncertainty or heatmaps of any kind.

Two properties of the system define the space that remains. Its cohort is private, and its data
availability statement records that no other publicly available dataset was used; the paper contains
no mention of ACDC or M&Ms. And its retrieval component is a conversational side channel, invoked
when a user asks a knowledge question. Its structured diagnostic report carries no citations at all.
Where citations do appear, they identify a document, not a passage.

#### 2.5.2 CardAIc-Agents

Zhang et al. (2025) present a multi-agent framework with a retrieval component (CardiacRAG) built on
Bio_ClinicalBERT embeddings, FAISS indexing, and hybrid dense and keyword retrieval. It performs
complexity assessment, stepwise plan refinement as evidence accumulates, and escalation to a
simulated multidisciplinary team for difficult cases. Its ablation shows that removing the adaptive
workflow reduces accuracy from 0.87 to 0.80, and removing retrieval reduces it to 0.77. It provides
visual review panels including labelled ECG waveforms and echocardiographic view identification.

The decisive fact is that the paper contains no mention of MRI, CMR or magnetic resonance anywhere.
It is an ECG, echocardiography and electronic health record system, evaluated on MIMIC-IV, PTB-XL
and the PTB Diagnostic ECG database. It is therefore not a cardiac MRI system and cannot serve as a
cardiac MRI baseline. The proposal listed it as one; this is corrected. It remains highly relevant
as prior art for two mechanisms: hybrid retrieval, and adaptive re-planning in response to
intermediate evidence.

Its knowledge base mixes clinical practice guidelines with material from the Mayo Clinic website,
the UK National Health Service website and MedlinePlus. Its citations return the original chunk,
which is passage level, but the chunks themselves carry no grading metadata.

### 2.6 Evaluating generated clinical reports

BERTScore (Zhang et al., 2020) computes similarity between a candidate and a reference text using
contextual embeddings. It is an embedding-based semantic metric, not a lexical one, and grouping it
with BLEU and ROUGE, as the proposal did, is a category error.

More seriously, BERTScore requires a reference report. ACDC, M&Ms and M&Ms-2 supply segmentation
masks and pathology labels. None of them supplies a radiology report. A BERTScore target is
therefore not measurable against these datasets without first constructing reference reports, and
the target proposed (0.85) is in any case below the 0.898 that BAAI already achieves against real
clinical reports. Section 3.8 replaces this with a metric that can actually be computed.

### 2.7 Research gap

Table 2.1 positions the present work against the two closest systems. The cohort size row is stated
plainly and is not favourable to this work. It is included because a reader who finds an omission
there will distrust the rest of the table.

**Table 2.1. Positioning against the closest comparable systems.**

| | BAAI Cardiac Agent (2026) | CardAIc-Agents (2025) | This work |
|---|---|---|---|
| Modality | CMR | ECG, echo, EHR. No MRI. | CMR |
| Agentic orchestration | Yes | Yes | Yes |
| Guideline retrieval | Yes, conversational side channel | Yes (CardiacRAG) | Yes, inside the diagnostic path |
| Citations in the diagnostic report | None | Not applicable (no imaging report) | Passage level, with source, section and page |
| Class of recommendation and level of evidence on chunks | No | No | Yes |
| Visual explainability | None | Panels, not linked to text | Saliency and uncertainty, linked to the cited sentence |
| Refusal on implausible segmentation | Not reported | Not reported | Yes, traced |
| Cohort | 2,413 patients | 11,939 patients | 830 studies |
| Cohort public | **No** | Yes (not MRI) | **Yes** |
| Evaluated on a public CMR benchmark | **No** | Not applicable | **Yes** |

The gap this work occupies can be stated in one sentence. **No prior CMR system grounds its
diagnostic statements in citable clinical guideline passages, links those citations to visual
explanations, or reports its results on public benchmarks.** Each clause is checkable against the
two papers.

---

## 3. Methodology

### 3.1 Research design

The system is a directed graph of four agents with typed contracts at every boundary. Each contract
is a schema, and a payload that does not validate cannot cross the boundary. This is the mechanism
by which an error in one agent is prevented from propagating silently into the next, and it is a
structural property of the implementation rather than a policy applied at runtime.

The pipeline is:

```
segment -> quantify -> [plausibility gate] -> retrieve -> generate -> verify -> explain
                             |
                             +- fail, budget remaining --> re-segment with augmentation
                             +- fail, budget exhausted --> REFUSE. No diagnosis is produced.
                             +- pass, value near a guideline cut-point --> re-measure
```

### 3.2 Datasets

Three public datasets are used. Table 3.1 gives the counts as measured from the data on disk, which
differ from the counts stated in the proposal.

**Table 3.1. Datasets, as measured.**

| Dataset | Studies with ground truth | Vendors | Centres | Pathology classes |
|---|---|---|---|---|
| ACDC | 150 | not recorded | 1 | 5 |
| M&Ms | 320 | 4 | 5 | 9 |
| M&Ms-2 | 360 | 3 | not recorded | 8 |
| **Total** | **830** | **4** | **5** | |

Two corrections are required. The proposal states that M&Ms contains 375 subjects across six
centres. The open release contains 345 subjects across five centres; the larger figure describes the
full challenge cohort, part of which was never released. Of those 345, twenty-five are unlabelled
and are not used, leaving 320 with ground truth. The proposal also quotes a total of 525 subjects
across two datasets, while the methodology used three. The corrected total is 830 studies with
ground truth across three datasets, or 855 including the unlabelled subset.

The three datasets provide end-diastolic and end-systolic frame indices by three different
mechanisms: a configuration file in ACDC, a metadata spreadsheet in M&Ms, and pre-extracted volumes
in M&Ms-2. The proposal specified automatic detection of these frames by time-gradient analysis.
This is unnecessary and would introduce error that could not be attributed. The provided indices are
used.

### 3.3 Agent 1: segmentation

The backbone is CineMA, using the released fine-tuned checkpoints. All three seeds are run and the
softmax outputs averaged, which yields both the segmentation and, as the entropy of the mean, a
voxel-wise uncertainty map at no additional cost. nnU-Net is trained separately as a baseline and as
the alternative backbone; its self-derived configuration is reported as a result rather than
specified in advance.

Three runs must be distinguished, and the distinction is not academic. The `acdc_sax` checkpoint was
fine-tuned on ACDC. Applying it to ACDC is a supervised evaluation. Applying it to M&Ms and M&Ms-2
is a genuine cross-dataset zero-shot evaluation, and it is that run, not the first, which answers
RQ3. Reporting a supervised result as zero-shot would be disprovable from the public model card.

**The label convention problem.** ACDC labels the right ventricular cavity as 1 and the left
ventricular cavity as 3. M&Ms and M&Ms-2 do the reverse. All three use the label set {0,1,2,3} and
the volumes are visually indistinguishable. A model trained on one convention and evaluated on the
other produces Dice scores, ejection fractions and heart failure categories that are wrong and yet
entirely plausible, with no error raised anywhere.

The correction is an assertion, not a convention. The myocardium forms a ring around the left
ventricular cavity, so the myocardial centroid must lie close to the left ventricular centroid and
far from the right ventricular one. Measured across all three datasets, the separation is
unambiguous: the distance to the correct cavity is at most 4.2 pixels, and the distance to the
incorrect one is 25 to 42 pixels. This test is applied to every ground-truth volume at load time and
to every predicted mask, because each released checkpoint emits its own dataset's convention.

### 3.4 Agent 2: quantification and the plausibility gate

Volumes are obtained by counting labelled voxels and multiplying by the voxel volume. Ejection
fraction follows. Myocardial mass is myocardial volume multiplied by 1.05 g/mL (Kawel-Boehm et al.,
2020). No model is involved and nothing is learned.

Heart failure category is assigned by the corrected rule:

```
LVEF <= 40          -> HFrEF
40 <  LVEF <  50    -> HFmrEF
LVEF >= 50          -> HFpEF
```

Both boundaries are closed, so no value falls through. A value within two percentage points of
either cut-point is flagged as near-boundary, and that flag drives the re-measurement described in
Section 3.7.

**The plausibility gate** rejects a mask whose derived measurements or geometry are not credible:
ejection fraction outside a physiological range, end-systolic volume not less than end-diastolic
volume, a structure fragmented into disconnected components, or a gap in the left ventricular stack
between two slices that both contain the left ventricle.

The gate is calibrated on expert ground truth. This is the point that makes it defensible. A
plausibility gate is meaningful only if it accepts what expert annotation produces; a gate that
rejects the reference standard measures nothing. The thresholds are therefore set so that at least
98 per cent of expert annotations pass, and anything the gate then rejects is, by construction, less
anatomically plausible than an expert's own contour. The calibration result is reported in
Section 4.3.

### 3.5 Agent 3: guideline retrieval and report generation

#### 3.5.1 The corpus

The corpus is built exclusively from clinical practice guidelines and society consensus documents.
No encyclopaedia articles, no patient information pages, no primary literature. This purity is
deliberate. The knowledge base of CardAIc-Agents mixes guidelines with the Mayo Clinic and NHS
websites, and that of BAAI mixes guidelines with PubMed and a general-purpose corpus. A
guideline-only corpus permits every retrieved passage to carry a recommendation grade.

Twelve documents are included, listed in Section 4.5. Passages are segmented at approximately 350
tokens with 50 tokens of overlap. Every chunk carries its source, section, page, class of
recommendation and level of evidence. The two societies grade recommendations on different scales
(ESC uses I, IIa, IIb, III with levels A, B, C; ACC/AHA uses 1, 2a, 2b, 3 with levels A, B-R, B-NR,
C-LD, C-EO), and both are normalised onto the ESC scale for comparability while the original tokens
are retained alongside.

The guidelines are copyrighted and cannot be redistributed. What can be released, and what will be,
is the acquisition and chunking pipeline together with a SHA-256 digest of every chunk, so that a
reader holding legal copies of the same documents can reconstruct the corpus exactly.

#### 3.5.2 Retrieval

Retrieval is hybrid. Dense retrieval uses MedCPT with exact inner-product search over a FAISS index
(Johnson et al., 2021); the corpus is small enough that approximate indexing would serve no purpose.
Sparse retrieval uses BM25. The two ranked lists are combined by reciprocal rank fusion. Dense-only,
sparse-only and BioBERT-embedding variants are retained as ablations.

Contradictions between retrieved passages are detected and resolved by recency, then by class of
recommendation. If a contradiction cannot be resolved, both positions are presented with
attribution, and neither is silently discarded. Section 5.2 reports what this mechanism actually
found.

#### 3.5.3 Constrained generation

The language model never sees an image and is never asked to measure anything. It receives the
measurements as structured fields and the retrieved passages as delimited reference text, and it
returns a report conforming to a fixed schema. Generation is schema-constrained, so a structurally
malformed report cannot be produced.

The model is an open-weights model run locally. This is a deliberate choice rather than a budgetary
one. A system whose central claim is reproducibility should not depend on a proprietary endpoint
whose behaviour may change without notice and which no reader can rerun. The model identifier and
the date of use are recorded in the run metadata of every experiment.

#### 3.5.4 Guardrails: enforcing grounding, and the security boundary

Schema validity guarantees the shape of a report, not its truth. An empty diagnosis and an invented
citation identifier are both schema-valid. Four invariants are therefore enforced after generation,
and a report that violates one is not emitted.

1. **Every citation resolves.** A citation identifier that does not exist in the corpus index is not
   a weak citation. It is a fabricated one, and it is removed.
2. **Every number is traceable.** Each numeric value asserted in the report must either match a
   measurement produced by Agent 2 within tolerance, or appear verbatim in a passage that the report
   actually cites. A guideline threshold quoted from a cited passage has provenance; the same figure
   with no citation does not.
3. **Arithmetic is not negotiable.** The heart failure category is computed deterministically from
   the ejection fraction. If the model disagrees with that computation, the model is wrong.
4. **No claims about unobserved data.** The system sees a short-axis cine segmentation. A report
   asserting a late gadolinium enhancement finding is describing something that was never acquired.
   Recommending such a study is permitted; reporting its result is not.

Violations trigger a bounded repair loop in which each violation is named back to the model. If the
report still violates an invariant after the repair budget is exhausted, no diagnosis is emitted.
The system does not fall back to a best-effort report with a poor score, because an ungrounded
clinical report is worse than no report. This is the same refusal posture the plausibility gate
adopts toward a broken mask, applied to a broken report.

**The security position, stated without inflation.** The only text reaching the language model is
guideline passages that the system ingested itself and numeric fields produced by its own code.
There is no user-supplied free text anywhere in the pipeline, so the prompt-injection surface is
genuinely small. That is a stronger and more honest position than claiming an elaborate defence
against an attacker who has no route in.

It is not zero, and the non-zero part is the corpus. A PDF is an untrusted byte stream, and a future
corpus, for instance a hospital's own protocol documents, would be considerably less trustworthy
than an ESC guideline. Passages are therefore treated as data and never as instructions. Chunks
carrying imperative or role-play patterns are quarantined at ingest, before they can enter the index
and therefore before any query can retrieve them. Structural tokens are neutralised so that a
passage cannot escape its delimiter. Most importantly, the model has no tools, no network access and
no filesystem access; it emits a single JSON object against a fixed schema. A successful injection
would have no action available to it. Section 4.7 reports the false-positive rate of the quarantine
scan on real clinical text, which matters as much as its detection rate: a filter that quarantines
genuine guideline passages would silently delete the corpus.

### 3.6 Agent 4: explainability

Seg-Grad-CAM (Vinogradova et al., 2020) is applied to the segmentation backbone, taking the gradient
of a region-summed class score rather than a scalar class logit. The ensemble entropy map from
Section 3.3 provides a second, independent visual channel.

The contribution is neither map on its own. It is the link between them and the text. Each cited
sentence in the report is associated with the anatomical structure it describes, and thence with the
saliency region and the uncertainty over that region. The output is an explicit object binding a
report sentence to a guideline passage identifier and to an image region. A reader can therefore
ask, of any sentence in the report, both which guideline licenses it and which part of the image
produced the number it rests on.

### 3.7 Orchestration

The graph is implemented as a typed state machine (LangGraph). Two behaviours are the substance;
the graph itself is plumbing.

**Refusal.** When the plausibility gate rejects a mask and the retry budget is exhausted, the
pipeline records a failure state and the reasoning agent does not run. No report is produced. Each
node's verdict is written to a per-study trace, so a refusal can be inspected and attributed. This
is the concrete answer to the question of how an error in the segmentation agent is prevented from
cascading into the quantification, reasoning and explanation agents.

**Decision-boundary-aware recomputation.** The proposal stated that guideline evidence could refine
segmentation masks. A text retrieval component has no imaging signal and cannot alter a voxel, so
that mechanism as described is not implementable. The reformulation is as follows. The reasoning
agent knows the guideline decision boundaries. When a measured value lands near one, for example an
ejection fraction of 39.4 per cent against the 40 per cent cut-point, the agent issues a
re-measurement request, and the segmentation agent re-runs with test-time augmentation and full
ensembling to tighten the estimate. The trigger is guideline knowledge; the action occurs in imaging
space; the loop is bounded at two iterations and every iteration is logged.

The nearest prior art is the adaptive workflow of CardAIc-Agents, which refines the *plan* as
evidence accumulates. The mechanism here refines the *measurement* when it approaches a clinical
decision boundary. The trigger, the action and the failure mode all differ, and the prior art is
cited rather than avoided.

### 3.8 Evaluation

Segmentation is evaluated by Dice coefficient and 95th-percentile Hausdorff distance, stratified by
vendor and by centre, because cross-vendor degradation is RQ3.

Measurement agreement against ground truth is evaluated by mean absolute error, Bland-Altman bias
and limits of agreement, and the intraclass correlation coefficient. These are reported because
their absence would be conspicuous, not because they distinguish this work; BAAI already reports
Bland-Altman and Pearson correlation.

Three metrics address clinical correctness, which is what RQ2 actually asks about and which
segmentation and text-similarity metrics do not measure.

**Pathology top-1 accuracy.** The top differential in the generated report is scored against the
dataset label. The label vocabularies differ across the three datasets and are not pooled silently;
an explicit mapping records which classes are comparable and which are dataset-specific. Section 5.2
reports a complication with this metric that must be stated openly.

**Heart failure category accuracy.** The category derived from the predicted ejection fraction is
compared against the category derived from ground truth. Restricted to studies whose ground-truth
ejection fraction lies between 35 and 55 per cent, this is the test for the boundary-aware
recomputation hypothesis.

**Report fidelity.** Two deterministic quantities. Numeric fidelity is the proportion of numbers in
the report that trace to the quantification agent or to a cited passage. Citation fidelity is the
proportion of citations that resolve to a real indexed passage. Neither BAAI nor CardAIc reports any
comparable measure; BAAI asserts the absence of hallucination qualitatively and does not quantify
it.

An important caveat governs how these last two are reported. Because the guardrail layer enforces
them, both read 1.0 on the deployed system by construction. That is the definition of the
enforcement working, not a finding. The scientifically informative number is how often the guardrail
had to intervene, and the unenforced fidelity of the raw model is reported alongside as the honest
baseline.

Significance is assessed by paired bootstrap with 95 per cent confidence intervals.

### 3.9 Baselines

Five baselines, all of which can actually be run.

1. **nnU-Net standalone.** The segmentation floor.
2. **CineMA.** The foundation-model comparison, reported separately as supervised and as
   cross-dataset.
3. **BAAI Cardiac Agent evaluated on ACDC and M&Ms.** Its code and weights are released under
   Apache 2.0. No agentic CMR system has been benchmarked on public cardiac MRI data, so this run is
   the first such evaluation and provides the one baseline that genuinely matters.
4. **Measurements to language model, no retrieval.** The ungrounded generation ablation, which is
   the direct test of RQ2.
5. **Template report generated from the quantification output, with no language model at all.** The
   zero-AI floor.

Baseline 5 will score close to perfectly on numeric fidelity, because a template cannot hallucinate,
and will therefore appear to beat the full system on that metric. It is included for exactly that
reason. The gap between the template and the full system is a direct measure of what the language
model contributes, and reporting the baseline that flatters the work least is what makes the
remaining comparisons credible.

MARL-Rad and CardAIc-Agents are cited as context and are not re-run. MARL-Rad has no public weights.
CardAIc-Agents is not a cardiac MRI system and cannot serve as a cardiac MRI baseline.

### 3.10 Components involving no training

Table 3.2 lists which components learn parameters. Exactly one does.

**Table 3.2. Trained and untrained components.**

| Component | Trained in this work? |
|---|---|
| nnU-Net segmentation | Yes. The only trained component. |
| CineMA | No. Released weights, inference only. |
| Quantification (volumes, ejection fraction, mass) | No. Deterministic arithmetic. |
| Plausibility gate | No. Rule-based. |
| Retrieval encoders (MedCPT, BioBERT) | No. Frozen, inference only. |
| FAISS index and BM25 | No. No learned parameters. |
| Report generation | No. Prompted, not fine-tuned. |
| Guardrail enforcement | No. Deterministic checks. |
| Seg-Grad-CAM and entropy maps | No. Gradients and statistics over a frozen model. |
| Orchestration | No. |

This is a property of the design, not an omission. Because only one component is trained, every
failure in the system is attributable to a specific, inspectable stage.

### 3.11 Ethical considerations

All three datasets are secondary data, released publicly for research by their originating consortia
under approved protocols with participant consent. No identifiable patient data is handled and no
image ever leaves the machine on which the pipeline runs.

The proposal stated that ethics clearance would therefore not be necessary. That statement is
replaced. A structured expert reader study is planned, in which practising cardiologists or
radiologists rate generated reports on measurement plausibility, guideline fidelity, citation
correctness, explanation usefulness and clinical actionability, with inter-rater agreement reported.
That study collects primary data from human participants and requires ethics approval, which will be
sought under the low-risk route.

The rubric of that study is where its novelty lies rather than in its existence. BAAI already
conducted a six-radiologist study. What it could not assess, because its reports carry no citations,
is citation correctness and guideline fidelity. Those are the axes this study is built around.

---

## 4. Results Obtained So Far

All results in this section were produced by the implemented system and are traceable to files
written by the pipeline. Where a result is incomplete, that is stated.

### 4.1 The data spine and the label-convention finding

The three datasets were unified into a single record format. The concentricity assertion described
in Section 3.3 was applied to all 1,660 ground-truth volumes (830 studies at end diastole and end
systole). After correction of the ACDC convention, all 1,660 pass.

Before correction, the ACDC volumes fail the assertion, which confirms that the inverted convention
is real and not an artefact of the reading code. Table 4.1 gives the measured centroid separations
for representative studies.

**Table 4.1. Myocardium-to-cavity centroid distance, in pixels.**

| Study | Dataset | Distance to label 1 | Distance to label 3 | Left ventricle is |
|---|---|---|---|---|
| patient001 | ACDC | 33.6 | 0.7 | label 3 |
| patient070 | ACDC | 25.9 | 1.1 | label 3 |
| E9H1U4 | M&Ms | 0.3 | 25.3 | label 1 |
| M4P7Q6 | M&Ms | 4.2 | 37.8 | label 1 |
| 002 | M&Ms-2 | 0.8 | 27.5 | label 1 |
| 150 | M&Ms-2 | 2.0 | 41.6 | label 1 |

The separation is unambiguous. Voxel counts alone would not have resolved it: for E9H1U4 the two
cavities contain 6,395 and 6,015 voxels respectively, which is uninformative.

### 4.2 Ground-truth quantification

Volumes, ejection fraction and mass were computed for all 830 studies from expert ground-truth
contours. This provides the reference against which every predicted measurement will be scored.

The distribution behaves as physiology requires, which is the primary check that the loader and the
arithmetic are correct. Table 4.2 gives ejection fraction by pathology.

**Table 4.2. Ground-truth left ventricular ejection fraction by pathology.**

| Pathology | n | Mean LVEF (%) | SD |
|---|---|---|---|
| Myocardial infarction (MINF) | 30 | 30.8 | 8.1 |
| Dilated cardiomyopathy (DCM) | 127 | 41.1 | 21.2 |
| Normal (NOR) | 184 | 60.2 | 6.8 |
| Hypertrophic cardiomyopathy (HCM) | 172 | 63.1 | 9.4 |

The infarction and dilated cardiomyopathy cohorts sit well below the normal cohort, and the
hypertrophic cohort sits at or above it, which is the expected pattern. Across all 830 studies,
ejection fraction ranges from 5.2 to 85.1 per cent and no study has an end-systolic volume greater
than or equal to its end-diastolic volume.

Heart failure categories across the cohort are: preserved 563, reduced 169, mildly reduced 98.

**Sizing the boundary hypothesis.** Of the 830 studies, **221 (26.6 per cent)** have a ground-truth
ejection fraction between 35 and 55 per cent, which is the band in which the boundary-aware
recomputation hypothesis is evaluated. Of these, **80 studies (9.6 per cent of the cohort)** lie
within two percentage points of an actual cut-point, and it is these on which the re-measurement
loop fires. Both figures are large enough to support a statistical test, which was not certain
before it was measured.

### 4.3 Calibration of the plausibility gate

The gate was calibrated against expert ground truth as described in Section 3.4. The calibrated gate
accepts **819 of 830 expert annotations (98.7 per cent)**.

The initial parameterisation of the gate rejected 65 per cent of the ground truth. The cause was
that one-pixel and two-pixel annotation islands, which occur routinely at structure boundaries, were
being counted as disconnected components, and that the myocardial ring legitimately separates into
two arcs near the basal outflow tract. A gate that rejects two thirds of the reference standard does
not measure implausibility; it measures nothing at all. This is recorded here because the failure
mode is instructive: a filter that rejects everything appears rigorous and is exactly as broken as
one that rejects nothing.

The eleven studies the calibrated gate still rejects are not segmentation failures. They are expert
annotations containing anatomically disconnected structures or a gap in the left ventricular stack.
They constitute a small data-quality finding in the reference standard itself.

### 4.4 Segmentation

The CineMA backbone runs on the available hardware and produces the expected accuracy. On four ACDC
test studies, evaluated against ground truth after canonicalisation, the three-seed ensemble
achieves Dice coefficients of 0.93 to 0.97 for the left ventricular cavity, 0.86 to 0.90 for the
myocardium and 0.90 to 0.95 for the right ventricular cavity.

Segmentation of the full cohort is in progress and is a compute-bound task rather than an
unresolved one. Cross-vendor and cross-centre stratification, which is the substance of RQ3, is
pending completion of that run and is reported as outstanding.

nnU-Net was verified end to end with a short training run. Its self-derived configuration is
recorded as a result: a 2D configuration at 1.5625 by 1.5625 mm with a 256 by 224 patch, and a 3D
full-resolution configuration at 5.0 by 1.5625 by 1.5625 mm with a 20 by 256 by 224 patch. Neither
matches the 1.5 by 1.5 by 8 mm resampling and 192 by 192 patch specified in the proposal, which is
the expected outcome of allowing the method to configure itself.

### 4.5 The guideline corpus

The corpus contains **2,592 passages from twelve documents**, of which **50.2 per cent carry a class
of recommendation and a level of evidence**. Table 4.3 gives the composition.

**Table 4.3. Guideline corpus composition.**

| Document | Passages | With CoR and LoE |
|---|---|---|
| 2022 ESC Ventricular Arrhythmias and Sudden Cardiac Death | 607 | 66.9% |
| 2023 ESC Cardiomyopathies | 407 | 46.7% |
| 2021 ESC Heart Failure | 402 | 41.8% |
| 2022 AHA/ACC/HFSA Heart Failure | 250 | 58.8% |
| 2021 ESC/EACTS Valvular Heart Disease | 235 | 42.6% |
| 2024 AHA/ACC Hypertrophic Cardiomyopathy | 177 | 68.9% |
| CMR in the ESC Guidelines (JCMR 2023) | 148 | 57.4% |
| 2021 AHA/ACC Chest Pain | 140 | 59.3% |
| SCMR reference values, indications, protocols, post-processing (4 documents) | 226 | 0% |
| **Total** | **2,592** | **50.2%** |

The four SCMR documents carry no recommendation grades, which is correct: they are reference-range
and technical consensus documents rather than recommendation guidelines. The 2023 ESC
Cardiomyopathies guideline is the most directly relevant addition, since dilated, hypertrophic and
arrhythmogenic cardiomyopathy are the pathology labels of the datasets themselves.

The proposal cited an "ESC 2022 Guidelines on Cardiac Magnetic Resonance Imaging" in three places.
No such document exists. ESC guidance on CMR is distributed across many disease-specific guidelines,
and the aggregated review of that guidance (von Knobelsdorff-Brenkenhoff and Schulz-Menger, 2023) is
included in its place.

### 4.6 Retrieval

Retrieval returns clinically appropriate passages with their recommendation grades attached. A query
for the ejection fraction threshold defining reduced heart failure returns a passage containing the
40 per cent cut-point. A query concerning cardiac magnetic resonance in dilated cardiomyopathy
returns the corresponding recommendation from the ESC Cardiomyopathies guideline at Class I, Level
B. A query concerning implantable defibrillator implantation for primary prevention returns the
recommendation at Class I, Level A.

All four retrieval modes and both embedders run, so every retrieval ablation planned in the
evaluation is executable.

### 4.7 Report generation and fidelity

The reasoning agent generates schema-valid reports carrying passage-level citations with source,
section, page and recommendation grade. Each citation is bound to the specific sentence of the
report that it grounds, which is the object the explainability agent consumes.

On the studies generated so far, all citations resolve and all numbers trace to the quantification
output. The informative result is the behaviour of the raw model before enforcement. **On half of the
studies, the unguarded model violated at least one grounding invariant.** The violations were
fabricated citation identifiers and numbers appearing in the report that the quantification agent
had never produced. Every violation was detected and repaired by the enforcement layer, and no
ungrounded report was emitted.

This is the central argument for enforcing grounding rather than requesting it. A 14-billion
parameter model, explicitly instructed not to invent numbers or citation identifiers, invented them
anyway in half of its outputs.

**Security.** The injection scan quarantines chunks carrying imperative or role-play patterns at
ingest. Across the 2,592 real clinical passages in the corpus, **zero were quarantined**, while five
representative injection payloads were detected in testing. The false-positive rate matters as much
as the detection rate, because a filter that removes genuine guideline text would quietly delete the
corpus.

### 4.8 Orchestration and refusal

The complete pipeline was executed over all 830 studies. **819 studies produced a report and 11 were
refused**, in every case because the plausibility gate rejected the mask and the retry budget was
exhausted. For a refused study, no diagnosis is produced. The per-study trace records which node
refused and why.

The eleven refusals correspond exactly to the eleven expert annotations that fail the calibrated
gate. This deserves to be said plainly rather than left for a reader to notice: in this configuration
the pipeline was supplied with ground-truth contours, so the refusals are refusals of *expert
annotation*, not of model output. The mechanism is the same either way, and the fact that the gate
declines to reason over contours a human drew is arguably the stronger demonstration, but it must be
stated and not glossed.

### 4.9 Summary of completion

**Table 4.4. Status of each component.**

| Component | Status |
|---|---|
| Data spine, 830 studies, label convention verified | Complete |
| Quantification and gate calibration | Complete |
| Guideline corpus, 12 documents, 2,592 passages | Complete |
| Retrieval, all modes and both embedders | Complete |
| Report generation with enforced grounding | Complete |
| Orchestration, refusal, bounded recomputation | Complete |
| Explainability, saliency and uncertainty | Implemented; overlays pending segmentation |
| Evaluation harness and figure generation | Complete |
| Segmentation of the full cohort | In progress (compute-bound) |
| Ablation grid across all configurations | Pending |
| nnU-Net full training | Pending (verified with a short run) |
| BAAI Cardiac Agent on public benchmarks | Pending (see Section 5.2) |
| Expert reader study | Pending ethics approval |

---

## 5. Discussion

### 5.1 Hypotheses

Three hypotheses follow from the objectives, and each maps onto an ablation that the system already
supports.

**H1. Grounding improves clinical correctness.** A report generated from retrieved guideline
passages will contain more clinically correct diagnostic statements than one generated by the same
model from the same measurements without retrieval. Tested by comparing the full configuration
against the no-retrieval configuration, on pathology accuracy, report fidelity and expert rating.

**H2. Boundary-aware recomputation helps where it matters.** Re-measuring when a value lands near a
guideline cut-point reduces heart failure category misclassification for studies in that region.
Tested by comparing the full configuration against the no-feedback configuration, restricted to the
221 studies whose ground-truth ejection fraction lies between 35 and 55 per cent.

**H3. Decomposition makes failure visible.** Per-agent gating catches implausible segmentations
before a report is emitted, at a rate a monolithic pipeline cannot match. Tested by comparing the
full configuration against the no-gate configuration on corrupted and difficult studies, measuring
the proportion of failures caught before emission.

### 5.2 Limitations and open problems, stated openly

**The conflict resolution mechanism has nothing to resolve.** The system detects contradictions
between retrieved guideline passages and resolves them by recency and recommendation class. That
mechanism was run against eight deliberately cross-society queries covering the heart failure
cut-points, natriuretic peptide thresholds, defibrillator and resynchronisation indications, chest
pain pathways and SGLT2 inhibitors. It found **no conflicts**, including on the six queries where
European and American passages were both retrieved. The two societies agree wherever this corpus can
see them. The mechanism is therefore presented as a **safety property**, meaning that the system
cannot silently choose a side, and not as a demonstrated capability. Claiming a demonstrated
conflict-resolution capability on this corpus would not survive a request to see one.

**The M&Ms dilated cardiomyopathy label is not defined by ejection fraction.** Forty-seven of the 97
M&Ms studies labelled DCM have a ground-truth ejection fraction above 55 per cent, with a maximum of
85.1 per cent, because M&Ms uses a clinical diagnosis independent of current function, whereas ACDC
defines the class partly by reduced ejection fraction. One M&Ms study labelled normal has an ejection
fraction of 21.2 per cent. An agent reasoning correctly from ejection fraction will be scored
incorrect on these studies through no fault of its own. The label vocabularies are therefore not
pooled silently, and pathology accuracy is reported per dataset with an explicit statement of which
classes are comparable.

**Report fidelity is 1.0 by construction on the deployed system.** This follows from enforcement and
is not a finding. The reported quantity of interest is the rate at which the guardrail intervened,
together with the unenforced fidelity of the raw model.

**Semantic hallucination is not solved and cannot be.** A report can satisfy every grounding
invariant and still reach a clinically wrong conclusion. No architecture eliminates this, and the
claim will not be made. The expert reader study exists precisely to measure what mechanical
verification cannot.

**Retrieval recall is not measured.** It is known that every citation resolves; it is not known
whether the most relevant passage was retrieved. Measuring that would require a labelled
query-to-passage relevance set, which does not exist for this corpus.

**The BAAI baseline is blocked on a licence, not on a method.** The model weights are released under
Apache 2.0 but are held in a gated repository requiring acceptance of terms. The evaluation script
is written and tested. The model additionally requires more memory than the available workstation
provides, since the multimodal component alone occupies approximately 15 GB in half precision, so
the run is scheduled for the university's high-performance computing facility.

---

## 6. Conclusion and Research Plan

### 6.1 Restatement of the objectives

The objectives are to build a four-agent CMR analysis system whose diagnostic statements are bound
to citable clinical guideline passages, whose visual explanations are linked to those citations, and
which is evaluated on public benchmark data; and to determine whether such grounding improves
clinical correctness, whether measurement refinement near a guideline threshold reduces
misclassification, and how the system generalises across vendors and centres.

### 6.2 Summary of findings so far

The full pipeline is implemented and runs end to end on 830 public CMR studies. Segmentation,
quantification, guideline retrieval, grounded report generation with enforced citation integrity,
refusal on implausible input, and the evaluation harness are all working.

Five findings are already reportable and were not anticipated at the proposal stage. The label
conventions of ACDC and M&Ms are mutually inverted, and the correction is a prerequisite for any
valid cross-dataset measurement. The plausibility gate must be calibrated against expert annotation
or it is meaningless, and once calibrated it accepts 98.7 per cent of ground truth while rejecting
eleven expert contours that are anatomically disconnected. A quarter of the cohort lies in the band
where the boundary hypothesis is tested, which makes that hypothesis viable. The two major guideline
societies do not in fact disagree anywhere this corpus can see, which reframes the conflict
resolution objective as a safety property. And an unguarded language model fabricates citations or
numbers in half of its reports, which is the empirical justification for enforcing grounding rather
than merely requesting it.

### 6.3 Continuation plan

**Table 6.1. Plan to submission.**

| Period | Work |
|---|---|
| July to August 2026 | Complete segmentation of the full cohort on all three datasets. Populate the explainability overlays. Submit the ethics application for the reader study. |
| August to September 2026 | Obtain the BAAI licence and evaluate it on ACDC and M&Ms on the HPC facility. Train nnU-Net to the full schedule. |
| September to October 2026 | Execute the complete ablation grid. Compute all evaluation metrics with confidence intervals, stratified by vendor and centre. |
| October to November 2026 | Conduct the expert reader study, subject to ethics approval. Complete the analysis and the write-up of the results chapter. |
| November 2026 | Candidature defence. |
| December 2026 to mid-2027 | Journal submission of the public-benchmark evaluation. Thesis writing. |
| Late 2027 | Thesis submission. |

The single item on the critical path that is not under the candidate's control is the ethics
approval for the reader study, and the application is being prepared accordingly.

---

## 7. Publications

Tahir, M. (2026). *Transparent Cardiac Intelligence: A Post-Hoc Explainability Framework for
Ensemble-Based Arrhythmia Classification*. Submitted 23 May 2026, under review.

> **Verify before submission.** The proposal records the journal as *BMJ Medical Informatics and
> Decision Making*. There is no such journal; there is a *BMC* Medical Informatics and Decision
> Making. Confirm the title from the submission portal.
>
> Note also that the proposal describes this paper as laying the theoretical groundwork for the
> explainability agent. It does not: it applies SHAP and LIME to a tabular ECG ensemble classifier,
> whereas the explainability agent applies gradient-based saliency to a segmentation network on
> images. Describe it as evidence of prior work in explainable cardiac AI, not as a foundation for
> Agent 4.

A second paper is planned from the public-benchmark evaluation of the BAAI Cardiac Agent, which
would be the first such evaluation reported.

---

## 8. References

Bernard, O., Lalande, A., Zotti, C., Cervenansky, F., Yang, X., Heng, P.-A., et al. (2018). Deep
learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: Is the
problem solved? *IEEE Transactions on Medical Imaging, 37*(11), 2514-2525.

Campello, V. M., Gkontra, P., Izquierdo, C., Martin-Isla, C., Sojoudi, A., Full, P. M., et al.
(2021). Multi-centre, multi-vendor and multi-disease cardiac segmentation: The M&Ms challenge. *IEEE
Transactions on Medical Imaging, 40*(12), 3543-3554.

Heidenreich, P. A., Bozkurt, B., Aguilar, D., Allen, L. A., Byun, J. J., Colvin, M. M., et al.
(2022). 2022 AHA/ACC/HFSA guideline for the management of heart failure. *Circulation, 145*(18),
e895-e1032.

Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: A
self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods,
18*(2), 203-211.

Johnson, J., Douze, M., & Jegou, H. (2021). Billion-scale similarity search with GPUs. *IEEE
Transactions on Big Data, 7*(3), 535-547.

Kawel-Boehm, N., Hetzel, S. J., Ambale-Venkatesh, B., Captur, G., Francois, C. J., Jerosch-Herold,
M., et al. (2020). Reference ranges ("normal values") for cardiovascular magnetic resonance in adults
and children: 2020 update. *Journal of Cardiovascular Magnetic Resonance, 22*(1), 87.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., et al. (2020).
Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information
Processing Systems, 33*, 9459-9474.

McDonagh, T. A., Metra, M., Adamo, M., Gardner, R. S., Baumbach, A., Bohm, M., et al. (2021). 2021
ESC guidelines for the diagnosis and treatment of acute and chronic heart failure. *European Heart
Journal, 42*(36), 3599-3726.

Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond.
*Foundations and Trends in Information Retrieval, 3*(4), 333-389.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2020). Grad-CAM:
Visual explanations from deep networks via gradient-based localization. *International Journal of
Computer Vision, 128*(2), 336-359.

Vinogradova, K., Dibrov, A., & Myers, G. (2020). Towards interpretable semantic segmentation via
gradient-weighted class activation mapping. *Proceedings of the AAAI Conference on Artificial
Intelligence, 34*(10), 13943-13944.

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating text
generation with BERT. *International Conference on Learning Representations*.

---

### References you must verify before submission

I have not fabricated any reference above, but the following could not be verified against the
published record from the working environment, and you must confirm the volume, page and author list
of each from the actual paper before this document is submitted. A thesis about citation fidelity
cannot afford a wrong citation.

| Reference | What to check |
|---|---|
| Qu, T., Zhang, H., et al. (2026). BAAI Cardiac Agent. arXiv:2604.04078 | Full author list, title, and whether it has since appeared in a peer-reviewed venue. |
| Zhang, Y., Bunting, K. V., et al. (2025). CardAIc-Agents. arXiv:2508.13256 | Same. |
| Fu, Y., et al. (2025). CineMA. arXiv:2506.00679 | Confirm the first author and the exact title. |
| Jin, Q., et al. (2023). MedCPT. *Bioinformatics* | Volume, issue, article number. |
| Martin-Isla, C., et al. M&Ms-2 challenge paper | **Not cited above because I could not confirm it.** Find and cite the M&Ms-2 paper; you are using the dataset. |
| von Knobelsdorff-Brenkenhoff, F., & Schulz-Menger, J. (2023). CMR in the ESC guidelines. *JCMR* | Confirm authors and volume. Note this is the correct attribution; the proposal attributed this document to Petersen et al. |
| Arbelo, E., et al. (2023). ESC Cardiomyopathies guideline | Full citation. Currently used in the corpus but not cited in the reference list. |
| Zeppenfeld, K., et al. (2022). ESC Ventricular Arrhythmias guideline | Same. |
| Gulati, M., et al. (2021). AHA/ACC Chest Pain guideline | Same. |
| Ommen, S. R., et al. (2024). AHA/ACC HCM guideline | Same. |
| Vahanian, A., et al. (2021). ESC/EACTS Valvular Heart Disease guideline | Same. |

Every guideline in the corpus must appear in the reference list. Add the five listed above.
