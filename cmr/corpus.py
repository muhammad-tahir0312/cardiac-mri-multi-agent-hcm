"""Guideline PDFs -> Chunks.

The point of this module is the metadata, not the text. Any RAG system can split a PDF.
What neither BAAI's Cardiac Agent nor CardAIc-Agents carries is the *strength* of the
recommendation a passage encodes — its Class of Recommendation and Level of Evidence.
Without CoR/LoE a citation is decoration: it tells you a guideline mentioned something,
not whether the guideline told you to do it. With CoR/LoE, Agent 3 can say "Class I,
Level A" and Agent 4 can rank two conflicting passages. That is the contribution.

Two societies, two grading vocabularies, three table layouts:

    ESC      recommendation text, then class+level fused by the PDF text layer:
             "An ACE-I is recommended ... death.110-113 IA"    -> I / A
    ACC/AHA  a COR|LOE cell, EITHER alone on its line above the recommendation:
             "1 B-NR" / "1. In patients with HF, vital signs ..."
             OR opening the recommendation line itself:
             "1 B-NR 1. In patients with suspected HCM, a TTE ..."

That second ACC/AHA layout is the one this module used to drop on the floor. Only the
standalone cell was matched, so every inline row went untagged — 44 of the 122 graded
rows in the 2024 HCM deck, 24 of 147 in the 2022 HF deck. Both layouts are handled now,
and they occur in the *same* document, so neither can be assumed away.

ACC/AHA's vocabulary is NOT ESC's with different glyphs. COR is 1/2a/2b/3 (and COR 3
splits into "No Benefit" vs "Harm"); LOE is A/B-R/B-NR/C-LD/C-EO. We normalise onto the
ESC scale (I/IIa/IIb/III, A/B/C) because that is the single axis every consumer already
reads — report.py, xai.py, and retrieve._COR_RANK, which has to rank an ESC passage
against an ACC/AHA one. The normalisation is lossy in exactly one direction, so the
verbatim tokens survive alongside it in Chunk.cor_raw / Chunk.loe_raw:

    "3: Harm" + B-NR  ->  cor=III  loe=B  cor_raw="3: Harm"  loe_raw="B-NR"

Rank on the normalised field; quote the raw one. Nothing is discarded, nothing is faked.

Documents with NO grading scheme at all (the four SCMR/JCMR consensus papers: reference
values, clinical indications, post-processing, protocols) are style="none" and are never
tagged. They are in the corpus because they answer "how is LV mass contoured" and "is CMR
indicated here", which the society guidelines do not. They necessarily *lower* the
headline CoR/LoE percentage, because that percentage is a property of corpus composition,
not of the extractor. Read the per-document table, never the single number.

Chunking is deliberately embedder-independent: chunk_id must not change when the
retrieval ablation switches from MedCPT to BioBERT, or the ablation rows would be
comparing different corpora. Token counts are therefore whitespace words, not wordpieces.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from cmr import config, guardrails
from cmr.types import Chunk

log = logging.getLogger("cmr.corpus")


# ─── the corpus registry ─────────────────────────────────────────────────────
# `url = None` means: not obtainable by script. See scripts/fetch_guidelines.py.


@dataclass(frozen=True)
class Source:
    source_id: str
    source: str
    filename: str
    url: str | None
    style: str  # esc | acc_aha | none  -> which CoR/LoE extractor to run
    year: int
    note: str = ""


SOURCES: tuple[Source, ...] = (
    Source(
        source_id="cmr_esc_2023",
        source="CMR in the ESC Guidelines (JCMR 2023)",
        filename="cmr_in_esc_guidelines_2023.pdf",
        url="https://jcmr-online.biomedcentral.com/counter/pdf/10.1186/s12968-023-00950-z.pdf",
        style="esc",
        year=2023,
        # PMC10364363. NB: authored by von Knobelsdorff-Brenkenhoff & Schulz-Menger,
        # NOT Petersen et al. as the thesis proposal states. CC-BY.
    ),
    Source(
        source_id="esc_hf_2021",
        source="2021 ESC Heart Failure Guideline",
        filename="esc_hf_2021.pdf",
        url="https://orbi.uliege.be/bitstream/2268/290864/1/ehab368.pdf",
        style="esc",
        year=2021,
        # Eur Heart J 42(36):3599. Publisher copy is behind Cloudflare; this is the
        # green-OA deposit in the University of Liege repository (listed by Unpaywall).
    ),
    Source(
        source_id="esc_hf_2023",
        source="2023 ESC Focused Update on Heart Failure",
        filename="esc_hf_2023_focused_update.pdf",
        url=None,  # MANUAL — see fetch_guidelines.py
        style="esc",
        year=2023,
    ),
    Source(
        source_id="aha_hf_2022",
        source="2022 AHA/ACC/HFSA Heart Failure Guideline (AHA slide set)",
        filename="aha_hf_2022_slides.pdf",
        url="https://professional.heart.org/en/science-news/-/media/832EA0F4E73948848612F228F7FA2D35.ashx",
        style="acc_aha",
        year=2022,
        note=(
            "The full Circulation text is NOT scriptable (Cloudflare) and the JACC/JCF "
            "co-publications are closed access. This is the AHA's own official slide set: "
            "it reproduces every recommendation table with COR/LOE verbatim, but not the "
            "narrative prose. Named as a slide set so no citation ever overstates it."
        ),
    ),
    Source(
        source_id="aha_chest_pain_2021",
        source="2021 AHA/ACC Chest Pain Guideline (AHA slide set)",
        filename="aha_chest_pain_2021_slides.pdf",
        url="https://professional.heart.org/-/media/DEC633637F414AD5BD6937B3059ECAB9.ashx",
        style="acc_aha",
        year=2021,
        note="Same caveat as aha_hf_2022. Circulation full text is closed (Unpaywall: is_oa=false).",
    ),
    Source(
        source_id="aha_hcm_2024",
        source="2024 AHA/ACC/AMSSM/HRS/PACES/SCMR Hypertrophic Cardiomyopathy Guideline (AHA slide set)",
        filename="aha_hcm_2024_slides.pdf",
        url=(
            "https://professional.heart.org/-/media/PHD-Files-2/Science-News/2/2024/"
            "2024-Guideline-for-HCM-Slide-Set.pdf?sc_lang=en"
        ),
        style="acc_aha",
        year=2024,
        note=(
            "Circulation full text (10.1161/CIR.0000000000001250) is CLOSED — Unpaywall "
            "is_oa=false, not in Europe PMC, JACC co-publication closed too. This is the "
            "AHA's own official slide set: every recommendation table with COR/LOE verbatim, "
            "but no narrative prose. Named as a slide set so no citation overstates it. "
            "HCM is a dataset pathology label, so this is worth having even truncated."
        ),
    ),
    # ── ESC guidelines whose pathologies ARE the dataset's labels ────────────
    Source(
        source_id="esc_cardiomyopathies_2023",
        source="2023 ESC Guidelines on Cardiomyopathies",
        filename="esc_cardiomyopathies_2023.pdf",
        url=(
            "https://pure.amsterdamumc.nl/ws/files/156880448/"
            "2023-esc-guidelines-for-the-management-of-cardiomyopathies.pdf"
        ),
        style="esc",
        year=2023,
        note=(
            "Eur Heart J 44(37):3503, doi 10.1093/eurheartj/ehad194. The publisher copy is "
            "'bronze' OA behind Cloudflare; this is the CC-BY author-accepted deposit in the "
            "Amsterdam UMC Pure repository, which Unpaywall lists as an OA location. "
            "DCM / HCM / ARVC — i.e. literally the ACDC pathology labels."
        ),
    ),
    Source(
        source_id="esc_va_scd_2022",
        source="2022 ESC Guidelines on Ventricular Arrhythmias and Sudden Cardiac Death",
        filename="esc_va_scd_2022.pdf",
        url=(
            "https://pure.amsterdamumc.nl/ws/files/156878041/"
            "2022-esc-guidelines-for-the-management-of-patients-with-ventricular-arrhythmias"
            "-and-the-prevention-of-sudden-cardiac-dea.pdf"
        ),
        style="esc",
        year=2022,
        note=(
            "Eur Heart J 43(40):3997, doi 10.1093/eurheartj/ehac262. Unpaywall oa_status=green; "
            "CC-BY deposit, Amsterdam UMC Pure. Carries the ICD primary-prevention LVEF "
            "cut-points and the heaviest CMR/LGE content of any ESC guideline."
        ),
    ),
    Source(
        source_id="esc_vhd_2021",
        source="2021 ESC/EACTS Guidelines on Valvular Heart Disease",
        filename="esc_vhd_2021.pdf",
        url=(
            "https://pure.amsterdamumc.nl/ws/files/139990392/"
            "2021-esceacts-guidelines-for-the-management-of-valvular-heart-disease.pdf"
        ),
        style="esc",
        year=2021,
        note="Eur Heart J 43(7):561, doi 10.1093/eurheartj/ehab395. CC-BY deposit, Amsterdam UMC Pure.",
    ),
    # ── Added because a DATASET PATHOLOGY LABEL had no guideline speaking to it. ──────
    # The corpus is not "every cardiology guideline"; provenance purity is a stated
    # contribution (PLAN.md §0.4), and an irrelevant guideline is retrieval noise, not
    # coverage. Each entry below is here because a class in ACDC / M&Ms / M&Ms-2 was
    # otherwise unanswerable. Deliberately NOT added: atrial fibrillation, sports
    # cardiology, pericardial disease — all downloadable, none corresponding to any label
    # in these three cohorts.
    Source(
        source_id="esc_achd_2020",
        source="2020 ESC Guidelines on Adult Congenital Heart Disease",
        filename="esc_achd_2020.pdf",
        url=(
            "https://pure.amsterdamumc.nl/ws/files/139989532/"
            "2020-esc-guidelines-for-the-management-of-adult-congenital-heart-disease.pdf"
        ),
        style="esc",
        year=2020,
        note=(
            "Eur Heart J 42(6):563, doi 10.1093/eurheartj/ehaa554. CC-BY deposit, Amsterdam UMC "
            "Pure. THE BIGGEST GAP THIS CLOSES: M&Ms-2 labels FALL (tetralogy of Fallot), CIA "
            "(interatrial communication) and TRI (tricuspid regurgitation), and NOTHING in the "
            "corpus previously spoke to any of them — the RV-focused congenital classes were "
            "being reasoned about with no guideline at all. Also the main source of RV volume "
            "and pulmonary-regurgitation criteria, which is where CMR is the reference standard."
        ),
    ),
    Source(
        source_id="esc_ph_2022",
        source="2022 ESC/ERS Guidelines on Pulmonary Hypertension",
        filename="esc_ph_2022.pdf",
        url=(
            "https://pure.amsterdamumc.nl/ws/files/139991498/"
            "2022-escers-guidelines-for-the-diagnosis-and-treatment-of-pulmonary-hypertension.pdf"
        ),
        style="esc",
        year=2022,
        note=(
            "Eur Heart J 43(38):3618, doi 10.1093/eurheartj/ehac237. CC-BY deposit, Amsterdam UMC "
            "Pure. Carries the RV-dysfunction and RV-dilatation criteria — i.e. ACDC's 'RV' class "
            "(abnormal right ventricle) and M&Ms-2's 'RV' (dilated right ventricle), which the "
            "HF and cardiomyopathy guidelines only address in passing."
        ),
    ),
    Source(
        source_id="esc_acs_2023",
        source="2023 ESC Guidelines on Acute Coronary Syndromes",
        filename="esc_acs_2023.pdf",
        url="https://orbi.uliege.be/bitstream/2268/329460/1/ehad191.pdf",
        style="esc",
        year=2023,
        note=(
            "Eur Heart J 44(38):3720, doi 10.1093/eurheartj/ehad191. Green-OA deposit, University "
            "of Liege ORBi (the publisher copy is bronze OA behind Cloudflare). Acute MI — the "
            "origin of ACDC's MINF class (prior myocardial infarction)."
        ),
    ),
    Source(
        source_id="esc_ccs_2024",
        source="2024 ESC Guidelines on Chronic Coronary Syndromes",
        filename="esc_ccs_2024.pdf",
        url="https://pure.amsterdamumc.nl/ws/files/136127232/ehae177.pdf",
        style="esc",
        year=2024,
        note=(
            "Eur Heart J 45(36):3415, doi 10.1093/eurheartj/ehae177. CC-BY deposit, Amsterdam UMC "
            "Pure. M&Ms's IHD class, and the guideline that actually grades CMR for ischaemia and "
            "myocardial viability — the single most CMR-relevant recommendation set outside the "
            "cardiomyopathy guideline."
        ),
    ),
    Source(
        source_id="esc_pacing_2021",
        source="2021 ESC Guidelines on Cardiac Pacing and Cardiac Resynchronization Therapy",
        filename="esc_pacing_2021.pdf",
        url=(
            "https://pure.amsterdamumc.nl/ws/files/139990292/"
            "2021-esc-guidelines-on-cardiac-pacing-and-cardiac-resynchronization-therapy.pdf"
        ),
        style="esc",
        year=2021,
        note=(
            "Eur Heart J 42(35):3427, doi 10.1093/eurheartj/ehab364. CC-BY deposit, Amsterdam UMC "
            "Pure. Carries the CRT LVEF cut-points (<=35%), which are LVEF DECISION BOUNDARIES the "
            "system can land near — i.e. exactly what H2's boundary-aware recomputation fires on, "
            "and a cut-point the corpus did not previously state outside the ICD context."
        ),
    ),
    Source(
        source_id="esc_htn_2024",
        source="2024 ESC Guidelines on Elevated Blood Pressure and Hypertension",
        filename="esc_htn_2024.pdf",
        url=None,  # MANUAL — see below
        style="esc",
        year=2024,
        note=(
            "Eur Heart J 45(38):3912, doi 10.1093/eurheartj/ehae178. WANTED for M&Ms's HHD class "
            "(hypertensive heart disease, n=15) — and because distinguishing hypertensive LV "
            "hypertrophy from HCM is a classic CMR question the corpus cannot currently answer. "
            "NOT SCRIPTABLE: bronze OA at the publisher (Cloudflare 403); the Bern (BORIS) and "
            "Glasgow (Enlighten) deposits serve an HTML login page, not the PDF, and the figshare "
            "record carries metadata with no file attached. All four checked 2026-07-13. Download "
            "by hand from https://academic.oup.com/eurheartj/article/45/38/3912/7741010 and save "
            "as <guidelines>/esc_htn_2024.pdf — corpus.py picks it up automatically."
        ),
    ),
    # ── SCMR/EACVI consensus. No CoR/LoE anywhere in these documents (verified,
    #    not assumed): they grade nothing, they specify how to acquire, post-process
    #    and report a CMR study. style="none" so we never invent a grade for them.
    Source(
        source_id="scmr_ref_2020",
        source="SCMR Reference Values (Kawel-Boehm 2020)",
        filename="scmr_reference_values_2020.pdf",
        url="https://jcmr-online.biomedcentral.com/track/pdf/10.1186/s12968-020-00683-3",
        style="none",  # reference ranges + the 1.05 g/mL density constant; no CoR/LoE
        year=2020,
    ),
    Source(
        source_id="scmr_indications_2020",
        source="SCMR Position Paper on Clinical Indications for CMR (Leiner 2020)",
        filename="scmr_indications_2020.pdf",
        url="https://jcmr-online.biomedcentral.com/track/pdf/10.1186/s12968-020-00682-4",
        style="none",
        year=2020,
        note=(
            "JCMR 22:76, doi 10.1186/s12968-020-00682-4. Gold OA, CC-BY. This is the document "
            "that answers 'is CMR indicated in <pathology>' — it grades indications on SCMR's "
            "own appropriateness scale, NOT on CoR/LoE, so it is correctly style='none'."
        ),
    ),
    Source(
        source_id="scmr_postproc_2020",
        source="SCMR Standardized Post-Processing (Schulz-Menger 2020 update)",
        filename="scmr_postprocessing_2020.pdf",
        url="https://jcmr-online.biomedcentral.com/track/pdf/10.1186/s12968-020-00610-6",
        style="none",
        year=2020,
        note=(
            "JCMR 22:19, doi 10.1186/s12968-020-00610-6. Gold OA, CC-BY. Defines how LV/RV "
            "volumes and mass are contoured — i.e. the provenance of Agent 2's arithmetic."
        ),
    ),
    Source(
        source_id="scmr_protocols_2020",
        source="SCMR Standardized CMR Imaging Protocols (Kramer 2020 update)",
        filename="scmr_protocols_2020.pdf",
        url="https://jcmr-online.biomedcentral.com/track/pdf/10.1186/s12968-020-00607-1",
        style="none",
        year=2020,
        note="JCMR 22:17, doi 10.1186/s12968-020-00607-1. Gold OA, CC-BY.",
    ),
)

SOURCE_BY_ID: dict[str, Source] = {s.source_id: s for s in SOURCES}


# ─── CoR / LoE extraction ────────────────────────────────────────────────────

# A CoR/LoE token alone is not enough: "IA" and "1 A" both occur in ordinary prose, and
# tagging those would make the reported coverage a lie. A chunk is only tagged if a token
# appears AND the chunk is plausibly a recommendation — by verb, by section, or by the
# table header. Terse ESC rows ("BNP/NT-proBNP  IB") have no verb at all, which is why
# the section/header signals are needed and not merely belt-and-braces.
_REC_CUE = re.compile(
    r"is\s+recommended|are\s+recommended|should\s+be\s+considered|may\s+be\s+considered"
    r"|is\s+indicated|are\s+indicated|is\s+not\s+recommended|are\s+not\s+recommended"
    r"|is\s+reasonable|can\s+be\s+useful|should\s+be\s+performed|is\s+of\s+benefit"
    r"|is\s+contraindicated|should\s+be\s+avoided|may\s+be\s+used",
    re.I,
)
_REC_SECTION = re.compile(r"recommendation", re.I)
_REC_TABLE_HDR = re.compile(r"Recommendations?\s+Class|Class\s*a?\s*Level|\bCOR\b\s+\bLOE\b", re.I)

# ESC: "...death.110-113 IA"  /  "...symptoms.  IIaB"
_ESC = re.compile(r"(?<![A-Za-z])(III|IIa|IIb|IIA|IIB|I)\s?([ABC])(?![A-Za-z])")

# ACC/AHA: "1 B-NR", "2a C-EO", "3: No Benefit B-R", "3: Harm C-LD".
# The COR is 1/2a/2b/3 and the LOE is A/B-R/B-NR/C-LD/C-EO — a DIFFERENT vocabulary from
# ESC's I/IIa/IIb/III + A/B/C, not a typographic variant of it.
_AHA_CELL_RE = r"(1|2a|2b|3)(\s*:\s*(?:No\s+Benefit|Harm))?\s+(A|B-R|B-NR|C-LD|C-EO)"
_AHA = re.compile(rf"(?<![\w-]){_AHA_CELL_RE}(?![\w-])")

# The same cell appears in TWO layouts, often in the same deck, and only the first was
# ever handled — which silently dropped every recommendation laid out the second way
# (most of the 2024 HCM deck, a third of the 2022 HF deck):
#
#   standalone  the cell is alone on its line, and the recommendation follows below:
#                   "1 B-NR"
#                   "1. In patients with HF, vital signs ..."
#   inline      the cell opens the line and the recommendation runs on from it:
#                   "1 B-NR 1. In patients with suspected HCM, a TTE is recommended ..."
_AHA_STANDALONE = re.compile(rf"^\s*{_AHA_CELL_RE}\s*$")
_AHA_INLINE = re.compile(rf"^\s*{_AHA_CELL_RE}\s+(?=\S)")

_COR_MAP = {"1": "I", "2a": "IIa", "2b": "IIb", "3": "III", "iia": "IIa", "iib": "IIb"}
_LOE_MAP = {"A": "A", "B-R": "B", "B-NR": "B", "C-LD": "C", "C-EO": "C"}


@dataclass(frozen=True)
class Grade:
    """A parsed recommendation grade: normalised for ranking, verbatim for quoting."""

    cor: str  # ESC scale, always:  I | IIa | IIb | III
    loe: str  # ESC scale, always:  A | B | C
    cor_raw: str  # as printed:  "3: Harm", "2a", "IIa"
    loe_raw: str  # as printed:  "B-NR", "C-EO", "B"
    scheme: str  # ESC | ACC_AHA


def _esc_grade(m: re.Match[str]) -> Grade:
    cor = _COR_MAP.get(m.group(1).lower(), m.group(1))
    return Grade(cor, m.group(2), cor, m.group(2), "ESC")


def _aha_grade(m: re.Match[str]) -> Grade:
    """Normalise an ACC/AHA cell onto the ESC scale, keeping what was actually printed.

    Lossy, deliberately and in one direction only:
        "3: No Benefit" and "3: Harm"  both collapse to  III
        B-R and B-NR                   both collapse to  B
        C-LD and C-EO                  both collapse to  C
    Those are real clinical distinctions, so cor_raw/loe_raw carry them through unharmed.
    Every downstream consumer (report.py, xai.py, retrieve._COR_RANK) reads the normalised
    field, which is why normalisation is not optional — but nothing is thrown away.
    """
    qualifier = (m.group(2) or "").strip()  # ": Harm" | ": No Benefit" | ""
    cor_raw = m.group(1) + (f"{qualifier}" if qualifier else "")
    loe_raw = m.group(3)
    return Grade(_COR_MAP[m.group(1)], _LOE_MAP[loe_raw], cor_raw, loe_raw, "ACC_AHA")


def extract_grade(text: str, style: str, section: str = "") -> Grade | None:
    """The full Grade for a passage, or None. Normalised + verbatim; see Grade.

    NOTE: build_corpus does NOT use this. It tags at the *row* level inside _read_pdf,
    where the class/level is still adjacent to the text it qualifies; this function can
    only see a finished chunk, and a 350-word chunk that straddles four table rows cannot
    say which row a given "IIa B" belongs to. This is kept as the standalone extractor for
    ad-hoc text (and it is what the unit tests pin the regexes against) — but the corpus is
    built by the stricter path, and the two must not be confused.
    """
    if style == "none":
        return None
    is_rec = _REC_CUE.search(text) or _REC_TABLE_HDR.search(text) or _REC_SECTION.search(section)
    if not is_rec:
        return None

    if style == "esc":
        m = _ESC.search(text)
        return _esc_grade(m) if m else None
    if style == "acc_aha":
        m = _AHA.search(text)
        return _aha_grade(m) if m else None
    return None


def extract_cor_loe(text: str, style: str, section: str = "") -> tuple[str | None, str | None]:
    """(class_of_recommendation, level_of_evidence), normalised onto the ESC scale.

    The narrow, back-compatible face of extract_grade().
    """
    g = extract_grade(text, style, section)
    return (g.cor, g.loe) if g else (None, None)


# ─── PDF -> (section, page, sentence) ────────────────────────────────────────

_LIGATURE = re.compile(r"/C\d+")  # Type-1 glyph codes that pypdf leaves behind
_NUMBERED_HEADING = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-Z].{2,80})$")
_REC_HEADING = re.compile(r"^\s*(Recommendations?\s+for\s+.{3,90})$", re.I)
_TABLE_HEADING = re.compile(r"^\s*(Table\s+\d+[.:]\s*.{3,90})$", re.I)
_REFS_HEADING = re.compile(r"^\s*(?:\d{1,2}\.?\s*)?References\s*$", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_ABBREV = re.compile(r"\b(e\.g|i\.e|vs|cf|Fig|Ref|Dr|Prof|approx|et\s+al)\.", re.I)

# A reference list is 20+ pages of an ESC guideline and is pure poison for retrieval:
# it is dense in exactly the medical nouns a query matches on, and grounds nothing.
_REF_MARKER = re.compile(r"et\s+al\.|\b(19|20)\d{2};\s*\d+|\bdoi:|https?://doi\.org")


def _clean(text: str) -> str:
    # NFKC folds the ligatures the ESC/AHA typesetters bake into the text layer: without
    # it the corpus literally says "atrial ﬁbrillation" (U+FB01) and BM25 can never match
    # a query that says "fibrillation".
    text = unicodedata.normalize("NFKC", text)
    text = _LIGATURE.sub(" ", text)
    text = text.replace("­", "")  # soft hyphen
    # ESC PDFs render ≤ and ≥ as "<_" and ">_". Left alone, every cut-point in the corpus
    # ("LVEF <_40%") is unreadable to both the reader and detect_conflicts' threshold regex.
    text = text.replace("<_", "≤").replace(">_", "≥")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate across line breaks
    return text


def _is_heading(line: str) -> str | None:
    """A heading, or None. Conservative: a false positive only mislabels `section`,
    but this must never swallow a numbered ACC/AHA recommendation ('1. In patients...')."""
    for pat in (_REC_HEADING, _TABLE_HEADING):
        m = pat.match(line)
        if m:
            return m.group(1).strip()

    m = _NUMBERED_HEADING.match(line)
    if not m:
        return None
    title = m.group(2).strip()
    # Headings are short, title-like, and are not sentences or table cells.
    if title.endswith(".") or "," in title or len(title.split()) > 12:
        return None
    if not re.search(r"[a-z]", title):  # e.g. the ACC/AHA cell "1 B-NR"
        return None
    return f"{m.group(1)} {title}"


def _sentences(block: str) -> list[str]:
    """Sentence split that will not cut inside a sentence, and survives 'e.g.'."""
    guarded = _ABBREV.sub(lambda m: m.group(0).replace(".", "\x00"), block)
    out = []
    for s in _SENT_SPLIT.split(guarded):
        s = " ".join(s.replace("\x00", ".").split())
        if s:
            out.append(s)
    return out


def _is_references(text: str) -> bool:
    """Bibliography prose, by marker density. Backstop for the 'References' heading."""
    words = max(len(text.split()), 1)
    return len(_REF_MARKER.findall(text)) / words > 0.02


@dataclass
class Unit:
    """One sentence of prose, or one whole recommendation row (`grade` set)."""

    section: str
    page: int
    text: str
    grade: Grade | None = None


# A recommendation row, detected on the *line* as the PDF lays it out.
#   ESC     the class+level close the row:  "...and death.110-113  IA"
#   ACC/AHA the COR|LOE cell either sits alone above the row, or opens the row itself.
_ESC_ROW_END = re.compile(r"(?:^|\s)(III|IIa|IIb|IIA|IIB|I)\s?([ABC])\s*$")
# Table footers ("ACE-I = angiotensin-converting enzyme inhibitor; ARB = ...") close a table.
_TABLE_FOOTER = re.compile(r"(\w+\s*=\s*\w+.*){2,}")
# Slide/page furniture that lands inside a recommendation row and is not part of it.
_ROW_NOISE = re.compile(r"^\s*\d{1,3}\s*$")

# A real recommendation row is a sentence or two. Anything longer means the table-close
# was missed and prose is bleeding into the row — refuse to tag it rather than assert a
# Class/Level over 600 words we did not actually parse.
_ROW_MAX_WORDS = 150


def _read_pdf(path: Path, style: str) -> list[Unit]:
    """-> Units in document order. Deterministic.

    Recommendation rows are recognised here, on the line, where the class/level is still
    adjacent to the text it qualifies. Doing it later — on a 350-word chunk that straddles
    four table rows — can only guess which row a given "IIa B" belongs to, and guessing is
    how a citation ends up asserting a strength the guideline never gave it.
    """
    reader = PdfReader(str(path))
    n_pages = len(reader.pages)
    section = "Front matter"
    units: list[Unit] = []
    prose: list[str] = []
    row: list[str] = []
    pending: Grade | None = None  # ACC/AHA: the cell that opens (or precedes) the row text
    in_table = False
    page_no = 1

    def flush_prose() -> None:
        for s in _sentences(" ".join(prose)):
            if not _is_references(s):
                units.append(Unit(section, page_no, s))
        prose.clear()

    def flush_row(grade: Grade | None) -> None:
        text = " ".join(" ".join(row).split())
        row.clear()
        if grade and 4 <= len(text.split()) <= _ROW_MAX_WORDS:
            units.append(Unit(section, page_no, text, grade))
        elif text:  # an un-tagged straggler, or an over-long one, is prose
            prose.append(text)

    for page_no, page in enumerate(reader.pages, start=1):
        raw = _clean(page.extract_text() or "")
        if not raw.strip():
            continue

        for line in raw.split("\n"):
            # The bibliography runs to the end of the document. Stop; do not chunk it.
            if _REFS_HEADING.match(line) and page_no > n_pages // 2:
                flush_row(pending)
                flush_prose()
                return units

            # An ACC/AHA cell is tested BEFORE anything else, and without consulting
            # in_table. "1 B-NR 1. In patients with HCM ..." is not a sentence that occurs
            # in prose by accident — the cell IS the table, wherever it is printed — and a
            # slide deck's tables are not reliably announced by a header line. Testing it
            # first also stops _is_heading from mistaking the cell for a numbered heading.
            if style == "acc_aha":
                cell = _AHA_STANDALONE.match(line) or _AHA_INLINE.match(line)
                if cell:
                    flush_row(pending)  # close the previous row
                    pending = _aha_grade(cell)
                    in_table = True
                    rest = line[cell.end() :].strip()  # inline layout: the row starts here
                    if rest:
                        row.append(rest)
                    continue

            head = _is_heading(line)
            if head:
                flush_row(pending)
                pending = None
                flush_prose()
                section = head
                in_table = bool(_REC_HEADING.match(line))
            if _REC_TABLE_HDR.search(line):
                in_table = True
            elif _TABLE_FOOTER.search(line):
                flush_row(pending)
                pending = None
                in_table = False

            if not in_table:
                prose.append(line)
                continue

            if style == "acc_aha":
                if pending and not _ROW_NOISE.match(line):
                    row.append(line)  # continuation of the open recommendation
                elif not pending:
                    prose.append(line)
                continue

            if style == "esc":
                end = _ESC_ROW_END.search(line)
                if end:
                    row.append(line[: end.start()])
                    flush_row(_esc_grade(end))
                    continue
                row.append(line)
                continue

            prose.append(line)

        # A row never spans a page break in these documents; a table may.
        flush_row(pending)
        pending = None
        flush_prose()

    return units


# ─── chunking ────────────────────────────────────────────────────────────────


def _pack(units: list[Unit], size: int, overlap: int) -> list[Unit]:
    """Sentence-aware packing to ~`size` words, carrying ~`overlap` words forward.

    A sentence is never split. A recommendation row is never merged with anything: it is
    its own chunk, so its CoR/LoE describes every word of the chunk that carries it.

    A section boundary is ALSO never crossed. Without this, a heading-detection miss (an
    ESC section title containing a comma, say — see _is_heading) or a row whose class/level
    never matched _ESC_ROW_END silently degrades into untagged prose, and that stray prose
    sits in `units` right next to genuinely unrelated content from the following section.
    Packing purely by word count then glues them into one chunk — which is exactly how the
    single sentence "Reduced LVEF is defined as <=40%... HFrEF" (2021 ESC HF, p.14, section
    "3.2 Terminology") ended up diluted inside ~180 words of an unrelated AF-management
    recommendation table, and neither BM25 nor the dense embedding could surface it for an
    HFrEF query anymore. A chunk's `section` field is a promise about what it contains; this
    is what keeps that promise instead of just reporting it.
    """
    chunks: list[Unit] = []
    cur: list[Unit] = []

    def flush() -> list[Unit]:
        if not cur:
            return []
        head = cur[0]
        chunks.append(Unit(head.section, head.page, " ".join(u.text for u in cur)))
        tail: list[Unit] = []
        used = 0
        for u in reversed(cur):
            w = len(u.text.split())
            if used + w > overlap:
                break
            tail.insert(0, u)
            used += w
        return tail

    n = 0
    for u in units:
        if u.grade:  # a recommendation row stands alone
            flush()
            cur, n = [], 0
            chunks.append(u)
            continue
        if cur and cur[-1].section != u.section:  # never blend two sections, even via overlap
            flush()
            cur, n = [], 0
        w = len(u.text.split())
        if cur and n + w > size:
            cur = flush()
            n = sum(len(x.text.split()) for x in cur)
        cur.append(u)
        n += w

    flush()
    return chunks


def build_corpus(cfg: config.Config) -> list[Chunk]:
    """PDFs -> chunks -> artifacts/corpus/chunks.jsonl.

    Re-runnable and deterministic: the same PDF yields byte-identical chunk_ids, because
    chunk_id = sha256(text)[:16] and nothing upstream of the text depends on wall time,
    dict order, or the embedder.
    """
    gdir = Path(cfg.paths.guidelines)
    size = int(cfg.retrieval.chunk_tokens)
    overlap = int(cfg.retrieval.chunk_overlap)

    chunks: list[Chunk] = []
    seen: set[str] = set()
    quarantined: list[tuple[str, int, list[str]]] = []

    for src in SOURCES:
        pdf = gdir / src.filename
        if not pdf.exists():
            log.warning("missing %-20s %s — run scripts/fetch_guidelines.py", src.source_id, pdf.name)
            continue

        units = _read_pdf(pdf, src.style)
        n_before = len(chunks)
        tagged = 0

        for u in _pack(units, size, overlap):
            # SECURITY BOUNDARY. A PDF is an untrusted byte stream: text can be white-on-white,
            # in a form field, or in a footnote nobody reads. An ESC guideline is not a realistic
            # attacker, but a future corpus (a hospital's own protocol documents) is far less
            # trustworthy, and the defence costs nothing. Quarantine at INGEST, so a poisoned
            # chunk never enters the index and no query can ever surface it.
            hits = guardrails.scan_injection(u.text)
            if hits:
                quarantined.append((src.source_id, u.page, hits))
                log.error(
                    "QUARANTINED chunk from %s p.%d — prompt-injection pattern(s): %s",
                    src.source_id, u.page, hits,
                )
                continue

            text = guardrails.sanitise(u.text)  # defang role markers / fences, keep the clinical text
            cid = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            if cid in seen:  # running headers/footers repeat verbatim; keep the first
                continue
            seen.add(cid)
            g = u.grade
            tagged += g is not None
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=text,
                    source=src.source,
                    source_id=src.source_id,
                    section=u.section,
                    page=u.page,
                    class_of_recommendation=g.cor if g else None,
                    level_of_evidence=g.loe if g else None,
                    cor_scheme=g.scheme if g else None,
                    cor_raw=g.cor_raw if g else None,
                    loe_raw=g.loe_raw if g else None,
                    tokens=len(text.split()),
                )
            )

        made = len(chunks) - n_before
        pct = 100.0 * tagged / made if made else 0.0
        log.info("%-20s %4d chunks  %4d with CoR/LoE (%.1f%%)", src.source_id, made, tagged, pct)

    out = config.artifacts(cfg, "corpus") / "chunks.jsonl"
    with open(out, "w") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")

    if quarantined:
        log.error("SECURITY: %d chunk(s) quarantined at ingest for prompt-injection "
                  "patterns and are NOT in the index", len(quarantined))
    else:
        log.info("security: 0 chunks quarantined (no injection patterns in the corpus)")

    total_tagged = sum(c.class_of_recommendation is not None for c in chunks)
    log.info(
        "corpus: %d chunks from %d documents, %d with CoR/LoE (%.1f%%) -> %s",
        len(chunks),
        len({c.source_id for c in chunks}),
        total_tagged,
        100.0 * total_tagged / len(chunks) if chunks else 0.0,
        out,
    )
    return chunks


def load_chunks(cfg: config.Config) -> list[Chunk]:
    path = config.artifacts(cfg, "corpus") / "chunks.jsonl"
    if not path.exists():
        return build_corpus(cfg)
    with open(path) as f:
        return [Chunk.model_validate_json(line) for line in f if line.strip()]


def chunk_index(cfg: config.Config) -> dict[str, Chunk]:
    """chunk_id -> Chunk. factcheck.py resolves every Citation against this."""
    return {c.chunk_id: c for c in load_chunks(cfg)}


def coverage(chunks: list[Chunk]) -> dict[str, dict]:
    """Per-document CoR/LoE coverage. Reported as-is — this number is not to be inflated."""
    total: Counter[str] = Counter()
    cor: Counter[str] = Counter()
    loe: Counter[str] = Counter()
    for c in chunks:
        total[c.source_id] += 1
        cor[c.source_id] += c.class_of_recommendation is not None
        loe[c.source_id] += c.level_of_evidence is not None
    return {
        sid: {
            "chunks": total[sid],
            "cor": cor[sid],
            "loe": loe[sid],
            "cor_pct": round(100.0 * cor[sid] / total[sid], 1),
            "loe_pct": round(100.0 * loe[sid] / total[sid], 1),
        }
        for sid in total
    }


if __name__ == "__main__":
    cfg = config.load()
    cs = build_corpus(cfg)
    log.info("coverage:\n%s", json.dumps(coverage(cs), indent=2))
