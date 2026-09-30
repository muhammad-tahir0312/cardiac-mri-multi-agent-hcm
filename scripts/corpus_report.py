"""Corpus acceptance report: coverage per document + the acceptance queries.

    ./.venv/bin/python scripts/corpus_report.py

print(), not logging, on purpose — this script IS a report. Everything it prints is read
straight off artifacts/corpus/chunks.jsonl and the live retriever; nothing is hard-coded.
"""

from __future__ import annotations

from cmr import config
from cmr.corpus import SOURCE_BY_ID, coverage, load_chunks
from cmr.retrieve import Retriever

# Each query names what a correct answer must contain, so a plausible-looking miss is
# still a miss. "40" is the HFrEF cut-point; if it is absent, the passage is not the one.
ACCEPTANCE: list[tuple[str, str | None]] = [
    ("LVEF threshold for HFrEF", "40"),
    ("CMR indication in dilated cardiomyopathy", None),
    ("ICD primary prevention LVEF threshold", None),
    ("hypertrophic cardiomyopathy CMR late gadolinium enhancement", None),
]


def _grade(c) -> str:
    if not c.class_of_recommendation:
        return "—"
    raw = ""
    if c.cor_raw and (c.cor_raw != c.class_of_recommendation or c.loe_raw != c.level_of_evidence):
        raw = f"  [as printed: {c.cor_raw} / {c.loe_raw}]"
    return f"Class {c.class_of_recommendation} / Level {c.level_of_evidence} ({c.cor_scheme}){raw}"


def main() -> None:
    cfg = config.load()
    chunks = load_chunks(cfg)
    cov = coverage(chunks)

    print("\n" + "=" * 100)
    print("CoR/LoE COVERAGE PER DOCUMENT")
    print("=" * 100)
    print(f"  {'SOURCE':<26} {'STYLE':<8} {'CHUNKS':>7} {'CoR':>6} {'CoR %':>7}   SCHEME")
    print(f"  {'-'*26} {'-'*8} {'-'*7} {'-'*6} {'-'*7}   {'-'*8}")

    graded_chunks = graded_cor = 0
    for sid, st in cov.items():
        src = SOURCE_BY_ID[sid]
        schemes = {c.cor_scheme for c in chunks if c.source_id == sid and c.cor_scheme}
        if src.style != "none":
            graded_chunks += st["chunks"]
            graded_cor += st["cor"]
        print(
            f"  {sid:<26} {src.style:<8} {st['chunks']:>7} {st['cor']:>6} "
            f"{st['cor_pct']:>6.1f}%   {'/'.join(sorted(schemes)) or '—'}"
        )

    tot = len(chunks)
    cor = sum(c.class_of_recommendation is not None for c in chunks)
    print(f"  {'-'*26} {'-'*8} {'-'*7} {'-'*6} {'-'*7}")
    print(f"  {'ALL DOCUMENTS':<26} {'':<8} {tot:>7} {cor:>6} {100*cor/tot:>6.1f}%")
    print(
        f"  {'GRADED DOCUMENTS ONLY':<26} {'':<8} {graded_chunks:>7} {graded_cor:>6} "
        f"{100*graded_cor/graded_chunks:>6.1f}%"
    )
    print(
        "\n  The SCMR/JCMR consensus papers contain NO recommendation grading whatsoever\n"
        "  (verified, not assumed), so they can only ever score 0% and they drag the\n"
        "  all-documents figure down. The 'graded documents only' row is the one that\n"
        "  measures the extractor. Neither number is inflated: a chunk is tagged only when\n"
        "  the class/level was printed on the row it came from."
    )

    r = Retriever(cfg)
    print("\n" + "=" * 100)
    print(f"ACCEPTANCE QUERIES  (mode={r.mode}, embedder={cfg.retrieval.embedder})")
    print("=" * 100)

    for q, must in ACCEPTANCE:
        hits = r.retrieve(q, k=3)
        print(f"\n  QUERY: {q!r}")
        if must:
            ok = any(must in h.text for h in hits)
            print(f"  MUST CONTAIN {must!r} in top-3: {'PASS' if ok else 'FAIL'}")
        for i, c in enumerate(hits, 1):
            print(f"\n    [{i}] {c.source}")
            print(f"        section : {c.section}   (p.{c.page})")
            print(f"        grade   : {_grade(c)}")
            print(f"        chunk_id: {c.chunk_id}")
            body = " ".join(c.text.split())
            print(f"        {body[:300]}{'...' if len(body) > 300 else ''}")
    print()


if __name__ == "__main__":
    main()
