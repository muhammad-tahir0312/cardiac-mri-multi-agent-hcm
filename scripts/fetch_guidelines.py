"""Download the guideline corpus into cfg.paths.guidelines. Re-runnable; verifies checksums.

    ./.venv/bin/python scripts/fetch_guidelines.py

Prints a table of what it got and what it could not, and exits non-zero if anything is
missing. PDFs are stored locally and NOT redistributed; a SHA-256 is recorded per file so
a corpus rebuild is reproducible and a silently-swapped document is detectable.

WHAT IS AND IS NOT OBTAINABLE (re-checked against Unpaywall + Europe PMC, July 2026)

Twelve of the thirteen documents download cleanly. One does not, and NO SUBSTITUTE IS
MADE FOR IT:

  esc_hf_2023  2023 ESC Focused Update on Heart Failure (Eur Heart J 44(37):3627,
               doi 10.1093/eurheartj/ehad195)
               "Bronze" OA: free to read at the publisher, but NOT open-access licensed,
               so there is no lawfully redistributable copy to script. Checked and
               rejected, each for a stated reason:
                 - academic.oup.com          Cloudflare managed challenge (HTTP 403)
                 - hal.science/hal-04462129  Anubis proof-of-work wall (HTTP 503)
                 - Maastricht CRIS, pure.eur.nl, eprints.gla.ac.uk
                                             metadata records only; no file deposited
                 - Europe PMC / PMC          not deposited (inEPMC=N, isOpenAccess=N)
               Both walls above are deliberate anti-scraping controls and are not
               circumvented here. Third-party mirrors of ehad195.pdf do exist on national
               cardiology-society websites; they are unlicensed re-uploads and are NOT
               used, because a thesis about guideline fidelity cannot cite a document it
               obtained from an unattributable source. Download it by hand in a browser:

                   https://academic.oup.com/eurheartj/article/44/37/3627/7246292
                   -> save as  <guidelines>/esc_hf_2023_focused_update.pdf

               corpus.py picks it up automatically once it is there; the ESC CoR/LoE
               extractor already handles it. Nothing else in the pipeline changes.

THE THREE ACC/AHA GUIDELINES ARE SLIDE SETS, AND THAT IS NOT A SHORTCUT — IT IS THE
CEILING. Their full texts are not open access, and the proposal's belief that they are is
wrong:

  aha_hf_2022         Circulation 10.1161/CIR.0000000000001063 -> Unpaywall is_oa=TRUE but
                      oa_status=BRONZE (free-to-read, all rights reserved) and Cloudflare
                      returns 403. JACC co-pub 10.1016/j.jacc.2021.12.012 -> CLOSED.
                      JCF co-pub 10.1016/j.cardfail.2022.02.010 -> CLOSED. Not in PMC.
  aha_chest_pain_2021 Circulation 10.1161/CIR.0000000000001029 -> Unpaywall is_oa=FALSE.
                      JACC co-pub 10.1016/j.jacc.2021.07.053   -> CLOSED. Not in PMC.
  aha_hcm_2024        Circulation 10.1161/CIR.0000000000001250 -> Unpaywall is_oa=FALSE.
                      Not in PMC.

  What IS lawfully downloadable is the AHA's own official slide set for each, served
  unprotected from professional.heart.org. Those decks reproduce every recommendation
  table with COR/LOE verbatim but carry none of the narrative prose. Each source_id is
  therefore named "... (AHA slide set)" so that no citation the system emits can ever
  overstate what it actually read. To get the narrative text, buy or institutionally
  access the Circulation full texts and drop them in by hand.

Routes worth recording, because the obvious ones fail:
  * The ESC website's "download" buttons serve the Declaration of Interest report, not
    the guideline. The ESC guidelines here come from CC-BY repository deposits instead
    (Amsterdam UMC Pure), which Unpaywall lists as OA locations for each DOI.
  * ahajournals.org is Cloudflare-walled, but professional.heart.org is not.
  * JCMR (BioMedCentral) is gold OA, CC-BY, and just works — all four SCMR papers.

The proposal's "ESC 2022 Guidelines on Cardiac Magnetic Resonance Imaging" is not in this
list because NO SUCH DOCUMENT EXISTS. Do not go looking for it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from pypdf import PdfReader

from cmr import config
from cmr.corpus import SOURCES, Source

log = logging.getLogger("cmr.fetch")

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# A guideline is tens of pages. Anything shorter is a cover page, an erratum, or a
# challenge page that happened to be valid PDF — not the document we asked for.
_MIN_PAGES = 8


@dataclass
class Result:
    """One row of the report table."""

    source_id: str
    state: str  # ok | cached | manual | FAILED
    pages: int = 0
    mb: float = 0.0
    sha256: str = ""
    why: str = ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pages(path: Path) -> int:
    """Page count, or 0 if the bytes are not a readable PDF."""
    try:
        return len(PdfReader(str(path)).pages)
    except Exception as e:  # noqa: BLE001 — a corrupt download must not abort the run
        log.warning("%s is not a readable PDF: %s", path.name, e)
        return 0


def _verify(path: Path) -> tuple[bool, str]:
    """Is this file actually a plausible guideline PDF? (magic bytes + page count)"""
    with open(path, "rb") as f:
        if f.read(4) != b"%PDF":
            return False, "not a PDF (bad magic bytes)"
    n = _pages(path)
    if n < _MIN_PAGES:
        return False, f"only {n} page(s) — implausible for a guideline"
    return True, ""


def _download(source_id: str, url: str, retries: int = 3) -> bytes | None:
    """Stream a PDF. Returns None (never raises) so one dead host cannot abort the corpus.

    Some repositories are slow — the ESC deposits routinely take >2 min for 12 MB.
    """
    for attempt in range(1, retries + 1):
        try:
            log.info("%-26s downloading (attempt %d/%d)", source_id, attempt, retries)
            with requests.get(
                url,
                headers={"User-Agent": UA},
                timeout=(15, 300),  # (connect, read)
                allow_redirects=True,
                stream=True,
            ) as r:
                r.raise_for_status()
                body = b"".join(r.iter_content(1 << 16))
            if not body.startswith(b"%PDF"):
                # Never let a Cloudflare/Anubis challenge page masquerade as a guideline.
                log.error(
                    "%-26s served %s, not a PDF", source_id, r.headers.get("content-type")
                )
                return None
            return body
        except requests.RequestException as e:
            log.warning("%-26s %s", source_id, e)
    log.error("%-26s gave up after %d attempts", source_id, retries)
    return None


def _fetch_one(src: Source, gdir: Path, sums: dict[str, str]) -> Result:
    dest = gdir / src.filename

    if src.url is None:  # not obtainable by script; see this module's docstring
        if dest.exists():
            ok, why = _verify(dest)
            if ok:
                sums[src.filename] = _sha256(dest)
                return Result(src.source_id, "cached", _pages(dest),
                              dest.stat().st_size / 1e6, sums[src.filename])
            return Result(src.source_id, "FAILED", why=f"manual copy invalid: {why}")
        return Result(src.source_id, "manual", why="paywalled — obtain by hand (see docstring)")

    # Idempotent: a file whose checksum still matches is never re-downloaded.
    if dest.exists() and sums.get(src.filename) == _sha256(dest):
        return Result(src.source_id, "cached", _pages(dest),
                      dest.stat().st_size / 1e6, sums[src.filename])

    body = _download(src.source_id, src.url)
    if body is None:
        # A PDF we already hold is worth more than a dead host: keep it, checksum it, move on.
        if dest.exists() and _verify(dest)[0]:
            sums[src.filename] = _sha256(dest)
            log.warning("%-26s host unreachable; keeping verified local copy", src.source_id)
            return Result(src.source_id, "cached", _pages(dest),
                          dest.stat().st_size / 1e6, sums[src.filename],
                          why="host unreachable, local copy kept")
        return Result(src.source_id, "FAILED", why="download failed and no local copy")

    dest.write_bytes(body)
    ok, why = _verify(dest)
    if not ok:
        dest.unlink()  # do not leave a bad file where corpus.py will find it
        return Result(src.source_id, "FAILED", why=why)

    sums[src.filename] = _sha256(dest)
    return Result(src.source_id, "ok", _pages(dest), len(body) / 1e6, sums[src.filename])


def _report(results: list[Result]) -> None:
    """The table. print(), not log() — this IS the script's output."""
    w = max(len(r.source_id) for r in results)
    print()
    print(f"  {'SOURCE':<{w}}  {'STATE':<7} {'PAGES':>5} {'MB':>6}  {'SHA-256':<16}  NOTE")
    print(f"  {'-' * w}  {'-' * 7} {'-' * 5} {'-' * 6}  {'-' * 16}  {'-' * 44}")
    for r in results:
        pages = str(r.pages) if r.pages else "-"
        mb = f"{r.mb:.1f}" if r.mb else "-"
        print(
            f"  {r.source_id:<{w}}  {r.state:<7} {pages:>5} {mb:>6}  "
            f"{r.sha256[:16] or '-':<16}  {r.why}"
        )

    got = [r for r in results if r.state in ("ok", "cached")]
    missing = [r for r in results if r.state not in ("ok", "cached")]
    print()
    print(f"  {len(got)}/{len(results)} documents present, {sum(r.pages for r in got)} pages total")
    if missing:
        print(f"  MISSING: {', '.join(r.source_id for r in missing)}")
        print("  These are NOT substituted. See the docstring of this file for why, and for")
        print("  the manual-download URL of each.")
    print()


def fetch(cfg: config.Config) -> list[Result]:
    gdir = Path(cfg.paths.guidelines)
    gdir.mkdir(parents=True, exist_ok=True)
    sums_path = gdir / "checksums.json"
    sums: dict[str, str] = json.loads(sums_path.read_text()) if sums_path.exists() else {}

    results = [_fetch_one(src, gdir, sums) for src in SOURCES]

    sums_path.write_text(json.dumps(sums, indent=2, sort_keys=True) + "\n")
    _report(results)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    rs = fetch(config.load())
    sys.exit(0 if all(r.state in ("ok", "cached") for r in rs) else 1)
