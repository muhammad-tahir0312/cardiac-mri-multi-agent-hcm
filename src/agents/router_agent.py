"""
src/agents/router_agent.py

Router Agent — Selects the most diagnostically relevant DICOM series from a
patient directory before any preprocessing occurs.

Design rationale
----------------
Short-axis (SAX) cardiac cine series are the gold standard for LV/RV
morphology quantification.  This agent assigns a numeric priority score to
every series in a patient directory, filters out series with too few slices
(noise / localizer), and returns the single highest-priority series path.

Priority hierarchy (descending)
--------------------------------
1. ``1-short``  — explicitly annotated short-axis series (highest value)
2. ``short``    — generic short-axis keyword
3. ``body``     — body/balanced-FFE series (fallback)
4. Any series with sufficient slice count (lowest priority)

Usage (programmatic)
---------------------
    from src.agents.router_agent import RouterAgent

    agent = RouterAgent(min_slices=10)
    best = agent.select_series(Path("data/raw/Normal/Directory_1"))
    print(best)  # e.g. PosixPath('data/raw/Normal/Directory_1/1-short')

Usage (CLI demo)
-----------------
    python -m src.agents.router_agent data/raw/Normal/Directory_1
"""

import argparse
import logging
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Priority keywords (lowest index = highest priority).
#: Matching is case-insensitive and substring-based.
PRIORITY_KEYWORDS: List[str] = ["1-short", "short", "body"]

#: Score awarded for each position in ``PRIORITY_KEYWORDS``
#: (index 0 → highest base score).
KEYWORD_BASE_SCORES: Dict[str, int] = {
    kw: len(PRIORITY_KEYWORDS) - idx
    for idx, kw in enumerate(PRIORITY_KEYWORDS)
}

#: Bonus slices-per-point normalization cap (prevents huge series dominating).
MAX_SLICE_BONUS: float = 1.0

#: Default minimum DICOM slices required to consider a series valid.
DEFAULT_MIN_SLICES: int = 10


# ---------------------------------------------------------------------------
# RouterAgent
# ---------------------------------------------------------------------------


class RouterAgent:
    """Selects the most diagnostically relevant series in a patient directory.

    Args:
        min_slices:       Minimum number of image files a series must
                          contain to be considered valid.
        priority_keywords: Ordered list of case-insensitive substring keywords.
                          Earlier entries receive higher scores.
        img_extension:    File extension used to count image slices (default ``.jpg``).

    Attributes:
        min_slices (int):           Validated series must have >= this many slices.
        priority_keywords (List[str]): Lower-cased priority keyword list.
    """

    def __init__(
        self,
        min_slices: int = DEFAULT_MIN_SLICES,
        priority_keywords: Optional[List[str]] = None,
        img_extension: str = ".jpg",
    ) -> None:
        self.min_slices = min_slices
        self.priority_keywords: List[str] = (
            [kw.lower() for kw in priority_keywords]
            if priority_keywords is not None
            else [kw.lower() for kw in PRIORITY_KEYWORDS]
        )
        # Build a per-instance score map so custom keyword lists work correctly.
        self._keyword_scores: Dict[str, int] = {
            kw: len(self.priority_keywords) - idx
            for idx, kw in enumerate(self.priority_keywords)
        }
        self._img_extension = img_extension

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_series(self, patient_dir: Path) -> Optional[Path]:
        """Return the path of the best series within ``patient_dir``.

        Scans all immediate subdirectories, scores each, discards those below
        ``min_slices``, and returns the highest-scoring path.  Returns
        ``None`` when no valid series is found.

        Args:
            patient_dir: Directory corresponding to a single patient
                         (e.g. ``data/raw/Normal/Directory_1``).

        Returns:
            Path to the selected series directory, or ``None``.

        Raises:
            FileNotFoundError: If ``patient_dir`` does not exist.
        """
        if not patient_dir.exists():
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

        candidates = self._gather_candidates(patient_dir)
        if not candidates:
            logger.warning(
                "No series with >= %d slices found in: %s",
                self.min_slices,
                patient_dir,
            )
            return None

        scored = [(self.score_series(path, n_slices), path) for path, n_slices in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_path = scored[0]
        logger.info(
            "Selected series '%s' (score=%.3f, %d slices) from %s",
            best_path.name,
            best_score,
            self._count_slices(best_path),
            patient_dir.name,
        )
        return best_path

    def score_series(self, series_path: Path, n_slices: Optional[int] = None) -> float:
        """Compute a numeric priority score for a single series directory.

        Score = keyword_base_score + normalised_slice_bonus

        The keyword base score is determined by the highest-priority keyword
        found as a substring of the series folder name (case-insensitive).
        If no keyword matches, the base score is 0.  A normalised slice bonus
        (capped at ``MAX_SLICE_BONUS``) prevents extremely large series from
        dominating over short-axis series with fewer frames.

        Args:
            series_path: Path to the series directory.
            n_slices:    Pre-computed DICOM slice count; if ``None`` it will
                         be computed automatically.

        Returns:
            Float priority score (higher is better).
        """
        name_lower = series_path.name.lower()

        # Keyword score — use the instance-level map so custom lists work.
        keyword_score: int = 0
        for kw in self.priority_keywords:
            if kw in name_lower:
                keyword_score = self._keyword_scores.get(kw, 0)
                break  # Use highest-priority match only

        # Slice bonus (normalised to [0, MAX_SLICE_BONUS])
        if n_slices is None:
            n_slices = self._count_slices(series_path)

        slice_bonus = min(n_slices / 100.0, MAX_SLICE_BONUS)

        return float(keyword_score) + slice_bonus

    def rank_all_series(self, patient_dir: Path) -> List[Tuple[float, int, Path]]:
        """Return all series sorted by score, including those below min_slices.

        Useful for diagnostics and debugging.

        Args:
            patient_dir: Patient directory to scan.

        Returns:
            List of ``(score, n_slices, series_path)`` tuples, sorted
            descending by score.

        Raises:
            FileNotFoundError: If ``patient_dir`` does not exist.
        """
        if not patient_dir.exists():
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

        series_dirs = sorted(
            [d for d in patient_dir.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )
        results: List[Tuple[float, int, Path]] = []
        for sd in series_dirs:
            n = self._count_slices(sd)
            s = self.score_series(sd, n)
            results.append((s, n, sd))

        results.sort(key=lambda x: x[0], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _gather_candidates(self, patient_dir: Path) -> List[Tuple[Path, int]]:
        """Return series directories that meet the minimum slice threshold.

        Args:
            patient_dir: Patient directory to scan.

        Returns:
            List of ``(series_path, n_slices)`` tuples.
        """
        candidates: List[Tuple[Path, int]] = []
        series_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]

        for sd in series_dirs:
            n = self._count_slices(sd)
            if n >= self.min_slices:
                candidates.append((sd, n))

        return candidates

    def _count_slices(self, series_dir: Path) -> int:
        """Count DICOM files within a series directory (recursive).

        Args:
            series_dir: Series directory path.

        Returns:
            Number of DICOM files found.
        """
        return len(list(series_dir.rglob(f"*{self._img_extension}")))


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestRouterAgent(unittest.TestCase):
    """Unit tests for :class:`RouterAgent` priority scoring logic.

    These tests use only in-memory :class:`pathlib.Path` objects — no real
    files are required.
    """

    def setUp(self) -> None:
        self.agent = RouterAgent(min_slices=10)

    # --- score_series tests -----------------------------------------------

    def test_1short_highest_priority(self) -> None:
        """The ``1-short`` series must outrank ``short`` and ``body``."""
        p1 = Path("data/raw/Normal/Directory_1/1-short")
        p2 = Path("data/raw/Normal/Directory_1/short")
        p3 = Path("data/raw/Normal/Directory_1/series0001-Body")

        s1 = self.agent.score_series(p1, n_slices=20)
        s2 = self.agent.score_series(p2, n_slices=20)
        s3 = self.agent.score_series(p3, n_slices=20)

        self.assertGreater(s1, s2, "1-short should score higher than short")
        self.assertGreater(s2, s3, "short should score higher than Body")

    def test_short_beats_body(self) -> None:
        """``short`` keyword must score above ``body``."""
        p_short = Path("/some/patient/2-short")
        p_body = Path("/some/patient/series0005-Body")

        self.assertGreater(
            self.agent.score_series(p_short, n_slices=15),
            self.agent.score_series(p_body,  n_slices=15),
        )

    def test_keyword_case_insensitive(self) -> None:
        """Keyword matching must be case-insensitive."""
        p_upper = Path("/patient/SHORT-AXIS")
        p_lower = Path("/patient/short-axis")
        self.assertAlmostEqual(
            self.agent.score_series(p_upper, n_slices=30),
            self.agent.score_series(p_lower, n_slices=30),
        )

    def test_no_keyword_match(self) -> None:
        """A series with no matching keyword should have base score 0."""
        p = Path("/patient/unknown_series")
        score = self.agent.score_series(p, n_slices=10)
        # Base is 0, only slice bonus contributes
        self.assertAlmostEqual(score, min(10 / 100.0, MAX_SLICE_BONUS), places=5)

    def test_slice_bonus_capped(self) -> None:
        """Slice bonus must not exceed ``MAX_SLICE_BONUS``."""
        p = Path("/patient/1-short")
        score_large = self.agent.score_series(p, n_slices=10_000)
        score_small = self.agent.score_series(p, n_slices=20)

        base = float(KEYWORD_BASE_SCORES["1-short"])
        self.assertAlmostEqual(score_large, base + MAX_SLICE_BONUS, places=5)
        self.assertLess(score_small, score_large + 1e-9)

    def test_more_slices_increase_score_up_to_cap(self) -> None:
        """Adding slices should monotonically increase score until cap."""
        p = Path("/patient/series-body")
        prev = self.agent.score_series(p, n_slices=0)
        for n in [5, 10, 50, 99, 100, 200]:
            curr = self.agent.score_series(p, n_slices=n)
            self.assertGreaterEqual(curr, prev)
            prev = curr

    def test_custom_keywords_override_defaults(self) -> None:
        """Custom keyword list must replace, not extend, defaults."""
        agent = RouterAgent(min_slices=5, priority_keywords=["sax", "lax"])
        p_sax = Path("/patient/SAX_cine")
        p_1short = Path("/patient/1-short")

        s_sax = agent.score_series(p_sax, n_slices=15)
        s_1short = agent.score_series(p_1short, n_slices=15)

        # "1-short" is NOT in custom list → base score 0
        self.assertGreater(s_sax, s_1short)

    # --- select_series / rank_all_series tests ----------------------------

    def test_select_series_missing_dir_raises(self) -> None:
        """``select_series`` must raise ``FileNotFoundError`` for missing dirs."""
        with self.assertRaises(FileNotFoundError):
            self.agent.select_series(Path("/nonexistent/path"))

    def test_rank_all_series_missing_dir_raises(self) -> None:
        """``rank_all_series`` must raise ``FileNotFoundError`` for missing dirs."""
        with self.assertRaises(FileNotFoundError):
            self.agent.rank_all_series(Path("/nonexistent/path"))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli() -> None:
    """Minimal CLI to demonstrate the RouterAgent on a real directory."""
    parser = argparse.ArgumentParser(
        description="RouterAgent demo: rank series in a patient directory."
    )
    parser.add_argument(
        "patient_dir", type=Path, nargs="?", default=None,
        help="Path to a patient directory (omit when using --run_tests).",
    )
    parser.add_argument(
        "--min_slices", type=int, default=DEFAULT_MIN_SLICES,
        help="Minimum DICOM slices for a series to be considered valid.",
    )
    parser.add_argument(
        "--run_tests", action="store_true",
        help="Run unit tests instead of scanning a directory.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.run_tests:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestRouterAgent)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

    if args.patient_dir is None:
        parser.error("patient_dir is required when --run_tests is not set.")

    agent = RouterAgent(min_slices=args.min_slices)

    print(f"\nRanking all series in: {args.patient_dir}\n")
    print(f"{'Score':>8}  {'Slices':>7}  Series")
    print("-" * 55)

    try:
        ranked = agent.rank_all_series(args.patient_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    for score, n_slices, path in ranked:
        marker = "  ← VALID" if n_slices >= args.min_slices else ""
        print(f"{score:8.3f}  {n_slices:7d}  {path.name}{marker}")

    best = agent.select_series(args.patient_dir)
    print(f"\nSelected series: {best}")


if __name__ == "__main__":
    _cli()
