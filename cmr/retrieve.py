"""Agent 3a — retrieval over the guideline corpus.

HYBRID, NOT DENSE-ONLY, and that is a deliberate correction to the proposal.

The proposal specifies BioBERT mean-pooling + plain dense FAISS. That retriever is
*strictly weaker* than the one in a paper the thesis itself cites: CardAIc-Agents argues
explicitly that "Dense Passage Retrieval ... often lacks semantic relevance", and its own
ablation shows vector-only and keyword-only both underperform hybrid. Shipping dense-only
BioBERT invites a reviewer to ask why a documented improvement in a cited paper was ignored.

So:
  * MedCPT, not BioBERT. BioBERT is a masked language model; MedCPT is *trained* for
    biomedical retrieval, as a two-tower query/article pair. Both are 768-dim, so nothing
    downstream changes. BioBERT stays as a config-switchable ablation row — it costs
    nothing and turns a weakness into a table entry.
  * Dense + BM25, fused by Reciprocal Rank Fusion. Each half is also selectable alone,
    because `hybrid | dense | bm25 | none` are four rows of the ablation grid.
  * FAISS IndexFlatIP over L2-normalised vectors — exact search. A thousand chunks;
    IVF/HNSW would be pure ceremony.

The dense index is written per-embedder. An index built with MedCPT and then read back
under `embedder: biobert` would still return five plausible-looking passages, and the
ablation row would be quietly meaningless. Separate files make that failure impossible.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

# isort: off
# torch MUST be imported before faiss. Both ship their own copy of libomp; on macOS,
# importing faiss first wins the load and the process then SIGSEGVs inside the first torch
# forward pass. Alphabetical import order (what isort wants) is exactly the fatal order,
# hence the guard. Do not "tidy" these two lines.
import torch
import faiss

# isort: on
import numpy as np
from rank_bm25 import BM25Okapi

from cmr import config
from cmr.corpus import SOURCE_BY_ID, load_chunks
from cmr.types import Chunk

log = logging.getLogger("cmr.retrieve")

__all__ = ["Retriever", "build_index", "build_query", "detect_conflicts"]

_WORD = re.compile(r"[a-z0-9]+")
_COR_RANK = {"I": 0, "IIa": 1, "IIb": 2, "III": 3}  # lower == stronger


def _tokenise(t: str) -> list[str]:
    # len > 2 would drop "40", "35", "50" — the cut-points the whole system turns on.
    return [w for w in _WORD.findall(t.lower()) if len(w) >= 2]


# ─── encoders ────────────────────────────────────────────────────────────────


class _Encoder:
    """MedCPT is a two-tower model: queries and articles get DIFFERENT encoders.

    Using the query encoder on documents (an easy mistake, and a silent one — it still
    returns 768-dim vectors and still ranks something) quietly destroys recall. The
    asymmetry is therefore explicit here rather than implied.
    """

    def __init__(self, cfg: config.Config) -> None:
        from transformers import AutoModel, AutoTokenizer

        which = cfg.retrieval.embedder
        m = cfg.retrieval.models
        if which not in ("medcpt", "biobert"):
            raise ValueError(f"unknown embedder '{which}' (medcpt | biobert)")

        self.device = config.device(cfg)
        self.two_tower = which == "medcpt"
        q_name = m[which]
        d_name = m["medcpt_doc"] if self.two_tower else m[which]
        log.info("embedder=%s query=%s doc=%s device=%s", which, q_name, d_name, self.device)

        self.q_tok = AutoTokenizer.from_pretrained(q_name)
        self.q_model = AutoModel.from_pretrained(q_name).to(self.device).eval()
        if self.two_tower:
            self.d_tok = AutoTokenizer.from_pretrained(d_name)
            self.d_model = AutoModel.from_pretrained(d_name).to(self.device).eval()
        else:
            self.d_tok, self.d_model = self.q_tok, self.q_model

    @torch.no_grad()
    def _encode(self, texts: list[str], tok, model, max_len: int, bs: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            b = tok(
                texts[i : i + bs],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(self.device)
            h = model(**b).last_hidden_state
            # MedCPT is trained to read the [CLS] token. BioBERT has no retrieval head, so
            # it is mean-pooled over the attention mask — the standard sentence recipe.
            if self.two_tower:
                v = h[:, 0, :]
            else:
                mask = b["attention_mask"].unsqueeze(-1).float()
                v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
        return np.vstack(out).astype("float32")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.q_tok, self.q_model, max_len=128)

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.d_tok, self.d_model, max_len=512)


def _paths(cfg: config.Config, embedder: str) -> tuple[Path, Path]:
    d = config.artifacts(cfg, "corpus")
    return d / f"embeddings_{embedder}.npy", d / f"index_{embedder}.faiss"


def build_index(cfg: config.Config) -> None:
    """Embed every chunk; write artifacts/corpus/{embeddings,index}_{embedder}.{npy,faiss}.

    PER-EMBEDDER ONLY. This used to also write unsuffixed `index.faiss` / `embeddings.npy`
    "canonical aliases", which was a trap: nothing ever loaded them (Retriever always opens
    the per-embedder path), and whichever embedder was built LAST silently won the alias — so
    after building medcpt and then biobert, a file called `index.faiss` held BioBERT vectors.
    A file that can quietly disagree with its own name is worse than no file, and the MedCPT
    vs BioBERT ablation is exactly the comparison it would have corrupted.
    """
    chunks = load_chunks(cfg)
    if not chunks:
        raise RuntimeError("no chunks — run scripts/fetch_guidelines.py, then `python -m cmr.corpus`")

    emb_p, idx_p = _paths(cfg, cfg.retrieval.embedder)
    enc = _Encoder(cfg)
    log.info("embedding %d chunks", len(chunks))
    vecs = enc.encode_docs([c.text for c in chunks])

    index = faiss.IndexFlatIP(vecs.shape[1])  # cosine, because the vectors are normalised
    index.add(vecs)

    np.save(emb_p, vecs)
    faiss.write_index(index, str(idx_p))
    log.info("indexed %d chunks (%d-dim) -> %s", len(chunks), vecs.shape[1], idx_p)


# ─── the retriever ───────────────────────────────────────────────────────────


class Retriever:
    def __init__(self, cfg: config.Config) -> None:
        self.cfg = cfg
        self.mode = cfg.retrieval.mode
        self.k = int(cfg.retrieval.top_k)
        self.fetch_k = int(cfg.retrieval.fetch_k)
        self.rrf_k = int(cfg.get_path("retrieval.rrf_k", 60))
        self.chunks = load_chunks(cfg)
        self._index = {c.chunk_id: c for c in self.chunks}

        # BM25 is free to build and is needed by both `bm25` and `hybrid`.
        if self.mode in ("bm25", "hybrid"):
            self.bm25 = BM25Okapi([_tokenise(c.text) for c in self.chunks])

        if self.mode in ("dense", "hybrid"):
            _, idx_p = _paths(cfg, cfg.retrieval.embedder)
            index = faiss.read_index(str(idx_p)) if idx_p.exists() else None
            if index is None or index.ntotal != len(self.chunks):
                log.info("dense index missing or stale (corpus changed) — rebuilding")
                build_index(cfg)
                index = faiss.read_index(str(idx_p))
            self.faiss = index
            self.enc = _Encoder(cfg)

    @property
    def index(self) -> dict[str, Chunk]:
        """chunk_id -> Chunk. factcheck.py resolves every citation against this."""
        return self._index

    def _dense(self, q: str, n: int) -> list[int]:
        v = self.enc.encode_queries([q])
        _, ids = self.faiss.search(v, min(n, len(self.chunks)))
        return [int(i) for i in ids[0] if i >= 0]

    def _sparse(self, q: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(_tokenise(q))
        return [int(i) for i in np.argsort(scores)[::-1][:n] if scores[i] > 0]

    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        k = k or self.k
        if self.mode == "none":
            return []
        if self.mode == "dense":
            ranked = self._dense(query, self.fetch_k)
        elif self.mode == "bm25":
            ranked = self._sparse(query, self.fetch_k)
        elif self.mode == "hybrid":
            ranked = _rrf(
                [self._dense(query, self.fetch_k), self._sparse(query, self.fetch_k)],
                k=self.rrf_k,
            )
        else:
            raise ValueError(f"unknown retrieval mode '{self.mode}' (hybrid|dense|bm25|none)")
        return [self.chunks[i] for i in ranked[:k]]


def _rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: score(d) = sum_r 1 / (k + rank_r(d)).

    Score-free — it fuses ORDERINGS, so a cosine similarity and a BM25 score (which live on
    incomparable scales) combine without any normalisation hand-waving. That is the whole
    reason to prefer it over the usual min-max-and-add.
    """
    score: dict[int, float] = {}
    for r in rankings:
        for rank, doc in enumerate(r):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=lambda d: -score[d])


# ─── query construction ──────────────────────────────────────────────────────


def build_query(measurements: dict, pathology_hint: str | None = None) -> str:
    """Agent 2's numbers -> a retrieval query.

    Deliberately verbose and clinical: the corpus is written in guideline prose, so a query
    phrased as guideline prose retrieves better than a bag of variable names. The raw
    numbers are weak retrieval keys on their own ("48.2" matches nothing), so each is
    paired with the clinical concept it implies.
    """
    parts: list[str] = []
    lvef = measurements.get("lvef_pct")
    rvef = measurements.get("rv_ef_pct")
    cat = measurements.get("hf_category")

    if lvef is not None:
        parts.append(f"left ventricular ejection fraction LVEF {lvef:.0f}%")
        parts.append("LVEF threshold for heart failure classification and management")
    if cat:
        expand = {
            "HFrEF": "heart failure with reduced ejection fraction HFrEF LVEF 40% or less",
            "HFmrEF": "heart failure with mildly reduced ejection fraction HFmrEF LVEF 41 to 49%",
            "HFpEF": "heart failure with preserved ejection fraction HFpEF LVEF 50% or greater",
        }
        parts.append(expand.get(cat, cat))
    if rvef is not None and rvef < 45:
        parts.append(f"right ventricular dysfunction RVEF {rvef:.0f}%")
    if measurements.get("lv_mass_g"):
        parts.append("left ventricular mass hypertrophy cardiac magnetic resonance reference values")
    if pathology_hint:
        parts.append(f"cardiovascular magnetic resonance findings and management in {pathology_hint}")

    parts.append("class of recommendation and level of evidence")
    return ". ".join(parts)


# ─── conflict detection (panel comment C11) ──────────────────────────────────

# ESC and ACC/AHA agree on the heart-failure cut-points (40 / 50). If this returns nothing
# on the HF corpus, THAT IS THE HONEST FINDING, and C11 must be reframed as a safety
# property the system holds rather than a capability it demonstrates. Do not manufacture a
# conflict that is not there.
_THRESHOLD = re.compile(
    r"(LVEF|ejection\s+fraction)[^.]{0,30}?([<>≤≥]=?|less\s+than|greater\s+than|of|at)?\s*(\d{2})\s*%",
    re.I,
)
_DIRECTION = {
    "<": "below", "<=": "below", "≤": "below", "less than": "below",
    ">": "above", ">=": "above", "≥": "above", "greater than": "above",
}
# Generic connective tissue of a guideline heading. Two passages sharing only these are
# not discussing the same clinical question.
_STOP = {
    "the", "of", "for", "in", "with", "and", "patients", "patient", "recommendation",
    "recommendations", "management", "treatment", "therapy", "diagnosis", "evaluation",
    "assessment", "syndrome", "disease", "chronic", "acute", "general", "clinical",
    "considerations", "improved", "table", "figure", "other", "specific", "use",
}
_MIN_SHARED = 2  # a single shared word is a coincidence, not a shared question
_MIN_JACCARD = 0.34


def _thresholds(c: Chunk) -> set[tuple[str, int]]:
    """The (direction, value) cut-points a passage actually asserts."""
    return {
        (_DIRECTION.get((op or "").strip().lower(), "at"), int(val))
        for _metric, op, val in _THRESHOLD.findall(c.text)
    }


def _topic(c: Chunk) -> frozenset[str]:
    """A passage's clinical topic — two cut-points only compete if they answer one question."""
    return frozenset(w for w in _tokenise(c.section) if w not in _STOP and len(w) > 3)


def _same_question(a: Chunk, b: Chunk) -> frozenset[str]:
    """The shared topic of two passages, or empty if they are not answering the same question.

    This gate is the difference between a conflict detector and a conflict *generator*. A
    guideline states many different LVEF cut-points — ≤35% for an ICD, ≤40% for HFrEF,
    ≥50% for HFpEF — and they disagree with each other by design. Pairing them up because
    both headings contain the word "management" invents contradictions that do not exist.
    """
    ta, tb = _topic(a), _topic(b)
    shared = ta & tb
    if len(shared) < _MIN_SHARED:
        return frozenset()
    if len(shared) / len(ta | tb) < _MIN_JACCARD:
        return frozenset()
    return shared


def detect_conflicts(passages: list[Chunk]) -> list[str]:
    """Contradicting thresholds among retrieved passages (C11).

    Resolved by recency, then by class of recommendation. When neither separates them BOTH
    are returned, attributed — the system never silently picks a side, because a silently
    picked side is indistinguishable to the reader from a hallucinated one.
    """
    conflicts: list[str] = []

    for i, a in enumerate(passages):
        for b in passages[i + 1 :]:
            if a.source_id == b.source_id:
                continue  # one document quoting several numbers is not a contradiction
            shared = _same_question(a, b)
            if not shared:
                continue

            for dir_a, val_a in _thresholds(a):
                for dir_b, val_b in _thresholds(b):
                    if dir_a != dir_b or val_a == val_b:
                        continue

                    ya = getattr(SOURCE_BY_ID.get(a.source_id), "year", 0)
                    yb = getattr(SOURCE_BY_ID.get(b.source_id), "year", 0)
                    if ya != yb:
                        winner = a if ya > yb else b
                        why = f"resolved by recency -> {winner.source} ({max(ya, yb)})"
                    else:
                        ra = _COR_RANK.get(a.class_of_recommendation or "", 9)
                        rb = _COR_RANK.get(b.class_of_recommendation or "", 9)
                        if ra != rb:
                            w = a if ra < rb else b
                            why = (
                                "same year; resolved by class of recommendation -> "
                                f"{w.source} (Class {w.class_of_recommendation})"
                            )
                        else:
                            why = "UNRESOLVED — both retained, neither discarded"

                    conflicts.append(
                        f"Conflicting cut-point on '{' '.join(sorted(shared))}': "
                        f"{_attrib(a, dir_a, val_a)}; {_attrib(b, dir_b, val_b)}. {why}."
                    )

    return conflicts


def _attrib(c: Chunk, direction: str, val: int) -> str:
    cor = c.class_of_recommendation or "-"
    loe = c.level_of_evidence or "-"
    return f"{c.source} (p{c.page}, Class {cor}/Level {loe}) gives {direction} {val}%"
