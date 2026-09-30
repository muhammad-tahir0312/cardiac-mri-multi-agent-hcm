"""The viewer. `cmr serve` -> http://127.0.0.1:8000

Two halves. Everything up to the "Live runs" section is READ-ONLY: it never runs a model,
never writes a file, never mutates a run. Everything it shows is already on disk; if it is
not on disk, the panel says so instead of inventing it.

It exists to make four things visible in sixty seconds, because those four things are
what BAAI Cardiac Agent and CardAIc-Agents cannot show:

  1  passage-level citations INSIDE the report      -> the citation chips
  2  the LINK heatmap <-> cited guideline sentence  -> the "linked evidence" panel
  3  the system REFUSING to diagnose (H3)           -> the REFUSED rows and their traces
  4  hallucination MEASURED, per report             -> the fidelity panel, in red

NIfTI is rendered to PNG here, server-side, with matplotlib. The browser gets pixels.

The "Live runs" section (/api/live/*) is the one deliberate exception: it uploads a new
subject's ED/ES phases, runs the real graph on them, and streams progress over SSE. See
that section's own docstring for the honesty rules it still obeys.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import uuid
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from . import config as cfgmod
from .checks import LabelOrderError, assert_canonical
from .cli import resolved_llm_model
from .seg import _CKPT_NATIVE_ORDER as _SEG_CHECKPOINTS
from .types import LABEL_NAMES, Subject

log = logging.getLogger("cmr.app")

CFG = cfgmod.load()
ART = Path(CFG.paths.artifacts)
RUNS = ART / "runs"
UPLOADS = ART / "uploads"  # raw bytes of live-uploaded subjects, so volumes() can re-render them
UI_CONFIG = ART / "ui_config.json"  # pipeline settings the Config page edits (see FIELDS)

STRUCTURE_COLOUR = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}  # Okabe-Ito, as in xai.py

# Only the SAX checkpoints are wired up in seg.py (LAX is a 2-D pipeline it raises on) —
# reuse ITS list rather than duplicating it, so a new checkpoint only has to be added once.
UPLOAD_CHECKPOINTS = tuple(c for c in _SEG_CHECKPOINTS if not c.endswith("lax_4c"))
_MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # a SAX cine phase is a few MB; generous, not unlimited
_MIN_NIFTI_BYTES = 352  # the NIfTI-1 header is 348 bytes; anything shorter cannot be one


# ─────────────────────────────────────────────────────────────────────────────
# Artifacts, read once, cached. The viewer is read-only, so a cache never goes stale
# within a session — and the panel must not stall for 830 subjects at the viva.
# ─────────────────────────────────────────────────────────────────────────────
def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


@lru_cache(maxsize=1)
def corpus() -> dict[str, dict]:
    p = ART / "corpus" / "chunks.jsonl"
    if not p.exists():
        log.warning("no corpus at %s — citations cannot be resolved", p)
        return {}
    out = {}
    with open(p) as f:
        for line in f:
            c = json.loads(line)
            out[c["chunk_id"]] = c
    log.info("corpus: %d chunks", len(out))
    return out


@lru_cache(maxsize=1)
def cohort() -> dict[str, dict]:
    """subject_id -> dataset / pathology / vendor / ground-truth LVEF. From the manifest."""
    import pandas as pd

    p = ART / "gt_measurements.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    keep = ["dataset", "split", "pathology", "vendor", "centre", "lvef_pct", "hf_category"]
    df = df.set_index("subject_id")[[c for c in keep if c in df.columns]]
    return {k: {kk: _clean(vv) for kk, vv in v.items()} for k, v in df.to_dict("index").items()}


def _clean(v: Any) -> Any:
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v.item() if isinstance(v, np.generic) else v


@lru_cache(maxsize=4)
def _sha256_of(path: Path, _mtime: float) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def corpus_sha256() -> str | None:
    """Hash of the corpus THIS SERVER is resolving citations against.

    A chunk_id is `sha256(chunk_text)[:16]` (see corpus.build_corpus) — it is content-
    addressed. So rebuilding the corpus with different chunk boundaries changes every
    chunk_id, and a report generated against an older build cites ids that no longer exist.
    That is a corpus-version mismatch, NOT a hallucination, and the two must never be
    confused: `cmr run` writes the corpus hash it used into run.json, and api_subject
    compares it with this. Getting this wrong would mean the viewer accusing the model of
    fabricating a citation it did not fabricate — in the one panel the whole thesis rests on.
    """
    p = ART / "corpus" / "chunks.jsonl"
    return _sha256_of(p, p.stat().st_mtime) if p.exists() else None


def _trace(cid: str, sid: str) -> list[dict]:
    p = RUNS / cid / "traces" / f"{sid}.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def _stamp(cid: str) -> float:
    """Cheap staleness key: a directory's mtime changes when a file is added to it. So a
    run that finishes while the viewer is open is picked up on the next request, and one
    that is not moving costs a single stat()."""
    rd = RUNS / cid
    return max((p.stat().st_mtime for p in (rd, rd / "reports", rd / "traces") if p.exists()),
               default=0.0)


def run_table(cid: str) -> list[dict]:
    return _run_table(cid, _stamp(cid))


@lru_cache(maxsize=8)
def _run_table(cid: str, _stale_key: float) -> list[dict]:
    """One row per subject that the graph actually touched.

    A REFUSED subject has a trace and NO report — that asymmetry is the whole point, so
    the table is keyed on traces, not on reports. Keying it on reports would make the
    11 refusals silently vanish, which is exactly the failure mode the thesis is about.
    """
    rd = RUNS / cid
    if not rd.is_dir():
        raise HTTPException(404, f"no run '{cid}'")
    rep_dir, tr_dir = rd / "reports", rd / "traces"

    if tr_dir.is_dir():
        sids = sorted(p.stem for p in tr_dir.glob("*.jsonl"))
    elif rep_dir.is_dir():
        sids = sorted(p.stem for p in rep_dir.glob("*.json"))
    else:
        return []

    rows = []
    meta_all = cohort()
    for sid in sids:
        meta = meta_all.get(sid, {})
        rep = _read_json(rep_dir / f"{sid}.json")
        if rep:
            m, f = rep.get("measurements") or {}, rep.get("fidelity") or {}
            r = rep.get("report") or {}
            rows.append({
                "subject_id": sid,
                # cohort() only knows ACDC/MnMs/MnMs2 (it is built from
                # gt_measurements.parquet); a live-uploaded subject falls back to the
                # dataset the report itself recorded (graph.write_report — see its docstring).
                "dataset": meta.get("dataset") or rep.get("dataset", ""),
                "pathology": meta.get("pathology", ""),
                "lvef_pct": _clean(m.get("lvef_pct")),
                "hf_category": m.get("hf_category") or "",
                "status": rep.get("status", "ok"),
                "refused": False,
                "numeric_fidelity": f.get("numeric_fidelity"),
                "citation_fidelity": f.get("citation_fidelity"),
                "n_citations": len(r.get("citations") or []),
                "n_unsupported": len(f.get("unsupported_numbers") or []),
                "n_dangling": len(f.get("dangling_citations") or []),
                "violations": "",
                "iterations": rep.get("iterations", 0),
            })
            continue

        # No report on disk. Reconstruct from the trace — this is a refusal (H3).
        ev = _trace(cid, sid)
        q = next((e for e in reversed(ev) if e.get("node") == "quantify"), {})
        g = next((e for e in reversed(ev) if e.get("node") == "gate"), {})
        refused = any(e.get("node") == "refuse" for e in ev)
        rows.append({
            "subject_id": sid,
            "dataset": meta.get("dataset", ""),
            "pathology": meta.get("pathology", ""),
            "lvef_pct": q.get("lvef"),
            "hf_category": q.get("hf") or "",
            "status": "failed_segmentation" if refused else "no_report",
            "refused": refused,
            "numeric_fidelity": None,
            "citation_fidelity": None,
            "n_citations": 0,
            "n_unsupported": 0,
            "n_dangling": 0,
            "violations": "; ".join(g.get("violations") or []),
            "iterations": max((e.get("iteration", 0) for e in ev), default=0),
        })
    log.info("run %s: %d subjects, %d refused", cid, len(rows), sum(r["refused"] for r in rows))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Volumes. A targeted loader: data.py only exposes generators, and walking 700 subjects
# of NIfTI to reach subject 701 is not a thing you do while a panel watches.
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=16)
def volumes(dataset: str, sid: str) -> dict[str, Any] | None:
    from . import data
    from .checks import canonicalise

    try:
        if dataset == "ACDC":
            root = Path(CFG.paths.acdc)
            pdir = next((root / s / sid for s in ("training", "testing")
                         if (root / s / sid).is_dir()), None)
            if pdir is None:
                return None
            info = data._read_info_cfg(pdir / "Info.cfg")
            ed, es = int(info["ED"]), int(info["ES"])
            img_ed, sp = data._load(pdir / f"{sid}_frame{ed:02d}.nii.gz")
            img_es, _ = data._load(pdir / f"{sid}_frame{es:02d}.nii.gz")
            gt_ed, _ = data._load(pdir / f"{sid}_frame{ed:02d}_gt.nii.gz")
            gt_es, _ = data._load(pdir / f"{sid}_frame{es:02d}_gt.nii.gz")
            gt_ed = canonicalise(gt_ed.astype(np.uint8), "acdc")
            gt_es = canonicalise(gt_es.astype(np.uint8), "acdc")

        elif dataset == "MnMs":
            root = Path(CFG.paths.mnms)
            sdir = next((d for d in root.glob(f"*/*/{sid}") if d.is_dir()), None) or next(
                (d for d in root.glob(f"*/{sid}") if d.is_dir()), None)
            if sdir is None:
                return None
            row = data._mnms_meta(CFG).loc[sid]
            ed, es = int(row["ED"]), int(row["ES"])
            img4, sp = data._load(sdir / f"{sid}_sa.nii.gz")
            gt4, _ = data._load(sdir / f"{sid}_sa_gt.nii.gz")
            img_ed, img_es = img4[..., ed], img4[..., es]
            gt_ed, gt_es = gt4[..., ed].astype(np.uint8), gt4[..., es].astype(np.uint8)

        elif dataset == "MnMs2":
            sdir = Path(CFG.paths.mnms2) / "dataset" / sid
            if not sdir.is_dir():
                return None
            img_ed, sp = data._load(sdir / f"{sid}_SA_ED.nii")
            img_es, _ = data._load(sdir / f"{sid}_SA_ES.nii")
            gt_ed, _ = data._load(sdir / f"{sid}_SA_ED_gt.nii")
            gt_es, _ = data._load(sdir / f"{sid}_SA_ES_gt.nii")
            gt_ed, gt_es = gt_ed.astype(np.uint8), gt_es.astype(np.uint8)

        elif dataset == "Upload":
            # No dataset tree to search: the raw bytes were saved at upload time (see
            # api_live_upload) to exactly these two paths.
            p_ed, p_es = UPLOADS / f"{sid}_ed.nii.gz", UPLOADS / f"{sid}_es.nii.gz"
            if not (p_ed.exists() and p_es.exists()):
                return None
            img_ed, sp = data._load(p_ed)
            img_es, _ = data._load(p_es)
            gt_p_ed, gt_p_es = UPLOADS / f"{sid}_ed_gt.nii.gz", UPLOADS / f"{sid}_es_gt.nii.gz"
            gt_ed = data._load(gt_p_ed)[0].astype(np.uint8) if gt_p_ed.exists() else None
            gt_es = data._load(gt_p_es)[0].astype(np.uint8) if gt_p_es.exists() else None
        else:
            return None
    except Exception as e:  # a missing file is a graceful "no image", not a 500
        log.warning("cannot load %s/%s: %s", dataset, sid, e)
        return None

    return {"img_ed": img_ed.astype(np.float32), "img_es": img_es.astype(np.float32),
            "gt_ed": gt_ed, "gt_es": gt_es, "spacing": sp}


def _pred(kind: str, dataset: str, sid: str, phase: str, ckpt: str | None = None) -> np.ndarray | None:
    """Agent 1's own output, if `cmr segment` (or a live run) has reached this subject.

    `ckpt` defaults to the global config's checkpoint — correct for every batch run, since
    the ablation grid never varies segmentation. A live upload lets the user pick a DIFFERENT
    checkpoint per run, so api_subject() passes the one that run's own run.json recorded
    instead of assuming the default.
    """
    ckpt = ckpt or CFG.segmentation.checkpoint
    if kind == "mask":
        p = ART / "masks" / ckpt / dataset / f"{sid}_{phase}.nii.gz"
        if not p.exists():
            return None
        import nibabel as nib

        return np.asanyarray(nib.load(str(p)).dataobj).astype(np.uint8)
    p = ART / "entropy" / ckpt / dataset / f"{sid}_{phase}.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


def _cam(dataset: str, sid: str, structure: str) -> np.ndarray | None:
    p = ART / "xai" / dataset / sid / f"cam_{structure.lower()}.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


def _best_slice(mask: np.ndarray | None, nz: int) -> int:
    if mask is None:
        return nz // 2
    areas = [(mask[:, :, z] == 2).sum() for z in range(mask.shape[2])]
    return int(np.argmax(areas)) if any(areas) else mask.shape[2] // 2


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────
def _png(fig) -> bytes:
    import matplotlib.pyplot as plt

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0,
                facecolor="#0b0d10")
    plt.close(fig)
    return buf.getvalue()


@lru_cache(maxsize=256)
def render(dataset: str, sid: str, phase: str, overlay: str, structure: str,
           ckpt: str | None = None, src: str = "pred") -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = volumes(dataset, sid)
    if v is None:
        raise HTTPException(404, f"no image volume for {dataset}/{sid}")
    img = v[f"img_{phase}"]
    gt = v[f"gt_{phase}"]

    # `src` is the backend the RUN BEING VIEWED actually used, not whatever happens to be
    # lying in the shared mask pool. artifacts/masks/ is keyed by (checkpoint, dataset,
    # subject) and is shared across runs, so a groundtruth-backend run would otherwise
    # display a predicted mask left behind by some earlier `cmr segment` — a mask that did
    # NOT produce the numbers on screen. Showing the wrong mask beside the right numbers is
    # precisely the silent, plausible-looking wrongness this codebase exists to prevent.
    mask = None if src == "gt" else _pred("mask", dataset, sid, phase, ckpt)
    if mask is None or mask.shape != img.shape:
        mask = gt
    z = _best_slice(mask, img.shape[2])

    base = img[:, :, z]
    base = (base - base.min()) / (np.ptp(base) + 1e-8)

    fig = plt.figure(figsize=(3.4, 3.4), dpi=110)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.imshow(base.T, cmap="gray", interpolation="nearest")

    if overlay == "mask" and mask is not None:
        for lab, colour in STRUCTURE_COLOUR.items():
            m = (mask[:, :, z] == lab).T
            if m.any():
                ax.contourf(np.ma.masked_where(~m, m.astype(float)), levels=[0.5, 1.5],
                            colors=[colour], alpha=0.5)
                ax.contour(m.astype(float), levels=[0.5], colors=[colour], linewidths=0.9)

    elif overlay == "entropy":
        if src == "gt":
            # Ground truth is one expert annotation, not an ensemble — it has no entropy.
            # Any entropy on disk belongs to a *segmentation* run, not to this one.
            raise HTTPException(
                404, f"{dataset}/{sid}: this run used ground-truth masks, which carry no "
                     f"ensemble uncertainty")
        ent = _pred("entropy", dataset, sid, phase, ckpt)
        if ent is None or ent.shape != img.shape:
            raise HTTPException(404, f"no entropy map for {dataset}/{sid}")
        # Normalised by ln(4), never min-maxed — see xai.uncertainty_map for why. Alpha
        # tracks the value, so a confident voxel stays transparent and the anatomy shows
        # through: a flat alpha would paint "the model is certain here" in solid colour.
        unc = np.clip(ent[:, :, z] / float(np.log(4)), 0, 1)
        ax.imshow(unc.T, cmap="cividis", alpha=np.clip(unc.T * 2.5, 0, 0.9), vmin=0, vmax=1,
                  interpolation="nearest")

    elif overlay == "cam":
        cam = _cam(dataset, sid, structure)
        if cam is None or cam.shape != img.shape:
            raise HTTPException(404, f"no Seg-Grad-CAM for {dataset}/{sid}/{structure}")
        c = cam[:, :, z]
        ax.imshow(c.T, cmap="inferno", alpha=np.clip(c.T * 1.4, 0, 0.85), vmin=0, vmax=1,
                  interpolation="nearest")
        lab = next((k for k, n in LABEL_NAMES.items() if n == structure), None)
        if lab is not None and mask is not None:
            ax.contour((mask[:, :, z] == lab).T, levels=[0.5], colors="w", linewidths=1.0,
                       linestyles="--")

    return _png(fig)


# ─────────────────────────────────────────────────────────────────────────────
# The link: report sentence -> guideline passage -> the structure's heatmap.
# ─────────────────────────────────────────────────────────────────────────────
def links(cid: str, sid: str, dataset: str, rep: dict, ckpt: str | None = None,
          src: str = "pred") -> list[dict]:
    """Prefer the Explanation objects the run wrote. If the XAI node never ran (the
    groundtruth backend skips it), rebuild the sentence->structure edge with the SAME
    table Agent 4 uses (xai.map_claim), and mark it derived. We never fabricate the
    heatmap itself: if the .npy is not on disk, the panel says it is not on disk.

    `ckpt` is only needed for the entropy check — CAMs live at a checkpoint-agnostic path
    (artifacts/xai/{dataset}/{sid}/...) because they belong to one specific run's segmenter,
    not a shared pool the way masks/entropy are (see _pred's docstring).
    """
    from .xai import map_claim

    report = rep.get("report") or {}
    stored = {e.get("chunk_id"): e for e in (rep.get("explanations") or [])}
    idx = corpus()
    out = []

    for c in report.get("citations") or []:
        cid_ = c.get("chunk_id", "")
        sentence = (c.get("supports") or "").strip() or report.get("diagnosis", "")
        ex = stored.get(cid_, {})
        struct = ex.get("structure")
        derived = False
        if not struct:
            lab = map_claim(sentence)
            struct = LABEL_NAMES.get(lab) if lab else None
            derived = True

        chunk = idx.get(cid_)
        has_cam = struct is not None and _cam(dataset, sid, struct) is not None
        # Ground truth carries no ensemble uncertainty — see render()'s entropy branch.
        has_ent = src == "pred" and _pred("entropy", dataset, sid, "ed", ckpt) is not None
        out.append({
            "sentence": sentence,
            "chunk_id": cid_,
            "resolved": chunk is not None,
            "chunk": chunk,
            "citation": c,
            "structure": struct,
            "structure_derived": derived,
            "mean_uncertainty": ex.get("mean_uncertainty"),
            "heatmap": (f"/img/{dataset}/{sid}.png?phase=ed&overlay=cam&structure={struct}"
                        f"&ckpt={ckpt}&src={src}" if has_cam else None),
            "entropy": (f"/img/{dataset}/{sid}.png?phase=ed&overlay=entropy"
                        f"&ckpt={ckpt}&src={src}" if has_ent else None),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline configuration — what the NEXT run does.
#
# WHAT THIS IS NOT: it is not an editor for a finished run. The header line
# "vertical_slice_f66e47 · backend=groundtruth · retrieval=none · llm=none · gate=true" is
# read from that run's artifacts/runs/<id>/run.json — the REPRODUCIBILITY RECORD of what
# actually happened, written once when the run started. Making it editable would let a
# result be rewritten after the fact, which is the precise failure this codebase exists to
# prevent (cli.write_provenance: "If a number in the thesis cannot be traced back to one of
# those directories, it does not go in the thesis"). run.json stays immutable, always.
#
# What this DOES: it sets the configuration the next live run executes with, and that run
# then records those settings in its OWN run.json, exactly like a batch `cmr run` does. Edit
# forward, never backward.
#
# Batch `cmr run` still reads configs/*.yaml and is untouched by this page — those YAML files
# are heavily commented documentation, and silently rewriting them from a browser click would
# destroy that. The overrides live in artifacts/ui_config.json instead.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Field:
    key: str  # dotted path into the config tree
    label: str
    kind: str  # choice | bool | int | str
    choices: tuple[str, ...] = ()
    help: str = ""
    minimum: int = 0
    maximum: int = 0


FIELDS: tuple[Field, ...] = (
    Field("segmentation.backend", "Agent 1 · backend", "choice", ("cinema", "groundtruth"),
          "cinema = run the real segmenter. groundtruth = use the uploaded expert mask instead "
          "(needs ground-truth files, and is how the downstream agents were built and tested)."),
    Field("segmentation.checkpoint", "Agent 1 · checkpoint", "choice", UPLOAD_CHECKPOINTS,
          "acdc_sax on ACDC is SUPERVISED; acdc_sax on M&Ms / M&Ms-2 is the true zero-shot "
          "cross-dataset run (RQ3). The Run tab can override this per upload."),
    Field("retrieval.mode", "Agent 3a · retrieval", "choice", ("hybrid", "dense", "bm25", "none"),
          "none = the H1 ablation: the LLM reasons from measurements with no guideline grounding, "
          "and can therefore carry no citations."),
    Field("retrieval.embedder", "Agent 3a · embedder", "choice", ("medcpt", "biobert"),
          "MedCPT is trained for biomedical retrieval; BioBERT is a masked LM. BioBERT is the "
          "ablation row promised to the panel."),
    Field("llm.provider", "Agent 3b · LLM provider", "choice",
          ("ollama", "local", "gemini", "groq", "openrouter", "none"),
          "ollama keeps every byte on this machine. none = the zero-LLM template floor "
          "(a template cannot hallucinate — the baseline that flatters you least)."),
    Field("llm.model", "Agent 3b · model id", "str", (),
          "Recorded verbatim in run.json, with the access date. Ignored when provider = none."),
    Field("orchestration.gate", "Plausibility gate", "bool", (),
          "off = the H3 ablation: implausible masks pass straight through to Agent 3 instead of "
          "being refused."),
    Field("orchestration.feedback", "Boundary-aware recomputation", "bool", (),
          "off = the H2 ablation: a value landing near a guideline cut-point is NOT re-measured."),
    Field("orchestration.max_iterations", "Max loop iterations", "int", (),
          "Bounds both feedback loops. 0 disables retrying entirely.", 0, 5),
)
FIELD_BY_KEY = {f.key: f for f in FIELDS}


def _provider_models() -> dict[str, str | None]:
    """Each provider's default model — 'qwen2.5:14b' is meaningless to Gemini, so the UI
    must re-fill the model box when the provider changes (llm.py enforces the same
    precedence: providers.<p>.model wins over the bare llm.model)."""
    return {p: (dict(v) or {}).get("model") for p, v in dict(CFG.llm.providers).items()}


def load_overrides() -> dict[str, Any]:
    return {k: v for k, v in (_read_json(UI_CONFIG) or {}).items() if k in FIELD_BY_KEY}


def _set_dotted(tree: dict, dotted: str, value: Any) -> None:
    """Write into a nested dict, copying each level on the way down so CFG is never mutated."""
    parts = dotted.split(".")
    node = tree
    for p in parts[:-1]:
        node[p] = dict(node[p])
        node = node[p]
    node[parts[-1]] = value


def effective_cfg() -> cfgmod.Config:
    """The base YAML config with the Config page's overrides applied on top."""
    tree = dict(CFG)
    ov = load_overrides()
    for k, v in ov.items():
        _set_dotted(tree, k, v)

    # llm.py resolves the model as: $CMR_LLM_MODEL > providers.<provider>.model > llm.model.
    # So writing llm.model alone would be silently ignored for any provider that carries its
    # own default (gemini, groq, …). Pin it on the provider too, or the box on the page would
    # be a lie.
    model = ov.get("llm.model")
    provider = tree["llm"]["provider"]
    if model and provider not in ("none",) and provider in dict(CFG.llm.providers):
        _set_dotted(tree, f"llm.providers.{provider}.model", model)
    return cfgmod.Config(tree)


def _coerce(f: Field, v: Any) -> Any:
    if f.kind == "choice":
        if v not in f.choices:
            raise HTTPException(400, f"{f.key}: {v!r} is not one of {list(f.choices)}")
        return v
    if f.kind == "bool":
        if not isinstance(v, bool):
            raise HTTPException(400, f"{f.key}: expected true/false, got {v!r}")
        return v
    if f.kind == "int":
        try:
            n = int(v)
        except (TypeError, ValueError):
            raise HTTPException(400, f"{f.key}: expected an integer, got {v!r}") from None
        if not (f.minimum <= n <= f.maximum):
            raise HTTPException(400, f"{f.key}: must be between {f.minimum} and {f.maximum}")
        return n
    return str(v).strip()


def _config_payload() -> dict:
    eff = effective_cfg()
    return {
        "fields": [
            {"key": f.key, "label": f.label, "kind": f.kind, "choices": list(f.choices),
             "help": f.help, "min": f.minimum, "max": f.maximum,
             "value": eff.get_path(f.key), "base": CFG.get_path(f.key)}
            for f in FIELDS
        ],
        "overridden": sorted(load_overrides()),
        "provider_models": _provider_models(),
        # The one-liner the header shows for a run — but for the config that WILL be used next.
        # The model is the RESOLVED one (llm.py's precedence), not the bare llm.model, or
        # switching to Gemini would still read "qwen2.5:14b" here.
        "summary": (f"backend={eff.segmentation.backend} · retrieval={eff.retrieval.mode}"
                    f" · llm={resolved_llm_model(eff) or 'none'}"
                    f" · gate={eff.orchestration.gate}"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Live runs — upload two NIfTI phases, watch the graph run, node by node.
#
# Everything above this line is read-only, by the module's own rule. This section is the
# one deliberate exception: it runs the real pipeline (cmr/graph.py) on user-supplied data.
# It still obeys the same honesty posture as the rest of the codebase:
#   - a bad upload is rejected with a specific reason, never silently coerced
#   - uploaded ground truth is checked with assert_canonical, exactly like every dataset
#     loader in data.py — an unverified label order would corrupt any GT comparison the
#     same silent way it would for ACDC/M&Ms (see cmr/checks.py)
#   - the checkpoint choice is always shown, never defaulted invisibly
#   - a finished live run is written to artifacts/runs/{run_id}/ in the SAME shape a batch
#     `cmr run` produces, so it is immediately browsable through the endpoints above —
#     live view and history view are the same data, not two different code paths
# ─────────────────────────────────────────────────────────────────────────────
def _read_upload_nifti(raw: bytes, field: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Bytes from an upload -> (volume, spacing_mm). Rejects anything implausible before
    it ever reaches a model, the same posture scripts/fetch_guidelines.py takes on PDFs.

    The magic-byte check only runs on the UNCOMPRESSED case: gzip does not support checking
    a fixed byte offset without decompressing from the start, and decompressing twice (once
    to peek, once via nibabel) would double the cost for no benefit. For a .nii.gz, nibabel's
    own load is the check — it raises a clear, specific error on anything that is not really
    a NIfTI, which is caught below and turned into the same kind of 400 either way.
    """
    import tempfile

    import nibabel as nib

    if len(raw) < _MIN_NIFTI_BYTES:
        raise HTTPException(400, f"{field}: only {len(raw)} bytes — too small to be a NIfTI file")
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"{field}: {len(raw) / 1e6:.0f} MB exceeds the upload limit "
                                  f"({_MAX_UPLOAD_BYTES / 1e6:.0f} MB)")

    is_gz = raw[:2] == b"\x1f\x8b"
    if not is_gz and raw[344:348] not in (b"n+1\x00", b"ni1\x00"):
        raise HTTPException(400, f"{field}: not a NIfTI-1 file (bad magic bytes at offset 344)")

    with tempfile.NamedTemporaryFile(suffix=".nii.gz" if is_gz else ".nii", delete=False) as f:
        f.write(raw)
        path = Path(f.name)
    try:
        img = nib.load(str(path))
        arr = np.asanyarray(img.dataobj)
        zooms = img.header.get_zooms()
    except Exception as e:
        raise HTTPException(400, f"{field}: not a readable NIfTI file ({e})") from e
    finally:
        path.unlink(missing_ok=True)

    if arr.ndim == 4:  # a full cine series was uploaded where one phase was asked for
        raise HTTPException(400, f"{field}: got a 4-D volume {arr.shape} — upload a single "
                                  f"ED or ES phase (3-D), not the whole cine series")
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise HTTPException(400, f"{field}: expected a 2-D or 3-D volume, got shape {arr.shape}")
    if not (1 <= arr.shape[2] <= 64):
        raise HTTPException(400, f"{field}: {arr.shape[2]} slices is implausible for a SAX stack")

    sx = float(zooms[0]) if len(zooms) > 0 else 1.0
    sy = float(zooms[1]) if len(zooms) > 1 else 1.0
    sz = float(zooms[2]) if len(zooms) > 2 else 1.0
    return arr.astype(np.float32), (sx, sy, sz)


def _save_upload_volume(arr: np.ndarray, spacing: tuple[float, float, float], path: Path) -> None:
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.diag([*spacing, 1.0])
    nib.save(nib.Nifti1Image(arr, affine), str(path))


def _read_upload_mask(raw: bytes | None, field: str, ref_shape: tuple[int, ...]) -> np.ndarray | None:
    """Optional ground truth for side-by-side comparison. Checked with the SAME
    concentricity assertion every dataset loader in data.py runs — an uploaded mask whose
    label order we silently trusted would corrupt the comparison exactly as silently as an
    unremapped ACDC volume would (see cmr/checks.py and PLAN.md's label-convention warning).
    """
    if raw is None:
        return None
    arr, _ = _read_upload_nifti(raw, field)
    mask = arr.astype(np.uint8)
    if mask.shape != ref_shape:
        raise HTTPException(400, f"{field}: shape {mask.shape} does not match the image {ref_shape}")
    if not set(np.unique(mask)) <= {0, 1, 2, 3}:
        raise HTTPException(400, f"{field}: labels must be 0-3 (background/LV/Myo/RV); got "
                                  f"{sorted(int(v) for v in np.unique(mask))}")
    try:
        assert_canonical(mask, field)
    except LabelOrderError as e:
        raise HTTPException(
            400,
            f"{field}: {e} This system assumes the CANONICAL label order (1=LV, 2=Myo, "
            f"3=RV) on any uploaded ground truth — see cmr/checks.py. If your mask uses "
            f"ACDC's native order (1=RV, 3=LV) instead, remap it before uploading.",
        ) from e
    return mask


class _LiveRun:
    """One upload's event log, not a mailbox.

    It keeps the full event HISTORY plus a done flag, rather than a consume-once Queue, so a
    browser that reloads (or a second one that opens the same URL) can re-attach mid-run and
    be replayed everything it missed. With a Queue, a refresh during a 3-minute run orphaned
    the viewer permanently — the run kept going server-side with nobody watching, and a
    client attaching after the end would block forever on an empty queue.

    The worker thread appends via emit(); the SSE endpoint reads by index and waits on the
    condition variable, off the event loop (run_in_executor), so it never blocks asyncio.
    """

    def __init__(self, run_id: str, subject_id: str) -> None:
        self.run_id = run_id
        self.subject_id = subject_id
        self.events: list[dict] = []
        self.done = False
        self.error: str | None = None
        self._cv = threading.Condition()

    def emit(self, ev: dict) -> None:
        with self._cv:
            self.events.append(ev)
            self._cv.notify_all()

    def finish(self, error: str | None = None) -> None:
        with self._cv:
            self.error = error
            self.done = True
            self._cv.notify_all()

    def wait_from(self, i: int, timeout: float = 0.5) -> None:
        """Block (in a worker thread) until there is an event past index `i`, or the run ends."""
        with self._cv:
            if i >= len(self.events) and not self.done:
                self._cv.wait(timeout)


class RunManager:
    """Runs live, single-subject pipeline requests one at a time.

    Segmenter/Retriever/LLM are heavy — a loaded CineMA checkpoint, a warm Ollama
    connection — and are built once, lazily, on first use, then reused. Running two
    uploads at once would contend for the same MPS device and Ollama process and just
    make both slower, so serialising through one worker thread is the honest choice, not
    merely the easy one (the same reasoning cmr/graph.py's Deps docstring gives for
    building the Segmenter once per batch instead of once per subject).
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # Keyed by the settings that actually determine the object, so changing the config on
        # the Config page swaps in the right one (and reuses it next time) instead of serving
        # a stale model built from the previous settings.
        self._segmenters: dict[str, Any] = {}                  # checkpoint
        self._retrievers: dict[tuple[str, str], Any] = {}      # (mode, embedder)
        self._llms: dict[tuple[str, str], Any] = {}            # (provider, model)
        self._runs: dict[str, _LiveRun] = {}
        self._jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True, name="cmr-live-run").start()

    def _segmenter(self, cfg, checkpoint: str):
        if checkpoint not in self._segmenters:
            from .seg import Segmenter

            seg_cfg = cfgmod.Config(
                dict(cfg)
                | {"segmentation": dict(cfg.segmentation)
                   | {"checkpoint": checkpoint, "backend": "cinema"}}
            )
            log.info("live run: loading CineMA checkpoint %s (first use)", checkpoint)
            self._segmenters[checkpoint] = Segmenter(seg_cfg)
        return self._segmenters[checkpoint]

    def _retriever(self, cfg):
        if cfg.retrieval.mode == "none":
            return None  # the H1 ablation — the graph emits an ungrounded, citation-free report
        key = (cfg.retrieval.mode, cfg.retrieval.embedder)
        if key not in self._retrievers:
            from .retrieve import Retriever

            log.info("live run: building retriever %s/%s (first use)", *key)
            self._retrievers[key] = Retriever(cfg)
        return self._retrievers[key]

    def _llm(self, cfg):
        if cfg.llm.provider == "none":
            return None  # the zero-LLM template floor; graph.py routes to template_report
        from .llm import LLM

        probe = LLM(cfg)  # cheap: resolves the model id per llm.py's own precedence rules
        key = (cfg.llm.provider, probe.model)
        if key not in self._llms:
            log.info("live run: LLM %s/%s (first use)", *key)
            self._llms[key] = probe
        return self._llms[key]

    def submit(self, subject: Subject, checkpoint: str) -> str:
        run_id = f"upload_{uuid.uuid4().hex[:10]}"
        live = _LiveRun(run_id, subject.subject_id)
        self._runs[run_id] = live
        self._jobs.put((run_id, subject, checkpoint))
        log.info("live run %s queued (%d ahead of it)", run_id, self._jobs.qsize() - 1)
        return run_id

    def get(self, run_id: str) -> _LiveRun:
        live = self._runs.get(run_id)
        if live is None:
            raise HTTPException(404, f"no live run '{run_id}'")
        return live

    def _worker(self) -> None:
        while True:
            run_id, subject, checkpoint = self._jobs.get()
            live = self._runs[run_id]
            try:
                self._execute(run_id, subject, checkpoint, live)
            except Exception as e:  # the run failed; tell the browser why, don't just hang
                log.exception("live run %s failed", run_id)
                live.emit({"node": "_error", "error": str(e)})
                live.finish(str(e))
            else:
                live.finish()

    def _execute(self, run_id: str, subject: Subject, checkpoint: str, live: _LiveRun) -> None:
        from . import graph
        from .cli import write_provenance

        # Whatever the Config page currently says. The per-upload checkpoint wins over the
        # configured one, because the Run tab offers it explicitly.
        base = effective_cfg()
        cfg = cfgmod.Config(
            dict(base)
            | {"experiment": "upload",
               "segmentation": dict(base.segmentation) | {"checkpoint": checkpoint}}
        )
        backend = cfg.segmentation.backend
        d = graph.Deps(
            cfg=cfg,
            segmenter=None if backend == "groundtruth" else self._segmenter(cfg, checkpoint),
            retriever=self._retriever(cfg),
            llm=self._llm(cfg),
            on_event=live.emit,
        )
        # This run records what it ACTUALLY ran with — same as a batch `cmr run`. That file is
        # the immutable record; the Config page can never edit it after the fact.
        write_provenance(cfg, run_id)
        st = graph.run_subject(subject, d, config_id=run_id)
        graph.write_trace(cfg, run_id, st)
        graph.write_report(cfg, run_id, st)
        if backend != "groundtruth":
            self._save_prediction(cfg, subject, checkpoint, d)
        # No cache to invalidate: run_table()'s _stamp(cid) staleness key already changes
        # the instant these files land, so the viewer picks this run up on its next request.
        log.info("live run %s: subject %s -> status %s", run_id, subject.subject_id, st.status)

    @staticmethod
    def _save_prediction(cfg, subject: Subject, checkpoint: str, d) -> None:
        """Write Agent 1's mask + entropy to the SAME checkpoint-keyed pool `cmr segment`
        writes to (see seg.py), so the viewer's existing _pred()/render() find them for
        free — a live run and a batch run share one mask store, not two.

        Never called for a groundtruth-backend run: its 'mask' IS the expert annotation, and
        filing that in the PREDICTED pool would make every later viewer of this subject show
        ground truth labelled 'Agent 1 (predicted)'.
        """
        if d.mask_ed is None:  # segmentation raised
            return
        from .seg import _paths as seg_paths
        from .seg import _save_mask as seg_save_mask

        p = seg_paths(cfg, subject.dataset, subject.subject_id, checkpoint)
        try:
            seg_save_mask(d.mask_ed, subject, p["mask_ed"])
            seg_save_mask(d.mask_es, subject, p["mask_es"])
            if d.ent_ed is not None:
                np.save(p["ent_ed"], d.ent_ed)
            if d.ent_es is not None:
                np.save(p["ent_es"], d.ent_es)
        except Exception:  # the run itself already succeeded; a display artefact must not undo that
            log.exception("live run %s: failed to persist predicted mask/entropy",
                          subject.subject_id)


_RUN_MANAGER: RunManager | None = None


def run_manager() -> RunManager:
    global _RUN_MANAGER
    if _RUN_MANAGER is None:
        _RUN_MANAGER = RunManager(CFG)
    return _RUN_MANAGER


async def _sse_events(live: _LiveRun):
    """Replay everything this run has already emitted, then stream the rest.

    Replaying is what makes re-attaching work: a browser that reloads mid-run, or one that
    opens the run's URL after it has finished, gets the whole history and a terminal `done`
    rather than hanging on an empty stream.
    """
    import asyncio

    loop = asyncio.get_event_loop()
    i = 0
    while True:
        while i < len(live.events):
            yield f"data: {json.dumps(live.events[i], default=str)}\n\n"
            i += 1
        if live.done:
            yield ("event: done\ndata: " + json.dumps(
                {"run_id": live.run_id, "subject_id": live.subject_id, "error": live.error}) + "\n\n")
            return
        await loop.run_in_executor(None, live.wait_from, i)


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="CMR viewer", docs_url=None, redoc_url=None)


@app.get("/api/live/checkpoints")
def api_live_checkpoints() -> dict:
    eff = effective_cfg()
    ckpt = eff.segmentation.checkpoint
    return {
        "checkpoints": list(UPLOAD_CHECKPOINTS),
        "default": ckpt if ckpt in UPLOAD_CHECKPOINTS else UPLOAD_CHECKPOINTS[0],
        "backend": eff.segmentation.backend,
        "llm_provider": eff.llm.provider,
        "llm_model": resolved_llm_model(eff),
        "retrieval_mode": eff.retrieval.mode,
        "summary": _config_payload()["summary"],
    }


@app.get("/api/config")
def api_config() -> dict:
    return _config_payload()


@app.post("/api/config")
def api_config_set(values: dict[str, Any]) -> dict:
    """Validate and persist the settings the NEXT run will use.

    Never touches any run.json — a finished run's provenance is immutable by design. See the
    section header above.
    """
    ov = load_overrides()
    for k, v in values.items():
        f = FIELD_BY_KEY.get(k)
        if f is None:
            raise HTTPException(400, f"unknown setting {k!r}")
        coerced = _coerce(f, v)
        # Store only genuine departures from the YAML, so the page can show what is overridden
        # and "reset" means something.
        if coerced == CFG.get_path(k):
            ov.pop(k, None)
        else:
            ov[k] = coerced

    UI_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    UI_CONFIG.write_text(json.dumps(ov, indent=2, sort_keys=True) + "\n")
    log.info("config updated: %s", ov or "(back to configs/default.yaml)")
    return _config_payload()


@app.post("/api/config/reset")
def api_config_reset() -> dict:
    UI_CONFIG.unlink(missing_ok=True)
    log.info("config reset to configs/default.yaml")
    return _config_payload()


@app.get("/api/config/presets")
def api_config_presets() -> dict:
    """The ablation grid from configs/experiments.yaml, as dotted settings this page can apply.

    The grid IS the thesis's Table 3.3, so the page offers the same rows rather than inviting
    you to hand-assemble one and hope it matches.
    """
    import yaml

    from .config import EXPERIMENTS

    short = {  # the grid's shorthand -> the real config tree (same map as config.load_experiment)
        "retriever": "retrieval.mode", "embedder": "retrieval.embedder",
        "llm": "llm.model", "provider": "llm.provider",
        "gate": "orchestration.gate", "feedback": "orchestration.feedback",
        "checkpoint": "segmentation.checkpoint", "backend": "segmentation.backend",
    }
    with open(EXPERIMENTS) as f:
        grid = yaml.safe_load(f) or {}
    return {
        "presets": [
            {"name": name,
             "values": {short[k]: v for k, v in spec.items() if k in short and short[k] in FIELD_BY_KEY}}
            for name, spec in grid.items()
        ]
    }


@app.post("/api/live/upload")
async def api_live_upload(
    ed_img: UploadFile = File(...),
    es_img: UploadFile = File(...),
    ed_gt: UploadFile | None = File(None),
    es_gt: UploadFile | None = File(None),
    checkpoint: str = Form(...),
    subject_id: str = Form(""),
) -> dict:
    if checkpoint not in UPLOAD_CHECKPOINTS:
        raise HTTPException(400, f"checkpoint must be one of {UPLOAD_CHECKPOINTS}")

    img_ed, spacing = _read_upload_nifti(await ed_img.read(), "ed_img")
    img_es, spacing_es = _read_upload_nifti(await es_img.read(), "es_img")
    if img_ed.shape != img_es.shape:
        raise HTTPException(400, f"ed_img/es_img shape mismatch: {img_ed.shape} vs "
                                  f"{img_es.shape} — both must be the same subject's grid")
    if spacing != spacing_es:
        log.warning("ed_img/es_img spacing differs (%s vs %s) — using ed_img's", spacing, spacing_es)

    gt_ed = _read_upload_mask(await ed_gt.read() if ed_gt else None, "ed_gt", img_ed.shape)
    gt_es = _read_upload_mask(await es_gt.read() if es_gt else None, "es_gt", img_es.shape)

    # The groundtruth backend does not segment — it reads the expert mask. Without one there
    # is nothing for it to read, and every downstream agent would receive an empty mask.
    if effective_cfg().segmentation.backend == "groundtruth" and not (gt_ed is not None
                                                                      and gt_es is not None):
        raise HTTPException(
            400, "segmentation.backend is 'groundtruth', which reasons over the expert mask "
                 "rather than segmenting — so ED and ES ground truth are required. Upload them, "
                 "or set the backend to 'cinema' on the Config page.")

    sid = subject_id.strip() or f"subject_{uuid.uuid4().hex[:8]}"
    subject = Subject(
        subject_id=sid, dataset="Upload", split="upload", pathology="UNKNOWN",
        ed_idx=0, es_idx=0, spacing_mm=spacing,
        img_ed=img_ed, img_es=img_es, gt_ed=gt_ed, gt_es=gt_es,
    )
    # So volumes()'s "Upload" branch can re-render this subject later — re-encoded through
    # nibabel rather than the raw upload bytes, so the on-disk copy is always .nii.gz
    # regardless of what format was uploaded, matching every other dataset's layout.
    _save_upload_volume(img_ed, spacing, UPLOADS / f"{sid}_ed.nii.gz")
    _save_upload_volume(img_es, spacing, UPLOADS / f"{sid}_es.nii.gz")
    if gt_ed is not None:
        _save_upload_volume(gt_ed, spacing, UPLOADS / f"{sid}_ed_gt.nii.gz")
    if gt_es is not None:
        _save_upload_volume(gt_es, spacing, UPLOADS / f"{sid}_es_gt.nii.gz")

    run_id = run_manager().submit(subject, checkpoint)
    return {"run_id": run_id, "subject_id": sid, "checkpoint": checkpoint}


@app.get("/api/live/{run_id}/stream")
async def api_live_stream(run_id: str) -> StreamingResponse:
    live = run_manager().get(run_id)
    return StreamingResponse(_sse_events(live), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/runs")
def api_runs() -> list[dict]:
    if not RUNS.is_dir():
        return []
    out = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        prov = _read_json(d / "run.json") or {}
        n_rep = len(list((d / "reports").glob("*.json"))) if (d / "reports").is_dir() else 0
        n_tr = len(list((d / "traces").glob("*.jsonl"))) if (d / "traces").is_dir() else 0
        out.append({
            "config_id": d.name,
            "experiment": prov.get("experiment", ""),
            "llm": prov.get("llm_model_id", ""),
            "retrieval": (prov.get("retrieval") or {}).get("mode", ""),
            "backend": (prov.get("segmentation") or {}).get("backend", ""),
            "gate": (prov.get("orchestration") or {}).get("gate"),
            "started_at": prov.get("started_at", ""),
            "n_subjects": n_tr or n_rep,
            "n_reports": n_rep,
            "n_refused": max(n_tr - n_rep, 0),
            "has_metrics": (d / "metrics.json").exists(),
        })
    return out


@app.get("/api/subjects")
def api_subjects(run: str) -> dict:
    rows = run_table(run)
    return {"config_id": run, "n": len(rows), "n_refused": sum(r["refused"] for r in rows),
            "rows": rows}


@app.get("/api/subject/{sid}")
def api_subject(sid: str, run: str) -> dict:
    rd = RUNS / run
    if not rd.is_dir():
        raise HTTPException(404, f"no run '{run}'")
    ev = _trace(run, sid)
    rep = _read_json(rd / "reports" / f"{sid}.json")
    if not ev and not rep:
        raise HTTPException(404, f"{sid} not in run {run}")

    meta = cohort().get(sid, {})
    # cohort() only knows ACDC/MnMs/MnMs2; a live-uploaded subject falls back to the
    # dataset the report itself recorded (see graph.write_report).
    dataset = meta.get("dataset") or (rep or {}).get("dataset") or ""
    row = next((r for r in run_table(run) if r["subject_id"] == sid), {})
    v = volumes(dataset, sid) if dataset else None

    # The ablation grid never varies segmentation, so every batch run shares one checkpoint
    # (CFG's default) and _pred()'s own default is correct. A live upload lets the user pick
    # a DIFFERENT checkpoint per run — read the one THIS run actually used from its own
    # run.json rather than assuming the global default, or its mask/entropy would silently
    # not be found.
    prov = _read_json(rd / "run.json") or {}
    seg = prov.get("segmentation") or {}
    ckpt = seg.get("checkpoint") or CFG.segmentation.checkpoint
    backend = seg.get("backend") or CFG.segmentation.backend
    src = "gt" if backend == "groundtruth" else "pred"  # see render()'s `src` comment
    q = f"&ckpt={ckpt}&src={src}"

    images = {}
    if v is not None:
        for ph in ("ed", "es"):
            images[f"mask_{ph}"] = f"/img/{dataset}/{sid}.png?phase={ph}&overlay=mask{q}"
            images[f"plain_{ph}"] = f"/img/{dataset}/{sid}.png?phase={ph}&overlay=none{q}"
        if src == "pred" and _pred("entropy", dataset, sid, "ed", ckpt) is not None:
            images["entropy_ed"] = f"/img/{dataset}/{sid}.png?phase=ed&overlay=entropy{q}"

    has_pred = bool(dataset) and _pred("mask", dataset, sid, "ed", ckpt) is not None

    # A citation that does not resolve is the headline failure this system claims to catch —
    # so be certain WHY it does not resolve before saying so. If the corpus has been rebuilt
    # since this run, its chunk_ids (content hashes) all changed, and a non-resolving id says
    # nothing about the model. See corpus_sha256().
    run_corpus, cur_corpus = prov.get("corpus_sha256"), corpus_sha256()
    corpus_stale = bool(run_corpus and cur_corpus and run_corpus != cur_corpus)

    lk = links(run, sid, dataset, rep, ckpt, src) if (rep and dataset) else []
    return {
        "subject_id": sid,
        "config_id": run,
        "meta": meta,
        "row": row,
        "refused": row.get("refused", False),
        "status": row.get("status", "unknown"),
        "measurements": (rep or {}).get("measurements"),
        "report": (rep or {}).get("report"),
        "fidelity": (rep or {}).get("fidelity"),
        "links": lk,
        "trace": ev,
        "images": images,
        "mask_source": (
            "ground truth — this run's backend" if src == "gt"
            else ("Agent 1 (predicted)" if has_pred else "ground truth (no predicted mask on disk)")
        ),
        "has_cam": any(e["heatmap"] for e in lk),
        "corpus_stale": corpus_stale,
        "run_corpus_sha256": run_corpus,
        "current_corpus_sha256": cur_corpus,
    }


@app.get("/img/{dataset}/{sid}.png")
def api_img(dataset: str, sid: str, phase: str = "ed", overlay: str = "mask",
            structure: str = "LV", ckpt: str | None = None, src: str = "pred") -> Response:
    if phase not in ("ed", "es"):
        raise HTTPException(400, "phase must be ed|es")
    if src not in ("gt", "pred"):
        raise HTTPException(400, "src must be gt|pred")
    png = render(dataset, sid, phase, overlay, structure, ckpt, src)
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/chunks")
def api_chunks(q: str = "", limit: int = Query(40, le=200)) -> dict:
    idx = corpus()
    terms = [t for t in q.lower().split() if t]
    hits = []
    for c in idx.values():
        hay = f"{c['text']} {c['source']} {c['section']}".lower()
        if all(t in hay for t in terms):
            hits.append(c)
    hits.sort(key=lambda c: (c["source"], c["page"]))
    return {"n_total": len(idx), "n_hits": len(hits), "chunks": hits[:limit]}


@app.get("/api/chunk/{chunk_id}")
def api_chunk(chunk_id: str) -> dict:
    c = corpus().get(chunk_id)
    if c is None:
        # A dangling citation. Saying so IS the result — do not 500, do not invent a chunk.
        raise HTTPException(404, f"chunk {chunk_id} does not resolve against the index")
    return c


@app.get("/api/compare")
def api_compare(a: str, b: str) -> dict:
    def load(cid: str) -> dict:
        m = _read_json(RUNS / cid / "metrics.json")
        return {"config_id": cid, "metrics": m, "run": _read_json(RUNS / cid / "run.json"),
                "missing": m is None}

    return {"a": load(a), "b": load(b)}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "index.html").read_text()


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    log.info("artifacts: %s", ART)
    log.info("http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
