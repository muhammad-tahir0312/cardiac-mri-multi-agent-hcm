"""The single entrypoint.

    cmr manifest                          build + verify the 830-subject manifest
    cmr quantify --gt                     ground-truth EDV/ESV/LVEF/mass (the denominator)
    cmr segment  --dataset ACDC --ckpt acdc_sax
    cmr corpus                            download-free build of the guideline chunk index
    cmr run      --config full            the whole graph, every subject
    cmr run      --config no_rag --limit 20
    cmr eval     --config full
    cmr figures  --config full
    cmr doctor                            what works on this machine right now
    cmr serve                             read-only viewer on http://127.0.0.1:8000

Every run writes artifacts/runs/{config_id}/run.json with the git sha, package versions,
the exact LLM model id and access date, and the corpus hash. If a number in the thesis
cannot be traced back to one of those directories, it does not go in the thesis.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from tqdm import tqdm

from . import config as cfgmod
from . import data, graph

log = logging.getLogger("cmr.cli")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "not-a-git-repo"


def resolved_llm_model(cfg) -> str | None:
    """The model id the run will ACTUALLY call — resolved exactly as cmr/llm.py resolves it:

        $CMR_LLM_MODEL  >  llm.providers.<provider>.model  >  llm.model

    Recording the bare `llm.model` instead (as this used to) is wrong wherever a provider
    carries its own default: a Gemini run would file itself under "qwen2.5:14b", and a run
    with provider=none would claim a model it never loaded. This IS the field the proposal
    was criticised over (PLAN.md §2.11 — "name a current model, record the exact version"),
    so it has to say what actually ran.
    """
    provider = cfg.llm.provider
    if provider == "none":
        return None  # the template floor calls no model at all
    import os

    return (os.environ.get("CMR_LLM_MODEL")
            or cfg.get_path(f"llm.providers.{provider}.model")
            or cfg.llm.model)


def write_provenance(cfg, config_id: str) -> Path:
    """The reproducibility record. PROCESS.md flags that the proposal names GPT-4
    (retired) while the slides say GPT-4o — 'name a current model and record the exact
    version and access date'. This file is where that lives."""
    pkgs = {}
    for p in ("torch", "numpy", "nibabel", "transformers", "faiss-cpu", "openai",
              "langgraph", "monai", "pydantic"):
        try:
            pkgs[p] = version(p)
        except Exception:
            pass
    corpus = Path(cfg.paths.artifacts) / "corpus" / "chunks.jsonl"
    rec = {
        "config_id": config_id,
        "experiment": cfg.get("experiment", "default"),
        "started_at": datetime.now(UTC).astimezone().isoformat(),
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": cfgmod.device(cfg),
        "packages": pkgs,
        "llm_provider": cfg.llm.provider,
        "llm_model_id": resolved_llm_model(cfg),
        "llm_access_date": datetime.now().date().isoformat(),
        "retrieval": dict(cfg.retrieval),
        "segmentation": dict(cfg.segmentation),
        "orchestration": dict(cfg.orchestration),
        "seeds": cfg.compute.seeds,
        "corpus_sha256": _sha256(corpus) if corpus.exists() else None,
    }
    out = cfgmod.artifacts(cfg, "runs", config_id) / "run.json"
    out.write_text(json.dumps(rec, indent=2))
    log.info("provenance -> %s", out)
    return out


def _sha256(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
def cmd_doctor(a) -> None:
    cfg = cfgmod.load(a.config_file)
    print(f"device            : {cfgmod.device(cfg)}")
    print(f"data_root         : {cfg.paths.data_root}")
    for name, p in [("ACDC", cfg.paths.acdc), ("M&Ms", cfg.paths.mnms), ("M&Ms-2", cfg.paths.mnms2)]:
        print(f"  {name:<8}        : {'OK' if Path(p).is_dir() else 'MISSING'}  {p}")
    art = Path(cfg.paths.artifacts)
    # The index is per-embedder (retrieve.build_index): there is no shared 'index.faiss', on
    # purpose — one would silently hold whichever embedder was built last.
    for f in ["manifest.parquet", "gt_measurements.parquet", "corpus/chunks.jsonl",
              f"corpus/index_{cfg.retrieval.embedder}.faiss"]:
        print(f"  {f:<24}: {'OK' if (art / f).exists() else '-'}")
    print(f"llm               : {cfg.llm.provider} / {cfg.llm.model}")
    try:
        from .llm import LLM

        LLM(cfg).ping()
        print("  llm reachable   : OK")
    except Exception as e:
        print(f"  llm reachable   : FAIL ({e})")


def cmd_manifest(a) -> None:
    cfg = cfgmod.load(a.config_file)
    df = data.build_manifest(cfg, verify=True)
    print(df.groupby(["dataset", "split"]).size().to_string())
    print(f"\n{len(df)} subjects, all passed the canonical label-order assertion.")


def cmd_quantify(a) -> None:
    from . import quantify

    cfg = cfgmod.load(a.config_file)
    df = quantify.run_groundtruth(cfg)
    print(df.groupby("hf_category").size().to_string())


def cmd_segment(a) -> None:
    from . import seg

    cfg = cfgmod.load(a.config_file)
    if a.ckpt:
        cfg["segmentation"] = dict(cfg.segmentation) | {"checkpoint": a.ckpt}
    seg.run_dataset(cfg, a.dataset, cfg.segmentation.checkpoint)


def cmd_corpus(a) -> None:
    from . import corpus, retrieve

    cfg = cfgmod.load(a.config_file)
    chunks = corpus.build_corpus(cfg)
    print(f"{len(chunks)} chunks")
    retrieve.build_index(cfg)


def cmd_run(a) -> None:
    cfg = cfgmod.load_experiment(a.config, cfgmod.load(a.config_file))
    cid = cfg.config_id
    write_provenance(cfg, cid)
    log.info("run %s -> artifacts/runs/%s", a.config, cid)

    d = graph.Deps(cfg=cfg)
    if cfg.segmentation.backend != "groundtruth":
        from .seg import Segmenter

        d.segmenter = Segmenter(cfg)
    if cfg.retrieval.mode != "none":
        from .retrieve import Retriever

        d.retriever = Retriever(cfg)
    if cfg.llm.provider != "none":
        from .llm import LLM

        d.llm = LLM(cfg)  # once per run, not once per subject

    subs = list(data.load_all(cfg, tuple(a.datasets)))
    if a.limit:
        subs = subs[: a.limit]

    ok = refused = failed = 0
    meas, segm, reports, fids = [], [], [], []
    for s in tqdm(subs, desc=a.config):
        try:
            st = graph.run_subject(s, d, cid)
            graph.write_trace(cfg, cid, st)
            graph.write_report(cfg, cid, st)

            if st.measurements:
                meas.append(
                    st.measurements.model_dump()
                    | {
                        "gate_passed": bool(st.gate.passed) if st.gate else True,
                        "gate_violations": ";".join(st.gate.violations) if st.gate else "",
                        "status": st.status.value,
                        "iterations": st.iteration,
                    }
                )
            # Dice/HD95 of the PREDICTED mask against this subject's ground truth. Only
            # meaningful when Agent 1 actually predicted something — with the groundtruth
            # backend the mask IS the ground truth and a Dice of 1.0 would be a lie.
            if cfg.segmentation.backend != "groundtruth" and s.has_gt and d.mask_ed is not None:
                from . import eval as _ev

                segm.append(
                    {"subject_id": s.subject_id}
                    | _ev.segmentation_metrics(d.mask_ed, s.gt_ed, s.spacing_mm)
                )
            if st.report:
                reports.append(st.report.model_dump())
            if st.fidelity:
                fids.append(st.fidelity.model_dump())

            refused += st.refused
            ok += not st.refused
        except Exception as e:
            failed += 1
            log.error("[%s] %s: %s", s.subject_id, type(e).__name__, e)
            if a.strict:
                raise

    # eval.py reads exactly these four. Without them it silently reports nothing, which
    # looks like "the system scored zero" rather than "nobody wrote the file".
    rd = cfgmod.artifacts(cfg, "runs", cid)
    if meas:
        __import__("pandas").DataFrame(meas).to_parquet(rd / "measurements.parquet", index=False)
    if segm:
        __import__("pandas").DataFrame(segm).to_parquet(rd / "seg_metrics.parquet", index=False)
    for name, rows in (("reports.jsonl", reports), ("fidelity.jsonl", fids)):
        if rows:
            (rd / name).write_text("\n".join(json.dumps(r, default=str) for r in rows))

    print(f"\n{a.config}: {ok} reported, {refused} REFUSED (gate), {failed} errored")
    print(f"artifacts/runs/{cid}/")


def cmd_eval(a) -> None:
    from . import eval as ev

    cfg = cfgmod.load_experiment(a.config, cfgmod.load(a.config_file))
    m = ev.evaluate_run(cfg, cfg.config_id)
    print(json.dumps(m, indent=2, default=str))


def cmd_figures(a) -> None:
    from . import figures

    cfg = cfgmod.load_experiment(a.config, cfgmod.load(a.config_file))
    for p in figures.all_figures(cfg, cfg.config_id):
        print(p)


def cmd_serve(a) -> None:
    from . import app as _app

    _app.serve(host=a.host, port=a.port)


def main() -> None:
    p = argparse.ArgumentParser("cmr", description="Multi-agent cardiac MRI analysis")
    p.add_argument("--config-file", default=None, help="override configs/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    sub.add_parser("manifest").set_defaults(fn=cmd_manifest)

    q = sub.add_parser("quantify")
    q.add_argument("--gt", action="store_true", default=True)
    q.set_defaults(fn=cmd_quantify)

    s = sub.add_parser("segment")
    s.add_argument("--dataset", required=True, choices=["ACDC", "MnMs", "MnMs2"])
    s.add_argument("--ckpt", default=None)
    s.set_defaults(fn=cmd_segment)

    sub.add_parser("corpus").set_defaults(fn=cmd_corpus)

    r = sub.add_parser("run")
    r.add_argument("--config", default="full", help="a row of configs/experiments.yaml")
    r.add_argument("--datasets", nargs="+", default=["ACDC", "MnMs", "MnMs2"])
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--strict", action="store_true", help="crash on the first subject error")
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("eval")
    e.add_argument("--config", default="full")
    e.set_defaults(fn=cmd_eval)

    f = sub.add_parser("figures")
    f.add_argument("--config", default="full")
    f.set_defaults(fn=cmd_figures)

    # The viewer. Read-only, localhost-only, and it never touches artifacts/.
    sv = sub.add_parser("serve", help="browse artifacts/ at http://127.0.0.1:8000")
    sv.add_argument("--host", default="127.0.0.1")  # NOT 0.0.0.0. No external exposure.
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(fn=cmd_serve)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
