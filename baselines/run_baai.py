"""BAAI Cardiac-Agent (Qu et al. 2026) — SAX cine segmentation expert, on ACDC / M&Ms.

WHY THIS IS THE HEADLINE BASELINE
    No agentic CMR system has ever been benchmarked on a public cardiac-MRI dataset.
    BAAI's 2,413-patient cohort is private; CardAIc-Agents' data is public but is not
    MRI (it is ECG/echo — it mentions MRI zero times and cannot be a CMR baseline at
    all). Running BAAI's released expert on ACDC and M&Ms is therefore the first public
    -benchmark evaluation of the state of the art.

WHAT WE RUN, AND WHAT WE DO NOT
    We run the *SAX cine segmentation expert* (`Cine_SAX_seg1.pth` + `Cine_SAX_seg2.pth`,
    a two-stage coarse->fine ResUNet cascade, ~548 MB total). We do NOT run the 7B
    LLaVA-Med agent that dispatches it: its weights are 15.1 GB in fp16 and do not fit
    in this Mac's 16 GB of unified memory (see README of this module's report). The
    segmentation expert is the only part that produces a number comparable to ours, so
    it is the part that matters for the Dice table. The agent is an HPC job.

LABEL ORDER — READ THIS
    BAAI does NOT use the canonical order, and it does not use ACDC's either. It is a
    THIRD convention. From `src/CMR/calculate_cardiac_metrics_cine_sa.py`:

        BACKGROUND_ID = 0
        LV_MYOCARDIUM_ID = 1      <- label 1 is MYOCARDIUM, not LV cavity
        LV_BLOOD_POOL_ID = 2      <- label 2 is the LV cavity
        RV_BLOOD_POOL_ID = 3
        RV_MYOCARDIUM_ID = 4      <- merged into 1 at output (predictor_DY.py:255)

    i.e. BAAI native = 1=Myo, 2=LV, 3=RV;  canonical = 1=LV, 2=Myo, 3=RV.  Swap 1<->2.

    (BAAI's own `serve/cine_sa_seg_worker.py` visualisation comments claim 1=LV cavity,
    which contradicts its metrics module. The metrics module is the one that computes
    the numbers BAAI reports, and it is internally consistent — it measures myocardial
    thickness radially outward from label 2 through label 1, which only works if 2 is
    the cavity and 1 is the ring. The comment is wrong; the constants are right.)

    We do not take that on faith. `detect_native_order()` re-derives the mapping from
    the ANATOMY of an actual prediction (the myocardium is the label that encloses a
    cavity), and `assert_canonical` guards the mask we write. If BAAI ever changes its
    convention, this script fails loudly instead of silently swapping LV and RV.

    python -m baselines.run_baai --dataset ACDC --limit 10
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes

from cmr import config
from cmr.checks import assert_canonical
from cmr.data import load_acdc, load_mnms
from cmr.types import LV, MYO, RV

log = logging.getLogger("baselines.run_baai")

HF_REPO = "TaipingQu/BAAI-Cardiac-Agent"
CKPT_1 = "cine_seg_first_SA/Cine_SAX_seg1.pth"  # stage 1: coarse heart localisation
CKPT_2 = "cine_seg_second_SA/Cine_SAX_seg2.pth"  # stage 2: fine 3-class segmentation

# What the source says. Used only as a cross-check against what we measure.
DECLARED_NATIVE = {1: MYO, 2: LV, 3: RV}

_GATE_MSG = f"""
The BAAI weights are behind a GATED Hugging Face repo and could not be downloaded.

    https://huggingface.co/{HF_REPO}

To unblock, ONCE:
  1. Log in to Hugging Face and click "Agree and access repository" on the page above.
  2. Create a token:  https://huggingface.co/settings/tokens   (read scope is enough)
  3. export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
  4. Re-run this script.

Nothing else about this script is blocked. It is the licence click that is missing.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Device. BAAI's predictor is hard-coded to CUDA — .cuda(gpu), torch.cuda.device(gpu),
# map_location="cuda:N", torch.cuda.empty_cache(). There is no device abstraction in it.
# On UM HPC that is fine and this shim is a no-op. On the Mac (MPS) or a CPU node it is
# not, so we redirect those calls. Contained, reversible, and touches no BAAI source.
# ─────────────────────────────────────────────────────────────────────────────
def _shim_cuda(dev: torch.device) -> None:
    if dev.type == "cuda":
        return
    log.warning("BAAI's predictor is CUDA-only; shimming .cuda() -> .to(%s)", dev)
    _load = torch.load
    torch.load = lambda f, *a, **k: _load(f, *a, **{**k, "map_location": dev})  # type: ignore[assignment]
    torch.Tensor.cuda = lambda self, *a, **k: self.to(dev)  # type: ignore[assignment]
    torch.nn.Module.cuda = lambda self, *a, **k: self.to(dev)  # type: ignore[assignment]
    torch.cuda.device = lambda *a, **k: contextlib.nullcontext()  # type: ignore[assignment]
    torch.cuda.empty_cache = lambda: None  # type: ignore[assignment]
    torch.cuda.is_available = lambda: True  # type: ignore[assignment]


def _predictor(repo: Path, dev: torch.device):
    """Import BAAI's SAX predictor out of its repo and load the two checkpoints."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    src = repo / "src" / "CINE_SA_SEG"
    if not src.is_dir():
        raise SystemExit(
            f"BAAI repo not found at {repo}. Clone it:\n"
            f"  git clone https://github.com/plantain-herb/Cardiac-Agent.git {repo}"
        )
    sys.path[:0] = [str(repo), str(src)]

    try:
        w1 = hf_hub_download(HF_REPO, CKPT_1)
        w2 = hf_hub_download(HF_REPO, CKPT_2)
    except GatedRepoError:
        raise SystemExit(_GATE_MSG) from None

    _shim_cuda(dev)
    from infer.predictor_DY import CtAbdomenSegDYModel, CtAbdomenSegDYPredictor  # type: ignore

    model = CtAbdomenSegDYModel(
        model_DY_f=w1,
        model_DY_crop_f=w2,
        network_DY_f=str(src / "train/config/seg_mrdy_stage1.py"),
        network_DY_crop_f=str(src / "train/config/seg_mrdy_stage2.py"),
        config_f=str(src / "example/heart_seg.yaml"),
    )
    # `gpu` is used both as a .cuda() arg and as a torch device= arg. An int means cuda:N;
    # off CUDA we hand it the device itself, which the shim above tolerates.
    return CtAbdomenSegDYPredictor(gpu=0 if dev.type == "cuda" else dev, model=model)


# ─────────────────────────────────────────────────────────────────────────────
# Empirical label-order detection. Anatomy, not trust.
# ─────────────────────────────────────────────────────────────────────────────
def detect_native_order(mask: np.ndarray) -> dict[int, int] | None:
    """Derive {native_label -> canonical_label} from the shape of a real prediction.

    The myocardium is the only structure that is a RING: fill its holes and it swallows
    a cavity. That cavity is the LV (the RV is a crescent stuck to the outside of it).
    Whatever is left over is the RV. This needs no prior knowledge of BAAI's convention,
    which is the entire point — their README does not state one and their own comments
    contradict their constants.
    """
    labs = [i for i in (1, 2, 3) if (mask == i).any()]
    if len(labs) < 3:
        return None  # apical/basal slabs missing a structure — undecidable, skip

    best: tuple[float, int, int] | None = None
    for ring in labs:
        filled = np.zeros(mask.shape, bool)
        for z in range(mask.shape[2]):
            filled[:, :, z] = binary_fill_holes(mask[:, :, z] == ring)
        interior = filled & (mask != ring)
        for cav in labs:
            if cav == ring:
                continue
            frac = float((interior & (mask == cav)).sum()) / max(1, int((mask == cav).sum()))
            if best is None or frac > best[0]:
                best = (frac, ring, cav)

    assert best is not None
    frac, myo, lv = best
    if frac < 0.5:  # nothing encloses anything: the mask is junk, do not guess
        log.warning("label-order detection inconclusive (best enclosure %.2f)", frac)
        return None
    rv = next(i for i in labs if i not in (myo, lv))
    log.info("detected BAAI native order: %d=Myo %d=LV %d=RV (enclosure %.2f)", myo, lv, rv, frac)
    return {lv: LV, myo: MYO, rv: RV}


def canonicalise_baai(mask: np.ndarray, order: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(mask)
    for native, canon in order.items():
        out[mask == native] = canon
    return out


# ─────────────────────────────────────────────────────────────────────────────
def _predict(predictor, img_xyz: np.ndarray, spacing_xyz: tuple[float, float, float]):
    """BAAI speaks SimpleITK, i.e. (Z, Y, X) arrays and (z, y, x) spacing. cmr speaks
    (X, Y, Z). Transpose in, transpose out. Getting this backwards silently transposes
    the heart, so it is done in exactly one place."""
    vol_zyx = np.ascontiguousarray(img_xyz.transpose(2, 1, 0).astype(np.float32))
    mask_zyx = predictor.DY_predict(vol_zyx, np.array(spacing_xyz[::-1], dtype=np.float64))
    return np.asarray(mask_zyx).transpose(2, 1, 0).astype(np.uint8)


def run(cfg: config.Config, dataset: str, repo: Path, limit: int | None, dev: torch.device) -> None:
    predictor = _predictor(repo, dev)
    out_dir = Path(cfg.paths.artifacts) / "masks" / "baai" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = load_acdc(cfg) if dataset == "ACDC" else load_mnms(cfg)
    order: dict[int, int] | None = None
    n = 0

    for i, s in enumerate(subjects):
        if limit and i >= limit:
            break
        for phase, img in (("ED", s.img_ed), ("ES", s.img_es)):
            raw = _predict(predictor, img, s.spacing_mm)

            if order is None:  # decide once, from the first usable prediction
                order = detect_native_order(raw)
                if order and order != DECLARED_NATIVE:
                    log.warning(
                        "detected order %s != order declared in BAAI's source %s. "
                        "Trusting the anatomy.", order, DECLARED_NATIVE
                    )
            if order is None:
                log.warning("%s/%s: order still undetermined — skipped", s.subject_id, phase)
                continue

            mask = canonicalise_baai(raw, order)
            assert_canonical(mask, f"baai/{dataset}/{s.subject_id}/{phase}")  # the guard

            aff = np.diag([*s.spacing_mm, 1.0])
            nib.save(nib.Nifti1Image(mask, aff), str(out_dir / f"{s.subject_id}_{phase}.nii.gz"))
            n += 1

    log.info("BAAI SAX expert: wrote %d canonical masks -> %s", n, out_dir)


def _device(cfg: config.Config) -> torch.device:
    return torch.device(config.device(cfg))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("ACDC", "MnMs"), default="ACDC")
    p.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("CARDIAC_AGENT_REPO", "third_party/Cardiac-Agent")),
        help="clone of github.com/plantain-herb/Cardiac-Agent",
    )
    p.add_argument("--config-file", default=None)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    cfg = config.load(a.config_file)
    run(cfg, a.dataset, a.repo, a.limit, _device(cfg))


if __name__ == "__main__":
    main()
