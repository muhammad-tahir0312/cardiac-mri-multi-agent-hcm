"""Agent 1 — segmentation. A thin wrapper around CineMA (Fu et al., 2025).

CineMA is a cine-CMR foundation model: a masked-autoencoder ViT/CNN hybrid (ConvUNetR)
pretrained on UK Biobank and fine-tuned per dataset. We do not train it. We run it, we
ensemble its three released seeds, and we take the entropy of that ensemble for free —
which is what Agent 4 uses as its uncertainty channel.

Install (the pins in CineMA's pyproject would downgrade numpy/pandas/monai and break the
rest of this codebase, hence --no-deps):

    pip install --no-deps git+https://github.com/mathpluscode/CineMA
    pip install einops timm omegaconf

THE LABEL-ORDER TRAP — read this before touching anything below.

Every CineMA segmentation checkpoint emits RV=1, MYO=2, LV=3 — the ACDC order, the
INVERSE of ours. Including the M&Ms ones. This is counter-intuitive (you would expect a
checkpoint fine-tuned on M&Ms to speak M&Ms) and it is the single easiest way to produce
a thesis full of wrong-but-plausible ejection fractions. The reason is upstream: CineMA
unifies every training set onto one convention before fine-tuning
(`cinema/__init__.py`: LV_LABEL=3, MYO_LABEL=2, RV_LABEL=1; and
`cinema/data/mnms/__init__.py`: MNMS_LABEL_MAP = {1: LV_LABEL, 2: MYO_LABEL, 3: RV_LABEL},
which swaps M&Ms' 1 and 3 at preprocessing time). So the fine-tuned weights never saw
M&Ms' native order at all.

Reading the source is not evidence, so it was also measured — see _CKPT_NATIVE_ORDER.
Everything this module writes to disk is canonical (1=LV, 2=MYO, 3=RV) and is put through
`assert_canonical` first.
"""

from __future__ import annotations

import logging
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import data
from .checks import LabelOrderError, assert_canonical, canonicalise
from .config import Config, artifacts, device
from .types import Subject

log = logging.getLogger("cmr.seg")

REPO = "mathpluscode/CineMA"

# Native label order emitted by each fine-tuned checkpoint, i.e. the argument to
# checks.canonicalise(). "acdc" means 1=RV / 3=LV and therefore needs the 1<->3 swap.
#
# HOW THIS WAS DETERMINED (not read off the README — measured, on real subjects):
#   1. Ran each checkpoint on an ACDC, an M&Ms and an M&Ms-2 subject.
#   2. Called cmr.checks.lv_label() on the RAW prediction. It answers "which of label 1
#      and label 3 is the myocardium concentric with?" — anatomy, not voxel counts. Every
#      checkpoint answered 3.
#   3. Cross-checked against ground truth, which is canonical: Dice of the raw prediction
#      was 0.00 for label 1 and 0.00 for label 3, and 0.95+ for both after swapping.
#      A 0.00-vs-0.95 gap is not a judgement call.
# So all four are "acdc". If a future checkpoint is added, run tests/test_seg.py — it will
# fail loudly rather than silently invert an EF.
_CKPT_NATIVE_ORDER = {
    "acdc_sax": "acdc",
    "mnms_sax": "acdc",
    "mnms2_sax": "acdc",
    "mnms2_lax_4c": "acdc",
}

# Channel permutation that turns native logits into canonical ones: native channel 3 (LV)
# becomes canonical channel 1, and vice versa. Identity for a hypothetical "mnms" model.
_CANON_PERM = {"acdc": [0, 3, 2, 1], "mnms": [0, 1, 2, 3]}

_PATCH = (192, 192, 16)  # from the checkpoints' own config.yaml — not our config's
_SPACING = (1.0, 1.0)  # in-plane mm/px CineMA was fine-tuned at


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing. CineMA's own eval pipeline, minus the parts that need ground truth.
#
# CineMA preprocesses offline: resample to 1x1x10 mm, then crop 192x192 around the LV
# centre *taken from the ground-truth mask*. At inference there is no ground truth, so we
# resample and then let cinema's sliding-window inference (50%-overlap patches, softmax-
# averaged) cover the whole field of view instead. Z is left on its native grid: the
# encoder never downsamples z (enc_patch_size=(4,4,1), enc_scale_factor=(2,2,1)), so a
# z-resample would buy nothing and would cost an interpolation of the output mask.
# ─────────────────────────────────────────────────────────────────────────────
def _preprocess(img: np.ndarray, spacing: tuple[float, float, float]) -> tuple[torch.Tensor, tuple]:
    x, y, z = img.shape
    nx = max(1, round(x * spacing[0] / _SPACING[0]))
    ny = max(1, round(y * spacing[1] / _SPACING[1]))

    t = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))
    if (nx, ny) != (x, y):
        t = F.interpolate(t.permute(2, 0, 1)[:, None], size=(nx, ny), mode="bilinear")
        t = t[:, 0].permute(1, 2, 0)

    t = (t - t.min()) / (t.max() - t.min() + 1e-8)  # MONAI ScaleIntensityd
    pad = [0, max(0, _PATCH[2] - z), 0, max(0, _PATCH[1] - ny), 0, max(0, _PATCH[0] - nx)]
    return F.pad(t, pad), (nx, ny, z)  # SpatialPadd(method="end")


def _postprocess(probs: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """(4, nx, ny, z) on the 1 mm grid -> (4, X, Y, Z) on the subject's own grid."""
    p = probs.permute(3, 0, 1, 2)  # (z, 4, nx, ny)
    p = F.interpolate(p, size=shape[:2], mode="bilinear")
    p = p.permute(1, 2, 3, 0)  # (4, X, Y, Z)
    return p / p.sum(0, keepdim=True).clamp_min(1e-8)


class _Canonical(nn.Module):
    """CineMA in this codebase's own terms: a plain tensor in, canonical logits out.

    ConvUNetR wants {"sax": t} and answers {"sax": logits} with LV on channel 3. Agent 4's
    Seg-Grad-CAM (xai.py) feeds a bare (1,1,X,Y,Z) tensor and asks for canonical class 1 to
    mean the LV. This adapter is the seam between those two facts, so that neither module
    has to know about the other's convention.
    """

    def __init__(self, net: nn.Module, native: str, view: str = "sax") -> None:
        super().__init__()
        self.net, self.view = net, view
        self.perm = _CANON_PERM[native]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # xai.py hands over the raw image, whose size need not suit the encoder's patch
        # grid. Pad up, then crop the logits back, so the CAM lands on the image's own grid.
        _, _, sx, sy, sz = x.shape
        px, py = (-sx) % 32, (-sy) % 32
        pz = max(0, _PATCH[2] - sz)
        x = F.pad(x, (0, pz, 0, py, 0, px))
        out = self.net({self.view: x})[self.view]
        return out[:, self.perm, :sx, :sy, :sz]


# ─────────────────────────────────────────────────────────────────────────────
class Segmenter:
    """Three seeds, one mask, one entropy map. Reused across subjects — loading 127M
    parameters x3 per patient would dominate the runtime (see graph.Deps)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.backend = cfg.segmentation.backend
        self.ckpt = cfg.segmentation.checkpoint
        self.tta = bool(cfg.segmentation.tta)  # graph.py flips this on mid-run
        self.device = torch.device(device(cfg))
        self.nets: list[nn.Module] = []
        self.model: nn.Module | None = None  # what xai.py hooks its CAM onto
        self._gt: dict[str, tuple] | None = None

        if self.backend == "groundtruth":
            self.native = "mnms"  # ground truth is already canonical
            return
        if self.backend != "cinema":
            raise ValueError(f"unknown segmentation backend {self.backend!r}")
        if self.ckpt not in _CKPT_NATIVE_ORDER:
            raise ValueError(f"unknown checkpoint {self.ckpt!r}. Known: {sorted(_CKPT_NATIVE_ORDER)}")
        if self.ckpt.endswith("lax_4c"):
            raise NotImplementedError("only the SAX checkpoints are wired up; LAX is a 2-D pipeline")

        self.native = _CKPT_NATIVE_ORDER[self.ckpt]
        seeds = list(cfg.compute.seeds) if cfg.segmentation.ensemble else [cfg.compute.seeds[0]]
        for s in seeds:
            self.nets.append(self._load(s))
        self.model = _Canonical(self.nets[0], self.native)
        log.info(
            "CineMA %s: %d seed(s) on %s, tta=%s, native order=%s",
            self.ckpt, len(self.nets), self.device, self.tta, self.native,
        )

    def _load(self, seed: int) -> nn.Module:
        from cinema import ConvUNetR

        net = ConvUNetR.from_finetuned(
            repo_id=self.cfg.segmentation.cinema_repo,
            model_filename=f"finetuned/segmentation/{self.ckpt}/{self.ckpt}_{seed}.safetensors",
            config_filename=f"finetuned/segmentation/{self.ckpt}/config.yaml",
        )
        return net.eval().to(self.device)

    # ── the public call ──────────────────────────────────────────────────────
    def segment(
        self, img: np.ndarray, spacing: tuple[float, float, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """(X, Y, Z) float32 -> (canonical uint8 mask, float32 entropy map).

        `spacing` is optional only because graph.py's node signature predates it; pass it
        whenever you have it (run_dataset always does). Without it the image is fed on its
        native pixel grid, which is a scale the model was not fine-tuned at.
        """
        if self.backend == "groundtruth":
            return self._groundtruth(img)

        probs = self._probs(img, spacing or (1.0, 1.0, 1.0))
        mask = probs.argmax(0).to(torch.uint8).cpu().numpy()
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(0)

        mask = canonicalise(mask, self.native)
        assert_canonical(mask, f"{self.ckpt} prediction")
        return mask, entropy.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _probs(self, img: np.ndarray, spacing: tuple[float, float, float]) -> torch.Tensor:
        """Mean softmax over seeds (and TTA flips). Still in the checkpoint's NATIVE order —
        the entropy must be computed before any relabelling, but relabelling is a
        permutation and entropy is permutation-invariant, so it makes no difference. Keeping
        it native means exactly one place (`segment`) does the swap."""
        from cinema.segmentation.train import segmentation_forward

        t, (nx, ny, z) = _preprocess(img, spacing)
        x = t[None, None].to(self.device)

        # Identity plus in-plane flips. The fine-tuning augmentations were affine (rotate/
        # translate/scale) and never flipped, so a flip is genuinely out-of-distribution —
        # which is the point: it perturbs the model without perturbing the anatomy.
        flips: list[tuple[int, ...]] = [(), (2,), (3,), (2, 3)] if self.tta else [()]

        total = None
        for net in self.nets:
            for f in flips:
                xf = torch.flip(x, f) if f else x
                logits = segmentation_forward(net, {"sax": xf}, {"sax": _PATCH}, torch.float32)["sax"]
                p = torch.softmax(logits, dim=1)
                if f:
                    p = torch.flip(p, f)
                total = p if total is None else total + p
        probs = total / (len(self.nets) * len(flips))

        return _postprocess(probs[0, :, :nx, :ny, :z], img.shape)

    # ── the groundtruth backend ──────────────────────────────────────────────
    def _groundtruth(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the subject's own ground truth, so the rest of the pipeline can be built
        before segmentation works. graph.py short-circuits to subj.gt_* and never reaches
        here; this path exists for anyone calling Segmenter directly.

        The index is keyed on the image bytes because segment() is handed an array, not a
        subject. Masks are held zlib-compressed (they are ~99% background, so this is ~50x)
        rather than as ~1 GB of live uint8. Scope the scan with
        `segmentation.groundtruth_datasets` if you only care about one dataset.
        """
        if self._gt is None:
            self._gt = {}
            sets = self.cfg.get_path("segmentation.groundtruth_datasets", ("ACDC", "MnMs", "MnMs2"))
            for s in data.load_all(self.cfg, tuple(sets)):
                if not s.has_gt:
                    continue
                for im, gt in ((s.img_ed, s.gt_ed), (s.img_es, s.gt_es)):
                    self._gt[_fingerprint(im)] = (zlib.compress(gt.tobytes()), gt.shape)
            log.info("groundtruth backend: indexed %d volumes", len(self._gt))

        hit = self._gt.get(_fingerprint(img))
        if hit is None:
            raise KeyError("groundtruth backend: this image is not in the dataset")
        blob, shape = hit
        mask = np.frombuffer(zlib.decompress(blob), dtype=np.uint8).reshape(shape)
        return mask.copy(), np.zeros(shape, dtype=np.float32)  # GT has no uncertainty


def _fingerprint(img: np.ndarray) -> str:
    import hashlib

    return hashlib.sha1(np.ascontiguousarray(img, dtype=np.float32).tobytes()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Disk layout. One directory per checkpoint, so an ablation cannot overwrite another's
# masks: artifacts/{masks,entropy}/{ckpt}/{dataset}/{sid}_{ed,es}.{nii.gz,npy}
# ─────────────────────────────────────────────────────────────────────────────
def _paths(cfg: Config, dataset: str, sid: str, ckpt: str) -> dict[str, Path]:
    m = artifacts(cfg, "masks", ckpt, dataset)
    e = artifacts(cfg, "entropy", ckpt, dataset)
    return {
        "mask_ed": m / f"{sid}_ed.nii.gz",
        "mask_es": m / f"{sid}_es.nii.gz",
        "ent_ed": e / f"{sid}_ed.npy",
        "ent_es": e / f"{sid}_es.npy",
    }


def run_dataset(cfg: Config, dataset: str, checkpoint: str | None = None) -> None:
    """Segment every subject of `dataset`. Resumable: a subject whose four files already
    exist is skipped, so an interrupted 20-hour MPS run picks up where it stopped."""
    ckpt = checkpoint or cfg.segmentation.checkpoint
    cfg = Config(dict(cfg) | {"segmentation": dict(cfg.segmentation) | {"checkpoint": ckpt}})

    seg = Segmenter(cfg)
    loader = {"ACDC": data.load_acdc, "MnMs": data.load_mnms, "MnMs2": data.load_mnms2}[dataset]

    done = skipped = failed = 0
    for s in loader(cfg):
        p = _paths(cfg, dataset, s.subject_id, ckpt)
        if all(f.exists() for f in p.values()):
            skipped += 1
            continue
        try:
            for phase, img, gt in (("ed", s.img_ed, s.gt_ed), ("es", s.img_es, s.gt_es)):
                if seg.backend == "groundtruth":
                    # We hold the Subject here, so there is no need to make the Segmenter
                    # reverse-engineer it from the pixels.
                    mask, ent = gt, np.zeros(gt.shape, dtype=np.float32)
                else:
                    mask, ent = seg.segment(img, s.spacing_mm)
                _save_mask(mask, s, p[f"mask_{phase}"])
                np.save(p[f"ent_{phase}"], ent)
        except LabelOrderError as e:
            # One unusable mask must not take the other 854 subjects down with it, and it
            # must not be written either. Nothing on disk == Status.FAILED_SEGMENTATION.
            failed += 1
            log.error("%s: %s", s.subject_id, e)
            for f in p.values():
                f.unlink(missing_ok=True)
            continue
        done += 1
        if done % 10 == 0:
            log.info("%s/%s: %d segmented, %d already on disk", ckpt, dataset, done, skipped)

    log.info("%s/%s: done. %d segmented, %d skipped, %d failed", ckpt, dataset, done, skipped, failed)


def _save_mask(mask: np.ndarray, s: Subject, path: Path) -> None:
    assert_canonical(mask, f"{s.dataset}/{s.subject_id} -> {path.name}")
    affine = np.diag([*s.spacing_mm, 1.0])
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(path))


def load_mask(cfg: Config, dataset: str, sid: str, ckpt: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (mask_ed, mask_es), canonical uint8."""
    p = _paths(cfg, dataset, sid, ckpt)
    out = []
    for k in ("mask_ed", "mask_es"):
        if not p[k].exists():
            raise FileNotFoundError(f"{p[k]} — run `cmr segment --dataset {dataset} --ckpt {ckpt}`")
        out.append(np.asanyarray(nib.load(str(p[k])).dataobj).astype(np.uint8))
    return out[0], out[1]


def load_entropy(cfg: Config, dataset: str, sid: str, ckpt: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (entropy_ed, entropy_es), float32, nats. Max is ln(4) — see xai.uncertainty_map."""
    p = _paths(cfg, dataset, sid, ckpt)
    out = []
    for k in ("ent_ed", "ent_es"):
        if not p[k].exists():
            raise FileNotFoundError(f"{p[k]} — run `cmr segment --dataset {dataset} --ckpt {ckpt}`")
        out.append(np.load(p[k]).astype(np.float32))
    return out[0], out[1]
