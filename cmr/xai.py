"""Agent 4 — explainability. The thesis's strongest surviving differentiator.

The competition, precisely:

    BAAI Cardiac Agent   no visual explainability at all. None. Zero heatmaps.
    CardAIc-Agents       visual panels, but they float free of the textual rationale:
                         nothing says WHICH pixels support WHICH sentence.

So the unclaimed contribution is not "a heatmap" and not "a citation". Both exist.
It is the LINK — a machine-checkable edge from

    a sentence in the report
      -> the guideline passage (chunk_id) that grounds it
        -> the Seg-Grad-CAM heatmap of the anatomical structure that claim is about
          -> the ensemble-entropy uncertainty over that same structure

and `Explanation` (types.py) is that edge, made a first-class object.

TWO INDEPENDENT VISUAL CHANNELS, and they answer different questions:

    Seg-Grad-CAM   "which pixels drove the network's decision here?"   (attribution)
    Ensemble       "does the network even agree with itself here?"     (uncertainty)

The second is free — seg.py already runs 3 seeds — and is arguably the more clinically
useful of the two, because a CAM is always confident-looking whereas entropy is not.
A CAM cannot tell you the model is guessing. Entropy can. We ship both, side by side.

SEG-GRAD-CAM, NOT VANILLA GRAD-CAM. This is a correctness issue, not a citation nicety.
Grad-CAM (Selvaraju 2017 — the only method the proposal currently cites) is defined for
CLASSIFIERS: it backprops one class logit, a single scalar, for the whole image. A
segmentation network has no such scalar; it has H*W*Z logits per class. Vinogradova et
al. (2020) supply the missing definition: pick a region M, and backprop the REGION-SUMMED
class score

    y_c = sum_{(i,j,k) in M} logit_c(i, j, k)

then weight the target layer's channels by the global-average-pooled gradient of y_c, as
usual. Choosing M = the predicted structure asks the question a radiologist would ask:
"what made you call THAT the left ventricle?" Vanilla Grad-CAM cannot ask it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .config import Config, artifacts
from .types import LABEL_NAMES, LV, MYO, RV, Explanation, Report, Subject

log = logging.getLogger("cmr.xai")

N_CLASSES = 4  # background, LV, Myo, RV
MAX_ENTROPY = float(np.log(N_CLASSES))  # 1.386 nats — a 4-way coin flip

# Okabe–Ito. Colour-blind-safe under all three common deficiencies.
STRUCTURE_COLOUR = {LV: "#0072B2", MYO: "#D55E00", RV: "#009E73"}


# ─────────────────────────────────────────────────────────────────────────────
# The claim -> structure tables.
#
# This is the spine of the contribution, so it is a TABLE — readable, auditable,
# arguable by a clinician who has never seen Python — and not a pile of regexes
# buried in a function body. A supervisor can review these lists. That is the point.
#
# A clinical claim carries two INDEPENDENT signals, so there are two tables rather
# than one. A single flat phrase->structure table cannot be made to work: it needs a
# row for every (side x topic) product, and it still gets "RV dilatation" wrong
# (whatever tie-break you pick, "dilatation" and "rv dilat" fight each other).
# Factor the two apart and the ambiguity disappears:
#
#     SIDE   which ventricle is this about?     left | right     (earliest mention wins)
#     TOPIC  which tissue is this about?        wall | cavity    (wall beats cavity)
#
# and then resolve on a 2x2 (see `map_claim`). Adding a new phrase means adding one
# row to one table, not N rows to a product.
# ─────────────────────────────────────────────────────────────────────────────
LEFT, RIGHT, WALL, CAVITY = "left", "right", "wall", "cavity"

# Leading space = word boundary. The sentence is space-padded and de-punctuated first,
# so " lv " matches "LV mass" but not "solve"; "left ventric" matches "left ventricular".
SIDE_TABLE: tuple[tuple[str, str], ...] = (
    ("left ventric", LEFT),
    ("lvef", LEFT),
    ("lvedv", LEFT),
    ("lvesv", LEFT),
    (" lv ", LEFT),
    ("right ventric", RIGHT),
    ("rvef", RIGHT),
    ("rvedv", RIGHT),
    ("rvesv", RIGHT),
    (" rv ", RIGHT),
)

TOPIC_TABLE: tuple[tuple[str, str], ...] = (
    # ── the muscle -> Myo. Any of these makes the claim a claim about the wall. ──
    ("hypertroph", WALL),  # hypertrophy / hypertrophic / hypertrophied
    ("wall thickness", WALL),
    ("wall thickening", WALL),
    ("myocardial mass", WALL),
    ("mass index", WALL),
    ("concentric remodel", WALL),
    ("myocardi", WALL),  # myocardium / myocardial
    ("septum", WALL),
    ("septal", WALL),
    ("mass", WALL),
    ("wall", WALL),
    # ── the blood pool -> LV / RV cavity ────────────────────────────────────────
    ("ejection fraction", CAVITY),
    ("end diastolic volume", CAVITY),  # hyphens are normalised to spaces
    ("end systolic volume", CAVITY),
    ("systolic dysfunction", CAVITY),
    ("systolic function", CAVITY),
    ("stroke volume", CAVITY),
    ("cardiac output", CAVITY),
    ("dilat", CAVITY),  # dilated / dilatation / dilation
    ("cardiomyopathy", CAVITY),
    ("heart failure", CAVITY),
    ("hfref", CAVITY),
    ("hfmref", CAVITY),
    ("hfpef", CAVITY),
    ("lvef", CAVITY),
    ("rvef", CAVITY),
    ("edv", CAVITY),
    ("esv", CAVITY),
    ("volume", CAVITY),
    ("ventricle", CAVITY),
    ("ventricular", CAVITY),
)


def _normalise(sentence: str) -> str:
    """Lowercase, punctuation -> space, space-padded. So 'end-diastolic' == 'end diastolic'
    and a leading-space phrase like ' rv ' means a whole word."""
    import re

    return " " + re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip() + " "


def map_claim(sentence: str) -> int | None:
    """Which anatomical structure is this report sentence ABOUT? None if undecidable.

    The 2x2, and the three judgement calls in it, stated openly:

        (left,  wall)   -> Myo      (right, wall)   -> None   [1]
        (left,  cavity) -> LV       (right, cavity) -> RV
        (none,  wall)   -> Myo      (none,  cavity) -> LV     [2]
        (side,  none)   -> that side's cavity       (none, none) -> None

      [1] ACDC / M&Ms / M&Ms-2 label ONLY the LV myocardium — label 2 is the LV wall.
          There is no RV free-wall label anywhere in this cohort. So a claim about RV
          hypertrophy has no structure to point at, and we say so instead of pointing
          at the LV wall and hoping nobody checks.
      [2] An unqualified "ejection fraction" in a CMR report means the LEFT ventricle's,
          by universal convention. This is the one place we assume rather than read.

    `None` is a real answer, not a failure. A claim we cannot ground is shown ungrounded;
    it is never linked to a plausible-looking wrong heatmap. A confidently mislinked
    explanation is worse than an absent one — it is exactly the failure mode this whole
    agent exists to prevent.
    """
    s = _normalise(sentence)

    sides = [(s.find(p), side) for p, side in SIDE_TABLE if p in s]
    side = min(sides)[1] if sides else None  # earliest mention = the grammatical subject

    topics = {topic for p, topic in TOPIC_TABLE if p in s}
    topic = WALL if WALL in topics else (CAVITY if CAVITY in topics else None)

    if topic is None and side is None:
        return None
    if topic == WALL:
        if side == RIGHT:
            log.info("RV wall is not a labelled structure in this cohort — %r left unlinked",
                     sentence[:60])
            return None
        return MYO
    return RV if side == RIGHT else LV


# ─────────────────────────────────────────────────────────────────────────────
# Seg-Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────
def _as_logits(out) -> torch.Tensor:
    """Segmentation models return tensors, tuples, or dicts. Take the logits."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        for k in ("logits", "out", "pred", "seg"):
            if k in out:
                return out[k]
        return next(v for v in out.values() if isinstance(v, torch.Tensor))
    if isinstance(out, list | tuple):
        return _as_logits(out[0])
    raise TypeError(f"cannot find logits in model output of type {type(out)}")


def auto_layer(model: nn.Module, n_classes: int = N_CLASSES) -> nn.Module:
    """The last convolution BEFORE the classification head.

    Not the head itself: the head's activations are already the per-class logits, and a
    CAM over them is a tautology ("the LV is where the model says the LV is"). The layer
    before it still carries K feature channels whose weighting is informative. We spot
    the head by its output width — it is the conv that emits n_classes channels.
    """
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d | nn.Conv3d)]
    if not convs:
        raise ValueError("no Conv2d/Conv3d in this model; pass `layer=` explicitly")
    body = [m for m in convs if m.out_channels != n_classes] or convs
    log.debug("auto target layer: %s", body[-1])
    return body[-1]


def _feed(img_t: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "slices":  # 2-D net: the Z axis becomes the batch
        return img_t.permute(2, 0, 1).unsqueeze(1)  # (Z, 1, X, Y)
    return img_t.unsqueeze(0).unsqueeze(0)  # (1, 1, X, Y, Z)


def _canon(logits: torch.Tensor, layout: str) -> torch.Tensor:
    """-> (C, X, Y, Z), whatever the network's own layout was."""
    return logits.permute(1, 2, 3, 0) if layout == "slices" else logits[0]


def model_patch_size() -> tuple[int, int, int] | None:
    """The exact input CineMA demands, asked of seg.py rather than guessed.

    CineMA's encoder is a ViT: it reshapes its tokens to a FIXED grid, so it accepts one
    input size and one only — 192x192x16. seg.py's own inference never notices, because it
    goes through cinema's sliding-window helper. Seg-Grad-CAM cannot: a sliding window
    stitches together many forward passes under no_grad, and there is no single graph left
    to differentiate. So the CAM is computed on one patch, the model's native one.

    (seg.py's `_Canonical.forward` says it takes "the raw image, which need not be a
    multiple of the patch grid" and pads X/Y to a multiple of 32. That is not enough for a
    ViT, and a bare full-volume call raises. Rather than edit another agent's module, we
    take its declared patch size and feed it what it actually wants.)
    """
    try:
        from .seg import _PATCH

        return tuple(_PATCH)  # type: ignore[return-value]
    except Exception:  # seg.py absent or restructured — fully-conv models don't need this
        return None


def _window(dim: int, size: int, centre: float) -> tuple[int, int]:
    """A `size`-long window over [0, dim), centred where we can, clamped where we cannot."""
    lo = int(round(centre - size / 2))
    return (max(0, min(lo, dim - size)), size) if dim >= size else (0, dim)


def _patch_cam(model, img, spacing, region, target_class, acts, dev, patch):
    """Seg-Grad-CAM for a model that only accepts one input size (CineMA).

    The patch is cut on the model's OWN grid — seg.py's `_preprocess` (1 mm in-plane,
    min-max scaled, padded) — so the CAM explains exactly the tensor the segmenter saw,
    not a lookalike on the subject's native voxel grid. It is then mapped back.
    """
    from .seg import _preprocess

    t, (nx, ny, z) = _preprocess(img, spacing)  # (NX, NY, NZ) on the model's grid
    grid = tuple(t.shape)
    r = torch.as_tensor(region.astype(np.float32))[None, None]
    r = F.interpolate(r, size=grid, mode="nearest")[0, 0]  # region -> the same grid

    idx = torch.nonzero(r)
    centre = idx.float().mean(0) if len(idx) else torch.tensor(grid).float() / 2
    (x0, px), (y0, py), (z0, pz) = (
        _window(grid[i], patch[i], float(centre[i])) for i in range(3)
    )
    win = (slice(x0, x0 + px), slice(y0, y0 + py), slice(z0, z0 + pz))

    # A volume thinner than the patch (ACDC is ~10 slices; the patch wants 16) is padded,
    # never stretched: stretching would move the anatomy relative to the heatmap.
    xw, rw = t[win].to(dev), r[win].to(dev)
    pad = [0, patch[2] - pz, 0, patch[1] - py, 0, patch[0] - px]
    xw, rw = F.pad(xw, pad), F.pad(rw, pad)

    logits = _as_logits(model(xw[None, None].requires_grad_(True)))[0]  # (C, ...)
    if logits.shape[0] <= target_class:
        raise ValueError(f"model emits {logits.shape[0]} classes; need {target_class + 1}")

    y_c = (logits[target_class] * rw).sum()  # THE region-summed score
    a = acts[-1]
    grads = torch.autograd.grad(y_c, a)[0]
    alpha = grads.mean(dim=tuple(range(2, a.ndim)), keepdim=True)
    cam = F.relu((alpha * a).sum(dim=1))[0]  # (x, y, z) on the activation grid

    cam = F.interpolate(cam[None, None], size=tuple(patch), mode="trilinear",
                        align_corners=False)[0, 0]
    full = torch.zeros(grid, device=dev)  # everything outside the patch is unexplained
    full[win] = cam[:px, :py, :pz]
    return F.interpolate(full[None, None], size=tuple(img.shape), mode="trilinear",
                         align_corners=False)[0, 0]


def seg_grad_cam(
    model: nn.Module,
    img: np.ndarray,
    target_class: int,
    region: np.ndarray | None = None,
    layer: nn.Module | None = None,
    patch: tuple[int, int, int] | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Seg-Grad-CAM (Vinogradova et al., 2020). See the module docstring for why.

        y_c = sum over pixels in `region` of the logit for class c
        alpha_k = GAP_spatial( d y_c / d A_k )
        cam = ReLU( sum_k alpha_k * A_k )        upsampled to the image grid

    Args:
        model:        any torch segmentation net; 3-D, 2-D (per-slice), or fixed-input-size
                      (CineMA) — auto-detected in that order.
        img:          (X, Y, Z) float32, the raw image. Min-max scaled here, as at training.
        target_class: 1=LV, 2=Myo, 3=RV (canonical labels — see types.py).
        region:       (X, Y, Z) bool, the M of the paper. Defaults to every pixel, which
                      recovers "explain this class everywhere"; pass the predicted
                      structure to ask the sharper question "why THAT LV?".
        layer:        target layer. Defaults to the last conv before the head.
        patch, spacing: only for a model that accepts a single fixed input size. `explain`
                      supplies both automatically; see `model_patch_size`.

    Returns:
        (X, Y, Z) float32 heatmap, normalised to [0, 1], on the image grid.
    """
    model.eval()
    dev = next(model.parameters()).device
    shape = tuple(img.shape)

    # Min-max to [0,1], NOT a z-score. This is not a free choice: it is what
    # seg.py's _preprocess does (MONAI ScaleIntensityd) and therefore what CineMA was
    # fine-tuned on. Explaining a model on an intensity distribution it never saw would
    # produce a heatmap of the mismatch rather than a heatmap of the anatomy.
    x_img = (img - img.min()) / (np.ptp(img) + 1e-8)
    img_t = torch.as_tensor(x_img, dtype=torch.float32, device=dev)

    if region is None:
        region = np.ones(shape, dtype=bool)
    if not region.any():
        log.warning("empty region for class %s — returning a zero heatmap", target_class)
        return np.zeros(shape, dtype=np.float32)
    reg_t = torch.as_tensor(region.astype(np.float32), device=dev)

    layer = layer or auto_layer(model)
    acts: list[torch.Tensor] = []
    handle = layer.register_forward_hook(lambda _m, _i, o: acts.append(_as_logits(o)))

    try:
        with torch.enable_grad():
            # Volume first: a fully-convolutional 3-D net takes the whole thing. A 2-D net
            # rejects the 5-D input outright (Conv2d: "expected 3-D or 4-D"), so it falls
            # through to the slice stack — safe rather than merely lucky. A fixed-input ViT
            # (CineMA) rejects both, and gets the patch path.
            for layout in ("volume", "slices"):
                acts.clear()
                try:
                    inp = _feed(img_t, layout).requires_grad_(True)
                    logits = _canon(_as_logits(model(inp)), layout)
                except (RuntimeError, ValueError, IndexError) as e:
                    log.debug("layout %s rejected by the model: %s", layout, e)
                    continue
                break
            else:
                if patch is None:
                    raise RuntimeError(
                        "model accepted neither a 3-D volume nor a 2-D slice stack, and no "
                        "`patch` was given. A fixed-input model (CineMA) needs one — see "
                        "xai.model_patch_size()."
                    )
                acts.clear()
                log.debug("fixed-input model: computing the CAM on one %s patch", patch)
                cam = _patch_cam(model, img, spacing or (1.0, 1.0, 1.0), region,
                                 target_class, acts, dev, patch)
                return _normalise_cam(cam, shape, target_class)

            if logits.shape[0] <= target_class:
                raise ValueError(f"model emits {logits.shape[0]} classes; need {target_class + 1}")

            # The model may work at its own resolution. Move the region to meet it.
            r = reg_t
            if tuple(logits.shape[1:]) != shape:
                r = F.interpolate(reg_t[None, None], size=logits.shape[1:], mode="nearest")[0, 0]

            y_c = (logits[target_class] * r).sum()  # THE region-summed score
            a = acts[-1]
            grads = torch.autograd.grad(y_c, a)[0]

            # GAP the gradient over spatial dims -> one weight per feature channel.
            dims = tuple(range(2, a.ndim))  # (2,3) for 2-D, (2,3,4) for 3-D
            alpha = grads.mean(dim=dims, keepdim=True)
            cam = F.relu((alpha * a).sum(dim=1))  # (B, h, w) or (B, x, y, z)

            cam = cam.permute(1, 2, 0) if layout == "slices" else cam[0]  # -> (x, y, z)
            cam = F.interpolate(
                cam[None, None], size=shape, mode="trilinear", align_corners=False
            )[0, 0]
    finally:
        handle.remove()

    return _normalise_cam(cam, shape, target_class)


def _normalise_cam(cam: torch.Tensor, shape: tuple, target_class: int) -> np.ndarray:
    """-> (X, Y, Z) float32 in [0, 1]. ReLU already floors at 0, so dividing by the peak
    is exactly [0, 1] — and 0 keeps its meaning of "no positive evidence here", which a
    min-max rescale would destroy."""
    out = cam.detach().float().cpu().numpy()
    peak = float(out.max())
    if peak <= 0:  # ReLU killed everything: no positive evidence for this class anywhere.
        log.warning("all-zero CAM for class %s — no positive evidence", target_class)
        return np.zeros(shape, dtype=np.float32)
    return (out / peak).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty — the second channel, free from seg.py's 3-seed ensemble
# ─────────────────────────────────────────────────────────────────────────────
def uncertainty_map(entropy: np.ndarray) -> np.ndarray:
    """Ensemble entropy -> [0, 1], normalised by ln(4), the entropy of a 4-way coin flip.

    Deliberately NOT min-max normalised. Min-max would rescale every subject to fill
    [0, 1], so a model that was confident everywhere would look exactly as uncertain as
    one that was guessing everywhere — which is the one comparison a clinician actually
    wants to make. Dividing by the theoretical maximum keeps 0.6 meaning the same thing
    on every subject in the cohort, and makes the colour bar comparable across figures.
    """
    return np.clip(np.asarray(entropy, dtype=np.float32) / MAX_ENTROPY, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# THE CONTRIBUTION: the link
# ─────────────────────────────────────────────────────────────────────────────
def _ed(x):
    """seg.py hands back (ED, ES) pairs; the figure is an ED figure. Accept either."""
    if isinstance(x, list | tuple):
        return x[0]
    return x


def explain(
    subject: Subject,
    mask,
    entropy,
    report: Report,
    cfg: Config,
    model: nn.Module | None = None,
) -> list[Explanation]:
    """One Explanation per Citation: sentence -> chunk_id -> CAM -> uncertainty.

    Every field of the returned object is an edge in that chain, and every edge is
    checkable by a third party: the chunk_id resolves against the guideline index, the
    heatmap is on disk, the mean uncertainty is recomputable from the entropy volume.
    Neither competitor system can produce this object.
    """
    mask_ed, ent_ed = _ed(mask), _ed(entropy)
    unc = uncertainty_map(ent_ed) if ent_ed is not None else None
    out_dir = artifacts(cfg, "xai", subject.dataset, subject.subject_id)

    ent_path = None
    if unc is not None:
        ent_path = out_dir / "uncertainty.npy"
        np.save(ent_path, unc)

    if not report.citations:
        log.warning("%s: report has no citations — nothing to link to", subject.subject_id)
        return []

    cam_cache: dict[int, Path | None] = {}
    explanations: list[Explanation] = []

    for cit in report.citations:
        sentence = cit.supports.strip() or report.diagnosis
        lab = map_claim(sentence)
        if lab is None:
            log.info("%s: no structure maps to %r — left unlinked", subject.subject_id, sentence[:60])

        # One CAM per structure, not per citation: several sentences cite the same anatomy.
        if lab is not None and lab not in cam_cache:
            cam_cache[lab] = _cam_to_disk(model, subject, mask_ed, lab, out_dir)

        region = (mask_ed == lab) if lab is not None else (mask_ed > 0)
        mean_u = float(unc[region].mean()) if unc is not None and region.any() else None

        explanations.append(
            Explanation(
                sentence=sentence,
                chunk_id=cit.chunk_id,
                heatmap_path=str(cam_cache[lab]) if lab is not None and cam_cache[lab] else None,
                entropy_path=str(ent_path) if ent_path else None,
                structure=LABEL_NAMES.get(lab) if lab is not None else None,
                mean_uncertainty=mean_u,
            )
        )

    linked = sum(e.heatmap_path is not None for e in explanations)
    log.info(
        "%s: %d citation(s) -> %d linked explanation(s) over %d structure(s)",
        subject.subject_id, len(report.citations), linked, len(cam_cache),
    )
    return explanations


def _cam_to_disk(model, subject: Subject, mask_ed, lab: int, out_dir: Path) -> Path | None:
    if model is None:
        log.debug("no model given — entropy-only explanation for %s", LABEL_NAMES[lab])
        return None
    region = mask_ed == lab
    if not region.any():
        log.warning("%s: %s absent from the mask — no CAM", subject.subject_id, LABEL_NAMES[lab])
        return None
    cam = seg_grad_cam(
        model, subject.img_ed, target_class=lab, region=region,
        patch=model_patch_size(),  # ignored unless the model demands a fixed input (CineMA)
        spacing=subject.spacing_mm,
    )
    p = out_dir / f"cam_{LABEL_NAMES[lab].lower()}.npy"
    np.save(p, cam)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# The figure. A reader must SEE the link, not infer it.
# ─────────────────────────────────────────────────────────────────────────────
def _best_slice(mask: np.ndarray) -> int:
    """The most informative slice: the one with the most myocardium (mid-ventricular)."""
    areas = [(mask[:, :, z] == MYO).sum() for z in range(mask.shape[2])]
    return int(np.argmax(areas)) if any(areas) else mask.shape[2] // 2


def _wrap(s: str, width: int = 95) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(s, width)[:3])


def render(
    subject: Subject,
    mask,
    explanations: list[Explanation],
    cfg: Config,
    out: Path,
    report: Report | None = None,
) -> Path:
    """One figure per subject. Four panels per claim, and the claim printed underneath.

    `report` is optional but you want it: Explanation carries the chunk_id, and the
    Citation carries the source / section / page that make the chunk_id human-readable.
    Without it the figure prints the chunk_id alone, which is verifiable but not legible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask_ed = _ed(mask)
    img = subject.img_ed
    z = _best_slice(mask_ed)
    cites = {c.chunk_id: c for c in (report.citations if report else [])}

    shown = [e for e in explanations if e.heatmap_path or e.entropy_path][:4]
    if not shown:
        raise ValueError(f"{subject.subject_id}: nothing to render")

    n = len(shown)
    fig = plt.figure(figsize=(15, 4.6 * n), dpi=300)
    gs = fig.add_gridspec(2 * n, 4, height_ratios=[3.2, 1.0] * n, hspace=0.28, wspace=0.06)

    base = img[:, :, z]
    base = (base - base.min()) / (np.ptp(base) + 1e-8)  # np.ptp: ndarray.ptp() died in NumPy 2

    for r, e in enumerate(shown):
        lab = next((k for k, v in LABEL_NAMES.items() if v == e.structure), None)
        axes = [fig.add_subplot(gs[2 * r, c]) for c in range(4)]
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        axes[0].imshow(base.T, cmap="gray")
        axes[0].set_title("cine SAX (ED)", fontsize=10)

        axes[1].imshow(base.T, cmap="gray")
        for label, colour in STRUCTURE_COLOUR.items():
            m = (mask_ed[:, :, z] == label).T
            if m.any():
                axes[1].contourf(
                    np.ma.masked_where(~m, m.astype(float)),
                    levels=[0.5, 1.5], colors=[colour], alpha=0.55,
                )
        axes[1].set_title("segmentation  LV / Myo / RV", fontsize=10)

        # ── CAM ──────────────────────────────────────────────────────────────
        if e.heatmap_path and Path(e.heatmap_path).exists():
            cam = np.load(e.heatmap_path)[:, :, z]
            axes[2].imshow(base.T, cmap="gray")
            im = axes[2].imshow(cam.T, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
            if lab is not None:  # the anatomy, outlined, so CAM-vs-truth is visible
                axes[2].contour((mask_ed[:, :, z] == lab).T, levels=[0.5],
                                colors="w", linewidths=1.1, linestyles="--")
            fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)
            axes[2].set_title(f"Seg-Grad-CAM — {e.structure}", fontsize=10)
        else:
            axes[2].imshow(base.T, cmap="gray")
            axes[2].set_title("no CAM (claim unmapped)", fontsize=10)

        # ── entropy ──────────────────────────────────────────────────────────
        if e.entropy_path and Path(e.entropy_path).exists():
            unc = np.load(e.entropy_path)[:, :, z]
            axes[3].imshow(base.T, cmap="gray")
            im = axes[3].imshow(unc.T, cmap="cividis", alpha=0.70, vmin=0, vmax=1)
            fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.02)
            u = f"{e.mean_uncertainty:.3f}" if e.mean_uncertainty is not None else "n/a"
            axes[3].set_title(f"ensemble entropy — mean over {e.structure or 'heart'} = {u}",
                              fontsize=10)
        else:
            axes[3].axis("off")

        # ── THE LINK, in words, under the panels it explains ─────────────────
        cap = fig.add_subplot(gs[2 * r + 1, :])
        cap.axis("off")
        c = cites.get(e.chunk_id or "")
        src = (
            f"{c.source} — {c.section}, p. {c.page}"
            + (f"   [Class {c.class_of_recommendation}" if c.class_of_recommendation else "")
            + (f" / Level {c.level_of_evidence}]" if c.level_of_evidence else
               ("]" if c.class_of_recommendation else ""))
            if c else "guideline passage"
        )
        colour = STRUCTURE_COLOUR.get(lab, "#666666")
        cap.text(
            0.01, 0.82, f'REPORT CLAIM:  "{_wrap(e.sentence)}"',
            fontsize=11, va="top", ha="left", weight="bold",
        )
        cap.text(
            0.01, 0.40,
            f"GROUNDED IN:  {src}\n"
            f"CHUNK:  {e.chunk_id}\n"
            f"SUPPORTED BY:  the {e.structure or '—'} heatmap and uncertainty above",
            fontsize=9.5, va="top", ha="left", family="monospace", color="#222222",
        )
        cap.add_patch(
            plt.Rectangle((0.0, 0.0), 0.004, 1.0, transform=cap.transAxes,
                          color=colour, clip_on=False)
        )

    fig.suptitle(
        f"Linked explanation — {subject.subject_id}  ({subject.dataset}, "
        f"{subject.pathology}, slice {z})\n"
        "each row: the report sentence, the guideline passage that grounds it, "
        "and the image evidence for it",
        fontsize=13, y=0.997,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("linked-XAI figure -> %s", out)
    return out
