"""Agent 1's one non-negotiable check.

If the label order regresses, every ejection fraction downstream inverts and nothing
raises. So: segment a real ACDC subject with the real checkpoint, and demand that the
prediction is (a) canonical and (b) actually correct.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cmr import config, data, seg
from cmr.checks import assert_canonical
from cmr.types import LABEL_NAMES


def dice(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    p, g = pred == label, gt == label
    denom = p.sum() + g.sum()
    return float("nan") if denom == 0 else 2.0 * float((p & g).sum()) / float(denom)


@pytest.mark.slow
def test_acdc_prediction_is_canonical_and_accurate():
    cfg = config.load()
    subject = next(itertools.islice(data.load_acdc(cfg), 1))

    cfg["segmentation"] = dict(cfg.segmentation) | {"backend": "cinema", "checkpoint": "acdc_sax"}
    mask, entropy = seg.Segmenter(cfg).segment(subject.img_ed, subject.spacing_mm)

    assert mask.shape == subject.img_ed.shape
    assert entropy.shape == subject.img_ed.shape
    assert set(np.unique(mask)) <= {0, 1, 2, 3}
    assert 0.0 <= entropy.min() and entropy.max() <= np.log(4) + 1e-5

    # The guard: myocardium must be concentric with label 1, not label 3.
    assert_canonical(mask, "acdc_sax prediction")

    # ...and it is not a vacuous guard: the inverted mask must fail it.
    swapped = mask.copy()
    swapped[mask == 1], swapped[mask == 3] = 3, 1
    with pytest.raises(AssertionError):
        assert_canonical(swapped, "deliberately inverted")

    for label, name in LABEL_NAMES.items():
        d = dice(mask, subject.gt_ed, label)
        assert d > 0.8, f"{name} Dice {d:.3f} <= 0.8 — check the label order"
