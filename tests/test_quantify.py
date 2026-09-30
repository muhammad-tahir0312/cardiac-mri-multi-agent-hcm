"""The boundary test.

The proposal's rule — (<40 = reduced, 40-50 = mildly reduced, >50 = preserved) — has a
hole at exactly 40.0 and exactly 50.0. Those are not pathological inputs; they are the
two most consequential values in the whole classification, and a boundary-aware system
(H2) is engineered to land on them. So they get a test, and it fails loudly if anyone
"tidies" the comparison operators back to strict inequalities.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmr import config
from cmr.quantify import gate, hf_category, is_near_boundary, measure
from cmr.types import LV, MYO, RV


@pytest.fixture(scope="module")
def cfg():
    return config.load()


# ─── the whole point of this file ────────────────────────────────────────────
@pytest.mark.parametrize(
    "lvef,expected",
    [
        (40.0, "HFrEF"),   # closed downward — the proposal leaves this undefined
        (50.0, "HFpEF"),   # closed upward   — likewise
        (49.9, "HFmrEF"),
        (40.1, "HFmrEF"),
        (39.9, "HFrEF"),
        (50.1, "HFpEF"),
        (0.0, "HFrEF"),
        (100.0, "HFpEF"),
    ],
)
def test_hf_boundaries(cfg, lvef, expected):
    assert hf_category(lvef, cfg) == expected


def test_near_boundary_flag(cfg):
    m = cfg.quantification.boundary_margin_pct  # 2.0
    assert is_near_boundary(40.0, cfg)
    assert is_near_boundary(50.0, cfg)
    assert is_near_boundary(40.0 + m, cfg)
    assert is_near_boundary(50.0 - m, cfg)
    assert not is_near_boundary(40.0 + m + 0.01, cfg)
    assert not is_near_boundary(65.0, cfg)


# ─── arithmetic ──────────────────────────────────────────────────────────────
def _phantom(lv_vox: int, myo_vox: int = 0, rv_vox: int = 0, nz: int = 4) -> np.ndarray:
    """A crude 3-D phantom with an exact voxel budget per label. Shape is irrelevant —
    these tests are about the arithmetic, not the anatomy."""
    m = np.zeros((100, 100, nz), np.uint8)
    flat = m.reshape(-1)
    flat[:lv_vox] = LV
    flat[lv_vox : lv_vox + myo_vox] = MYO
    flat[lv_vox + myo_vox : lv_vox + myo_vox + rv_vox] = RV
    return flat.reshape(m.shape)


def test_volumes_and_ef_are_exact(cfg):
    spacing = (1.0, 1.0, 10.0)  # voxel = 10 mm^3 = 0.01 mL
    ed = _phantom(lv_vox=10_000, myo_vox=5_000, rv_vox=8_000)
    es = _phantom(lv_vox=4_000, rv_vox=4_000)
    m = measure(ed, es, spacing, "phantom", cfg, source="groundtruth")

    assert m.edv_ml == pytest.approx(100.0)
    assert m.esv_ml == pytest.approx(40.0)
    assert m.sv_ml == pytest.approx(60.0)
    assert m.lvef_pct == pytest.approx(60.0)
    assert m.lv_mass_g == pytest.approx(50.0 * 1.05)  # Kawel-Boehm density
    assert m.rv_ef_pct == pytest.approx(50.0)
    assert m.hf_category == "HFpEF"
    assert m.source == "groundtruth"


def test_empty_lv_does_not_raise(cfg):
    m = measure(_phantom(0), _phantom(0), (1.0, 1.0, 10.0), "empty", cfg)
    assert m.lvef_pct == 0.0
    assert not gate(m, _phantom(0), _phantom(0), cfg).passed


# ─── the gate ────────────────────────────────────────────────────────────────
def test_gate_catches_esv_ge_edv(cfg):
    ed, es = _phantom(4_000), _phantom(5_000)
    m = measure(ed, es, (1.0, 1.0, 10.0), "backwards", cfg)
    g = gate(m, ed, es, cfg)
    assert not g.passed
    assert "esv_ge_edv" in g.violations


def test_gate_catches_lv_slice_gap(cfg):
    ed = np.zeros((40, 40, 5), np.uint8)
    ed[10:20, 10:20, 0] = LV
    ed[10:20, 10:20, 2] = LV  # slice 1 is empty, sandwiched -> a hole in the stack
    es = ed.copy()
    es[10:15, 10:20, :] = 0
    m = measure(ed, es, (1.0, 1.0, 10.0), "gap", cfg)
    assert "lv_slice_discontinuity_ED" in gate(m, ed, es, cfg).violations


def _stack(*blobs: tuple[slice, slice, slice]) -> np.ndarray:
    m = np.zeros((40, 40, 3), np.uint8)
    for b in blobs:
        m[b] = LV
    return m


def test_gate_catches_a_detached_island(cfg):
    """A real spurious blob is caught by the 3-D stray-fraction check, not by the
    per-slice component count — the component count is deliberately loose (see below)."""
    ed = _stack((slice(5, 15), slice(5, 15), slice(None)),
                (slice(28, 38), slice(28, 38), slice(1, 2)))  # 100 px, 25% of the LV
    es = _stack((slice(7, 13), slice(7, 13), slice(None)))
    v = gate(measure(ed, es, (1.0, 1.0, 10.0), "island", cfg), ed, es, cfg).violations
    assert "disconnected_LV_ED" in v


def test_gate_ignores_an_annotation_speck(cfg):
    """THE calibration property. Expert ground truth is full of 1-2 px border islands.
    A gate that counts them rejects 65% of the ground truth and therefore measures
    nothing. 819/830 GT subjects pass with the speck filter; 287/830 without it.
    """
    ed = _stack((slice(5, 15), slice(5, 15), slice(None)),
                (slice(30, 33), slice(30, 33), slice(1, 2)))  # 9 px < min_component_px
    es = _stack((slice(7, 13), slice(7, 13), slice(None)))
    assert gate(measure(ed, es, (1.0, 1.0, 10.0), "speck", cfg), ed, es, cfg).passed


def test_gate_catches_three_components_on_a_slice(cfg):
    ed = _stack((slice(2, 12), slice(2, 12), slice(None)),
                (slice(15, 25), slice(2, 12), slice(1, 2)),
                (slice(28, 38), slice(2, 12), slice(1, 2)))  # 3 real blobs on one slice
    es = _stack((slice(4, 10), slice(4, 10), slice(None)))
    v = gate(measure(ed, es, (1.0, 1.0, 10.0), "islands", cfg), ed, es, cfg).violations
    assert "multiple_components_LV_ED" in v


def test_gate_passes_a_clean_mask(cfg):
    ed = np.zeros((40, 40, 3), np.uint8)
    es = np.zeros((40, 40, 3), np.uint8)
    for z in range(3):
        ed[14:26, 14:26, z] = LV
        es[17:23, 17:23, z] = LV
    m = measure(ed, es, (1.5, 1.5, 8.0), "clean", cfg)
    assert gate(m, ed, es, cfg).passed


# ─── the diagnosis normaliser (it feeds pathology_top1, so a bug here is a fake metric) ──
@pytest.mark.parametrize(
    "text,expected",
    [
        ("dilated cardiomyopathy", "DCM"),
        ("Tricuspid regurgitation, severe", "TRI"),
        ("ARVC", "ARV"),
        ("normal study", "NOR"),
        # "tri" is a substring of ven-TRI-cular. Whole-word matching or this returns TRI.
        ("left ventricular hypertrophy", None),
        ("pericardial effusion", None),
        ("", None),
    ],
)
def test_diagnosis_normaliser_does_not_match_substrings(text, expected):
    from cmr.eval import normalise_diagnosis

    assert normalise_diagnosis(text) == expected


# ─── the real data ───────────────────────────────────────────────────────────
@pytest.mark.slow
def test_dcm_ef_below_nor(cfg):
    """If this fails, the loader is broken — most likely the ACDC LV/RV remap.

    A DCM cohort measured with the LV and RV labels swapped still produces plausible
    volumes, so nothing else in the stack would notice.
    """
    import pandas as pd

    from cmr.config import artifacts
    from cmr.quantify import run_groundtruth

    p = artifacts(cfg) / "gt_measurements.parquet"
    df = pd.read_parquet(p) if p.exists() else run_groundtruth(cfg)

    dcm = df[df.pathology == "DCM"].lvef_pct
    nor = df[df.pathology == "NOR"].lvef_pct
    assert len(dcm) and len(nor)
    assert dcm.mean() < nor.mean(), f"DCM {dcm.mean():.1f}% !< NOR {nor.mean():.1f}%"

    acdc = df[df.dataset == "ACDC"]
    assert acdc[acdc.pathology == "DCM"].lvef_pct.mean() < 30.0  # textbook DCM
    assert acdc[acdc.pathology == "NOR"].lvef_pct.mean() > 50.0
