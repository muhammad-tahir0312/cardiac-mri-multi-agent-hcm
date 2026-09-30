"""The gate must accept the ground truth.

A plausibility gate is only meaningful if it accepts what expert annotation produces.
If it rejects the ground truth, it is not strict — it is wrong, and every H3 number
computed with it is meaningless.

An earlier version of the gate counted 1-2 pixel annotation islands as "multiple
connected components" and rejected 65% of expert annotations. This test exists so that
cannot silently happen again.
"""

import pandas as pd
import pytest

from cmr import config


@pytest.mark.slow
def test_gate_accepts_expert_ground_truth():
    cfg = config.load()
    df = pd.read_parquet(f"{cfg.paths.artifacts}/gt_measurements.parquet")
    rate = df.gate_passed.mean()
    assert rate >= 0.98, (
        f"the plausibility gate rejects {(1 - rate) * 100:.1f}% of EXPERT GROUND TRUTH. "
        f"A gate that refuses the ground truth measures nothing. Re-calibrate it "
        f"(cmr/quantify.py::gate) before trusting any H3 result."
    )
