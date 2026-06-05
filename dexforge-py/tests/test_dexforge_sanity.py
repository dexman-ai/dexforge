"""Sanity check for dexforge_py supported arms.

Confirms that each ``dexforge_py`` supported arm:

1. Exposes ``solve(arm, T)`` and ``fk(arm, q)`` functions.
2. Returns at least one IK candidate for an FK-seeded target.
3. Every returned candidate FK-closes within ``1e-6`` (a packaging smoke
   test, not a solver correctness test; tighter per-arm tolerances are in
   test_dexforge_uniform_fuzz.py).

This mirrors ssik/tests/test_prebuilt_sanity.py but for the Rust
dexforge_py bindings.
"""

from __future__ import annotations

import numpy as np
import pytest

import dexforge_py
from conftest import ARMS_METADATA


@pytest.mark.parametrize(
    "arm_name",
    list(ARMS_METADATA),
    ids=list(ARMS_METADATA),
)
def test_dexforge_arm_solves_fk_seeded_target(arm_name: str) -> None:
    """dexforge_py must solve FK-seeded targets and FK-close."""
    arm = ARMS_METADATA[arm_name]

    # Public surface check: fk and solve functions exist.
    assert callable(dexforge_py.fk), "dexforge_py.fk must be callable"
    assert callable(dexforge_py.solve), "dexforge_py.solve must be callable"

    # FK on sample configuration.
    q = np.array(arm.sample_q, dtype=np.float64)
    assert len(q) == arm.dof, f"sample q length mismatch for {arm_name}"

    T = np.array(dexforge_py.fk(arm_name, q.tolist()))
    assert T.shape == (4, 4), f"{arm_name}: FK result must be 4×4"

    # Solve the FK-seeded target.
    sols = dexforge_py.solve(arm_name, T, respect_limits=False)
    assert sols, f"{arm_name}: solve() returned empty list for FK-seeded target"

    # FK closure on every returned candidate.
    # 1e-6 is the packaging smoke-test bar; per-arm tighter tolerances
    # are in test_dexforge_uniform_fuzz.py.
    for sol_q in sols:
        T_check = np.array(dexforge_py.fk(arm_name, sol_q))
        residual = np.linalg.norm(T_check - T, ord="fro")
        assert (
            residual < 1e-6
        ), f"{arm_name}: FK residual {residual:.2e} > 1e-6 for solution {sol_q}"
