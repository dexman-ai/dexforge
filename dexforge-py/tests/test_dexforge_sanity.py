"""Sanity check for every dexforge_py arm driver.

Confirms that each arm exposed by ``dexforge_py``:

1. Has callable ``solve(arm, T)`` and ``fk(arm, q)`` functions.
2. Returns at least one IK candidate for an FK-seeded target.
3. Every returned candidate FK-closes within ``1e-6`` (looser than the
   per-arm bulletproof contracts; this is a packaging smoke test, not a
   solver correctness test).

The per-arm parametrisation is driven by ``conftest.ARMS_METADATA``.
Adding a new arm therefore needs no edits in this file — add the arm to
``ARMS_METADATA`` and expose it in ``dexforge_py/src/lib.rs`` and the
test picks it up automatically.

Mirrors the pattern from ssik's ``test_prebuilt_sanity.py`` (#132).
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
def test_dexforge_arm_imports_and_solves(arm_name: str) -> None:
    """Each arm in ARMS_METADATA must import cleanly and solve FK-seeded targets."""
    arm = ARMS_METADATA[arm_name]
    
    # Public surface check.
    assert hasattr(dexforge_py, "solve"), "dexforge_py missing solve()"
    assert hasattr(dexforge_py, "fk"), "dexforge_py missing fk()"

    q = np.array(arm.sample_q, dtype=np.float64)
    assert len(q) == arm.dof, f"sample_q length mismatch for {arm_name}"

    # FK call
    T = dexforge_py.fk(arm_name, q.tolist())
    T = np.array(T)
    assert T.shape == (4, 4), f"{arm_name}: fk() did not return 4x4 matrix"

    # IK call
    solutions = dexforge_py.solve(arm_name, T, respect_limits=False)
    assert solutions, f"{arm_name}: solve() returned empty list for FK-seeded target"

    # FK closure on every returned candidate. 1e-6 is the smoke test bar;
    # per-arm bulletproof contracts (in test_dexforge_uniform_fuzz.py) gate
    # machine precision separately.
    for sol in solutions:
        q_sol = np.array(sol, dtype=np.float64)
        T_fk = np.array(dexforge_py.fk(arm_name, q_sol.tolist()))
        residual = np.linalg.norm(T_fk - T, ord='fro')
        assert residual < 1e-6, (
            f"{arm_name}: FK residual {residual:.2e} > 1e-6 for solution {sol}"
        )
