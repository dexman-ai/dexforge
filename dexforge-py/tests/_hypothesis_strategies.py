"""Shared Hypothesis strategies for dexforge_py IK property tests.

Mirrors the strategy from ssik/tests/_hypothesis_strategies.py, filtering
singularities in 6R and 7R kinematic structures.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

# Angle range avoiding 0 and π singularities.
_ANGLE = st.floats(
    min_value=-np.pi + 0.3,
    max_value=np.pi - 0.3,
    allow_nan=False,
    width=64,
)


@st.composite
def non_singular_q6r(draw: st.DrawFn) -> np.ndarray:
    """6R q-vector with singularity-prone axes (q[1], q[2], q[3], q[4])
    at least arcsin(0.2) ≈ 11.5° away from 0 and π.

    Filters:
    - q[1] ≈ 0 or π: shoulder pitch parallel to base.
    - q[2] ≈ 0 or π: upper-arm / elbow alignment.
    - q[3] ≈ 0 or π: elbow alignment (dominant cause of near-double roots).
    - q[4] ≈ 0 or π: wrist 2 alignment (spherical-wrist gimbal).

    q[0] (shoulder yaw) and q[5] (flange roll) are left free.
    """
    q = np.array([draw(_ANGLE) for _ in range(6)])
    assume(abs(np.sin(q[1])) > 0.2)
    assume(abs(np.sin(q[2])) > 0.2)
    assume(abs(np.sin(q[3])) > 0.2)
    assume(abs(np.sin(q[4])) > 0.2)
    return q


@st.composite
def non_singular_q7r(draw: st.DrawFn) -> np.ndarray:
    """7R q-vector with singularity-prone middle axes (q[1] through q[5])
    at least arcsin(0.2) ≈ 11.5° away from 0 and π.

    q[0] (base yaw) and q[6] (flange roll) are left free.

    Applies uniformly to SRS-class (iiwa14, Gen3) and non-SRS 7R
    (Franka, Rizon, Kassow, xArm7).
    """
    q = np.array([draw(_ANGLE) for _ in range(7)])
    assume(abs(np.sin(q[1])) > 0.2)
    assume(abs(np.sin(q[2])) > 0.2)
    assume(abs(np.sin(q[3])) > 0.2)
    assume(abs(np.sin(q[4])) > 0.2)
    assume(abs(np.sin(q[5])) > 0.2)
    return q
