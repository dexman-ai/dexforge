#!/usr/bin/env python3
"""Pinocchio ground-truth generator for the dexforge dynamics kernels.

The dexforge codegen (``codegen/generate_dynamics.py``) emits the manipulator
dynamics — ``gravity(q, gravity)`` and ``inverse_dynamics(q, qd, qdd, gravity)``
— as pure ``#![no_std]`` Rust derived symbolically (energy/Lagrangian method)
from the arm's kinematic chain and the link inertials in its URDF.

To *prove* that generated Rust is correct, this script computes an independent
ground truth with the reference rigid-body-dynamics library **Pinocchio**
(https://github.com/stack-of-tasks/pinocchio):

* Sample ``N`` (default 1000) random, physically valid joint states
  ``(q, qd, qdd)`` with ``q`` drawn inside the arm's mechanical joint limits and
  ``qd``/``qdd`` inside symmetric velocity/acceleration envelopes.
* For each state, evaluate the exact required gravity-compensation torque
    ``g(q) = pin.computeGeneralizedGravity`` and the inverse-dynamics torque
    ``tau = pin.rnea(q, qd, qdd)`` (recursive Newton-Euler).
* Emit the input/output pairs as a ``#![no_std]``-friendly Rust table
  (``dynamics_test_data.rs``) consumed by the driver's dynamics parity test,
  which asserts the generated Rust reproduces this golden reference.

The Pinocchio model and the generated Rust are sourced from the **same** URDF,
so the only thing under test is the dexforge symbolic-to-Rust pipeline; the two
independent dynamics formulations (Lagrangian vs. Newton-Euler) must agree.

Usage::

    pip install pin numpy
    python codegen/generate_dynamics_test_data.py \
        --urdf path/to/xarm7.urdf \
        --output path/to/out/dynamics_test_data.rs \
        --samples 1000 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_CODEGEN_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CODEGEN_DIR.parent

# Default magnitude of the symmetric joint velocity / acceleration envelopes the
# random states are sampled from (rad/s, rad/s^2).  Modest, physically plausible
# values that exercise the Coriolis/centrifugal and inertial torque terms.
_DEFAULT_QD_LIMIT = 1.0
_DEFAULT_QDD_LIMIT = 2.0

# Gravity in the base frame; must match the vector baked into the generated Rust
# (`codegen/generate_dynamics.py --gravity`, default 0 0 -9.81).
_DEFAULT_GRAVITY = (0.0, 0.0, -9.81)


def _format_f64(x: float) -> str:
    """Round-trip-exact Rust ``f64`` literal."""
    return repr(float(x))


def _row(values) -> str:
    return "[" + ", ".join(_format_f64(v) for v in values) + "]"


def build_dataset(
    urdf_path: Path,
    samples: int,
    seed: int,
    qd_limit: float,
    qdd_limit: float,
    gravity: Tuple[float, float, float],
):
    """Sample valid states and compute Pinocchio gravity + inverse dynamics.

    :returns: ``(dof, cases)`` where each case is the tuple
        ``(q, qd, qdd, gravity_torque, inverse_dynamics_torque)`` of length-DOF
        ``f64`` lists.
    """
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf_path))
    model.gravity.linear = np.asarray(gravity, dtype=np.float64)
    data = model.createData()
    dof = model.nv

    lower = np.asarray(model.lowerPositionLimit, dtype=np.float64)
    upper = np.asarray(model.upperPositionLimit, dtype=np.float64)
    # Guard against unbounded (continuous) joints that URDF reports as +/-inf.
    finite = np.isfinite(lower) & np.isfinite(upper)
    lower = np.where(finite, lower, -np.pi)
    upper = np.where(finite, upper, np.pi)

    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(samples):
        q = rng.uniform(lower, upper)
        qd = rng.uniform(-qd_limit, qd_limit, dof)
        qdd = rng.uniform(-qdd_limit, qdd_limit, dof)

        g = np.asarray(pin.computeGeneralizedGravity(model, data, q), dtype=np.float64)
        tau = np.asarray(pin.rnea(model, data, q, qd, qdd), dtype=np.float64)

        cases.append((q, qd, qdd, g, tau))
    return dof, cases


def emit_rust(arm: str, dof: int, cases, gravity: Tuple[float, float, float]) -> str:
    """Render the golden dataset as a ``#![no_std]`` Rust table."""
    out: List[str] = []
    out.append(f"//! Auto-generated Pinocchio dynamics ground truth for the {arm}.")
    out.append("//!")
    out.append("//! Each case bundles a physically valid joint state `(q, qd, qdd)`")
    out.append("//! sampled inside the arm's mechanical limits with the exact")
    out.append("//! gravity-compensation torque `g(q)` and inverse-dynamics torque")
    out.append("//! `tau = M(q) qdd + C(q, qd) qd + g(q)`, both computed with")
    out.append("//! Pinocchio (recursive Newton-Euler). The driver's dynamics test")
    out.append("//! asserts the generated Rust reproduces this golden reference.")
    out.append("//!")
    out.append(
        f"//! Gravity base-frame vector: [{gravity[0]}, {gravity[1]}, {gravity[2]}]."
    )
    out.append("//! Do not edit by hand. Regenerate with")
    out.append("//! `codegen/generate_dynamics_test_data.py`.")
    out.append("")
    out.append("#![allow(clippy::all)]")
    out.append("")
    out.append(f"/// Degrees of freedom of the {arm}.")
    out.append(f"pub const DOF: usize = {dof};")
    out.append("")
    out.append("/// A single ground-truth dynamics sample.")
    out.append("///")
    out.append("/// `(q, qd, qdd, gravity_torque, inverse_dynamics_torque)`.")
    out.append(
        "pub type Case = (\n"
        f"    [f64; {dof}],\n"
        f"    [f64; {dof}],\n"
        f"    [f64; {dof}],\n"
        f"    [f64; {dof}],\n"
        f"    [f64; {dof}],\n"
        ");"
    )
    out.append("")
    out.append(f"/// Pinocchio ground-truth dynamics dataset ({len(cases)} cases).")
    out.append(f"pub static CASES: [Case; {len(cases)}] = [")
    for q, qd, qdd, g, tau in cases:
        out.append(
            "    ("
            f"{_row(q)}, {_row(qd)}, {_row(qdd)}, {_row(g)}, {_row(tau)}),"
        )
    out.append("];")
    out.append("")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--urdf",
        type=Path,
        required=True,
        help="URDF carrying <inertial> blocks for the arm (same one the generated "
        "Rust dynamics is derived from).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination .rs file (default: write to stdout).",
    )
    parser.add_argument(
        "--arm",
        default=None,
        help="Human-readable arm name for the doc comment (default: URDF stem).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of random valid joint states to sample (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed for reproducible sampling (default: 0).",
    )
    parser.add_argument(
        "--qd-limit",
        type=float,
        default=_DEFAULT_QD_LIMIT,
        help="Symmetric joint velocity envelope in rad/s (default: 1.0).",
    )
    parser.add_argument(
        "--qdd-limit",
        type=float,
        default=_DEFAULT_QDD_LIMIT,
        help="Symmetric joint acceleration envelope in rad/s^2 (default: 2.0).",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        nargs=3,
        metavar=("GX", "GY", "GZ"),
        default=_DEFAULT_GRAVITY,
        help="Gravitational acceleration vector in the base frame (default: 0 0 -9.81).",
    )
    args = parser.parse_args(argv)

    if not args.urdf.exists():
        print(f"[generate_dynamics_test_data] ERROR: URDF not found: {args.urdf}", file=sys.stderr)
        return 1

    try:
        import pinocchio  # noqa: F401
    except ImportError:
        print(
            "[generate_dynamics_test_data] ERROR: Pinocchio not installed. "
            "Install with `pip install pin`.",
            file=sys.stderr,
        )
        return 1

    arm = args.arm or args.urdf.stem
    dof, cases = build_dataset(
        args.urdf,
        args.samples,
        args.seed,
        args.qd_limit,
        args.qdd_limit,
        tuple(args.gravity),
    )
    print(
        f"[generate_dynamics_test_data] {arm}: {len(cases)} states, {dof} DOF "
        f"from {args.urdf} (gravity={tuple(args.gravity)})",
        file=sys.stderr,
    )

    rust = emit_rust(arm, dof, cases, tuple(args.gravity))
    if args.output is None:
        sys.stdout.write(rust)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rust)
        print(f"[generate_dynamics_test_data] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
