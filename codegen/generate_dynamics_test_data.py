#!/usr/bin/env python3
"""Pinocchio ground-truth generator for the dexforge dynamics kernels.

The dexforge codegen (``codegen/generate_dynamics.py``) emits the manipulator
dynamics — ``gravity``, ``mass_matrix``, ``coriolis`` and ``inverse_dynamics`` —
as pure ``#![no_std]`` Rust derived symbolically (energy/Lagrangian method)
from the arm's kinematic chain and the link inertials in its URDF.

To *prove* that generated Rust is correct, this one-time script computes an
independent ground truth with **Pinocchio**
(https://github.com/stack-of-tasks/pinocchio):

* Sample ``N`` (default **10_000**) random, physically valid joint states
  ``(q, qd, qdd)`` with ``q`` inside the arm's mechanical joint limits and
  ``qd``/``qdd`` inside symmetric velocity/acceleration envelopes.
* For each state, evaluate with Pinocchio:

  - gravity torque ``g(q) = computeGeneralizedGravity``
  - mass matrix ``M(q)`` via ``crba`` (symmetrised)
  - Coriolis/centrifugal torque ``C(q, qd) qd`` via ``computeCoriolisMatrix``
  - inverse-dynamics torque ``tau = rnea(q, qd, qdd)``

* Write a compact little-endian ``f64`` binary blob
  (``dynamics_test_data.bin``) plus a thin ``#![no_std]`` Rust loader
  (``dynamics_test_data.rs``). The driver's dynamics parity test compares
  dexforge against this golden reference — **Pinocchio is not needed to run
  the tests**.

## Conventions (read before trusting a mismatch)

**Joint order.** Pinocchio ``model.nv`` indices follow the URDF actuated-joint
order (for the UR5: ``shoulder_pan``, ``shoulder_lift``, ``elbow``,
``wrist_1``, ``wrist_2``, ``wrist_3``). dexforge's generated dynamics use the
same ordering as the ssik artifact / URDF serial chain. The script prints the
Pinocchio joint names at generation time — they must line up with the driver.

**Gravity sign.** Both sides use the same base-frame gravitational
acceleration vector (default ``[0, 0, -9.81]``). Pinocchio stores it on
``model.gravity.linear``; dexforge takes it as the ``gravity`` / ``g_world``
argument to ``gravity`` and ``inverse_dynamics``. A flipped sign looks like
every gravity torque (and the static part of inverse dynamics) flipping sign.

**Same URDF.** The Pinocchio model and the generated Rust must be sourced from
the *same* URDF the driver was built from, so the only thing under test is the
dexforge symbolic-to-Rust pipeline (Lagrangian vs. Newton-Euler).

Usage::

    pip install pin numpy
    python codegen/generate_dynamics_test_data.py \\
        --urdf repos/dexforge-ur/rust/src/drivers/ur5/ur5.urdf \\
        --output-dir repos/dexforge-ur/rust/src/drivers/ur5 \\
        --arm ur5 --samples 10000 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_DEFAULT_SAMPLES = 10_000
_DEFAULT_QD_LIMIT = 1.0
_DEFAULT_QDD_LIMIT = 2.0
_DEFAULT_GRAVITY = (0.0, 0.0, -9.81)


def _format_f64(x: float) -> str:
    return repr(float(x))


def _symmetrize_mass_matrix(m: np.ndarray) -> np.ndarray:
    """Pinocchio CRBA fills only the upper triangle; mirror it."""
    upper = np.triu(m)
    return upper + upper.T - np.diag(np.diag(upper))


def build_dataset(
    urdf_path: Path,
    samples: int,
    seed: int,
    qd_limit: float,
    qdd_limit: float,
    gravity: Tuple[float, float, float],
):
    """Sample states and compute Pinocchio g, M, C·qd, and RNEA torques.

    :returns: ``(dof, joint_names, cases)`` where each case is
        ``(q, qd, qdd, g, M, c, tau)``.
    """
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf_path))
    model.gravity.linear = np.asarray(gravity, dtype=np.float64)
    data = model.createData()
    dof = int(model.nv)

    # Skip the universe/root name; actuated joints are 1..njoints-1 for a
    # simple serial arm loaded from URDF.
    joint_names = [model.names[i] for i in range(1, model.njoints)]
    print(
        f"[generate_dynamics_test_data] Pinocchio joint order ({dof} DOF): "
        f"{joint_names}",
        file=sys.stderr,
    )
    print(
        f"[generate_dynamics_test_data] gravity.linear = {tuple(model.gravity.linear)} "
        f"(must match dexforge gravity / g_world argument)",
        file=sys.stderr,
    )

    lower = np.asarray(model.lowerPositionLimit, dtype=np.float64)
    upper = np.asarray(model.upperPositionLimit, dtype=np.float64)
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

        pin.crba(model, data, q)
        mass = _symmetrize_mass_matrix(np.asarray(data.M, dtype=np.float64))

        pin.computeCoriolisMatrix(model, data, q, qd)
        c_mat = np.asarray(data.C, dtype=np.float64)
        c_vec = c_mat @ qd

        tau = np.asarray(pin.rnea(model, data, q, qd, qdd), dtype=np.float64)

        cases.append((q, qd, qdd, g, mass, c_vec, tau))
    return dof, joint_names, cases


def _pack_cases(dof: int, cases: Sequence) -> bytes:
    """Little-endian f64 blob: per case q,qd,qdd,g,M_row_major,c,tau."""
    parts: List[bytes] = []
    for q, qd, qdd, g, mass, c_vec, tau in cases:
        row = np.concatenate(
            [
                np.asarray(q, dtype=np.float64).reshape(-1),
                np.asarray(qd, dtype=np.float64).reshape(-1),
                np.asarray(qdd, dtype=np.float64).reshape(-1),
                np.asarray(g, dtype=np.float64).reshape(-1),
                np.asarray(mass, dtype=np.float64).reshape(-1),
                np.asarray(c_vec, dtype=np.float64).reshape(-1),
                np.asarray(tau, dtype=np.float64).reshape(-1),
            ]
        )
        # q,qd,qdd,g (4*dof) + M (dof*dof) + c,tau (2*dof)
        expected = 6 * dof + dof * dof
        if row.size != expected:
            raise RuntimeError(f"case width {row.size} != expected {expected}")
        parts.append(row.astype("<f8", copy=False).tobytes())
    return b"".join(parts)


def emit_rust_loader(
    arm: str,
    dof: int,
    n_cases: int,
    gravity: Tuple[float, float, float],
    joint_names: Sequence[str],
    bin_filename: str,
) -> str:
    """Thin ``#![no_std]`` Rust module that `include_bytes!` the binary blob."""
    # q,qd,qdd,g (4*dof) + M (dof*dof) + c,tau (2*dof)
    floats_per_case = 6 * dof + dof * dof
    names_list = ", ".join(f'"{n}"' for n in joint_names)
    g_lit = ", ".join(_format_f64(v) for v in gravity)
    out: List[str] = []
    out.append(f"//! Auto-generated Pinocchio dynamics ground truth for the {arm}.")
    out.append("//!")
    out.append("//! Binary layout of `dynamics_test_data.bin` (little-endian `f64`):")
    out.append("//! for each of [`N_CASES`] cases, concatenated:")
    out.append(
        f"//!   `q[{dof}]`, `qd[{dof}]`, `qdd[{dof}]`, `gravity_torque[{dof}]`,"
    )
    out.append(
        f"//!   `mass_matrix[{dof}*{dof}]` (row-major), `coriolis_torque[{dof}]`,"
    )
    out.append(f"//!   `inverse_dynamics_torque[{dof}]`.")
    out.append("//!")
    out.append("//! ## Conventions")
    out.append("//!")
    out.append(
        "//! * **Joint order** (Pinocchio `nv` / URDF actuated): "
        + ", ".join(joint_names)
        + "."
    )
    out.append(
        "//! * **Gravity** base-frame vector: "
        f"[{g_lit}] — must match the argument passed to dexforge "
        "`gravity` / `inverse_dynamics`."
    )
    out.append("//!")
    out.append("//! Do not edit by hand. Regenerate with:")
    out.append("//! ```text")
    out.append("//! pip install pin numpy")
    out.append(
        "//! python codegen/generate_dynamics_test_data.py \\"
    )
    out.append(
        f"//!     --urdf <path/to/{arm}.urdf> --output-dir <driver-dir> \\"
    )
    out.append(f"//!     --arm {arm} --samples {n_cases} --seed 0")
    out.append("//! ```")
    out.append("")
    out.append("#![allow(clippy::all)]")
    out.append("")
    out.append(f"/// Degrees of freedom of the {arm}.")
    out.append(f"pub const DOF: usize = {dof};")
    out.append("")
    out.append("/// Number of Pinocchio ground-truth samples in the binary blob.")
    out.append(f"pub const N_CASES: usize = {n_cases};")
    out.append("")
    out.append("/// `f64` count per case (see module docs for field layout).")
    out.append(f"pub const FLOATS_PER_CASE: usize = {floats_per_case};")
    out.append("")
    out.append("/// Gravity vector baked into this dataset (base frame).")
    out.append(f"pub const GRAVITY: [f64; 3] = [{g_lit}];")
    out.append("")
    out.append("/// Pinocchio / URDF actuated joint names in index order.")
    out.append(f"pub const JOINT_NAMES: [&str; {dof}] = [{names_list}];")
    out.append("")
    out.append("/// Raw little-endian `f64` payload (`N_CASES * FLOATS_PER_CASE` values).")
    out.append(f'pub static RAW: &[u8] = include_bytes!("{bin_filename}");')
    out.append("")
    out.append("/// One decoded ground-truth sample.")
    out.append("#[derive(Clone, Copy, Debug)]")
    out.append("pub struct Case {")
    out.append(f"    pub q: [f64; {dof}],")
    out.append(f"    pub qd: [f64; {dof}],")
    out.append(f"    pub qdd: [f64; {dof}],")
    out.append(f"    pub gravity_torque: [f64; {dof}],")
    out.append(f"    pub mass_matrix: [[f64; {dof}]; {dof}],")
    out.append(f"    pub coriolis_torque: [f64; {dof}],")
    out.append(f"    pub inverse_dynamics_torque: [f64; {dof}],")
    out.append("}")
    out.append("")
    out.append("#[inline]")
    out.append("fn read_f64(bytes: &[u8], idx: usize) -> f64 {")
    out.append("    let start = idx * 8;")
    out.append("    let mut buf = [0u8; 8];")
    out.append("    buf.copy_from_slice(&bytes[start..start + 8]);")
    out.append("    f64::from_le_bytes(buf)")
    out.append("}")
    out.append("")
    out.append("#[inline]")
    out.append(f"fn read_vec(bytes: &[u8], offset: usize) -> [f64; {dof}] {{")
    out.append(f"    let mut out = [0.0f64; {dof}];")
    out.append(f"    for i in 0..{dof} {{")
    out.append("        out[i] = read_f64(bytes, offset + i);")
    out.append("    }")
    out.append("    out")
    out.append("}")
    out.append("")
    out.append("/// Decode ground-truth case `index` (`0 .. N_CASES`).")
    out.append("pub fn case(index: usize) -> Case {")
    out.append('    assert!(index < N_CASES, "dynamics_test_data case index out of range");')
    out.append("    let base = index * FLOATS_PER_CASE;")
    out.append("    // Sanity: blob length matches the declared layout.")
    out.append("    debug_assert_eq!(RAW.len(), N_CASES * FLOATS_PER_CASE * 8);")
    out.append(f"    let q = read_vec(RAW, base);")
    out.append(f"    let qd = read_vec(RAW, base + {dof});")
    out.append(f"    let qdd = read_vec(RAW, base + {2 * dof});")
    out.append(f"    let gravity_torque = read_vec(RAW, base + {3 * dof});")
    out.append(f"    let mut mass_matrix = [[0.0f64; {dof}]; {dof}];")
    out.append(f"    let m_off = base + {4 * dof};")
    out.append(f"    for r in 0..{dof} {{")
    out.append(f"        for c in 0..{dof} {{")
    out.append(f"            mass_matrix[r][c] = read_f64(RAW, m_off + r * {dof} + c);")
    out.append("        }")
    out.append("    }")
    out.append(f"    let coriolis_torque = read_vec(RAW, base + {4 * dof + dof * dof});")
    out.append(
        f"    let inverse_dynamics_torque = read_vec(RAW, base + {5 * dof + dof * dof});"
    )
    # Offsets: q@0, qd@dof, qdd@2dof, g@3dof, M@4dof, c@(4dof+dof²), τ@(5dof+dof²)
    out.append(
        "    Case { q, qd, qdd, gravity_torque, mass_matrix, coriolis_torque, inverse_dynamics_torque }"
    )
    out.append("}")
    out.append("")
    return "\n".join(out)


def emit_dynamics_parity_rs(arm: str, dof: int) -> str:
    """Emit a DOF-parameterized ``dynamics_parity`` unit-test module."""
    return f"""//! Pinocchio ground-truth parity for the symbolic {arm} dynamics.
//!
//! Compares `generated_dynamics::{{gravity, mass_matrix, coriolis,
//! inverse_dynamics}}` against the independent Newton-Euler reference in
//! [`dynamics_test_data`] (produced by `codegen/generate_dynamics_test_data.py`).
//!
//! Before treating a mismatch as a dynamics bug, confirm:
//! * joint order matches [`dynamics_test_data::JOINT_NAMES`]
//! * gravity vector matches [`dynamics_test_data::GRAVITY`]

#![cfg(test)]

#[path = "dynamics_test_data.rs"]
mod dynamics_test_data;

use super::generated_dynamics::{{coriolis, gravity, inverse_dynamics, mass_matrix}};

/// Absolute torque / inertia tolerance. Lagrangian vs. recursive Newton-Euler
/// from the same URDF agree to well below this on commodity `f64`.
const ATOL: f64 = 1e-6;

fn max_abs_diff_vec(a: &[f64; {dof}], b: &[f64; {dof}]) -> f64 {{
    let mut m = 0.0f64;
    for i in 0..{dof} {{
        let d = libm::fabs(a[i] - b[i]);
        if d > m {{
            m = d;
        }}
    }}
    m
}}

fn max_abs_diff_mat(a: &[[f64; {dof}]; {dof}], b: &[[f64; {dof}]; {dof}]) -> f64 {{
    let mut m = 0.0f64;
    for i in 0..{dof} {{
        for j in 0..{dof} {{
            let d = libm::fabs(a[i][j] - b[i][j]);
            if d > m {{
                m = d;
            }}
        }}
    }}
    m
}}

#[test]
fn conventions_match_pinocchio_dataset() {{
    assert_eq!(dynamics_test_data::DOF, {dof});
    assert_eq!(
        dynamics_test_data::GRAVITY,
        [0.0, 0.0, -9.81],
        "gravity sign/convention drift — update generator --gravity or the test"
    );
    assert_eq!(dynamics_test_data::JOINT_NAMES.len(), {dof});
    assert!(dynamics_test_data::N_CASES > 0);
    assert_eq!(
        dynamics_test_data::RAW.len(),
        dynamics_test_data::N_CASES * dynamics_test_data::FLOATS_PER_CASE * 8
    );
}}

#[test]
fn matches_pinocchio_gravity_mass_coriolis_and_inverse_dynamics() {{
    let g_world = dynamics_test_data::GRAVITY;
    let mut worst = [0.0f64; 4]; // g, M, C, tau
    let mut worst_idx = [0usize; 4];

    for idx in 0..dynamics_test_data::N_CASES {{
        let c = dynamics_test_data::case(idx);

        let g_got = gravity(&c.q, &g_world);
        let m_got = mass_matrix(&c.q);
        let c_got = coriolis(&c.q, &c.qd);
        let tau_got = inverse_dynamics(&c.q, &c.qd, &c.qdd, &g_world);

        let dg = max_abs_diff_vec(&g_got, &c.gravity_torque);
        let dm = max_abs_diff_mat(&m_got, &c.mass_matrix);
        let dc = max_abs_diff_vec(&c_got, &c.coriolis_torque);
        let dtau = max_abs_diff_vec(&tau_got, &c.inverse_dynamics_torque);

        for (slot, d) in [(0, dg), (1, dm), (2, dc), (3, dtau)] {{
            if d > worst[slot] {{
                worst[slot] = d;
                worst_idx[slot] = idx;
            }}
        }}

        assert!(
            dg < ATOL,
            "case {{idx}}: gravity mismatch max|Δ|={{dg:e}}\\n got={{g_got:?}}\\n ref={{:?}}",
            c.gravity_torque
        );
        assert!(
            dm < ATOL,
            "case {{idx}}: mass_matrix mismatch max|Δ|={{dm:e}}"
        );
        assert!(
            dc < ATOL,
            "case {{idx}}: coriolis mismatch max|Δ|={{dc:e}}\\n got={{c_got:?}}\\n ref={{:?}}",
            c.coriolis_torque
        );
        assert!(
            dtau < ATOL,
            "case {{idx}}: inverse_dynamics mismatch max|Δ|={{dtau:e}}\\n got={{tau_got:?}}\\n ref={{:?}}",
            c.inverse_dynamics_torque
        );
    }}

    assert!(
        worst.iter().all(|&d| d < ATOL),
        "worst Δ: g={{:e}} (case {{}}), M={{:e}} (case {{}}), C={{:e}} (case {{}}), τ={{:e}} (case {{}})",
        worst[0],
        worst_idx[0],
        worst[1],
        worst_idx[1],
        worst[2],
        worst_idx[2],
        worst[3],
        worst_idx[3],
    );
}}

#[test]
fn inverse_dynamics_equals_m_qdd_plus_c_plus_g() {{
    let g_world = dynamics_test_data::GRAVITY;
    let n_check = dynamics_test_data::N_CASES.min(64);
    for idx in 0..n_check {{
        let c = dynamics_test_data::case(idx);
        let m = mass_matrix(&c.q);
        let cor = coriolis(&c.q, &c.qd);
        let g = gravity(&c.q, &g_world);
        let tau = inverse_dynamics(&c.q, &c.qd, &c.qdd, &g_world);

        let mut composed = [0.0f64; {dof}];
        for i in 0..{dof} {{
            let mut mqdd = 0.0;
            for j in 0..{dof} {{
                mqdd += m[i][j] * c.qdd[j];
            }}
            composed[i] = mqdd + cor[i] + g[i];
        }}
        let d = max_abs_diff_vec(&tau, &composed);
        assert!(
            d < ATOL,
            "case {{idx}}: τ ≠ M qdd + C qd + g (max|Δ|={{d:e}})"
        );
    }}
}}
"""


def write_ground_truth(
    urdf: Path,
    output_dir: Path,
    arm: Optional[str] = None,
    samples: int = _DEFAULT_SAMPLES,
    seed: int = 0,
    qd_limit: float = _DEFAULT_QD_LIMIT,
    qdd_limit: float = _DEFAULT_QDD_LIMIT,
    gravity: Tuple[float, float, float] = _DEFAULT_GRAVITY,
) -> Tuple[Path, Path, Path]:
    """Generate Pinocchio golden data + parity tests into *output_dir*.

    :returns: ``(bin_path, loader_rs, parity_rs)``.
    """
    try:
        import pinocchio  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Pinocchio not installed. Install with `pip install pin`."
        ) from exc

    arm_name = arm or urdf.stem
    dof, joint_names, cases = build_dataset(
        urdf, samples, seed, qd_limit, qdd_limit, gravity
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_name = "dynamics_test_data.bin"
    bin_path = output_dir / bin_name
    loader_rs = output_dir / "dynamics_test_data.rs"
    parity_rs = output_dir / "dynamics_parity.rs"

    blob = _pack_cases(dof, cases)
    bin_path.write_bytes(blob)
    loader_rs.write_text(
        emit_rust_loader(
            arm_name, dof, len(cases), gravity, joint_names, bin_name
        ),
        encoding="utf-8",
    )
    parity_rs.write_text(emit_dynamics_parity_rs(arm_name, dof), encoding="utf-8")
    print(
        f"[generate_dynamics_test_data] wrote {bin_path} ({len(blob)} bytes), "
        f"{loader_rs}, and {parity_rs}",
        file=sys.stderr,
    )
    return bin_path, loader_rs, parity_rs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        required=True,
        help="URDF carrying <inertial> blocks (same one the generated Rust used).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Driver directory that receives dynamics_test_data.rs + .bin.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,  # backwards-compat alias ignored; use --output-dir
    )
    parser.add_argument("--arm", default=None, help="Arm name for docs (default: URDF stem).")
    parser.add_argument(
        "--samples",
        type=int,
        default=_DEFAULT_SAMPLES,
        help=f"Number of random joint states (default: {_DEFAULT_SAMPLES}).",
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed (default: 0).")
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
        help="Base-frame gravity vector (default: 0 0 -9.81).",
    )
    args = parser.parse_args(argv)

    if not args.urdf.exists():
        print(f"[generate_dynamics_test_data] ERROR: URDF not found: {args.urdf}", file=sys.stderr)
        return 1

    out_dir = args.output_dir
    # Legacy: if someone still passes --output pointing at a .rs file, use its parent.
    if args.output is not None:
        out_dir = args.output.parent if args.output.suffix == ".rs" else args.output

    try:
        write_ground_truth(
            urdf=args.urdf,
            output_dir=out_dir,
            arm=args.arm,
            samples=args.samples,
            seed=args.seed,
            qd_limit=args.qd_limit,
            qdd_limit=args.qdd_limit,
            gravity=tuple(args.gravity),
        )
    except ImportError as exc:
        print(f"[generate_dynamics_test_data] ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — surface Pinocchio/URDF failures
        print(f"[generate_dynamics_test_data] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
