#!/usr/bin/env python3
"""One-time symbolic compiler: ssik artifact -> SymPy -> zero-dependency Rust.

This script acts as an offline code generator.  It sources the robot's
kinematics **directly from the ssik Python artifact** (the same ``*_ik.py``
prebuilt the IK pipeline produces — no URDF parsing required), derives **exact**
closed-form Forward Kinematics (FK) and the geometric Jacobian symbolically with
SymPy, and emits pure, allocation-free Rust for the dexforge hard-real-time
kernel.

Because the ssik artifact already encodes the full kinematic chain as a
product-of-exponentials — ``T = prod_i  T_left[i] . Rot(axis[i], q_i) . T_right[i]``
via the ``_JOINT_T_LEFTS`` / ``_JOINT_AXES`` / ``_JOINT_T_RIGHTS`` constants (the
same data the ``consts.rs`` generators bake) — this generator must run **after**
the ssik artifact is available, exactly like ``generate_<arm>_consts.py``.

Why generate Rust instead of linking a C/C++ dynamics library?  The generated
kernel has **zero dependencies** beyond ``libm`` (for ``sin``/``cos``/``sqrt``),
so it cross-compiles instantly and cleanly for any ARM target / Raspberry Pi.

Real-time kernel constraints honoured by the emitted Rust
---------------------------------------------------------
* **Zero heap allocation.** No ``Vec``/``Box``/``DMatrix``; every output is a
  fixed-size primitive array (``[[f64; 4]; 4]`` for FK, ``[[f64; N]; 6]`` for the
  Jacobian) whose dimensions are baked from the artifact's degrees of freedom
  ``N``.
* **``#![no_std]`` math.** ``core`` has no ``f64::sin``/``cos``/``sqrt`` and no
  ``powi``; the custom printer below rewrites every transcendental as a free
  ``libm::*`` call and expands integer powers into multiplications.  ``pi`` maps
  to ``core::f64::consts::PI``.
* **Strict O(1) execution.** The body is a flat list of ``sympy.cse()`` temporary
  ``let`` bindings followed by a single array literal — no ``if``/``else``, no
  ``for``/``while``.

Usage::

    pip install ssik sympy numpy
    # From a ssik prebuilt artifact name (looks up ssik.prebuilt.<name>):
    python codegen/generate_dynamics.py --artifact ur5_ik

    # From a local artifact .py file, writing to a driver module:
    python codegen/generate_dynamics.py --artifact /path/to/my_arm_ik.py \
        --output src/drivers/my_arm/dynamics.rs
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import sympy as sp
from sympy.printing.rust import RustCodePrinter

# Repository root — all relative paths are anchored here.
_CODEGEN_DIR = Path(__file__).resolve().parent   # codegen/
_REPO_ROOT = _CODEGEN_DIR.parent                 # repository root

# Tolerance for snapping a baked constant to the nearest integer (0, +/-1).  The
# artifact matrices carry rounding noise like ``6.12e-17`` for an exact ``0`` and
# ``1.0`` for an exact axis component; snapping keeps the emitted FK compact and
# numerically faithful to the reference.
_SNAP_TOL = 1e-12


# ---------------------------------------------------------------------------
# ssik artifact loading
# ---------------------------------------------------------------------------


def load_artifact(artifact: str) -> types.ModuleType:
    """Import a ssik artifact by prebuilt name or by path to a ``*_ik.py`` file.

    :param artifact: Either a bare module name (e.g. ``"ur5_ik"``) resolved
        against ``ssik.prebuilt``, or a filesystem path to an artifact module.
    :returns: The imported module exposing ``_JOINT_AXES``, ``_JOINT_T_LEFTS``
        and ``_JOINT_T_RIGHTS`` (and usually ``DOF``).
    :raises ImportError: If neither resolution strategy succeeds.
    """
    candidate = Path(artifact)
    if candidate.exists():
        spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load artifact module from {candidate}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    name = artifact[:-3] if artifact.endswith(".py") else artifact
    try:
        return importlib.import_module(f"ssik.prebuilt.{name}")
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Cannot find {artifact!r} as a file or as ssik.prebuilt.{name}. "
            "Install ssik (`pip install ssik`) or pass a path to the artifact file."
        ) from exc


def _snap(value: float) -> sp.Expr:
    """Snap a baked float to an exact integer when within ``_SNAP_TOL``."""
    rounded = round(value)
    if abs(value - rounded) < _SNAP_TOL:
        return sp.Integer(rounded)
    # 17 significant digits round-trip an IEEE-754 f64 exactly, so the emitted
    # literal reproduces the artifact constant bit-for-bit.
    return sp.Float(value, 17)


def _to_matrix(arr) -> sp.Matrix:
    a = np.asarray(arr, dtype=np.float64)
    return sp.Matrix([[_snap(a[i, j]) for j in range(a.shape[1])] for i in range(a.shape[0])])


def _to_vector(arr) -> sp.Matrix:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    return sp.Matrix([_snap(x) for x in a])


def extract_chain(
    mod: types.ModuleType,
) -> Tuple[List[sp.Matrix], List[sp.Matrix], List[sp.Matrix], int]:
    """Pull the product-of-exponentials chain out of a ssik artifact.

    :returns: ``(t_lefts, axes, t_rights, dof)`` as cleaned SymPy objects.
    """
    try:
        axes_raw = mod._JOINT_AXES
        tl_raw = mod._JOINT_T_LEFTS
        tr_raw = mod._JOINT_T_RIGHTS
    except AttributeError as exc:
        raise ValueError(
            f"artifact {mod.__name__!r} does not expose the expected "
            "_JOINT_AXES / _JOINT_T_LEFTS / _JOINT_T_RIGHTS constants"
        ) from exc

    axes = [_to_vector(a) for a in axes_raw]
    t_lefts = [_to_matrix(m) for m in tl_raw]
    t_rights = [_to_matrix(m) for m in tr_raw]
    dof = int(getattr(mod, "DOF", len(axes)))
    if not (len(axes) == len(t_lefts) == len(t_rights) == dof):
        raise ValueError(
            f"inconsistent chain lengths: axes={len(axes)}, T_left={len(t_lefts)}, "
            f"T_right={len(t_rights)}, DOF={dof}"
        )
    return t_lefts, axes, t_rights, dof


# ---------------------------------------------------------------------------
# Symbolic SE(3) construction
# ---------------------------------------------------------------------------


def _axis_angle_rot(axis: sp.Matrix, angle: sp.Expr) -> sp.Matrix:
    """Homogeneous Rodrigues rotation about a (constant) ``axis`` by ``angle``.

    Mirrors the Rust ``rot_homogeneous`` kernel so the symbolic FK reproduces
    the ssik product-of-exponentials bit-compatibly.
    """
    x, y, z = axis
    c, s = sp.cos(angle), sp.sin(angle)
    oc = 1 - c
    r = sp.Matrix(
        [
            [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s],
            [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s],
            [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc],
        ]
    )
    m = sp.eye(4)
    m[:3, :3] = r
    return m


def build_fk(
    t_lefts: List[sp.Matrix], axes: List[sp.Matrix], t_rights: List[sp.Matrix]
) -> Tuple[sp.Matrix, List[sp.Symbol], List[sp.Matrix]]:
    """Build the symbolic base->ee transform from the ssik chain.

    Returns ``(T_ee, q_syms, frames)`` where ``frames[i]`` is the base->joint-i
    *axis frame* transform (after ``T_left[i]``, before the joint rotation) —
    exactly the data the geometric Jacobian needs.
    """
    q_syms: List[sp.Symbol] = []
    frames: List[sp.Matrix] = []
    t_cum = sp.eye(4)
    for i, (tl, axis, tr) in enumerate(zip(t_lefts, axes, t_rights)):
        q = sp.Symbol(f"q{i}")
        q_syms.append(q)
        frame = t_cum * tl
        frames.append(frame)
        t_cum = frame * _axis_angle_rot(axis, q) * tr
    return t_cum, q_syms, frames


def build_jacobian(
    t_ee: sp.Matrix, axes: List[sp.Matrix], frames: List[sp.Matrix]
) -> sp.Matrix:
    """Geometric Jacobian ``[Jv; Jw]`` (6 x N) for the all-revolute ssik chain."""
    p_e = t_ee[:3, 3]
    cols = []
    for axis, frame in zip(axes, frames):
        r = frame[:3, :3]
        p_i = frame[:3, 3]
        z_i = r * axis  # joint axis expressed in the base frame
        jv = z_i.cross(p_e - p_i)
        jw = z_i
        cols.append(sp.Matrix.vstack(jv, jw))
    return sp.Matrix.hstack(*cols)


# ---------------------------------------------------------------------------
# no_std Rust printer
# ---------------------------------------------------------------------------


class NoStdRustPrinter(RustCodePrinter):
    """Rust printer that targets ``#![no_std]`` + ``libm`` instead of ``std``.

    ``core`` exposes neither ``f64::sin``/``cos``/``sqrt`` nor ``powi``/``powf``,
    so every transcendental becomes a free ``libm::*`` call and integer powers
    expand into plain multiplications.
    """

    _LIBM = {
        "sin": "libm::sin",
        "cos": "libm::cos",
        "tan": "libm::tan",
        "asin": "libm::asin",
        "acos": "libm::acos",
        "atan": "libm::atan",
        "atan2": "libm::atan2",
        "sinh": "libm::sinh",
        "cosh": "libm::cosh",
        "tanh": "libm::tanh",
        "exp": "libm::exp",
        "log": "libm::log",
        "sqrt": "libm::sqrt",
        "Abs": "libm::fabs",
    }

    def _print_Function(self, expr):  # noqa: N802 (SymPy API)
        # Map transcendentals SymPy would otherwise emit as `std`-only method
        # calls (e.g. `x.sin()`) to free `libm::*` functions so the output
        # compiles under `#![no_std]`.
        name = expr.func.__name__
        if name in self._LIBM:
            args = ", ".join(self._print(a) for a in expr.args)
            return f"{self._LIBM[name]}({args})"
        return super()._print_Function(expr)

    def _print_Abs(self, expr):  # noqa: N802
        return f"libm::fabs({self._print(expr.args[0])})"

    def _print_Integer(self, expr):  # noqa: N802
        # `core` arrays are `[f64; _]`; bare integer literals do not coerce to
        # f64 in Rust, so every integer must carry a decimal point.
        return f"{int(expr)}.0"

    def _print_Rational(self, expr):  # noqa: N802
        return f"({int(expr.p)}.0 / {int(expr.q)}.0)"

    def _print_Pow(self, expr):  # noqa: N802
        base = self._print(expr.base)
        exp = expr.exp
        if exp == sp.Rational(1, 2):
            return f"libm::sqrt({base})"
        if exp == sp.Rational(-1, 2):
            return f"(1.0 / libm::sqrt({base}))"
        if exp.is_Integer:
            n = int(exp)
            if n == 0:
                return "1.0"
            # Unroll small integer powers into repeated multiplication (cheaper
            # and `core`-safe, unlike `powi`). 8 is an arbitrary cap that keeps
            # the emitted expression compact; larger exponents fall back to
            # `libm::pow`. FK/Jacobian exponents are almost always 1 or 2.
            if 1 <= n <= 8:
                return "(" + " * ".join([base] * n) + ")"
            if -8 <= n <= -1:
                return "(1.0 / (" + " * ".join([base] * (-n)) + "))"
        return f"libm::pow({base}, {self._print(expr.exp)})"


def rust_literal(expr: sp.Expr, printer: NoStdRustPrinter) -> str:
    return printer.doprint(expr)


# ---------------------------------------------------------------------------
# Rust emission
# ---------------------------------------------------------------------------


def _emit_matrix(
    name: str,
    matrix: sp.Matrix,
    printer: NoStdRustPrinter,
    rows: int,
    cols: int,
) -> List[str]:
    lines = [f"    let {name}: [[f64; {cols}]; {rows}] = ["]
    flat = list(matrix)
    for i in range(rows):
        row_exprs = [rust_literal(flat[i * cols + j], printer) for j in range(cols)]
        lines.append("        [" + ", ".join(row_exprs) + "],")
    lines.append("    ];")
    return lines


def _unpack_q(q_syms: List[sp.Symbol], printer: NoStdRustPrinter) -> List[str]:
    return [f"    let {printer.doprint(q)} = q[{i}];" for i, q in enumerate(q_syms)]


def _emit_cse(replacements, printer: NoStdRustPrinter) -> List[str]:
    return [
        f"    let {printer.doprint(sym)} = {rust_literal(sub, printer)};"
        for sym, sub in replacements
    ]


def generate_rust(
    arm: str,
    n: int,
    t_ee: sp.Matrix,
    jac: sp.Matrix,
    q_syms: List[sp.Symbol],
    script_name: str,
) -> str:
    printer = NoStdRustPrinter({"contract": False})

    out: List[str] = []
    out.append(f"//! Auto-generated symbolic dynamics for the {arm} ({n}-DOF).")
    out.append("//!")
    out.append("//! Forward kinematics and the geometric Jacobian, derived")
    out.append("//! symbolically from the ssik artifact with SymPy and emitted as")
    out.append(f"//! pure `#![no_std]` Rust. Regenerate with `codegen/{script_name}`.")
    out.append("//!")
    out.append("//! Do not edit by hand. Every transcendental is a `libm` call and")
    out.append("//! every output is a fixed-size array: no heap, no branches, no loops.")
    out.append("")
    out.append("#![allow(clippy::all)]")
    out.append("#![allow(unused_parens)]")
    out.append("#![allow(unused_variables)]")
    out.append("#![allow(dead_code)]")
    out.append("")
    out.append(f"/// Degrees of freedom of the {arm}.")
    out.append(f"pub const DOF: usize = {n};")
    out.append("")

    # --- fk ---
    repls_fk, reduced_fk = sp.cse(list(t_ee), optimizations="basic")
    reduced_fk_mat = sp.Matrix(4, 4, reduced_fk)

    out.append("/// Forward kinematics: base -> end-effector SE(3) pose.")
    out.append("///")
    out.append("/// Returns the 4x4 homogeneous transform as a row-major array.")
    out.append(f"pub fn fk(q: &[f64; {n}]) -> [[f64; 4]; 4] {{")
    out.extend(_unpack_q(q_syms, printer))
    out.extend(_emit_cse(repls_fk, printer))
    out.extend(_emit_matrix("m", reduced_fk_mat, printer, 4, 4))
    out.append("    m")
    out.append("}")
    out.append("")

    # --- jacobian ---
    repls_j, reduced_j = sp.cse(list(jac), optimizations="basic")
    reduced_j_mat = sp.Matrix(6, n, reduced_j)

    out.append("/// Geometric Jacobian `[Jv; Jw]` (6 x DOF) of the end-effector.")
    out.append("///")
    out.append("/// Rows 0-2 are linear velocity, rows 3-5 are angular velocity.")
    out.append(f"pub fn jacobian(q: &[f64; {n}]) -> [[f64; {n}]; 6] {{")
    out.extend(_unpack_q(q_syms, printer))
    out.extend(_emit_cse(repls_j, printer))
    out.extend(_emit_matrix("j", reduced_j_mat, printer, 6, n))
    out.append("    j")
    out.append("}")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--artifact",
        required=True,
        help="ssik artifact name (e.g. 'ur5_ik') or path to a *_ik.py file.",
    )
    parser.add_argument(
        "--arm",
        default=None,
        help="Human-readable arm name for the doc comment (default: artifact stem).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination .rs file (default: write to stdout).",
    )
    args = parser.parse_args(argv)

    try:
        mod = load_artifact(args.artifact)
        t_lefts, axes, t_rights, n = extract_chain(mod)
    except (ImportError, ValueError) as exc:
        print(f"[generate_dynamics] ERROR: {exc}", file=sys.stderr)
        return 1

    stem = Path(args.artifact).stem
    arm = args.arm or stem.removesuffix("_ik")
    print(f"[generate_dynamics] {arm}: {n} DOF from ssik artifact {stem!r}", file=sys.stderr)

    t_ee, q_syms, frames = build_fk(t_lefts, axes, t_rights)
    jac = build_jacobian(t_ee, axes, frames)
    rust = generate_rust(arm, n, t_ee, jac, q_syms, Path(__file__).name)

    if args.output is None:
        sys.stdout.write(rust)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rust)
        print(f"[generate_dynamics] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
