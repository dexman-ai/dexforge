#!/usr/bin/env python3
"""One-time symbolic compiler: URDF -> SymPy -> zero-dependency ``#![no_std]`` Rust.

This script acts as an offline code generator.  It parses a robot URDF, derives
**exact** closed-form Forward Kinematics (FK) and the geometric Jacobian
symbolically with SymPy, and emits pure, allocation-free Rust suitable for the
dexforge hard-real-time kernel.

Why generate Rust instead of linking a C/C++ dynamics library?  The generated
kernel has **zero dependencies** beyond ``libm`` (for ``sin``/``cos``/``sqrt``),
so it cross-compiles instantly and cleanly for any ARM target / Raspberry Pi
through a plain ``pip install`` of this script's own deps (``sympy``) — no Eigen,
no Boost, no toolchain surgery.

Real-time kernel constraints honoured by the emitted Rust
---------------------------------------------------------
* **Zero heap allocation.** No ``Vec``/``Box``/``DMatrix``; every output is a
  fixed-size primitive array (``[[f64; 4]; 4]`` for FK, ``[[f64; N]; 6]`` for the
  Jacobian) whose dimensions are baked from the URDF's degrees of freedom ``N``.
* **``#![no_std]`` math.** ``core`` has no ``f64::sin``/``cos``/``sqrt`` and no
  ``powi``; the custom printer below rewrites every transcendental as a free
  ``libm::*`` call and expands integer powers into multiplications.  ``pi`` maps
  to ``core::f64::consts::PI``.
* **Strict O(1) execution.** The body is a flat list of ``sympy.cse()`` temporary
  ``let`` bindings followed by a single array literal — no ``if``/``else``, no
  ``for``/``while``.

Usage::

    pip install sympy
    # Defaults to the checked-in UR5 fixture and prints to stdout:
    python codegen/generate_dynamics.py

    # Explicit URDF, base/end-effector links, and output file:
    python codegen/generate_dynamics.py \
        --urdf src/drivers/ur5/ur5.urdf \
        --base base_link --ee ee_link \
        --output src/drivers/ur5/dynamics.rs
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import sympy as sp
from sympy.printing.rust import RustCodePrinter

# Repository root — all relative paths are anchored here.
_CODEGEN_DIR = Path(__file__).resolve().parent   # codegen/
_REPO_ROOT = _CODEGEN_DIR.parent                 # repository root
_DEFAULT_URDF = _REPO_ROOT / "src" / "drivers" / "ur5" / "ur5.urdf"

# Joint types that contribute a controllable degree of freedom.
_ACTUATED = ("revolute", "continuous", "prismatic")


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------


@dataclass
class Joint:
    """A single URDF joint along the kinematic chain."""

    name: str
    jtype: str
    parent: str
    child: str
    xyz: Tuple[float, float, float]
    rpy: Tuple[float, float, float]
    axis: Tuple[float, float, float]

    @property
    def actuated(self) -> bool:
        return self.jtype in _ACTUATED


def _parse_triple(text: Optional[str], default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if text is None:
        return default
    parts = [float(v) for v in text.split()]
    if len(parts) != 3:
        raise ValueError(f"expected 3 values, got {text!r}")
    return (parts[0], parts[1], parts[2])


def parse_urdf(path: Path) -> List[Joint]:
    """Parse every joint from a URDF into a flat list (declaration order)."""
    root = ET.parse(path).getroot()
    joints: List[Joint] = []
    for j in root.findall("joint"):
        jtype = j.get("type", "fixed")
        parent = j.find("parent")
        child = j.find("child")
        if parent is None or child is None:
            raise ValueError(f"joint {j.get('name')!r} missing parent/child")
        origin = j.find("origin")
        xyz = _parse_triple(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
        rpy = _parse_triple(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
        axis_el = j.find("axis")
        axis = _parse_triple(axis_el.get("xyz") if axis_el is not None else None, (0.0, 0.0, 1.0))
        joints.append(
            Joint(
                name=j.get("name", "?"),
                jtype=jtype,
                parent=parent.get("link"),
                child=child.get("link"),
                xyz=xyz,
                rpy=rpy,
                axis=axis,
            )
        )
    return joints


def order_chain(joints: List[Joint], base: Optional[str], ee: Optional[str]) -> List[Joint]:
    """Order joints into the serial chain that connects ``base`` to ``ee``.

    When ``base``/``ee`` are omitted they are inferred from the tree topology:
    the base is the unique link that is never a child, and the end-effector is a
    leaf link that is never a parent.
    """
    by_parent = {}
    for jt in joints:
        by_parent.setdefault(jt.parent, []).append(jt)

    children = {jt.child for jt in joints}
    parents = {jt.parent for jt in joints}

    if base is None:
        roots = [p for p in parents if p not in children]
        if len(roots) != 1:
            raise ValueError(
                f"cannot infer base link (candidates: {sorted(roots)}); pass --base"
            )
        base = roots[0]
    if ee is None:
        leaves = [c for c in children if c not in parents]
        if len(leaves) != 1:
            raise ValueError(
                f"cannot infer ee link (candidates: {sorted(leaves)}); pass --ee"
            )
        ee = leaves[0]

    chain: List[Joint] = []
    link = base
    while link != ee:
        outgoing = by_parent.get(link)
        if not outgoing:
            raise ValueError(f"no joint leaves link {link!r}; chain to {ee!r} is broken")
        if len(outgoing) != 1:
            raise ValueError(
                f"link {link!r} branches into {[j.name for j in outgoing]}; "
                "URDF is not a serial chain"
            )
        jt = outgoing[0]
        chain.append(jt)
        link = jt.child
    return chain


# ---------------------------------------------------------------------------
# Symbolic SE(3) construction
# ---------------------------------------------------------------------------

_PI_MULTIPLE_TOL = 1e-9


def _num(value: float) -> sp.Expr:
    """Convert a URDF float into a clean SymPy expression.

    Angles that are integer multiples of ``pi/2`` (the overwhelmingly common
    URDF case) collapse to exact symbolic constants so that, e.g.,
    ``cos(pi/2)`` folds to ``0`` and the generated FK stays compact and exact.
    """
    if value == 0.0:
        return sp.Integer(0)
    ratio = value / float(sp.pi)
    nearest = round(ratio * 2) / 2  # nearest multiple of 1/2
    if nearest != 0 and abs(ratio - nearest) < _PI_MULTIPLE_TOL:
        return sp.nsimplify(nearest) * sp.pi
    return sp.Float(value, 17)


def _rot_x(a: sp.Expr) -> sp.Matrix:
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: sp.Expr) -> sp.Matrix:
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: sp.Expr) -> sp.Matrix:
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rpy_to_rot(roll: sp.Expr, pitch: sp.Expr, yaw: sp.Expr) -> sp.Matrix:
    # URDF fixed-axis convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    return _rot_z(yaw) * _rot_y(pitch) * _rot_x(roll)


def _homogeneous(rot: sp.Matrix, trans: sp.Matrix) -> sp.Matrix:
    m = sp.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = trans
    return m


def _axis_angle_rot(axis: sp.Matrix, angle: sp.Expr) -> sp.Matrix:
    """Rodrigues rotation about a (constant) unit ``axis`` by symbolic ``angle``."""
    x, y, z = axis
    c, s = sp.cos(angle), sp.sin(angle)
    oc = 1 - c
    return sp.Matrix(
        [
            [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s],
            [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s],
            [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc],
        ]
    )


def _num_axis(value: float) -> sp.Expr:
    # Joint axes are unit vectors with components in {-1, 0, 1} for virtually all
    # URDFs; keep them exact when possible, otherwise carry the literal.
    rounded = round(value)
    if abs(value - rounded) < _PI_MULTIPLE_TOL:
        return sp.Integer(rounded)
    return sp.Float(value, 17)


def build_fk(chain: List[Joint]) -> Tuple[sp.Matrix, List[sp.Symbol], List[sp.Matrix]]:
    """Build the symbolic base->ee transform and the per-joint frame transforms.

    Returns ``(T_ee, q_syms, frames)`` where ``frames[i]`` is the base->joint-i
    *frame* transform (after the joint origin, before applying the joint motion)
    aligned with ``q_syms[i]`` — exactly the data the geometric Jacobian needs.
    """
    q_syms: List[sp.Symbol] = []
    frames: List[sp.Matrix] = []
    t_cum = sp.eye(4)
    dof = 0
    for jt in chain:
        roll, pitch, yaw = (_num(v) for v in jt.rpy)
        origin = _homogeneous(_rpy_to_rot(roll, pitch, yaw), sp.Matrix(jt.xyz))
        t_frame = t_cum * origin
        if jt.actuated:
            q = sp.Symbol(f"q{dof}")
            q_syms.append(q)
            frames.append(t_frame)
            dof += 1
            axis = sp.Matrix([_num_axis(v) for v in jt.axis])
            if jt.jtype == "prismatic":
                motion = _homogeneous(sp.eye(3), axis * q)
            else:  # revolute / continuous
                motion = _homogeneous(_axis_angle_rot(axis, q), sp.zeros(3, 1))
            t_cum = t_frame * motion
        else:  # fixed joint: pure static transform
            t_cum = t_frame
    return t_cum, q_syms, frames


def build_jacobian(
    t_ee: sp.Matrix, chain: List[Joint], q_syms: List[sp.Symbol], frames: List[sp.Matrix]
) -> sp.Matrix:
    """Geometric Jacobian ``[Jv; Jw]`` (6 x N) of the end-effector pose."""
    p_e = t_ee[:3, 3]
    actuated = [jt for jt in chain if jt.actuated]
    cols = []
    for jt, frame in zip(actuated, frames):
        r = frame[:3, :3]
        p_i = frame[:3, 3]
        axis = sp.Matrix([_num_axis(v) for v in jt.axis])
        z_i = r * axis  # joint axis expressed in the base frame
        if jt.jtype == "prismatic":
            jv = z_i
            jw = sp.zeros(3, 1)
        else:  # revolute / continuous
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


def generate_rust(
    arm: str,
    n: int,
    t_ee: sp.Matrix,
    jac: sp.Matrix,
    q_syms: List[sp.Symbol],
    script_name: str,
) -> str:
    printer = NoStdRustPrinter({"contract": False})

    # Shared common-subexpression elimination across FK and Jacobian so the two
    # functions stay compact; each function only re-binds the temporaries it uses.
    fk_flat = list(t_ee)
    jac_flat = list(jac)

    out: List[str] = []
    out.append(f"//! Auto-generated symbolic dynamics for the {arm} ({n}-DOF).")
    out.append("//!")
    out.append("//! Forward kinematics and the geometric Jacobian, derived")
    out.append("//! symbolically from the URDF with SymPy and emitted as pure")
    out.append(f"//! `#![no_std]` Rust. Regenerate with `codegen/{script_name}`.")
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
    repls_fk, reduced_fk = sp.cse(fk_flat, optimizations="basic")
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
    repls_j, reduced_j = sp.cse(jac_flat, optimizations="basic")
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


def _unpack_q(q_syms: List[sp.Symbol], printer: NoStdRustPrinter) -> List[str]:
    lines = []
    for i, q in enumerate(q_syms):
        lines.append(f"    let {printer.doprint(q)} = q[{i}];")
    return lines


def _emit_cse(replacements, printer: NoStdRustPrinter) -> List[str]:
    lines = []
    for sym, sub in replacements:
        lines.append(f"    let {printer.doprint(sym)} = {rust_literal(sub, printer)};")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--urdf",
        type=Path,
        default=_DEFAULT_URDF,
        help="Path to the robot URDF (default: checked-in UR5 fixture).",
    )
    parser.add_argument("--base", default=None, help="Base link name (default: inferred).")
    parser.add_argument("--ee", default=None, help="End-effector link name (default: inferred).")
    parser.add_argument(
        "--arm",
        default=None,
        help="Human-readable arm name for the doc comment (default: URDF stem).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination .rs file (default: write to stdout).",
    )
    args = parser.parse_args(argv)

    if not args.urdf.exists():
        parser.error(f"URDF file not found: {args.urdf}")

    arm = args.arm or args.urdf.stem
    joints = parse_urdf(args.urdf)
    chain = order_chain(joints, args.base, args.ee)
    n = sum(1 for jt in chain if jt.actuated)
    if n == 0:
        parser.error("chain has zero actuated joints; nothing to generate")

    print(f"[generate_dynamics] {arm}: {n} DOF over {len(chain)} chain joints", file=sys.stderr)

    t_ee, q_syms, frames = build_fk(chain)
    jac = build_jacobian(t_ee, chain, q_syms, frames)
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
