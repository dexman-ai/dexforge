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

    # Also emit inverse-dynamics (gravity / mass-matrix / Coriolis) by sourcing
    # the link masses + inertias from a URDF whose actuated-joint order matches
    # the artifact (the URDF supplies the *physical* properties the kinematics
    # artifact does not carry):
    python codegen/generate_dynamics.py --artifact ur5_ik \
        --urdf src/drivers/ur5/ur5_inertial.urdf \
        --output src/drivers/ur5/dynamics.rs

Dynamics (runtime gravity, mass matrix, Coriolis, inverse dynamics)
-----------------------------------------------------------
When ``--urdf`` points at a URDF that carries ``<inertial>`` blocks (link
masses, centres of mass and inertia tensors), the generator also derives the
manipulator dynamics with the energy (Lagrangian) method and emits matching
``#![no_std]`` Rust:

* ``gravity(q, gravity) -> [f64; N]`` — holding torque ``tau = dU/dq`` from the
    total potential energy ``U = -sum_i m_i (g . p_com_i)``.
* ``mass_matrix(q) -> [[f64; N]; N]`` — symmetric inertia matrix
  ``M = sum_i (m_i Jv_i^T Jv_i + Jw_i^T R_i I_i R_i^T Jw_i)``.
* ``coriolis(q, qd) -> [f64; N]`` — ``C(q, qd) qd`` from the Christoffel symbols
  of the first kind.
* ``inverse_dynamics(q, qd, qdd, gravity) -> [f64; N]`` — ``M qdd + C qd + g``
    for the MIT-mode torque feed-forward.

The dynamics path is all-revolute (every arm currently in the repo); prismatic
joints contribute to the linear Jacobian only and carry no angular inertia term.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import types
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# URDF inertial properties (link masses / centres of mass / inertia tensors)
#
# The ssik kinematics artifact deliberately carries no physical properties, so
# the manipulator dynamics is sourced from a URDF whose actuated-joint order
# matches the artifact chain.  Only the ``<inertial>`` blocks are consumed here;
# the kinematics (and hence FK/Jacobian) still come from the artifact.
# ---------------------------------------------------------------------------

_PI_MULTIPLE_TOL = 1e-9


@dataclass
class Inertial:
    """Mass, centre-of-mass pose and inertia tensor of a single URDF link."""

    mass: float
    com_xyz: Tuple[float, float, float]
    com_rpy: Tuple[float, float, float]
    # Inertia tensor about the centre of mass, expressed in the inertial frame.
    ixx: float
    ixy: float
    ixz: float
    iyy: float
    iyz: float
    izz: float


@dataclass
class UrdfJoint:
    """A single actuated/fixed URDF joint with its child link inertial (if any)."""

    name: str
    jtype: str
    parent: str
    child: str
    axis: Tuple[float, float, float]
    child_inertial: Optional[Inertial] = field(default=None)

    @property
    def actuated(self) -> bool:
        return self.jtype in ("revolute", "continuous", "prismatic")


def _parse_triple(text: Optional[str], default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if text is None:
        return default
    parts = [float(v) for v in text.split()]
    if len(parts) != 3:
        raise ValueError(f"expected 3 values, got {text!r}")
    return (parts[0], parts[1], parts[2])


def _parse_inertials(root: ET.Element) -> Dict[str, Inertial]:
    """Collect ``<inertial>`` blocks keyed by link name (links without one are skipped)."""
    out: Dict[str, Inertial] = {}
    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if name is None or inertial is None:
            continue
        mass_el = inertial.find("mass")
        inertia_el = inertial.find("inertia")
        if mass_el is None or inertia_el is None:
            continue
        origin = inertial.find("origin")
        com_xyz = _parse_triple(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
        com_rpy = _parse_triple(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))

        def _i(attr: str) -> float:
            return float(inertia_el.get(attr, "0"))

        out[name] = Inertial(
            mass=float(mass_el.get("value", "0")),
            com_xyz=com_xyz,
            com_rpy=com_rpy,
            ixx=_i("ixx"),
            ixy=_i("ixy"),
            ixz=_i("ixz"),
            iyy=_i("iyy"),
            iyz=_i("iyz"),
            izz=_i("izz"),
        )
    return out


def parse_urdf_joints(path: Path) -> List[UrdfJoint]:
    """Parse the serial joint chain plus each child link's inertial block."""
    root = ET.parse(path).getroot()
    inertials = _parse_inertials(root)
    joints: List[UrdfJoint] = []
    for j in root.findall("joint"):
        parent = j.find("parent")
        child = j.find("child")
        if parent is None or child is None:
            raise ValueError(f"joint {j.get('name')!r} missing parent/child")
        axis_el = j.find("axis")
        axis = _parse_triple(axis_el.get("xyz") if axis_el is not None else None, (0.0, 0.0, 1.0))
        child_link = child.get("link")
        joints.append(
            UrdfJoint(
                name=j.get("name", "?"),
                jtype=j.get("type", "fixed"),
                parent=parent.get("link"),
                child=child_link,
                axis=axis,
                child_inertial=inertials.get(child_link),
            )
        )
    return joints


def order_urdf_chain(joints: List[UrdfJoint]) -> List[UrdfJoint]:
    """Order URDF joints into the unique serial base->ee chain (topological walk)."""
    by_parent: Dict[str, List[UrdfJoint]] = {}
    for jt in joints:
        by_parent.setdefault(jt.parent, []).append(jt)
    children = {jt.child for jt in joints}
    parents = {jt.parent for jt in joints}
    roots = [p for p in parents if p not in children]
    leaves = [c for c in children if c not in parents]
    if len(roots) != 1 or len(leaves) != 1:
        raise ValueError(
            f"URDF is not a single serial chain (roots={sorted(roots)}, leaves={sorted(leaves)})"
        )
    chain: List[UrdfJoint] = []
    link = roots[0]
    ee = leaves[0]
    while link != ee:
        outgoing = by_parent.get(link)
        if not outgoing or len(outgoing) != 1:
            raise ValueError(f"URDF chain branches or breaks at link {link!r}")
        jt = outgoing[0]
        chain.append(jt)
        link = jt.child
    return chain


def _rpy_to_rot(roll: float, pitch: float, yaw: float) -> sp.Matrix:
    """URDF fixed-axis RPY: ``R = Rz(yaw) Ry(pitch) Rx(roll)`` (numeric, constant)."""

    def rot_x(a):
        c, s = sp.Float(np.cos(a), 17), sp.Float(np.sin(a), 17)
        return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rot_y(a):
        c, s = sp.Float(np.cos(a), 17), sp.Float(np.sin(a), 17)
        return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rot_z(a):
        c, s = sp.Float(np.cos(a), 17), sp.Float(np.sin(a), 17)
        return sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    return rot_z(yaw) * rot_y(pitch) * rot_x(roll)


def _snap_matrix(m: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(m.rows, m.cols, [_snap(float(v)) for v in m])


@dataclass
class RigidBody:
    """A massive link: its base->COM symbolic pose plus inertia, ready for energy terms."""

    mass: sp.Expr
    p_com: sp.Matrix          # 3x1 COM position in the base frame (function of q)
    r_inertia: sp.Matrix      # 3x3 orientation of the inertia frame in the base frame
    inertia_local: sp.Matrix  # 3x3 inertia about the COM in its own frame
    n_preceding: int          # number of actuated joints that move this body


def build_bodies(
    axes: List[sp.Matrix],
    frames: List[sp.Matrix],
    t_rights: List[sp.Matrix],
    q_syms: List[sp.Symbol],
    urdf_chain: List[UrdfJoint],
) -> List[RigidBody]:
    """Pair the artifact's per-joint link frames with the URDF inertial blocks.

    The artifact joint ``i`` and URDF actuated joint ``i`` describe the same
    revolute axis.  The child link of URDF joint ``i`` is rigidly attached to
    that joint, so its body frame is ``frames[i] . Rot(axis_i, q_i)`` (the joint
    frame after the joint rotates).  The URDF inertial ``<origin>`` (centre of
    mass) is expressed in that child-link frame, hence ``T_right[i]`` — which
    only translates on towards the *next* joint — is intentionally not applied.
    """
    actuated = [jt for jt in urdf_chain if jt.actuated]
    if len(actuated) != len(axes):
        raise ValueError(
            f"URDF actuated joints ({len(actuated)}) != artifact DOF ({len(axes)}); "
            "the URDF chain must match the kinematics artifact. Check that all "
            "revolute/continuous/prismatic joints in the URDF correspond to joints "
            "in the artifact chain."
        )

    bodies: List[RigidBody] = []
    for i, (axis, frame, q, jt) in enumerate(zip(axes, frames, q_syms, actuated)):
        inertial = jt.child_inertial
        if inertial is None or inertial.mass <= 0.0:
            continue
        link_t = frame * _axis_angle_rot(axis, q)
        r_link = link_t[:3, :3]
        p_link = link_t[:3, 3]
        com_local = sp.Matrix([_snap(v) for v in inertial.com_xyz])
        r_com = _snap_matrix(_rpy_to_rot(*inertial.com_rpy))
        p_com = p_link + r_link * com_local
        r_inertia = r_link * r_com
        inertia_local = _snap_matrix(
            sp.Matrix(
                [
                    [inertial.ixx, inertial.ixy, inertial.ixz],
                    [inertial.ixy, inertial.iyy, inertial.iyz],
                    [inertial.ixz, inertial.iyz, inertial.izz],
                ]
            )
        )
        bodies.append(
            RigidBody(
                mass=_snap(inertial.mass),
                p_com=p_com,
                r_inertia=r_inertia,
                inertia_local=inertia_local,
                n_preceding=i + 1,
            )
        )
    return bodies


def build_gravity(
    bodies: List[RigidBody], q_syms: List[sp.Symbol], gravity_syms: List[sp.Symbol]
) -> sp.Matrix:
    """Gravity (holding) torque ``tau_j = dU/dq_j`` from total potential energy."""
    g_vec = sp.Matrix(gravity_syms)
    u = sp.Integer(0)
    for b in bodies:
        u -= b.mass * (g_vec.dot(b.p_com))
    n = len(q_syms)
    tau = sp.Matrix([sp.diff(u, q_syms[j]) for j in range(n)])
    return tau


def _body_jacobians(
    body: RigidBody, q_syms: List[sp.Symbol], axes: List[sp.Matrix], frames: List[sp.Matrix]
) -> Tuple[sp.Matrix, sp.Matrix]:
    """Linear (COM) and angular Jacobians (3xN each) of a single rigid body."""
    n = len(q_syms)
    jv = body.p_com.jacobian(sp.Matrix(q_syms))
    jw_cols = []
    for j in range(n):
        if j < body.n_preceding:
            r = frames[j][:3, :3]
            jw_cols.append(r * axes[j])  # revolute axis in the base frame
        else:
            jw_cols.append(sp.zeros(3, 1))
    jw = sp.Matrix.hstack(*jw_cols)
    return jv, jw


def build_mass_matrix(
    bodies: List[RigidBody],
    q_syms: List[sp.Symbol],
    axes: List[sp.Matrix],
    frames: List[sp.Matrix],
) -> sp.Matrix:
    """Symmetric joint-space inertia matrix ``M(q)`` (NxN) via the energy method."""
    n = len(q_syms)
    m = sp.zeros(n, n)
    for b in bodies:
        jv, jw = _body_jacobians(b, q_syms, axes, frames)
        i_world = b.r_inertia * b.inertia_local * b.r_inertia.T
        m += b.mass * (jv.T * jv) + jw.T * i_world * jw
    return m


def build_coriolis(
    mass_matrix: sp.Matrix, q_syms: List[sp.Symbol], qd_syms: List[sp.Symbol]
) -> sp.Matrix:
    """Coriolis/centrifugal torque ``C(q, qd) qd`` from Christoffel symbols."""
    n = len(q_syms)
    tau = sp.zeros(n, 1)
    for i in range(n):
        acc = sp.Integer(0)
        for j in range(n):
            for k in range(n):
                c_ijk = sp.Rational(1, 2) * (
                    sp.diff(mass_matrix[i, j], q_syms[k])
                    + sp.diff(mass_matrix[i, k], q_syms[j])
                    - sp.diff(mass_matrix[j, k], q_syms[i])
                )
                acc += c_ijk * qd_syms[k] * qd_syms[j]
        tau[i] = acc
    return tau


@dataclass
class Dynamics:
    """Symbolic dynamics terms ready for Rust emission."""

    q_syms: List[sp.Symbol]
    qd_syms: List[sp.Symbol]
    qdd_syms: List[sp.Symbol]
    gravity_syms: List[sp.Symbol]
    gravity: sp.Matrix
    mass_matrix: sp.Matrix
    coriolis: sp.Matrix
    inverse_dynamics: sp.Matrix


def build_dynamics(
    axes: List[sp.Matrix],
    frames: List[sp.Matrix],
    t_rights: List[sp.Matrix],
    q_syms: List[sp.Symbol],
    urdf_path: Path,
) -> Dynamics:
    """Derive gravity / mass-matrix / Coriolis / inverse-dynamics symbolically."""
    urdf_chain = order_urdf_chain(parse_urdf_joints(urdf_path))
    bodies = build_bodies(axes, frames, t_rights, q_syms, urdf_chain)
    if not bodies:
        raise ValueError(
            f"URDF {urdf_path} has no positive-mass <inertial> blocks on the chain; "
            "cannot derive dynamics. Ensure each link in the URDF has an <inertial> "
            "block with a positive mass value."
        )

    n = len(q_syms)
    qd_syms = [sp.Symbol(f"qd{i}") for i in range(n)]
    qdd_syms = [sp.Symbol(f"qdd{i}") for i in range(n)]
    gravity_syms = [sp.Symbol(f"g{i}") for i in range(3)]

    grav = build_gravity(bodies, q_syms, gravity_syms)
    mass = build_mass_matrix(bodies, q_syms, axes, frames)
    cor = build_coriolis(mass, q_syms, qd_syms)
    qdd_vec = sp.Matrix(qdd_syms)
    inv_dyn = mass * qdd_vec + cor + grav

    return Dynamics(
        q_syms=q_syms,
        qd_syms=qd_syms,
        qdd_syms=qdd_syms,
        gravity_syms=gravity_syms,
        gravity=grav,
        mass_matrix=mass,
        coriolis=cor,
        inverse_dynamics=inv_dyn,
    )


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


def _emit_vector(
    name: str,
    vector: sp.Matrix,
    printer: NoStdRustPrinter,
    n: int,
) -> List[str]:
    exprs = [rust_literal(vector[i], printer) for i in range(n)]
    return [f"    let {name}: [f64; {n}] = [" + ", ".join(exprs) + "];"]


def _unpack_array(name: str, syms: List[sp.Symbol], printer: NoStdRustPrinter) -> List[str]:
    return [f"    let {printer.doprint(s)} = {name}[{i}];" for i, s in enumerate(syms)]


def _unpack_q(q_syms: List[sp.Symbol], printer: NoStdRustPrinter) -> List[str]:
    return _unpack_array("q", q_syms, printer)


def _emit_cse(replacements, printer: NoStdRustPrinter) -> List[str]:
    return [
        f"    let {printer.doprint(sym)} = {rust_literal(sub, printer)};"
        for sym, sub in replacements
    ]


def _emit_fn_matrix(
    out: List[str],
    signature: str,
    ret_name: str,
    rows: int,
    cols: int,
    matrix: sp.Matrix,
    arg_unpacks: List[str],
    printer: NoStdRustPrinter,
) -> None:
    repls, reduced = sp.cse(list(matrix), optimizations="basic")
    reduced_mat = sp.Matrix(rows, cols, reduced)
    out.append(signature)
    out.extend(arg_unpacks)
    out.extend(_emit_cse(repls, printer))
    out.extend(_emit_matrix(ret_name, reduced_mat, printer, rows, cols))
    out.append(f"    {ret_name}")
    out.append("}")
    out.append("")


def _emit_fn_vector(
    out: List[str],
    signature: str,
    ret_name: str,
    n: int,
    vector: sp.Matrix,
    arg_unpacks: List[str],
    printer: NoStdRustPrinter,
) -> None:
    repls, reduced = sp.cse(list(vector), optimizations="basic")
    reduced_vec = sp.Matrix(reduced)
    out.append(signature)
    out.extend(arg_unpacks)
    out.extend(_emit_cse(repls, printer))
    out.extend(_emit_vector(ret_name, reduced_vec, printer, n))
    out.append(f"    {ret_name}")
    out.append("}")
    out.append("")


def generate_rust(
    arm: str,
    n: int,
    t_ee: sp.Matrix,
    jac: sp.Matrix,
    q_syms: List[sp.Symbol],
    script_name: str,
    dynamics: Optional["Dynamics"] = None,
) -> str:
    printer = NoStdRustPrinter({"contract": False})

    out: List[str] = []
    kind = "dynamics" if dynamics is not None else "kinematics"
    out.append(f"//! Auto-generated symbolic {kind} for the {arm} ({n}-DOF).")
    out.append("//!")
    if dynamics is not None:
        out.append("//! Forward kinematics, the geometric Jacobian and the inverse")
        out.append("//! dynamics (gravity, mass matrix, Coriolis), derived")
    else:
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
    out.append("/// Forward kinematics: base -> end-effector SE(3) pose.")
    out.append("///")
    out.append("/// Returns the 4x4 homogeneous transform as a row-major array.")
    _emit_fn_matrix(
        out,
        f"pub fn fk(q: &[f64; {n}]) -> [[f64; 4]; 4] {{",
        "m",
        4,
        4,
        t_ee,
        _unpack_q(q_syms, printer),
        printer,
    )

    # --- jacobian ---
    out.append("/// Geometric Jacobian `[Jv; Jw]` (6 x DOF) of the end-effector.")
    out.append("///")
    out.append("/// Rows 0-2 are linear velocity, rows 3-5 are angular velocity.")
    _emit_fn_matrix(
        out,
        f"pub fn jacobian(q: &[f64; {n}]) -> [[f64; {n}]; 6] {{",
        "j",
        6,
        n,
        jac,
        _unpack_q(q_syms, printer),
        printer,
    )

    if dynamics is not None:
        _emit_dynamics(out, n, dynamics, printer)

    return "\n".join(out)


def _emit_dynamics(
    out: List[str], n: int, dyn: "Dynamics", printer: NoStdRustPrinter
) -> None:
    q_unpack = _unpack_q(dyn.q_syms, printer)
    gravity_unpack = _unpack_array("gravity", dyn.gravity_syms, printer)
    qd_unpack = _unpack_array("qd", dyn.qd_syms, printer)
    qdd_unpack = _unpack_array("qdd", dyn.qdd_syms, printer)

    out.append("/// Gravity (holding) torque `tau = dU/dq` for each joint.")
    out.append("///")
    out.append("/// Derived from the total potential energy of the link masses.")
    _emit_fn_vector(
        out,
        f"pub fn gravity(q: &[f64; {n}], gravity: &[f64; 3]) -> [f64; {n}] {{",
        "g",
        n,
        dyn.gravity,
        q_unpack + gravity_unpack,
        printer,
    )

    out.append("/// Joint-space inertia (mass) matrix `M(q)` (symmetric, DOF x DOF).")
    _emit_fn_matrix(
        out,
        f"pub fn mass_matrix(q: &[f64; {n}]) -> [[f64; {n}]; {n}] {{",
        "mm",
        n,
        n,
        dyn.mass_matrix,
        q_unpack,
        printer,
    )

    out.append("/// Coriolis & centrifugal torque `C(q, qd) qd` for each joint.")
    _emit_fn_vector(
        out,
        f"pub fn coriolis(q: &[f64; {n}], qd: &[f64; {n}]) -> [f64; {n}] {{",
        "c",
        n,
        dyn.coriolis,
        q_unpack + qd_unpack,
        printer,
    )

    out.append("/// Inverse dynamics torque `tau = M(q) qdd + C(q, qd) qd + g(q)`.")
    out.append("///")
    out.append("/// MIT-mode feed-forward torque for a desired joint acceleration.")
    _emit_fn_vector(
        out,
        f"pub fn inverse_dynamics(q: &[f64; {n}], qd: &[f64; {n}], qdd: &[f64; {n}], gravity: &[f64; 3]) "
        f"-> [f64; {n}] {{",
        "tau",
        n,
        dyn.inverse_dynamics,
        q_unpack + qd_unpack + qdd_unpack + gravity_unpack,
        printer,
    )


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
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help=(
            "Optional URDF carrying <inertial> blocks (link masses/inertias). When "
            "given, also emit gravity/mass_matrix/coriolis/inverse_dynamics. The "
            "URDF's actuated-joint order must match the kinematics artifact."
        ),
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

    dynamics = None
    if args.urdf is not None:
        if not args.urdf.exists():
            print(f"[generate_dynamics] ERROR: URDF not found: {args.urdf}", file=sys.stderr)
            return 1
        try:
            dynamics = build_dynamics(axes, frames, t_rights, q_syms, args.urdf)
        except ValueError as exc:
            print(f"[generate_dynamics] ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"[generate_dynamics] {arm}: dynamics from {args.urdf} (runtime gravity)", file=sys.stderr)

    rust = generate_rust(arm, n, t_ee, jac, q_syms, Path(__file__).name, dynamics)

    if args.output is None:
        sys.stdout.write(rust)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rust)
        print(f"[generate_dynamics] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
