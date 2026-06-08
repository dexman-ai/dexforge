"""Numerical validation of ``codegen/generate_dynamics.py`` dynamics.

Builds a synthetic planar RR (2-DOF, revolute-about-z) artifact plus a matching
URDF carrying ``<inertial>`` blocks, derives the symbolic dynamics and checks it
against an independent closed-form model:

* ``gravity`` equals the finite-difference gradient of the potential energy,
* ``mass_matrix`` equals the energy-method mass matrix built from numeric
  Jacobians (and is symmetric),
* ``inverse_dynamics`` equals ``M qdd + C qd + g``,
* the Coriolis term satisfies the passivity identity
  ``qd^T Mdot qd == 2 qd^T (C qd)`` (Christoffel skew-symmetry).

These run offline only (SymPy is not a CI dependency); the test skips cleanly
when SymPy / NumPy are unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sympy")
import sympy as sp  # noqa: E402  (after importorskip)

_CODEGEN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODEGEN))
import generate_dynamics as G  # noqa: E402

# Planar RR geometry shared between the generator fixtures and the reference.
L1, LC1, M1, IZ1 = 0.5, 0.25, 2.0, 0.03
L2, LC2, M2, IZ2 = 0.4, 0.2, 1.5, 0.007
GRAVITY = (0.0, -9.81, 0.0)


def _write_fixtures(tmp_path: Path):
    artifact = tmp_path / "synth_ik.py"
    artifact.write_text(
        "import numpy as np\n"
        "DOF = 2\n"
        "def _tx(d):\n"
        "    m = np.eye(4); m[0, 3] = d; return m\n"
        "_JOINT_AXES = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])]\n"
        "_JOINT_T_LEFTS = [np.eye(4), np.eye(4)]\n"
        f"_JOINT_T_RIGHTS = [_tx({L1}), _tx({L2})]\n"
    )
    urdf = tmp_path / "synth.urdf"
    urdf.write_text(
        f"""<robot name="rr">
  <link name="base"/>
  <link name="l1">
    <inertial>
      <origin xyz="{LC1} 0 0" rpy="0 0 0"/>
      <mass value="{M1}"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="{IZ1}"/>
    </inertial>
  </link>
  <link name="l2">
    <inertial>
      <origin xyz="{LC2} 0 0" rpy="0 0 0"/>
      <mass value="{M2}"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.006" iyz="0" izz="{IZ2}"/>
    </inertial>
  </link>
  <joint name="j0" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="j1" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="{L1} 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
  </joint>
</robot>
"""
    )
    return artifact, urdf


def _com_positions(q):
    q0, q1 = q
    p1 = np.array([LC1 * np.cos(q0), LC1 * np.sin(q0), 0.0])
    p2 = np.array(
        [
            L1 * np.cos(q0) + LC2 * np.cos(q0 + q1),
            L1 * np.sin(q0) + LC2 * np.sin(q0 + q1),
            0.0,
        ]
    )
    return p1, p2


def _potential(q):
    # U = -sum_i m_i (g . p_i); g = (0, -9.81, 0)  ->  U = 9.81 * sum m_i p_iy
    p1, p2 = _com_positions(q)
    return 9.81 * (M1 * p1[1] + M2 * p2[1])


def _gravity_ref(q):
    eps = 1e-6
    out = np.zeros(2)
    for i in range(2):
        a, b = q.copy(), q.copy()
        a[i] += eps
        b[i] -= eps
        out[i] = (_potential(a) - _potential(b)) / (2 * eps)
    return out


def _mass_ref(q):
    eps = 1e-6
    jv = [np.zeros((3, 2)), np.zeros((3, 2))]
    for body in range(2):
        for j in range(2):
            a, b = q.copy(), q.copy()
            a[j] += eps
            b[j] -= eps
            jv[body][:, j] = (_com_positions(a)[body] - _com_positions(b)[body]) / (2 * eps)
    jw = [
        np.array([[0, 0], [0, 0], [1, 0]], float),
        np.array([[0, 0], [0, 0], [1, 1]], float),
    ]
    iz, mass = [IZ1, IZ2], [M1, M2]
    m = np.zeros((2, 2))
    for body in range(2):
        i_world = np.diag([0.0, 0.0, iz[body]])
        m += mass[body] * jv[body].T @ jv[body] + jw[body].T @ i_world @ jw[body]
    return m


def test_dynamics_matches_planar_rr_reference(tmp_path):
    artifact, urdf = _write_fixtures(tmp_path)
    mod = G.load_artifact(str(artifact))
    t_lefts, axes, t_rights, _ = G.extract_chain(mod)
    _, q_syms, frames = G.build_fk(t_lefts, axes, t_rights)
    dyn = G.build_dynamics(axes, frames, t_rights, q_syms, urdf)

    grav_f = sp.lambdify(q_syms + dyn.gravity_syms, dyn.gravity, "numpy")
    mass_f = sp.lambdify(q_syms, dyn.mass_matrix, "numpy")
    cor_f = sp.lambdify(q_syms + dyn.qd_syms, dyn.coriolis, "numpy")
    invd_f = sp.lambdify(
    q_syms + dyn.qd_syms + dyn.qdd_syms + dyn.gravity_syms,
    dyn.inverse_dynamics,
    "numpy",
    )

    rng = np.random.default_rng(0)
    for _ in range(64):
        q = rng.uniform(-np.pi, np.pi, 2)
        qd = rng.uniform(-2.0, 2.0, 2)
        qdd = rng.uniform(-2.0, 2.0, 2)

        g_sym = np.asarray(grav_f(*q, *GRAVITY), float).reshape(-1)
        assert np.allclose(g_sym, _gravity_ref(q), atol=1e-6)

        m_sym = np.asarray(mass_f(*q), float)
        assert np.allclose(m_sym, m_sym.T, atol=1e-12)  # symmetric
        assert np.allclose(m_sym, _mass_ref(q), atol=1e-6)

        c_sym = np.asarray(cor_f(*q, *qd), float).reshape(-1)
        id_sym = np.asarray(invd_f(*q, *qd, *qdd, *GRAVITY), float).reshape(-1)
        assert np.allclose(id_sym, m_sym @ qdd + c_sym + g_sym, atol=1e-9)

        # Passivity: qd^T Mdot qd == 2 qd^T (C qd) for Christoffel-form Coriolis.
        eps = 1e-6
        m_dot = np.zeros((2, 2))
        for k in range(2):
            a, b = q.copy(), q.copy()
            a[k] += eps
            b[k] -= eps
            m_dot += (
                (np.asarray(mass_f(*a), float) - np.asarray(mass_f(*b), float))
                / (2 * eps)
                * qd[k]
            )
        assert abs(qd @ (m_dot @ qd) - 2 * qd @ c_sym) < 1e-6
