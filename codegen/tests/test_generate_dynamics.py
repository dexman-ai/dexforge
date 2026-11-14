"""Numerical validation of symbolic dynamics in ``dexforge_build``.

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
        'BASE_LINK = "base"\n'
        'EE_LINK = "l2"\n'
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
  <link name="tool"/>
  <joint name="j0" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="j1" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="{L1} 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="tool_mount" type="fixed">
    <parent link="l2"/><child link="tool"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""
    )
    return artifact, urdf


def test_order_urdf_chain_accepts_side_branches_when_endpoints_are_known():
    joints = [
        G.UrdfJoint(name="j0", jtype="revolute", parent="base", child="l1", axis=(0.0, 0.0, 1.0)),
        G.UrdfJoint(name="j1", jtype="revolute", parent="l1", child="l2", axis=(0.0, 0.0, 1.0)),
        G.UrdfJoint(name="tool_mount", jtype="fixed", parent="l2", child="tool", axis=(0.0, 0.0, 1.0)),
        G.UrdfJoint(name="camera_mount", jtype="fixed", parent="base", child="camera", axis=(0.0, 0.0, 1.0)),
    ]

    chain = G.order_urdf_chain(joints, base_link=None, ee_link="tool")

    assert [joint.name for joint in chain] == ["j0", "j1", "tool_mount"]


def test_order_urdf_chain_handles_helper_branch_off_base_link():
    joints = [
        G.UrdfJoint(name="shoulder_pan_joint", jtype="revolute", parent="base_link", child="shoulder_link", axis=(0.0, 0.0, 1.0)),
        G.UrdfJoint(name="base_link-base_fixed_joint", jtype="fixed", parent="base_link", child="base", axis=(1.0, 0.0, 0.0)),
        G.UrdfJoint(name="shoulder_lift_joint", jtype="revolute", parent="shoulder_link", child="upper_arm_link", axis=(0.0, 1.0, 0.0)),
        G.UrdfJoint(name="tool_mount", jtype="fixed", parent="upper_arm_link", child="tool0", axis=(1.0, 0.0, 0.0)),
    ]

    chain = G.order_urdf_chain(joints, base_link="base_link", ee_link="tool0")

    assert [joint.name for joint in chain] == [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "tool_mount",
    ]


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
    dyn = G.build_dynamics(
        axes,
        frames,
        t_rights,
        q_syms,
        urdf,
        *G.extract_base_ee_links(mod),
    )

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


def test_pinocchio_chain_xml_strips_world_transform_and_gripper_joints(tmp_path):
    import generate_dynamics_test_data as GD
    import xml.etree.ElementTree as ET

    urdf = tmp_path / "with_gripper.urdf"
    urdf.write_text(
        f"""<robot name="rr_gripper">
  <link name="world"/>
  <joint name="world_to_base" type="fixed">
    <parent link="world"/><child link="base"/>
    <origin xyz="0 0 1.0" rpy="1.5707963267948966 0 0"/>
  </joint>
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
  <link name="finger">
    <inertial>
      <origin xyz="0.02 0 0" rpy="0 0 0"/>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
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
  <joint name="finger_joint" type="prismatic">
    <parent link="l2"/><child link="finger"/>
    <origin xyz="{L2} 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )

    xml_str = GD._build_chain_urdf_xml(urdf, base_link="base", ee_link="l2")
    root = ET.fromstring(xml_str)
    joint_names = [j.get("name") for j in root.findall("joint")]
    link_names = [l.get("name") for l in root.findall("link")]
    assert joint_names == ["j0", "j1"]
    assert link_names == ["base", "l1", "l2"]


def test_emit_dynamics_parity_rs_checks_finiteness_and_custom_gravity():
    import generate_dynamics_test_data as GD

    rs = GD.emit_dynamics_parity_rs("demo", 6)
    assert 'assert!(d.is_finite(), "non-finite dynamics comparison");' in rs
    assert 'dynamics_test_data::GRAVITY.iter().all(|g| g.is_finite())' in rs
    assert "[0.0, 0.0, -9.81]" not in rs


def test_non_unit_urdf_axis_is_normalized_and_zero_axis_rejected(tmp_path):
    artifact, urdf = _write_fixtures(tmp_path)
    scaled_urdf = tmp_path / "scaled_axis.urdf"
    scaled_urdf.write_text(
        urdf.read_text(encoding="utf-8").replace('axis xyz="0 0 1"', 'axis xyz="0 0 2"'),
        encoding="utf-8",
    )
    mod = G.load_artifact(str(artifact))
    t_lefts, axes, t_rights, _ = G.extract_chain(mod)
    _, q_syms, frames = G.build_fk(t_lefts, axes, t_rights)
    dyn_unit = G.build_dynamics(axes, frames, t_rights, q_syms, urdf, "base", "l2")
    dyn_scaled = G.build_dynamics(axes, frames, t_rights, q_syms, scaled_urdf, "base", "l2")
    assert sp.simplify(dyn_unit.gravity - dyn_scaled.gravity) == sp.zeros(2, 1)
    assert sp.simplify(dyn_unit.mass_matrix - dyn_scaled.mass_matrix) == sp.zeros(2, 2)

    zero_urdf = tmp_path / "zero_axis.urdf"
    zero_urdf.write_text(
        urdf.read_text(encoding="utf-8").replace('axis xyz="0 0 1"', 'axis xyz="0 0 0"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="zero"):
        G.build_dynamics(axes, frames, t_rights, q_syms, zero_urdf, "base", "l2")


def test_rotated_base_frame_aligns_horizontal_gravity_with_pinocchio(tmp_path):
    pytest.importorskip("pinocchio")
    import generate_dynamics_test_data as GD

    _, urdf = _write_fixtures(tmp_path)
    # Artifact uses a base frame rotated 180 deg about Z relative to the URDF:
    # at q=0, link 1 extends along -X in the artifact frame and +X in the URDF frame.
    artifact = tmp_path / "rot_rr_ik.py"
    artifact.write_text(
        "import numpy as np\n"
        'BASE_LINK = "base"\n'
        'EE_LINK = "l2"\n'
        "DOF = 2\n"
        "def _tx(d):\n"
        "    m = np.eye(4); m[0, 3] = d; return m\n"
        "_JOINT_AXES = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])]\n"
        f"_JOINT_T_LEFTS = [np.eye(4), _tx({-L1})]\n"
        f"_JOINT_T_RIGHTS = [np.eye(4), _tx({-L2})]\n",
        encoding="utf-8",
    )
    mod = G.load_artifact(str(artifact))
    t_lefts, axes, t_rights, _ = G.extract_chain(mod)
    _, q_syms, frames = G.build_fk(t_lefts, axes, t_rights)
    dyn = G.build_dynamics(axes, frames, t_rights, q_syms, urdf, "base", "l2")
    grav_f = sp.lambdify(q_syms + dyn.gravity_syms, dyn.gravity, "numpy")

    horiz_g = (3.5, -4.2, -9.81)
    dof, _, cases = GD.build_dataset(
        urdf,
        samples=8,
        seed=0,
        qd_limit=1.0,
        qdd_limit=2.0,
        gravity=horiz_g,
        base_link="base",
        ee_link="l2",
        arm="rot_rr",
        output_dir=tmp_path,
    )
    assert dof == 2
    for q, _qd, _qdd, g_pin, _m, _c, _tau in cases:
        g_sym = np.asarray(grav_f(*q, *horiz_g), dtype=float).reshape(-1)
        assert np.allclose(g_sym, g_pin, atol=1e-6)


