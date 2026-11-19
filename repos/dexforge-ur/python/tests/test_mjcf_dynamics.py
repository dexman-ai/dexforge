"""The generated dynamics must agree with MuJoCo's, on a model built from the same URDF.

This is the check that tells us whether the MJCF inertials or the generated
dynamics are wrong: at rest MuJoCo's bias force is gravity, and its mass matrix
is M(q). The model is built with armature 0 because the generated dynamics have
no rotor inertia.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
import urdf_to_mjcf as G  # noqa: E402

import dexforge_ur  # noqa: E402

ur5 = dexforge_ur.ur5
RNG = np.random.default_rng(7)


@pytest.fixture(scope="module")
def model():
    urdf, ee = G.ARMS["ur5"]
    m = mujoco.MjModel.from_xml_string(G.build_mjcf(urdf, ee, armature=0.0))
    return m, mujoco.MjData(m)


def _set(model, q, qd=None):
    m, d = model
    d.qpos[:] = q
    d.qvel[:] = 0.0 if qd is None else qd
    mujoco.mj_forward(m, d)


def test_model_shape(model):
    m, _ = model
    assert m.nq == m.nv == m.nu == 6
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(6)]
    assert names == ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    assert np.allclose(m.opt.gravity, [0, 0, -ur5.standard_gravity()])  # one g for sim and model
    # direct torque: ctrl range is the URDF effort limit
    assert np.allclose(m.actuator_ctrlrange[:, 1], [150, 150, 150, 28, 28, 28])


def test_gravity_matches_mujoco_bias_at_rest(model):
    m, d = model
    worst = 0.0
    for _ in range(50):
        q = RNG.uniform(-3, 3, 6)
        _set(model, q)
        worst = max(worst, np.abs(np.array(ur5.gravity(q.tolist())) - d.qfrc_bias).max())
    assert worst < 1e-6, f"gravity mismatch vs MuJoCo qfrc_bias: {worst:.3e} N*m"


def test_mass_matrix_matches_mujoco(model):
    m, d = model
    worst = 0.0
    for _ in range(50):
        q = RNG.uniform(-3, 3, 6)
        _set(model, q)
        M = np.zeros((6, 6))
        mujoco.mj_fullM(m, d, M)
        worst = max(worst, np.abs(np.array(ur5.mass_matrix(q.tolist())) - M).max())
    assert worst < 1e-6, f"mass matrix mismatch vs MuJoCo: {worst:.3e}"


def test_coriolis_matches_mujoco_bias(model):
    m, d = model
    worst = 0.0
    for _ in range(50):
        q, qd = RNG.uniform(-3, 3, 6), RNG.uniform(-1, 1, 6)
        _set(model, q, qd)
        expected = np.array(ur5.gravity(q.tolist())) + np.array(ur5.coriolis(q.tolist(), qd.tolist()))
        worst = max(worst, np.abs(expected - d.qfrc_bias).max())
    assert worst < 1e-6, f"gravity+coriolis mismatch vs MuJoCo qfrc_bias: {worst:.3e}"


def test_tcp_site_matches_generated_fk(model):
    m, d = model
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    for _ in range(20):
        q = RNG.uniform(-3, 3, 6)
        _set(model, q)
        T = np.array(ur5.fk(q.tolist()))
        assert np.allclose(d.site_xpos[sid], T[:3, 3], atol=1e-9)
        assert np.allclose(d.site_xmat[sid].reshape(3, 3), T[:3, :3], atol=1e-9)


def test_gravity_compensation_holds_the_arm(model):
    """Feeding the generated gravity torque back as the motor command keeps the arm still."""
    m, d = model
    q0 = np.array([0.3, -1.0, 1.2, -0.5, 0.8, 0.2])
    _set(model, q0)
    for _ in range(500):  # 0.5 s
        d.ctrl[:] = np.clip(ur5.gravity(d.qpos.tolist()), m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        mujoco.mj_step(m, d)
    assert np.abs(d.qpos - q0).max() < 1e-3, f"arm drifted {np.abs(d.qpos - q0).max():.2e} rad"
