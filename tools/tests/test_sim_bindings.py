"""Bindings the simulator relies on: dynamics, manifest, buffer layout."""

import numpy as np
import pytest

import dexforge_ur as ur

ARMS = ["ur5", "ur3e"]


@pytest.fixture(params=ARMS)
def arm_name(request):
    return request.param


@pytest.fixture
def arm(arm_name):
    return getattr(ur, arm_name)


RNG = np.random.default_rng(0)


def _q():
    return RNG.uniform(-1.5, 1.5, 6).tolist()


def test_layout_is_contiguous_and_matches_manifest(arm, arm_name):
    m = arm.manifest()
    for layout, total in ((ur.state_layout(), m["state_len"]), (ur.command_layout(), m["command_len"])):
        n = layout.pop("len")
        assert n == total  # no aux tail
        spans = sorted(layout.values())
        pos = 0
        for off, length in spans:
            assert off == pos, "gap or overlap in buffer layout"
            pos += length
        assert pos == n


def test_known_offsets():
    s, c = ur.state_layout(), ur.command_layout()
    assert s["q"] == (0, 8) and s["qd"] == (8, 8)  # MAX_DOF = 8
    assert c["control_mode"] == (0, 1) and c["target_q"] == (1, 8)  # explicit mode slot first
    assert s["fault_code"][1] == 1


def test_manifest_matches_yaml(arm, arm_name):
    m = arm.manifest()
    assert m["name"] == arm_name and m["dof"] == 6
    assert m["control_freq_hz"] == 500
    assert m["command_dispatch_cycles"] == 2 and m["state_staleness_cycles"] == 1
    assert len(m["effort_limits"]) == 6 and m["effort_limits"][0] == {"ur5": 150.0, "ur3e": 54.0}[arm_name]
    assert m["max_dof"] == 8


def test_mass_matrix_symmetric_positive_definite(arm, arm_name):
    M = np.array(arm.mass_matrix(_q()))
    assert np.allclose(M, M.T, atol=1e-9)
    assert np.linalg.eigvalsh(M).min() > 0


def test_inverse_dynamics_consistency(arm, arm_name):
    q, qd = _q(), RNG.uniform(-1, 1, 6).tolist()
    zero = [0.0] * 6
    g = np.array(arm.gravity(q))
    assert np.allclose(arm.inverse_dynamics(q, zero, zero), g, atol=1e-8)
    C = np.array(arm.coriolis(q, qd))
    assert np.allclose(np.array(arm.inverse_dynamics(q, qd, zero)) - g, C, atol=1e-8)
    qdd = RNG.uniform(-1, 1, 6)
    M = np.array(arm.mass_matrix(q))
    full = np.array(arm.inverse_dynamics(q, qd, qdd.tolist()))
    assert np.allclose(full, g + C + M @ qdd, atol=1e-7)


def test_jacobian_matches_finite_difference_of_fk(arm, arm_name):
    q = np.array(_q())
    J = np.array(arm.jacobian(q.tolist()))
    p0 = np.array(arm.fk(q.tolist()))[:3, 3]
    eps = 1e-6
    for i in range(6):
        dq = q.copy(); dq[i] += eps
        num = (np.array(arm.fk(dq.tolist()))[:3, 3] - p0) / eps
        assert np.allclose(num, J[:3, i], atol=1e-4)  # geometric Jacobian is [Jv; Jw]


def test_bad_lengths_raise(arm, arm_name):
    with pytest.raises(ValueError):
        arm.gravity([0.0] * 5)
    with pytest.raises(ValueError):
        arm.coriolis([0.0] * 6, [0.0] * 5)


def test_fk_pose_matches_fk_matrix(arm, arm_name):
    from scipy.spatial.transform import Rotation as R

    q = _q()
    T = np.array(arm.fk(q))
    p = np.array(arm.fk_pose(q))
    assert np.allclose(p[:3], T[:3, 3], atol=1e-12)
    assert np.allclose(R.from_quat(p[3:]).as_matrix(), T[:3, :3], atol=1e-9)  # x, y, z, w


def test_ik_one_returns_seed_branch_and_enforces_the_bound(arm, arm_name):
    q = np.array(_q())
    pose = arm.fk_pose(q.tolist())
    sol = arm.ik_one(pose, q.tolist(), "near")
    assert sol is not None and np.allclose(sol, q, atol=1e-6)
    sol = arm.ik_one(pose, q.tolist(), "fast")
    assert sol is not None and np.allclose(sol, q, atol=1e-6)
    far = (q + 2.5).tolist()  # a seed far from every solution of this pose
    assert arm.ik_one(pose, far, "near", 0.01) is None
    assert arm.ik_one(pose, far, "fast", 0.01) is None
    assert arm.ik_default_max_joint_step() == pytest.approx(np.pi / 2)
    with pytest.raises(ValueError):
        arm.ik_one(pose[:6], q.tolist(), "near")
    with pytest.raises(ValueError):
        arm.ik_one(pose, q.tolist(), "bogus")


def test_control_modes_and_manifest_support(arm, arm_name):
    modes = arm.control_modes()
    assert np.isnan(modes["idle"])
    assert modes["joint_pos"] == 1.0 and modes["cartesian_pose"] == 4.0
    assert arm.supports_mode(modes["joint_pos"]) and arm.supports_mode(modes["joint_vel"])
    assert arm.supports_mode(modes["cartesian_pose"]) and arm.supports_mode(modes["idle"])
    assert not arm.supports_mode(modes["joint_torque"])  # manifest offers three modes
    assert arm.supports_mode(42.0) is None  # names no mode
