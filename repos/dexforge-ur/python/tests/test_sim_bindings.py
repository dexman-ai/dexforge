"""Bindings the simulator relies on: dynamics, manifest, buffer layout."""

import numpy as np
import pytest

import dexforge_ur as ur

ur5 = ur.ur5
RNG = np.random.default_rng(0)


def _q():
    return RNG.uniform(-1.5, 1.5, 6).tolist()


def test_layout_is_contiguous_and_matches_manifest():
    m = ur5.manifest()
    for layout, total in ((ur.state_layout(), m["state_len"]), (ur.command_layout(), m["command_len"])):
        n = layout.pop("len")
        assert n == total  # UR5 has no aux tail
        spans = sorted(layout.values())
        pos = 0
        for off, length in spans:
            assert off == pos, "gap or overlap in buffer layout"
            pos += length
        assert pos == n


def test_known_offsets():
    s, c = ur.state_layout(), ur.command_layout()
    assert s["q"] == (0, 8) and s["qd"] == (8, 8)  # MAX_DOF = 8
    assert c["target_q"] == (0, 8)
    assert s["fault_code"][1] == 1


def test_manifest_matches_yaml():
    m = ur5.manifest()
    assert m["name"] == "ur5" and m["dof"] == 6
    assert m["control_freq_hz"] == 500
    assert m["command_dispatch_cycles"] == 2 and m["state_staleness_cycles"] == 1
    assert len(m["effort_limits"]) == 6 and m["effort_limits"][0] == 150.0
    assert m["max_dof"] == 8


def test_mass_matrix_symmetric_positive_definite():
    M = np.array(ur5.mass_matrix(_q()))
    assert np.allclose(M, M.T, atol=1e-9)
    assert np.linalg.eigvalsh(M).min() > 0


def test_inverse_dynamics_consistency():
    q, qd = _q(), RNG.uniform(-1, 1, 6).tolist()
    zero = [0.0] * 6
    g = np.array(ur5.gravity(q))
    assert np.allclose(ur5.inverse_dynamics(q, zero, zero), g, atol=1e-8)
    C = np.array(ur5.coriolis(q, qd))
    assert np.allclose(np.array(ur5.inverse_dynamics(q, qd, zero)) - g, C, atol=1e-8)
    qdd = RNG.uniform(-1, 1, 6)
    M = np.array(ur5.mass_matrix(q))
    full = np.array(ur5.inverse_dynamics(q, qd, qdd.tolist()))
    assert np.allclose(full, g + C + M @ qdd, atol=1e-7)


def test_jacobian_matches_finite_difference_of_fk():
    q = np.array(_q())
    J = np.array(ur5.jacobian(q.tolist()))
    p0 = np.array(ur5.fk(q.tolist()))[:3, 3]
    eps = 1e-6
    for i in range(6):
        dq = q.copy(); dq[i] += eps
        num = (np.array(ur5.fk(dq.tolist()))[:3, 3] - p0) / eps
        assert np.allclose(num, J[:3, i], atol=1e-4)  # geometric Jacobian is [Jv; Jw]


def test_bad_lengths_raise():
    with pytest.raises(ValueError):
        ur5.gravity([0.0] * 5)
    with pytest.raises(ValueError):
        ur5.coriolis([0.0] * 6, [0.0] * 5)


def test_fk_pose_matches_fk_matrix():
    from scipy.spatial.transform import Rotation as R

    q = _q()
    T = np.array(ur5.fk(q))
    p = np.array(ur5.fk_pose(q))
    assert np.allclose(p[:3], T[:3, 3], atol=1e-12)
    assert np.allclose(R.from_quat(p[3:]).as_matrix(), T[:3, :3], atol=1e-9)  # x, y, z, w


def test_ik_nearest_returns_seed_branch_and_enforces_the_bound():
    q = np.array(_q())
    pose = ur5.fk_pose(q.tolist())
    sol = ur5.ik_nearest(pose, q.tolist())
    assert sol is not None and np.allclose(sol, q, atol=1e-6)
    far = (q + 2.5).tolist()  # a seed far from every solution of this pose
    assert ur5.ik_nearest(pose, far, 0.01) is None
    assert ur5.ik_default_max_joint_step() == pytest.approx(np.pi / 2)
    with pytest.raises(ValueError):
        ur5.ik_nearest(pose[:6], q.tolist())
