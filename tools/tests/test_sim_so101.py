"""The SO-ARM101 in the simulator: bindings, MuJoCo dynamics cross-check, and SimDriver behaviour.

The UR tests (``test_sim_*.py``) are written for six-joint arms at 500 Hz. This arm has five
joints and a 100 Hz manifest, so it gets its own file. Three layers, bottom up:

1. the ``dexforge_lerobot.so101`` bindings (generated FK / IK / dynamics, manifest, buffer layout);
2. the generated dynamics against MuJoCo's own, on a model built from the same URDF
   (independent of the symbolic compiler: a mismatch means the MJCF inertials or the generated
   dynamics are wrong);
3. ``SimDriver("so101")`` through the contract buffers, including the five-joint specifics: a pose
   outside the arm's 5-D reachable set runs on the nearest reachable configuration when it misses
   by less than ``ik_tolerance``, and is rejected otherwise; tracking settles everywhere in the
   workspace (diagonal-mass gains did not).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

mujoco = pytest.importorskip("mujoco")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import sim_driver as S  # noqa: E402
import urdf_to_mjcf as G  # noqa: E402

import dexforge_lerobot as lr  # noqa: E402

ARM = "so101"
DOF = 5
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# URDF joint limits (rad); the limits of the five real joints
LIMITS = np.array([(-1.91986, 1.91986), (-1.74533, 1.74533), (-1.69, 1.69), (-1.65806, 1.65806), (-2.74385, 2.84121)])

Q0 = np.array([0.0, -0.3, 0.5, -0.2, 0.0])  # SimDriver's start pose
GOAL = Q0 + np.array([0.4, 0.3, -0.3, 0.2, -0.2])
RNG = np.random.default_rng(11)


def _q(span=1.3):
    return RNG.uniform(-span, span, DOF).tolist()


@pytest.fixture(scope="module")
def arm():
    return getattr(lr, ARM)


# ----------------------------------------------------------------------------- 1. bindings


def test_layout_is_contiguous_and_matches_manifest(arm):
    m = arm.manifest()
    for layout, total in ((lr.state_layout(), m["state_len"]), (lr.command_layout(), m["command_len"])):
        n = layout.pop("len")
        assert n == total
        pos = 0
        for off, length in sorted(layout.values()):
            assert off == pos, "gap or overlap in the buffer layout"
            pos += length
        assert pos == n


def test_manifest_matches_yaml(arm):
    m = arm.manifest()
    assert m["name"] == ARM and m["dof"] == DOF and m["max_dof"] == 8
    assert m["control_freq_hz"] == 100
    assert m["command_dispatch_cycles"] == 1 and m["state_staleness_cycles"] == 1
    assert m["effort_limits"][:DOF] == [10.0] * DOF
    assert np.allclose(m["joint_limits_min"][:DOF], LIMITS[:, 0]) and np.allclose(m["joint_limits_max"][:DOF], LIMITS[:, 1])


def test_dynamics_are_five_dimensional(arm):
    q, qd = _q(), _q(1.0)
    assert np.array(arm.jacobian(q)).shape == (6, DOF)
    assert np.array(arm.mass_matrix(q)).shape == (DOF, DOF)
    assert len(arm.gravity(q)) == DOF and len(arm.coriolis(q, qd)) == DOF
    with pytest.raises(ValueError):
        arm.gravity([0.0] * 6)  # a six-joint vector is wrong for this arm
    with pytest.raises(ValueError):
        arm.coriolis([0.0] * DOF, [0.0] * 6)


def test_mass_matrix_symmetric_positive_definite(arm):
    M = np.array(arm.mass_matrix(_q()))
    assert np.allclose(M, M.T, atol=1e-9)
    assert np.linalg.eigvalsh(M).min() > 0


def test_inverse_dynamics_consistency(arm):
    q, qd = _q(), _q(1.0)
    zero = [0.0] * DOF
    g = np.array(arm.gravity(q))
    assert np.allclose(arm.inverse_dynamics(q, zero, zero), g, atol=1e-8)
    C = np.array(arm.coriolis(q, qd))
    assert np.allclose(np.array(arm.inverse_dynamics(q, qd, zero)) - g, C, atol=1e-8)
    qdd = RNG.uniform(-1, 1, DOF)
    M = np.array(arm.mass_matrix(q))
    assert np.allclose(arm.inverse_dynamics(q, qd, qdd.tolist()), g + C + M @ qdd, atol=1e-7)


def test_jacobian_matches_finite_difference_of_fk(arm):
    q = np.array(_q())
    J = np.array(arm.jacobian(q.tolist()))
    p0 = np.array(arm.fk(q.tolist()))[:3, 3]
    for i in range(DOF):
        dq = q.copy()
        dq[i] += 1e-6
        assert np.allclose((np.array(arm.fk(dq.tolist()))[:3, 3] - p0) / 1e-6, J[:3, i], atol=1e-4)


def test_fk_pose_matches_fk_matrix(arm):
    q = _q()
    T, p = np.array(arm.fk(q)), np.array(arm.fk_pose(q))
    assert np.allclose(p[:3], T[:3, 3], atol=1e-12)
    assert np.allclose(R.from_quat(p[3:]).as_matrix(), T[:3, :3], atol=1e-9)


def test_ik_nearest_recovers_the_seed_branch(arm):
    for _ in range(20):
        q = np.array(_q(1.2))
        sol = arm.ik_nearest(arm.fk_pose(q.tolist()), q.tolist())
        assert sol is not None and np.allclose(sol, q, atol=1e-6)


def test_ik_has_no_solution_for_poses_outside_the_reachable_set(arm):
    """A five-joint arm cannot orient the tool arbitrarily: rotate the tool 0.3 rad about the base y axis."""
    q = np.array(_q(1.0))
    T = np.array(arm.fk(q.tolist()))
    T[:3, :3] = R.from_euler("y", 0.3).as_matrix() @ T[:3, :3]
    assert arm.solve(T, respect_limits=False) == []
    pose = np.r_[T[:3, 3], R.from_matrix(T[:3, :3]).as_quat()]
    assert arm.ik_nearest(pose.tolist(), q.tolist()) is None


def test_ik_project_is_exact_for_reachable_poses_and_reports_the_miss_otherwise(arm):
    q = np.array(_q(1.0))
    pose = arm.fk_pose(q.tolist())
    sol, dp, da = arm.ik_project(pose, q.tolist())
    assert dp < 1e-6 and da < 1e-5 and np.allclose(sol, q, atol=1e-5)

    noisy = np.array(pose)
    noisy[:3] += [0.002, -0.001, 0.0015]  # a couple of millimetres off the reachable set
    _, dp, da = arm.ik_project(noisy.tolist(), q.tolist())
    assert dp < 0.0035 and da < 0.05  # never farther than the true configuration was

    T = np.array(arm.fk(q.tolist()))
    T[:3, :3] = R.from_euler("y", 0.3).as_matrix() @ T[:3, :3]  # an orientation five joints cannot absorb
    far = np.r_[T[:3, 3], R.from_matrix(T[:3, :3]).as_quat()]
    sol, dp, da = arm.ik_project(far.tolist(), q.tolist())
    assert da > 1e-3 and da <= 0.3 + 1e-9  # a genuine miss, reported honestly and no worse than not moving
    assert np.allclose(np.array(arm.fk(sol))[:3, 3], np.array(arm.fk(sol))[:3, 3])  # a real configuration

    assert (arm.ik_project(pose, q.tolist(), True, 0.1)[1] < 1e-6)
    for bad in ((pose[:6], q.tolist()), (pose, q.tolist()[:4]), ([float("nan")] * 7, q.tolist())):
        with pytest.raises(ValueError):
            arm.ik_project(*bad)
    with pytest.raises(ValueError):
        arm.ik_project(pose, q.tolist(), True, 0.0)


def test_control_modes_and_manifest_support(arm):
    modes = arm.control_modes()
    assert arm.supports_mode(modes["joint_pos"]) and arm.supports_mode(modes["joint_vel"])
    assert arm.supports_mode(modes["cartesian_pose"]) and arm.supports_mode(modes["idle"])
    assert not arm.supports_mode(modes["joint_torque"])


# ----------------------------------------------------------------------------- 2. MuJoCo cross-check


@pytest.fixture(scope="module")
def model():
    urdf, ee = G.ARMS[ARM]
    m = mujoco.MjModel.from_xml_string(G.build_mjcf(urdf, ee, armature=0.0))
    return m, mujoco.MjData(m)


def _set(model, q, qd=None):
    m, d = model
    d.qpos[:] = q
    d.qvel[:] = 0.0 if qd is None else qd
    mujoco.mj_forward(m, d)


def test_mjcf_shape(model, arm):
    m, _ = model
    assert m.nq == m.nv == m.nu == DOF
    assert [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(DOF)] == JOINT_NAMES
    assert np.allclose(m.opt.gravity, [0, 0, -arm.standard_gravity()])
    assert np.allclose(m.actuator_ctrlrange[:, 1], 10.0)  # URDF effort limit
    assert m.nmesh == 6  # one merged mesh per link


def test_gravity_matches_mujoco_bias_at_rest(model, arm):
    _, d = model
    worst = 0.0
    for _ in range(50):
        q = np.array(_q(1.5))
        _set(model, q)
        worst = max(worst, np.abs(np.array(arm.gravity(q.tolist())) - d.qfrc_bias).max())
    assert worst < 1e-6, f"gravity mismatch vs MuJoCo qfrc_bias: {worst:.3e} N*m"


def test_mass_matrix_matches_mujoco(model, arm):
    m, d = model
    worst = 0.0
    for _ in range(50):
        q = np.array(_q(1.5))
        _set(model, q)
        M = np.zeros((DOF, DOF))
        mujoco.mj_fullM(m, d, M)
        worst = max(worst, np.abs(np.array(arm.mass_matrix(q.tolist())) - M).max())
    assert worst < 1e-6, f"mass matrix mismatch vs MuJoCo: {worst:.3e}"


def test_coriolis_matches_mujoco_bias(model, arm):
    _, d = model
    worst = 0.0
    for _ in range(50):
        q, qd = np.array(_q(1.5)), np.array(_q(1.0))
        _set(model, q, qd)
        expected = np.array(arm.gravity(q.tolist())) + np.array(arm.coriolis(q.tolist(), qd.tolist()))
        worst = max(worst, np.abs(expected - d.qfrc_bias).max())
    assert worst < 1e-6, f"gravity+coriolis mismatch vs MuJoCo qfrc_bias: {worst:.3e}"


def test_tcp_site_matches_generated_fk(model, arm):
    m, d = model
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    for _ in range(20):
        q = np.array(_q(1.5))
        _set(model, q)
        T = np.array(arm.fk(q.tolist()))
        assert np.allclose(d.site_xpos[sid], T[:3, 3], atol=1e-9)
        assert np.allclose(d.site_xmat[sid].reshape(3, 3), T[:3, :3], atol=1e-9)


def test_gravity_compensation_holds_the_arm(model, arm):
    m, d = model
    _set(model, Q0)
    for _ in range(500):
        d.ctrl[:] = np.clip(arm.gravity(d.qpos.tolist()), m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        mujoco.mj_step(m, d)
    assert np.abs(d.qpos - Q0).max() < 1e-3


# ----------------------------------------------------------------------------- 3. SimDriver


@pytest.fixture
def drv():
    d = S.SimDriver(ARM)
    d.connect()
    yield d
    d.disconnect()


def state(d):
    buf = np.full(d.manifest()["state_len"], np.nan)
    d.read_state(buf)
    lay = lr.state_layout()
    return buf, lambda name: buf[lay[name][0] : lay[name][0] + lay[name][1]]


def _cmd(d, mode, field, values):
    c = d.idle_command()
    lay = lr.command_layout()
    c[lay["control_mode"][0]] = getattr(lr, ARM).control_modes()[mode]
    off = lay[field][0]
    c[off : off + len(values)] = values
    return c


def cmd_q(d, q):
    return _cmd(d, "joint_pos", "target_q", q)


def cmd_qd(d, qd):
    return _cmd(d, "joint_vel", "target_qd", qd)


def cmd_pose(d, pose):
    return _cmd(d, "cartesian_pose", "target_pose", pose)


def test_contract_lengths_and_initial_state(drv):
    m = drv.manifest()
    assert m["state_len"] == lr.state_layout()["len"] and m["command_len"] == lr.command_layout()["len"]
    buf, f = state(drv)
    assert len(buf) == m["state_len"]
    assert np.allclose(f("q")[:DOF], Q0) and f("fault_code")[0] == 0.0
    assert np.isnan(f("wrench")).all()  # unsupported fields stay NaN
    with pytest.raises(ValueError):
        drv.write_command(np.zeros(3))


def test_gravity_compensation_holds_at_rest(drv):
    drv.step(200)  # 2 s with no command
    _, f = state(drv)
    assert np.abs(f("q")[:DOF] - Q0).max() < 1e-4 and np.abs(f("qd")[:DOF]).max() < 1e-3


def test_joint_position_round_trip(drv):
    drv.write_command(cmd_q(drv, GOAL))
    drv.step(400)
    _, f = state(drv)
    assert np.abs(f("q")[:DOF] - GOAL).max() < 1e-3
    assert (np.abs(f("torques")[:DOF]) <= np.array(drv.manifest()["effort_limits"][:DOF]) + 1e-9).all()


def test_joint_velocity_mode(drv):
    drv.write_command(cmd_qd(drv, [0.2, 0, 0, 0, 0]))
    drv.step(100)  # 1 s
    _, f = state(drv)
    assert abs((f("q")[0] - Q0[0]) - 0.2) < 0.03


def test_cartesian_pose_uses_the_five_joint_ik(drv, arm):
    pose = np.array(arm.fk_pose(GOAL.tolist()))
    drv.write_command(cmd_pose(drv, pose))
    drv.step(400)
    _, f = state(drv)
    got = f("tcp_pose")
    assert np.abs(got[:3] - pose[:3]).max() < 1e-3
    assert abs(abs(np.dot(got[3:], pose[3:])) - 1.0) < 1e-4  # quaternion, up to sign
    assert f("fault_code")[0] == 0.0


def test_unreachable_position_faults_and_holds_until_cleared(drv):
    drv.write_command(cmd_pose(drv, [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    drv.step(50)
    _, f = state(drv)
    assert f("fault_code")[0] == S.FAULT_IK_NO_SOLUTION
    assert np.abs(f("q")[:DOF] - Q0).max() < 1e-4

    drv.write_command(cmd_q(drv, GOAL))  # latched: ignored until cleared
    drv.step(50)
    assert np.abs(state(drv)[1]("q")[:DOF] - Q0).max() < 1e-4

    drv.clear_faults()
    drv.write_command(cmd_q(drv, GOAL))
    drv.step(400)
    _, f = state(drv)
    assert f("fault_code")[0] == 0.0 and np.abs(f("q")[:DOF] - GOAL).max() < 1e-3


def test_the_sim_never_silently_executes_a_command_that_misses_by_more_than_the_tolerance(drv, arm):
    """A pose the five-joint arm cannot match exactly: it either faults (held) or lands within the tolerance."""
    T = np.array(arm.fk(GOAL.tolist()))
    T[:3, :3] = R.from_euler("y", 0.3).as_matrix() @ T[:3, :3]
    pose = np.r_[T[:3, 3], R.from_matrix(T[:3, :3]).as_quat()]
    drv.write_command(cmd_pose(drv, pose))
    drv.step(400)
    _, f = state(drv)
    if f("fault_code")[0] != 0.0:
        assert f("fault_code")[0] == S.FAULT_IK_NO_SOLUTION
        assert np.abs(f("q")[:DOF] - Q0).max() < 1e-4  # rejected, held
        return
    got = f("tcp_pose")
    pos_tol, rot_tol = drv.ik_tolerance
    assert np.linalg.norm(got[:3] - pose[:3]) <= pos_tol + 1e-3
    angle = 2.0 * np.arccos(np.clip(abs(float(np.dot(got[3:], pose[3:]))), 0.0, 1.0))
    assert angle <= rot_tol + 0.01


def test_the_exact_only_driver_rejects_what_the_arm_cannot_match(arm):
    d = S.SimDriver(ARM, ik_tolerance=None)
    d.connect()
    T = np.array(arm.fk(GOAL.tolist()))
    T[:3, :3] = R.from_euler("y", 0.3).as_matrix() @ T[:3, :3]
    d.write_command(cmd_pose(d, np.r_[T[:3, 3], R.from_matrix(T[:3, :3]).as_quat()]))
    d.step(50)
    _, f = state(d)
    assert f("fault_code")[0] == S.FAULT_IK_NO_SOLUTION and np.abs(f("q")[:DOF] - Q0).max() < 1e-4
    d.disconnect()


def _noisy_pose(arm, q, pos=(0.003, -0.002, 0.001), rot=0.02):
    T = np.array(arm.fk(q.tolist()))
    T[:3, 3] += pos
    T[:3, :3] = R.from_rotvec([rot * 0.6, -rot * 0.5, rot * 0.6]).as_matrix() @ T[:3, :3]
    return np.r_[T[:3, 3], R.from_matrix(T[:3, :3]).as_quat()]


def test_a_nearly_reachable_pose_moves_to_the_nearest_reachable_one(drv, arm):
    """A pose a few mm and ~1 degree off the reachable set: no fault, the arm gets as close as it can."""
    assert drv.ik_tolerance == (0.005, 0.1)
    pose = _noisy_pose(arm, GOAL)
    drv.write_command(cmd_pose(drv, pose))
    drv.step(400)
    _, f = state(drv)
    assert f("fault_code")[0] == 0.0
    got = f("tcp_pose")
    assert np.linalg.norm(got[:3] - pose[:3]) < 0.005  # within the tolerance of what was asked
    assert np.abs(f("q")[:DOF] - GOAL).max() < 0.05  # and on the configuration the pose came from, to a few degrees


def test_the_exact_only_behaviour_is_still_available(arm):
    d = S.SimDriver(ARM, ik_tolerance=None)
    d.connect()
    assert d.ik_tolerance is None
    d.write_command(cmd_pose(d, _noisy_pose(arm, GOAL)))
    d.step(50)
    _, f = state(d)
    assert f("fault_code")[0] == S.FAULT_IK_NO_SOLUTION
    assert np.abs(f("q")[:DOF] - Q0).max() < 1e-4  # rejected, held
    d.disconnect()


def test_a_miss_beyond_the_tolerance_is_rejected(drv, arm):
    pose = _noisy_pose(arm, GOAL, pos=(0.0, 0.0, 0.0), rot=0.6)  # ~35 degrees off an attainable orientation
    drv.write_command(cmd_pose(drv, pose))
    drv.step(50)
    _, f = state(drv)
    assert f("fault_code")[0] == S.FAULT_IK_NO_SOLUTION
    assert np.abs(f("q")[:DOF] - Q0).max() < 1e-4


def test_a_custom_tolerance_changes_what_is_accepted(arm):
    pose = _noisy_pose(arm, GOAL, pos=(0.0, 0.0, 0.0), rot=0.3)
    tight, loose = S.SimDriver(ARM, ik_tolerance=(0.005, 0.01)), S.SimDriver(ARM, ik_tolerance=(0.05, 0.5))
    for d in (tight, loose):
        d.connect()
        d.write_command(cmd_pose(d, pose))
        d.step(400)
    assert state(tight)[1]("fault_code")[0] == S.FAULT_IK_NO_SOLUTION
    assert state(loose)[1]("fault_code")[0] == 0.0
    for d in (tight, loose):
        d.disconnect()


def test_tolerance_is_validated():
    with pytest.raises(ValueError):
        S.SimDriver(ARM, ik_tolerance=(-0.1, 0.1))
    with pytest.raises(ValueError):
        S.SimDriver(ARM, ik_tolerance=(float("nan"), 0.1))
    with pytest.raises(ValueError):  # the UR bindings have no nearest-reachable IK, and need none
        S.SimDriver("ur5", ik_tolerance=(0.005, 0.1))
    assert S.SimDriver("ur5").ik_tolerance is None  # the UR arms keep the exact behaviour


def test_out_of_limits_is_rejected(drv):
    bad = Q0.copy()
    bad[3] = 2.0  # wrist flex limit is +-1.658
    drv.write_command(cmd_q(drv, bad))
    drv.step(20)
    _, f = state(drv)
    assert f("fault_code")[0] == S.FAULT_OUT_OF_LIMITS
    assert np.abs(f("q")[:DOF] - Q0).max() < 1e-4


def test_mode_is_explicit_not_inferred(drv):
    c = cmd_q(drv, GOAL)
    off = lr.command_layout()["target_pose"][0]
    c[off : off + 7] = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # would fault if it were used
    drv.write_command(c)
    drv.step(400)
    _, f = state(drv)
    assert f("fault_code")[0] == 0.0 and np.abs(f("q")[:DOF] - GOAL).max() < 1e-3


def test_unsupported_mode_is_rejected(drv, arm):
    c = drv.idle_command()
    c[lr.command_layout()["control_mode"][0]] = arm.control_modes()["joint_torque"]
    with pytest.raises((ValueError, NotImplementedError)):
        drv.write_command(c)


def test_latency_queues_shift_by_manifest_cycles():
    def first_change(d):
        d.write_command(cmd_q(d, GOAL))
        for i in range(1, 20):
            d.step()
            if np.abs(state(d)[1]("q")[:DOF] - Q0).max() > 1e-9:
                return i
        raise AssertionError("never moved")

    d = S.SimDriver(ARM)
    d.connect()
    m = d.manifest()
    assert first_change(d) == m["command_dispatch_cycles"] + m["state_staleness_cycles"] + 1
    d.disconnect()


def test_tracking_settles_everywhere_in_the_workspace():
    """Regression: diagonal-mass gains went unstable in strongly coupled configurations (8 of 30 random
    targets never settled), though they held the start pose. Computed torque settles them all."""
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]
    rng = np.random.default_rng(0)
    for _ in range(30):
        goal = rng.uniform(lo, hi) * 0.8
        d = S.SimDriver(ARM)
        d.connect()
        d.write_command(cmd_q(d, goal))
        d.step(500)
        _, f = state(d)
        assert np.abs(f("q")[:DOF] - goal).max() < 2e-3, f"did not settle at {goal.round(3).tolist()}"
        assert np.abs(f("qd")[:DOF]).max() < 0.05
        d.disconnect()


def test_the_default_controller_is_computed_torque():
    d = S.SimDriver(ARM)
    assert d._full_mass and d._wn == 20.0
    assert not S.SimDriver("ur5")._full_mass  # the UR arms keep their diagonal-mass gains


def test_sim_entry_point_demo_passes_headless(capsys):
    import sim

    assert sim.main(["--arm", ARM, "--headless"]) == 0
    out = capsys.readouterr().out
    assert out.count("ok") == 6 and "FAIL" not in out
