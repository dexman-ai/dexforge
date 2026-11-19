"""SimDriver behaviour through the driver-contract buffers (see tools/SIM_SPEC.md)."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import sim_driver as S  # noqa: E402

import dexforge_ur  # noqa: E402

ur5 = dexforge_ur.ur5
Q0 = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])
GOAL = Q0 + np.array([0.4, 0.3, -0.3, 0.2, -0.2, 0.3])


@pytest.fixture
def drv():
    d = S.SimDriver("ur5")
    d.connect()
    yield d
    d.disconnect()


def state(d):
    buf = np.full(d.manifest()["state_len"], np.nan)
    d.read_state(buf)
    lay = dexforge_ur.state_layout()
    f = lambda name: buf[lay[name][0] : lay[name][0] + lay[name][1]]  # noqa: E731
    return buf, f


def cmd_q(d, q):
    c = d.idle_command()
    lay = dexforge_ur.command_layout()
    c[lay["control_mode"][0]] = ur5.control_modes()["joint_pos"]
    off = lay["target_q"][0]
    c[off : off + 6] = q
    return c


def cmd_pose(d, pose):
    c = d.idle_command()
    lay = dexforge_ur.command_layout()
    c[lay["control_mode"][0]] = ur5.control_modes()["cartesian_pose"]
    off = lay["target_pose"][0]
    c[off : off + 7] = pose
    return c


def cmd_qd(d, qd):
    c = d.idle_command()
    lay = dexforge_ur.command_layout()
    c[lay["control_mode"][0]] = ur5.control_modes()["joint_vel"]
    off = lay["target_qd"][0]
    c[off : off + 6] = qd
    return c


def test_buffer_lengths_come_from_the_contract(drv):
    m = drv.manifest()
    assert m["state_len"] == dexforge_ur.state_layout()["len"]
    assert m["command_len"] == dexforge_ur.command_layout()["len"]
    buf, f = state(drv)
    assert len(buf) == m["state_len"]
    assert np.allclose(f("q")[:6], Q0)
    assert f("fault_code")[0] == 0.0
    assert np.isnan(f("wrench")).all()  # unsupported fields stay NaN
    with pytest.raises(ValueError):
        drv.write_command(np.zeros(3))


def test_gravity_compensation_holds_at_rest(drv):
    drv.step(500)  # 1 s with no command
    _, f = state(drv)
    assert np.abs(f("q")[:6] - Q0).max() < 1e-4
    assert np.abs(f("qd")[:6]).max() < 1e-3


def test_joint_position_round_trip(drv):
    drv.write_command(cmd_q(drv, GOAL))
    drv.step(1500)  # 3 s
    _, f = state(drv)
    err = np.abs(f("q")[:6] - GOAL).max()
    assert err < 1e-3, f"tracking error {err:.2e} rad"
    assert (np.abs(f("torques")[:6]) <= np.array(drv.manifest()["effort_limits"]) + 1e-9).all()


def test_reference_respects_velocity_limits(drv):
    drv.write_command(cmd_q(drv, GOAL))
    vmax = np.array(drv.manifest()["velocity_limits"])
    worst = 0.0
    for _ in range(200):
        drv.step()
        _, f = state(drv)
        worst = max(worst, (np.abs(f("qd")[:6]) / vmax).max())
    assert worst < 1.2  # tracks a velocity-limited reference, no runaway


def test_joint_velocity_mode(drv):
    qd = np.array([0.2, 0, 0, 0, 0, 0])
    drv.write_command(cmd_qd(drv, qd))
    drv.step(500)  # 1 s
    _, f = state(drv)
    assert abs((f("q")[0] - Q0[0]) - 0.2) < 0.02


def test_cartesian_pose_uses_generated_ik(drv):
    pose = np.array(ur5.fk_pose(GOAL.tolist()))
    drv.write_command(cmd_pose(drv, pose))
    drv.step(1500)
    _, f = state(drv)
    got = f("tcp_pose")
    assert np.abs(got[:3] - pose[:3]).max() < 1e-3
    assert abs(abs(np.dot(got[3:], pose[3:])) - 1.0) < 1e-4  # quaternion, up to sign
    assert f("fault_code")[0] == 0.0


def test_unreachable_pose_faults_and_holds_until_cleared(drv):
    far = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    drv.write_command(cmd_pose(drv, far))
    drv.step(200)
    _, f = state(drv)
    assert f("fault_code")[0] == S.FAULT_IK_NO_SOLUTION
    assert np.abs(f("q")[:6] - Q0).max() < 1e-4  # the arm did not move

    drv.write_command(cmd_q(drv, GOAL))  # latched: ignored until cleared
    drv.step(200)
    _, f = state(drv)
    assert np.abs(f("q")[:6] - Q0).max() < 1e-4

    drv.clear_faults()  # drops everything sent while latched, so send the command again
    drv.write_command(cmd_q(drv, GOAL))
    drv.step(1500)
    _, f = state(drv)
    assert f("fault_code")[0] == 0.0
    assert np.abs(f("q")[:6] - GOAL).max() < 1e-3


def test_ik_step_bound_fault():
    d = S.SimDriver("ur5", ik_max_joint_step=0.01)
    d.connect()
    d.write_command(cmd_pose(d, np.array(ur5.fk_pose(GOAL.tolist()))))
    d.step(50)
    _, f = state(d)
    assert f("fault_code")[0] == S.FAULT_IK_STEP_TOO_LARGE
    assert np.abs(f("q")[:6] - Q0).max() < 1e-4
    d.disconnect()


def test_out_of_limits_is_rejected(drv):
    bad = Q0.copy()
    bad[2] = 3.5  # elbow limit is +-pi
    drv.write_command(cmd_q(drv, bad))
    drv.step(100)
    _, f = state(drv)
    assert f("fault_code")[0] == S.FAULT_OUT_OF_LIMITS
    assert np.abs(f("q")[:6] - Q0).max() < 1e-4


def test_mode_is_explicit_not_inferred(drv):
    """With control_mode = joint_pos, a stray target_pose is ignored."""
    c = cmd_q(drv, GOAL)
    off = dexforge_ur.command_layout()["target_pose"][0]
    c[off : off + 7] = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # would fault if it were used
    drv.write_command(c)
    drv.step(1500)
    _, f = state(drv)
    assert f("fault_code")[0] == 0.0
    assert np.abs(f("q")[:6] - GOAL).max() < 1e-3


def test_idle_holds(drv):
    drv.write_command(cmd_q(drv, GOAL))
    drv.step(1500)
    drv.write_command(drv.idle_command())  # control_mode NaN
    drv.step(500)
    _, f = state(drv)
    assert np.abs(f("q")[:6] - GOAL).max() < 1e-3 and np.abs(f("qd")[:6]).max() < 1e-3


def test_write_command_rejects_what_the_contract_rejects(drv):
    lay = dexforge_ur.command_layout()
    modes = ur5.control_modes()

    unsupported = drv.idle_command()  # joint_torque is not in the manifest
    unsupported[lay["control_mode"][0]] = modes["joint_torque"]
    with pytest.raises(NotImplementedError):
        drv.write_command(unsupported)

    unknown = drv.idle_command()
    unknown[lay["control_mode"][0]] = 42.0
    with pytest.raises(ValueError):
        drv.write_command(unknown)

    missing = drv.idle_command()  # joint_pos with no target_q
    missing[lay["control_mode"][0]] = modes["joint_pos"]
    with pytest.raises(ValueError):
        drv.write_command(missing)

    drv.step(5)  # rejected commands were never latched
    _, f = state(drv)
    assert f("fault_code")[0] == 0.0


def first_change_cycle(d):
    d.write_command(cmd_q(d, GOAL))
    for i in range(1, 20):
        d.step()
        _, f = state(d)
        if np.abs(f("q")[:6] - Q0).max() > 1e-9:
            return i
    raise AssertionError("never moved")


def test_latency_queues_shift_by_manifest_cycles():
    d = S.SimDriver("ur5")
    d.connect()
    m = d.manifest()
    expected = m["command_dispatch_cycles"] + m["state_staleness_cycles"] + 1
    assert first_change_cycle(d) == expected
    d.disconnect()

    z = S.SimDriver("ur5")
    z._manifest["command_dispatch_cycles"] = 0
    z._manifest["state_staleness_cycles"] = 0
    z.connect()
    assert first_change_cycle(z) == 1
    z.disconnect()


def test_n_cycles_produce_n_timestamps(drv):
    _, f = state(drv)
    t0 = f("timestamp")[0]
    drv.step(10)
    _, f = state(drv)
    stale = drv.manifest()["state_staleness_cycles"]
    assert abs((f("timestamp")[0] - t0) - (10 - stale) * drv.dt) < 1e-12


def test_threaded_loop_latches_commands():
    d = S.SimDriver("ur5", realtime=False)
    d.connect(threaded=True)
    d.write_command(cmd_q(d, GOAL))
    deadline = time.time() + 20
    ok = False
    while time.time() < deadline:
        _, f = state(d)
        if np.abs(f("q")[:6] - GOAL).max() < 1e-3:
            ok = True
            break
        time.sleep(0.05)
    with pytest.raises(RuntimeError):
        d.step()
    d.disconnect()
    assert ok, "threaded sim did not converge"


def test_sim_entry_point_demo_passes_headless(capsys):
    import sim

    assert sim.main(["--arm", "ur5", "--headless"]) == 0
    out = capsys.readouterr().out
    assert out.count("ok") == 6 and "FAIL" not in out


def _stream(d, make_cmd, targets_q):
    """Send one new command per control cycle (as a kernel does); return the worst tracking error."""
    worst = 0.0
    n = len(targets_q)
    for i, tq in enumerate(targets_q):
        d.write_command(make_cmd(i))
        d.step()
        _, f = state(d)
        if i > 20:  # skip the start-up transient
            lag = d.manifest()["command_dispatch_cycles"] + d.manifest()["state_staleness_cycles"] + 1
            worst = max(worst, np.abs(f("q")[:6] - targets_q[max(0, i - lag)]).max())
    return worst, n


def test_streamed_joint_trajectory_tracks_without_fault(drv):
    """A smooth sweep, one new target per cycle (slow enough for the velocity limits)."""
    n = 1000
    s = (1 - np.cos(np.linspace(0, np.pi, n))) / 2  # 0 -> 1, zero end velocities
    traj = Q0 + np.outer(s, np.array([0.5, 0.3, -0.3, 0.2, -0.2, 0.3]))
    worst, _ = _stream(drv, lambda i: cmd_q(drv, traj[i]), traj)
    _, f = state(drv)
    assert f("fault_code")[0] == 0.0
    assert worst < 0.02, f"tracking lag {worst:.3f} rad while streaming"
    drv.step(300)
    _, f = state(drv)
    assert np.abs(f("q")[:6] - traj[-1]).max() < 1e-3  # and it settles on the final target


def test_streamed_cartesian_line_follows_ik_branch(drv):
    """A short TCP line streamed as poses; each is solved by the generated IK on the current branch."""
    n = 1000
    p0 = np.array(ur5.fk_pose(Q0.tolist()))
    poses = []
    for s in np.linspace(0.0, 1.0, n):
        p = p0.copy()
        p[:3] += s * np.array([0.0, 0.0, -0.10])  # 10 cm straight down
        poses.append(p)
    for pose in poses:
        drv.write_command(cmd_pose(drv, pose))
        drv.step()
    drv.step(300)
    _, f = state(drv)
    assert f("fault_code")[0] == 0.0
    assert np.abs(f("tcp_pose")[:3] - poses[-1][:3]).max() < 1e-3


def test_clear_faults_drops_the_rejected_command(drv):
    far = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    drv.write_command(cmd_pose(drv, far))
    drv.step(20)
    drv.clear_faults()  # the client sends nothing new
    drv.step(50)
    _, f = state(drv)
    assert f("fault_code")[0] == 0.0, "the rejected command was applied again"
    drv.write_command(cmd_q(drv, GOAL))  # and a new command is accepted
    drv.step(1500)
    _, f = state(drv)
    assert np.abs(f("q")[:6] - GOAL).max() < 1e-3
