"""The SO-101 hardware bridge's planning and safety logic, without hardware or LeRobot.

``so101_bridge`` imports LeRobot only inside its hardware functions, so the model side (joint
limits, step caps, minimum-jerk path, Cartesian planning through ``ik_project``) is tested here
against the real generated bindings. The hardware side is exercised by hand with a dry run.
"""

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import so101_bridge as B  # noqa: E402

lr = pytest.importorskip("dexforge_lerobot")
MODEL = lr.so101

Q_UPRIGHT = [-13.4, 6.0, -81.0, -3.6, 11.5]  # the pose in the photo: arm straight up


def test_unit_roundtrip():
    q = [10.0, -20.0, 30.0, -40.0, 50.0]
    assert B.deg(B.rad(q)) == pytest.approx(q)


def test_joint_limits_match_the_manifest():
    lo, hi = B.joint_limits_deg(MODEL)
    assert len(lo) == len(hi) == 5
    assert lo[1] == pytest.approx(-100.0, abs=0.01) and hi[1] == pytest.approx(100.0, abs=0.01)
    assert all(a < b for a, b in zip(lo, hi))


def test_model_zero_is_upper_arm_vertical_forearm_horizontal():
    # LeRobot's calibration pose; the bridge relies on LeRobot degrees == model degrees
    x, y, z = B.tcp_mm(MODEL, [0, 0, 0, 0, 0])
    assert (x, z) == pytest.approx((391.0, 226.0), abs=2.0) and abs(y) < 0.1
    # upright arm (the photo): the tip is high above the base
    assert B.tcp_mm(MODEL, Q_UPRIGHT)[2] > 480.0


def test_signs_match_the_hand_check_on_the_real_arm():
    base = [0.0] * 5
    p0 = B.tcp_mm(MODEL, base)
    # + shoulder_lift tilts the arm forward and down
    d = [b - a for a, b in zip(p0, B.tcp_mm(MODEL, [0, 10, 0, 0, 0]))]
    assert d[0] > 0 and d[2] < 0
    # + shoulder_pan swings the tip to the right (clockwise from above): y goes negative
    assert B.tcp_mm(MODEL, [10, 0, 0, 0, 0])[1] < 0


def test_check_target_accepts_a_small_move_and_refuses_the_rest():
    now = list(Q_UPRIGHT)
    ok = [now[0] + 5, now[1], now[2], now[3], now[4]]
    assert B.check_target(MODEL, now, ok, 15.0) == []
    # past the limit (pan limit is 110 deg)
    bad = list(now)
    bad[0] = 109.0
    assert any("outside the model's limits" in p for p in B.check_target(MODEL, now, bad, 200.0))
    # inside the limit but inside the margin too
    near = list(now)
    near[0] = 108.0
    assert B.check_target(MODEL, now, near, 200.0)
    # too big a step
    big = list(now)
    big[1] += 20.0
    assert any("more than" in p for p in B.check_target(MODEL, now, big, 15.0))


def test_a_joint_resting_at_its_limit_may_stay_but_not_go_further_out():
    now = [0.0, 0.0, -96.2, 0.0, 0.0]  # the real arm rests here, past the 3 deg margin of the model's -96.8
    stay = list(now)
    stay[0] = 5.0  # only pan moves
    assert B.check_target(MODEL, now, stay, 15.0) == []
    deeper = list(now)
    deeper[2] = -98.0
    assert B.check_target(MODEL, now, deeper, 15.0)
    back = list(now)
    back[2] = -80.0
    assert B.check_target(MODEL, now, back, 20.0) == []


def test_a_pose_outside_the_model_limits_can_still_move_back_in():
    now = [119.0, 0.0, 0.0, 0.0, 0.0]  # the real arm reaches 119 deg; the model stops at 110
    back = [100.0, 0.0, 0.0, 0.0, 0.0]
    assert B.check_target(MODEL, now, back, 30.0) == []


def test_trajectory_is_smooth_bounded_and_ends_exactly():
    q0, q1 = [0, 0, 0, 0, 0], [10, -6, 0, 3, 0]
    pts, duration = B.trajectory(q0, q1, speed_dps=15.0)
    assert pts[-1] == pytest.approx(q1)
    assert duration >= 1.5
    # never faster than the cap, never overshoots, monotone per joint
    dt = 1.0 / B.RATE_HZ
    prev = q0
    for p in pts:
        for a, b, lo, hi in zip(prev, p, q0, q1):
            assert abs(b - a) / dt <= 15.0 + 1e-6
            assert min(lo, hi) - 1e-9 <= b <= max(lo, hi) + 1e-9
        prev = p
    # starts and ends at rest: the first and last steps are tiny
    first = max(abs(a - b) for a, b in zip(q0, pts[0]))
    last = max(abs(a - b) for a, b in zip(pts[-2], pts[-1]))
    assert first < 0.05 and last < 0.05
    # per-command step (the thing LeRobot's max_relative_target caps) is far below the 5 deg cap
    assert max(max(abs(a - b) for a, b in zip(prev2, p)) for prev2, p in zip([q0] + pts[:-1], pts)) < B.MAX_RELATIVE_TARGET_DEG / 2


def test_trajectory_speed_is_clamped_and_validated():
    _, slow = B.trajectory([0] * 5, [10, 0, 0, 0, 0], speed_dps=10.0)
    _, fast = B.trajectory([0] * 5, [10, 0, 0, 0, 0], speed_dps=1000.0)  # clamped to MAX_SPEED_DPS
    assert fast >= 1.875 * 10.0 / B.MAX_SPEED_DPS - 1e-9 and fast < slow
    with pytest.raises(ValueError):
        B.trajectory([0] * 5, [1, 0, 0, 0, 0], speed_dps=0.0)


def test_cartesian_plan_lands_where_asked_and_holds_orientation():
    now = [0.0, 20.0, -30.0, 10.0, 0.0]
    p0 = B.tcp_mm(MODEL, now)
    plan = B.plan_cartesian(MODEL, now, [0.0, 0.0, 20.0])
    assert plan is not None
    q_t, perr, oerr = plan
    assert perr < B.POS_TOL_MM and oerr < B.ORI_TOL_RAD
    p1 = B.tcp_mm(MODEL, q_t)
    assert p1[2] - p0[2] == pytest.approx(20.0, abs=B.POS_TOL_MM)
    assert abs(p1[0] - p0[0]) < B.POS_TOL_MM + 1.0
    # a 20 mm move is a modest joint change, inside the cap a Cartesian move gets
    assert max(abs(a - b) for a, b in zip(q_t, now)) < B.DEFAULT_MAX_MOVE_JOINT_STEP_DEG
    assert B.check_target(MODEL, now, q_t, B.DEFAULT_MAX_MOVE_JOINT_STEP_DEG) == []


def test_a_small_cartesian_step_that_needs_a_big_joint_swing_is_refused():
    # stretched arm, 30 mm forward: the elbow would have to swing ~44 deg
    now = [0.0, 20.0, -30.0, 10.0, 0.0]
    plan = B.plan_cartesian(MODEL, now, [30.0, 0.0, 0.0])
    q_t = plan[0]
    assert B.check_target(MODEL, now, q_t, B.DEFAULT_MAX_MOVE_JOINT_STEP_DEG)


def test_cartesian_plan_sideways_is_a_pan_move():
    now = [0.0, 20.0, -30.0, 10.0, 0.0]
    q_t, perr, _ = B.plan_cartesian(MODEL, now, [0.0, -15.0, 0.0])  # to the right: pan goes up
    assert perr < B.POS_TOL_MM
    assert q_t[0] > now[0]


def test_unreachable_cartesian_move_is_reported_not_guessed():
    far = B.plan_cartesian(MODEL, [0.0, 20.0, -30.0, 10.0, 0.0], [0.0, 0.0, 5000.0])  # 5 m up
    # either no solution, or one that misses by far more than the bridge tolerates
    assert far is None or far[1] > B.POS_TOL_MM


def test_parse_joint_args():
    q = B.parse_joint_args(["shoulder_pan=+5", "wrist_roll=-10"], [1, 2, 3, 4, 5], absolute=False)
    assert q == [6.0, 2, 3, 4, -5.0]
    q = B.parse_joint_args(["elbow_flex=7"], [1, 2, 3, 4, 5], absolute=True)
    assert q == [1, 2, 7.0, 4, 5]
    for bad in ("gripper=5", "shoulder_pan", "shoulder_pan=abc"):
        with pytest.raises(SystemExit):
            B.parse_joint_args([bad], [0] * 5, absolute=False)


def test_dry_run_sends_nothing_and_needs_no_hardware(capsys):
    rc = B.main(["joint", "shoulder_pan=+5", "--from-q=" + ",".join(str(x) for x in Q_UPRIGHT)])
    out = capsys.readouterr().out
    assert rc == 0 and "DRY RUN" in out and "Nothing was sent" in out


def test_dry_run_refuses_an_unsafe_target(capsys):
    rc = B.main(["joint", "shoulder_lift=+40", "--from-q=" + ",".join(str(x) for x in Q_UPRIGHT)])
    assert rc == 3 and "REFUSED" in capsys.readouterr().out


def test_live_cannot_be_combined_with_a_given_start_pose():
    with pytest.raises(SystemExit):
        B.main(["joint", "shoulder_pan=+5", "--live", "--from-q=0,0,0,0,0"])


def test_max_relative_target_is_a_float():
    # LeRobot raises TypeError on an int
    assert isinstance(B.MAX_RELATIVE_TARGET_DEG, float)
    assert math.isfinite(B.MAX_RELATIVE_TARGET_DEG) and B.MAX_RELATIVE_TARGET_DEG > 0


def test_trim_closes_a_steady_state_error_without_overshoot_or_exceeding_its_cap():
    target = [0.0, 0.0, 86.5, 0.0, 0.0]
    moved = [False, False, True, False, False]
    trim = [0.0] * 5
    # a servo that stops 1.9 deg short of whatever goal it is given (a stand-in for load + friction)
    for _ in range(12):
        present = [0.0, 0.0, target[2] + trim[2] + 1.9, 0.0, 0.0]  # present = goal + 1.9 (goal = target + trim)
        trim = B.update_trim(trim, target, present, moved)
    assert trim[2] == pytest.approx(-1.9, abs=0.1) and all(t == 0.0 for i, t in enumerate(trim) if i != 2)
    # a huge error is capped
    huge = B.update_trim([0.0] * 5, target, [0, 0, 86.5 - 50.0, 0, 0], moved)
    assert huge[2] == B.TRIM_MAX_DEG
    # joints that were not moved are never trimmed
    assert B.update_trim([1.0] * 5, target, [9.0] * 5, [False] * 5) == [0.0] * 5
