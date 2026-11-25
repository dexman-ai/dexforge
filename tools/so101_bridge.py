#!/usr/bin/env python3
"""Bridge between the dexforge SO-101 model and a real SO-ARM101 follower. DRY RUN unless --live.

Run it with LeRobot's interpreter (it needs ``lerobot``); the dexforge bindings are found in
dexforge's own ``.venv`` and added at the END of ``sys.path``, so nothing is installed into
LeRobot's environment and LeRobot's packages keep priority::

    PY=/path/to/lerobot/.venv/bin/python
    $PY tools/so101_bridge.py --port /dev/tty.usbmodemXXXX state
    $PY tools/so101_bridge.py --port ... joint shoulder_pan=+5 wrist_roll=-10      # dry run
    $PY tools/so101_bridge.py --port ... move --dz 20                             # dry run, mm
    $PY tools/so101_bridge.py joint shoulder_pan=+5 --from-q=-13,6,-81,-4,11     # dry run, no arm needed
    $PY tools/so101_bridge.py --port ... joint shoulder_pan=+5 --live             # really moves

Subcommands
  state      read-only: joint angles, the model's TCP position, where each joint is in its limits
  joint      move joints by degrees (relative; ``--absolute`` for absolute targets)
  move       move the TCP by dx/dy/dz millimetres in the base frame (x forward, y left, z up),
             orientation held as well as five joints allow (``ik_project``)
  release    switch the torque off (the arm drops unless you support it)

Safety, all on by default:
  * nothing is sent without ``--live``, and then only after you type ``move`` (``--yes`` skips it);
  * targets must lie inside the model's joint limits (3 deg margin), and one command moves each joint
    by at most 15 deg (25 deg for a Cartesian move, which holds the orientation) / the TCP by at most 30 mm (``--allow-large`` lifts those two caps, not the limits);
  * motion is a minimum-jerk path at <= 15 deg/s (``--speed-dps``, never above 45), sent at 20 Hz;
  * LeRobot's ``max_relative_target`` (a float, 5 deg) caps every single command against the position
    the servos report, and the run stops and holds if a joint lags its goal by more than 12 deg;
  * the gripper is never commanded;
  * torque is never switched off by this script, not on exit, not on Ctrl-C, not on an error: a limp
    arm falls. Use ``release`` (with the arm supported) when you want it limp.

The model and LeRobot share the same zero and signs for all five joints (checked by hand on this arm);
the model's radians are LeRobot's degrees * pi/180. Verify any new arm's signs with ``so101_probe.py
--watch`` before trusting ``--live`` on it.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]  # the model's five joints
MARGIN_DEG = 3.0  # stay this far inside the model's limits
DEFAULT_MAX_JOINT_STEP_DEG = 15.0
DEFAULT_MAX_MOVE_JOINT_STEP_DEG = 25.0  # a Cartesian move holds the orientation, which costs wrist/elbow motion
DEFAULT_MAX_MOVE_MM = 30.0
DEFAULT_SPEED_DPS = 15.0
MAX_SPEED_DPS = 45.0
RATE_HZ = 20.0
MAX_RELATIVE_TARGET_DEG = 5.0  # LeRobot's per-command cap; it must be a float
TRIM_MAX_DEG = 2.0  # host-side trim: the most the goal may be shifted to close a steady-state error
TRIM_GAIN = 0.8  # fraction of the remaining error added to the trim each round (< 1: no overshoot)
TRIM_TOL_DEG = 0.15  # stop trimming once every moved joint is this close
TRIM_ROUNDS = 8
TRIM_ROUND_S = 0.4
SETTLE_S = 1.5  # keep commanding the final goal this long, so the servos can finish the last degree
ABORT_TRACKING_DEG = 12.0  # stop if a joint lags its goal by more than this
POS_TOL_MM = 2.0  # a Cartesian move must land this close ...
ORI_TOL_RAD = 0.1  # ... and keep the orientation this close, or it is refused


# ---------------------------------------------------------------------------------------------
# The model side: pure functions, no hardware, no LeRobot. These are what the tests exercise.
# ---------------------------------------------------------------------------------------------
def load_model():
    """The dexforge SO-101 bindings: installed, or found in dexforge's own ``.venv``."""
    try:
        from dexforge_lerobot import so101

        return so101
    except ImportError:
        pass
    for site in sorted((ROOT / ".venv" / "lib").glob("python3*/site-packages")):
        sys.path.append(str(site))  # appended: LeRobot's packages keep priority
    try:
        from dexforge_lerobot import so101

        return so101
    except ImportError as exc:
        raise SystemExit(
            f"cannot import dexforge_lerobot ({exc}).\n"
            f"Build it into dexforge's venv: see tools/sim_bindings_lerobot/README.md "
            f"(it must be the same Python version as the interpreter running this script: {sys.version.split()[0]})."
        ) from exc


def deg(q):
    return [math.degrees(x) for x in q]


def rad(q):
    return [math.radians(x) for x in q]


def joint_limits_deg(model):
    m = model.manifest()
    return deg(m["joint_limits_min"]), deg(m["joint_limits_max"])


def tcp_mm(model, q_deg):
    T = model.fk(rad(q_deg))
    return [float(T[i][3]) * 1000.0 for i in range(3)]


def check_target(model, q_now_deg, q_tgt_deg, max_step_deg, margin_deg=MARGIN_DEG):
    """Reasons this move must not be sent (an empty list means it may)."""
    lo, hi = joint_limits_deg(model)
    problems = []
    for i, name in enumerate(ARM):
        t = q_tgt_deg[i]
        band_lo, band_hi = lo[i] + margin_deg, hi[i] - margin_deg

        def outside(v, a=band_lo, b=band_hi):
            return max(a - v, v - b, 0.0)  # how far beyond the safe band

        # refuse a target that is outside the band AND further out than the arm already is: a joint
        # resting at its limit may stay there, or move back in, but never go further out
        if outside(t) > outside(q_now_deg[i]) + 0.1:
            problems.append(
                f"{name}: target {t:.1f} deg is outside the model's limits "
                f"[{band_lo:.1f}, {band_hi:.1f}] (limit {lo[i]:.1f}..{hi[i]:.1f}, {margin_deg:g} deg margin)"
            )
        step = abs(t - q_now_deg[i])
        if step > max_step_deg:
            problems.append(f"{name}: moves {step:.1f} deg, more than the {max_step_deg:g} deg allowed per command")
    return problems


def min_jerk(tau: float) -> float:
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def trajectory(q0_deg, q1_deg, speed_dps=DEFAULT_SPEED_DPS, rate_hz=RATE_HZ, min_time=1.5):
    """Minimum-jerk waypoints from q0 to q1 (degrees), excluding q0. Peak joint speed <= speed_dps."""
    speed_dps = min(float(speed_dps), MAX_SPEED_DPS)
    if speed_dps <= 0:
        raise ValueError("speed must be positive")
    dmax = max(abs(b - a) for a, b in zip(q0_deg, q1_deg))
    duration = max(min_time, 1.875 * dmax / speed_dps)  # min-jerk peak speed is 1.875x the mean
    n = max(2, math.ceil(duration * rate_hz))
    pts = [[a + (b - a) * min_jerk(k / n) for a, b in zip(q0_deg, q1_deg)] for k in range(1, n + 1)]
    return pts, duration


def update_trim(trim_deg, target_deg, present_deg, moved, max_trim=TRIM_MAX_DEG, gain=TRIM_GAIN):
    """One round of host-side trim (integral action on the servos' steady-state error).

    A position servo with a low P gain stops short of its goal under load. Shifting the goal by (a
    fraction of) the remaining error closes that. Only the joints that were moved are trimmed, and the
    total shift per joint is capped at ``max_trim`` degrees. Returns the new trim vector.
    """
    out = []
    for i, t in enumerate(trim_deg):
        if not moved[i]:
            out.append(0.0)
            continue
        err = target_deg[i] - present_deg[i]
        out.append(max(-max_trim, min(max_trim, t + gain * err)))
    return out


def plan_cartesian(model, q_now_deg, d_mm, rot_weight=None):
    """Joint target (deg) for moving the TCP by d_mm (x, y, z) with the orientation held; or None.

    Returns (q_target_deg, position_error_mm, orientation_error_rad). A five-joint arm cannot hold
    all six pose DOF, so the orientation error is what you pay for an exactly placed position.
    """
    q = rad(q_now_deg)
    pose = [float(v) for v in model.fk_pose(q)]  # [x, y, z, qx, qy, qz, qw]; the orientation is kept as it is
    for i in range(3):
        pose[i] += d_mm[i] / 1000.0
    res = model.ik_project(pose, q) if rot_weight is None else model.ik_project(pose, q, True, rot_weight)
    if res is None:
        return None
    q_t, perr_m, oerr = res
    return deg(q_t), perr_m * 1000.0, oerr


def parse_joint_args(items, q_now_deg, absolute):
    q = list(q_now_deg)
    for item in items:
        name, sep, val = item.partition("=")
        if not sep or name not in ARM:
            raise SystemExit(f"bad joint spec '{item}': use NAME=DEGREES with NAME in {ARM}")
        try:
            v = float(val)
        except ValueError:
            raise SystemExit(f"bad joint spec '{item}': '{val}' is not a number") from None
        i = ARM.index(name)
        q[i] = v if absolute else q[i] + v
    return q


# ---------------------------------------------------------------------------------------------
# The hardware side (LeRobot is imported here, so the functions above import without it).
# ---------------------------------------------------------------------------------------------
def make_robot(port, robot_id):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    cfg = SO101FollowerConfig(
        port=port,
        id=robot_id,
        max_relative_target=float(MAX_RELATIVE_TARGET_DEG),  # a float: LeRobot raises TypeError on an int
        disable_torque_on_disconnect=False,  # a limp arm falls
    )
    return SO101Follower(cfg)


def read_deg(robot):
    pos = robot.bus.sync_read("Present_Position")  # degrees for the joints (the calibration is loaded)
    return [float(pos[n]) for n in ARM]


def hold_here(robot):
    """Goal position := present position (raw), so the arm stays where it is. Torque is not touched."""
    raw = robot.bus.sync_read("Present_Position", normalize=False)
    robot.bus.sync_write("Goal_Position", raw, normalize=False)


def torque_state(robot):
    return {n: robot.bus.read("Torque_Enable", n, normalize=False) for n in robot.bus.motors}


def open_read_only(robot):
    """Open the port and ping the servos. Writes nothing."""
    robot.bus.connect()
    if not robot.calibration:
        raise SystemExit(
            f"no LeRobot calibration found for id '{robot.id}' ({robot.calibration_fpath}); run lerobot-calibrate first"
        )


def go_live(robot):
    """Enable torque the safe way: goal := present first, so the servos cannot jump to a stale goal."""
    if not robot.is_calibrated:
        raise SystemExit(
            "the calibration stored in the servos does not match the calibration file; "
            "run lerobot-calibrate again. Refusing to move."
        )
    hold_here(robot)  # torque is still off here (or already holding); the goal now equals the pose
    robot.configure()  # LeRobot's motor setup; it ends with torque enabled


def print_state(model, q_deg):
    lo, hi = joint_limits_deg(model)
    print(f"{'joint':14s} {'deg':>8s} {'rad':>8s}   model limits (deg)")
    for i, n in enumerate(ARM):
        flag = "" if lo[i] <= q_deg[i] <= hi[i] else "   <-- outside the model's limits"
        print(f"{n:14s} {q_deg[i]:8.2f} {math.radians(q_deg[i]):8.4f}   {lo[i]:7.1f} .. {hi[i]:6.1f}{flag}")
    x, y, z = tcp_mm(model, q_deg)
    print(f"model TCP (base frame, x fwd / y left / z up): x={x:7.1f}  y={y:7.1f}  z={z:7.1f} mm")


def execute(robot, model, q_now, q_tgt, speed_dps, assume_yes, trim_on=True):
    pts, duration = trajectory(q_now, q_tgt, speed_dps)
    print(f"\nLIVE: {len(pts)} waypoints over {duration:.1f} s at {RATE_HZ:g} Hz.")
    print("Torque will be switched ON and the arm will hold its present pose first. Keep clear of it.")
    if not assume_yes and input("Type 'move' to send this to the arm: ").strip() != "move":
        print("not confirmed; nothing sent")
        return 1
    go_live(robot)
    period = 1.0 / RATE_HZ
    try:
        for wp in pts:
            t0 = time.perf_counter()
            robot.send_action({f"{n}.pos": float(wp[i]) for i, n in enumerate(ARM)})
            now = read_deg(robot)
            lag = max(abs(a - b) for a, b in zip(wp, now))
            if lag > ABORT_TRACKING_DEG:
                hold_here(robot)
                print(f"STOPPED: a joint lags its goal by {lag:.1f} deg (blocked?). Holding here, torque left on.")
                return 2
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        final_goal = {f"{n}.pos": float(pts[-1][i]) for i, n in enumerate(ARM)}
        for _ in range(int(SETTLE_S * RATE_HZ)):
            t0 = time.perf_counter()
            robot.send_action(final_goal)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        # host-side trim: close the servos' steady-state error by shifting the goal, within a small cap
        lo, hi = joint_limits_deg(model)
        moved = [abs(t - c) > 0.5 for t, c in zip(q_tgt, q_now)]
        trim = [0.0] * len(ARM)
        for rnd in range(TRIM_ROUNDS if trim_on else 0):
            present = read_deg(robot)
            worst = max((abs(t - p) for t, p, m in zip(q_tgt, present, moved) if m), default=0.0)
            if worst <= TRIM_TOL_DEG:
                break
            trim = update_trim(trim, q_tgt, present, moved)
            goal = {
                f"{n}.pos": float(min(hi[i], max(lo[i], q_tgt[i] + trim[i]))) for i, n in enumerate(ARM)
            }  # never past the model's own limits
            for _ in range(int(TRIM_ROUND_S * RATE_HZ)):
                t0 = time.perf_counter()
                robot.send_action(goal)
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        if trim_on and any(trim):
            print("trim applied (deg): " + "  ".join(f"{n[:5]}={x:+.2f}" for n, x in zip(ARM, trim)))
    except BaseException:  # Ctrl-C included: hold where we are, never go limp
        try:
            hold_here(robot)
        finally:
            print("\ninterrupted: holding the present pose, torque left on")
        raise
    final = read_deg(robot)
    print("done. final joint error vs target (deg): " + "  ".join(f"{n[:5]}={f - t:+.2f}" for n, f, t in zip(ARM, final, q_tgt)))
    print("TCP now (mm): x=%.1f y=%.1f z=%.1f" % tuple(tcp_mm(model, final)))
    print("Torque is ON: the arm is holding. `release` switches it off (support the arm first).")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=os.environ.get("SO101_PORT"), help="serial port (or set SO101_PORT)")
    ap.add_argument("--id", default="my_follower", help="the LeRobot calibration id")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--live", action="store_true", help="really send it (default: dry run)")
        p.add_argument("--yes", action="store_true", help="skip the typed confirmation (with --live)")
        p.add_argument("--speed-dps", type=float, default=DEFAULT_SPEED_DPS)
        p.add_argument("--no-trim", action="store_true", help="do not close the servos' steady-state error by shifting the goal")
        p.add_argument("--allow-large", action="store_true", help="lift the per-command size caps (not the joint limits)")
        p.add_argument("--from-q", help="dry run only: plan from these 5 joint degrees (comma-separated) without reading the arm")

    sub.add_parser("state", help="read-only joint angles and model TCP")
    pj = sub.add_parser("joint", help="NAME=DEG ...")
    pj.add_argument("joints", nargs="+")
    pj.add_argument("--absolute", action="store_true")
    common(pj)
    pm = sub.add_parser("move", help="TCP offset in mm, base frame")
    for a in ("dx", "dy", "dz"):
        pm.add_argument(f"--{a}", type=float, default=0.0)
    pm.add_argument("--rot-weight", type=float, default=None)
    common(pm)
    pr = sub.add_parser("release", help="switch the torque off (the arm falls unless supported)")
    pr.add_argument("--live", action="store_true")
    args = ap.parse_args(argv)

    model = load_model()
    from_q = getattr(args, "from_q", None)
    if from_q:
        if getattr(args, "live", False):
            raise SystemExit("--from-q is for dry runs only")
        q_now = [float(x) for x in from_q.split(",")]
        if len(q_now) != 5:
            raise SystemExit("--from-q needs 5 comma-separated degrees")
        robot = None
    else:
        if not args.port:
            raise SystemExit("--port is required (or set SO101_PORT)")
        robot = make_robot(args.port, args.id)
        open_read_only(robot)

    try:
        if robot is not None:
            q_now = read_deg(robot)
        print("present pose" + (" (given)" if robot is None else "") + ":")
        print_state(model, q_now)
        if args.cmd == "state":
            if robot is not None:
                print("torque:", ", ".join(f"{n}={v}" for n, v in torque_state(robot).items()))
            return 0
        if args.cmd == "release":
            if not args.live:
                print("\nDRY RUN: `release --live` would switch the torque off. The arm FALLS unless you support it.")
                return 0
            if input("Support the arm, then type 'release': ").strip() != "release":
                print("not confirmed; torque unchanged")
                return 1
            robot.bus.disable_torque()
            print("torque off")
            return 0

        # joint / move: build the target
        if args.speed_dps <= 0 or args.speed_dps > MAX_SPEED_DPS:
            raise SystemExit(f"--speed-dps must be in (0, {MAX_SPEED_DPS:g}]")
        max_step = DEFAULT_MAX_JOINT_STEP_DEG if args.cmd == "joint" else DEFAULT_MAX_MOVE_JOINT_STEP_DEG
        if args.allow_large:
            max_step = 90.0
        if args.cmd == "joint":
            q_tgt = parse_joint_args(args.joints, q_now, args.absolute)
        else:
            d = [args.dx, args.dy, args.dz]
            if not args.allow_large and math.sqrt(sum(x * x for x in d)) > DEFAULT_MAX_MOVE_MM:
                raise SystemExit(f"that is {math.sqrt(sum(x * x for x in d)):.1f} mm; the cap is {DEFAULT_MAX_MOVE_MM:g} mm (--allow-large)")
            plan = plan_cartesian(model, q_now, d, args.rot_weight)
            if plan is None:
                raise SystemExit("no reachable configuration near this target; nothing to send")
            q_tgt, perr, oerr = plan
            print(f"\nIK: lands {perr:.2f} mm from the requested point, orientation off by {oerr:.3f} rad")
            if perr > POS_TOL_MM or oerr > ORI_TOL_RAD:
                raise SystemExit(f"refused: tolerance is {POS_TOL_MM:g} mm / {ORI_TOL_RAD:g} rad")

        problems = check_target(model, q_now, q_tgt, max_step)
        print("\ntarget pose:")
        print_state(model, q_tgt)
        print("joint change (deg): " + "  ".join(f"{n[:5]}={t - c:+.1f}" for n, c, t in zip(ARM, q_now, q_tgt)))
        if problems:
            print("\nREFUSED:")
            for p in problems:
                print("  -", p)
            return 3
        if not args.live:
            pts, duration = trajectory(q_now, q_tgt, args.speed_dps)
            print(f"\nDRY RUN: would send {len(pts)} waypoints over {duration:.1f} s. Nothing was sent. Add --live to move.")
            return 0
        return execute(robot, model, q_now, q_tgt, args.speed_dps, args.yes, trim_on=not args.no_trim)
    finally:
        if robot is not None:
            robot.bus.disconnect(disable_torque=False)  # never switch the torque off here


if __name__ == "__main__":
    sys.exit(main())
