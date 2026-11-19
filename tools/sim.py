#!/usr/bin/env python3
"""Run a DexForge arm in MuJoCo behind the driver contract, with an optional viewer.

Starts a ``SimDriver`` (generated IK / FK / gravity / mass matrix; MuJoCo for the
physics) and a small demo *client* that talks to it only through the contract
buffers (``write_command`` / ``read_state``), exactly as a real client would:
joint-position waypoints, then cartesian poses solved by the generated IK.

    mjpython tools/sim.py --arm ur5              # viewer (macOS needs mjpython)
    python   tools/sim.py --arm ur5 --headless   # no window; prints tracking errors
    python   tools/sim.py --arm ur5 --render out.png
    mjpython tools/sim.py --arm ur5 --interactive  # viewer + type commands in this terminal

With ``--no-demo`` nothing is commanded and the arm holds its pose against gravity.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_driver as S  # noqa: E402


class Client:
    """What a driver consumer does: write a command buffer, read a state buffer."""

    def __init__(self, drv: S.SimDriver, wait: Callable[[float], None]) -> None:
        import importlib

        self.drv, self._wait = drv, wait
        mod = importlib.import_module(S._ARMS[drv.arm][0])
        self._cmd, self._st = mod.command_layout(), mod.state_layout()
        self._codes = getattr(mod, drv.arm).control_modes()
        self._fk_pose = getattr(mod, drv.arm).fk_pose
        self._buf = np.full(drv.manifest()["state_len"], np.nan)

    def state(self, name: str) -> np.ndarray:
        self.drv.read_state(self._buf)
        off, n = self._st[name]
        return self._buf[off : off + n]

    def send(self, field: str, values) -> None:
        mode = {"target_q": "joint_pos", "target_qd": "joint_vel", "target_pose": "cartesian_pose"}[field]
        c = self.drv.idle_command()
        c[self._cmd["control_mode"][0]] = self._codes[mode]
        off, _ = self._cmd[field]
        c[off : off + len(values)] = values
        self.drv.write_command(c)

    def goto(self, field: str, values, check: Callable[[], float], tol: float, timeout: float = 6.0) -> float:
        """Send a target, wait until ``check()`` (an error) is within ``tol`` or the timeout; return the error."""
        self.send(field, values)
        t = 0.0
        while t < timeout:
            self._wait(0.05)
            t += 0.05
            if self.state("fault_code")[0] != 0.0 or check() < tol:
                break
        return check()


def demo(c: Client) -> int:
    """Joint waypoints, then cartesian poses of the same targets. Returns the number of failures."""
    n = c.drv.dof
    q0 = c.state("q")[:n].copy()
    deltas = [
        [0.5, 0.3, -0.3, 0.2, -0.2, 0.3],
        [-0.5, 0.2, 0.2, -0.3, 0.3, -0.3],
        [0.0, -0.3, 0.4, 0.2, 0.0, 0.0],
    ]
    failures = 0
    for i, dq in enumerate(deltas, 1):
        goal = q0 + np.array(dq)
        err = c.goto("target_q", goal, lambda: np.abs(c.state("q")[:n] - goal).max(), 1e-3)
        ok = err < 1e-3
        failures += not ok
        print(f"[joint  #{i}] max joint error {err:.2e} rad  {'ok' if ok else 'FAIL'}")

    for i, dq in enumerate(deltas, 1):
        pose = np.array(c._fk_pose((q0 + np.array(dq)).tolist()))
        err = c.goto("target_pose", pose, lambda: np.abs(c.state("tcp_pose")[:3] - pose[:3]).max(), 1e-3)
        fault = c.state("fault_code")[0]
        ok = err < 1e-3 and fault == 0.0
        failures += not ok
        print(f"[cart   #{i}] TCP position error {err * 1000:.2f} mm, fault {fault:g}  {'ok' if ok else 'FAIL'}")
    return failures


HELP = """commands (angles in radians unless noted):
  state                       joints, TCP pose, fault code
  joint q1 q2 q3 q4 q5 q6     joint-position command
  jointdeg q1 ... q6          same, in degrees
  nudge i delta               move joint i (1-6) by delta radians
  pose x y z qx qy qz qw      cartesian command (generated IK, current branch)
  move dx dy dz               move the TCP by (dx, dy, dz) metres, keeping orientation
  home                        back to the start pose
  clear                       clear a latched fault
  help | quit"""


def repl(c: Client, home: np.ndarray, readline=input) -> None:
    """Type commands; each is sent through the contract buffers and the result is printed."""
    n = c.drv.dof
    print(HELP)

    def settle(field, target, check, tol):
        err = c.goto(field, target, check, tol, timeout=8.0)
        fault = c.state("fault_code")[0]
        print(f"  error {err:.2e}  fault {fault:g}" + ("  (rejected; type 'clear')" if fault else ""))

    while True:
        try:
            line = readline("sim> ").strip()
        except EOFError:
            return
        if not line:
            continue
        cmd, *a = line.split()
        try:
            if cmd in ("quit", "exit", "q"):
                return
            elif cmd == "help":
                print(HELP)
            elif cmd == "state":
                q, p = c.state("q")[:n], c.state("tcp_pose")
                print(f"  q   {np.round(q, 4).tolist()}\n  tcp {np.round(p, 4).tolist()}  fault {c.state('fault_code')[0]:g}")
            elif cmd in ("joint", "jointdeg"):
                q = np.array([float(x) for x in a])
                q = np.radians(q) if cmd == "jointdeg" else q
                assert len(q) == n, f"need {n} values"
                settle("target_q", q, lambda: np.abs(c.state("q")[:n] - q).max(), 1e-3)
            elif cmd == "nudge":
                i, dq = int(a[0]) - 1, float(a[1])
                if not 0 <= i < n:
                    raise IndexError(f"joint index must be 1..{n}")
                q = c.state("q")[:n].copy()
                q[i] += dq
                settle("target_q", q, lambda: np.abs(c.state("q")[:n] - q).max(), 1e-3)
            elif cmd in ("pose", "move"):
                if cmd == "move":
                    assert len(a) == 3, "need dx dy dz"
                    pose = c.state("tcp_pose")[:7].copy()
                    pose[:3] += np.array([float(x) for x in a])
                else:
                    pose = np.array([float(x) for x in a])
                assert len(pose) == 7, "need x y z qx qy qz qw"
                settle("target_pose", pose, lambda: np.abs(c.state("tcp_pose")[:3] - pose[:3]).max(), 1e-3)
            elif cmd == "home":
                settle("target_q", home, lambda: np.abs(c.state("q")[:n] - home).max(), 1e-3)
            elif cmd == "clear":
                c.drv.clear_faults()
                print("  fault cleared")
            else:
                print(f"  unknown command {cmd!r}; type 'help'")
        except (ValueError, IndexError, AssertionError) as e:
            print(f"  bad input: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="ur5", choices=sorted(S._ARMS))
    ap.add_argument("--headless", action="store_true", help="no viewer; run the demo as fast as possible")
    ap.add_argument("--demo", action=argparse.BooleanOptionalAction, default=True, help="run the demo client")
    ap.add_argument("--interactive", action="store_true", help="type commands in the terminal (no demo)")
    ap.add_argument("--seconds", type=float, default=None, help="with the viewer (not --interactive): close after this long")
    ap.add_argument("--render", type=Path, help="write a PNG of the final pose and exit (implies --headless)")
    args = ap.parse_args(argv)
    headless = args.headless or args.render is not None

    if args.interactive:
        args.demo = False
    drv = S.SimDriver(args.arm, realtime=not headless)
    drv.connect(threaded=not headless)
    failures = 0
    try:
        if headless:
            client = Client(drv, lambda s: drv.step(round(s / drv.dt)))
            failures = demo(client) if args.demo else 0
            if args.interactive:
                repl(client, client.state("q")[: drv.dof].copy())
            if args.render:
                _render(drv, args.render)
                print(f"wrote {args.render}")
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(drv.model, drv.data) as viewer:
                t_end = None if args.seconds is None else time.time() + args.seconds

                def wait(seconds: float) -> None:
                    end = time.time() + seconds
                    while time.time() < end and viewer.is_running():
                        with drv.physics_lock:
                            viewer.sync()
                        time.sleep(1 / 60)

                client = Client(drv, time.sleep if args.interactive else wait)
                if args.interactive:
                    # the viewer syncs on its own thread so the terminal prompt can block
                    import threading

                    def pump() -> None:
                        while viewer.is_running():
                            with drv.physics_lock:
                                viewer.sync()
                            time.sleep(1 / 60)

                    threading.Thread(target=pump, daemon=True).start()
                    repl(client, client.state("q")[: drv.dof].copy())
                    return 0
                if args.demo:
                    failures = demo(client) if viewer.is_running() else 0
                while viewer.is_running() and (t_end is None or time.time() < t_end):
                    wait(0.1)
    finally:
        drv.disconnect()
    return 1 if failures else 0


def _render(drv: S.SimDriver, out: Path) -> None:
    import mujoco
    from PIL import Image

    r = mujoco.Renderer(drv.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.4]
    cam.distance, cam.azimuth, cam.elevation = 2.2, 135, -20
    with drv.physics_lock:
        r.update_scene(drv.data, cam)
    try:
        Image.fromarray(r.render()).save(out)
    finally:
        r.close()


if __name__ == "__main__":
    sys.exit(main())
