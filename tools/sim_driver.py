"""SimDriver: a simulated arm behind the dexhal driver contract, backed by MuJoCo.

It stands in for a real driver. A client calls ``write_command`` / ``read_state``
with the same flat ``f64`` buffers a real driver uses (layouts and lengths come
from the compiled ``dexforge_<oem>`` bindings, never hard-coded here). All arm
math is our generated code (IK, FK, gravity, mass matrix); only the hardware side
is replaced by MuJoCo. See ``tools/SIM_SPEC.md``.

Each control cycle (``1 / control_freq_hz``):

1. the latest command is pushed through a ``command_dispatch_cycles`` delay queue;
2. the applied command picks a mode: cartesian pose (generated IK, nearest to the
   current branch), joint position, or joint velocity;
3. a reference is slewed toward the target within the manifest velocity and
   acceleration limits (accelerate, cruise, brake to arrive);
4. torque = generated gravity compensation + PD tracking, clamped to the effort
   limits, applied to MuJoCo as direct joint torque, MuJoCo is stepped;
5. the state is filled (``q``, ``qd``, applied torques, TCP pose from the
   generated FK, time, mode, fault) and passed through a
   ``state_staleness_cycles`` delay queue.

The PD gains are scaled by the diagonal of the generated mass matrix, so the loop
stays stable at the control rate for both the heavy shoulder and the light wrist.

A rejected command never moves the arm. Faults latch until ``clear_faults()``.
"""

from __future__ import annotations

import importlib
import math
import threading
import time
from collections import deque
from typing import Dict, Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

import urdf_to_mjcf as G

# Fault codes written to ``state.fault_code`` (0.0 = healthy).
FAULT_NONE = 0.0
FAULT_IK_NO_SOLUTION = 1.0
FAULT_IK_STEP_TOO_LARGE = 2.0
FAULT_OUT_OF_LIMITS = 3.0
FAULT_UNSUPPORTED = 4.0

ROBOT_MODE_RUNNING = 7.0  # UR convention

# arm -> (bindings module, MJCF-builder arm key, initial joints)
_ARMS = {
    "ur5": ("dexforge_ur", "ur5", [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]),
}

_UNSUPPORTED_COMMAND_FIELDS = ("target_torques", "target_twist", "target_wrench")


class SimDriver:
    """Simulated driver. Use ``connect()``, then ``write_command`` / ``read_state``."""

    def __init__(
        self,
        arm: str = "ur5",
        *,
        realtime: bool = False,
        wn: float = 30.0,
        zeta: float = 1.0,
        ik_max_joint_step: Optional[float] = None,
        q0: Optional[np.ndarray] = None,
    ) -> None:
        if arm not in _ARMS:
            raise ValueError(f"unknown arm {arm!r}; supported: {sorted(_ARMS)}")
        mod_name, self._arm_key, home = _ARMS[arm]
        self._mod = importlib.import_module(mod_name)
        self._math = getattr(self._mod, arm)
        self.arm = arm
        self.realtime = realtime
        self._wn, self._zeta = wn, zeta

        self._manifest: Dict = self._math.manifest()
        m = self._manifest
        self.dof = int(m["dof"])
        self.dt = 1.0 / float(m["control_freq_hz"])
        self._state_layout = self._mod.state_layout()
        self._cmd_layout = self._mod.command_layout()
        self._qmin = np.array(m["joint_limits_min"])
        self._qmax = np.array(m["joint_limits_max"])
        self._vmax = np.array(m["velocity_limits"])
        self._amax = np.array(m["acceleration_limits"])
        self._effort = np.array(m["effort_limits"])
        self._ik_step = (
            self._math.ik_default_max_joint_step() if ik_max_joint_step is None else float(ik_max_joint_step)
        )
        self._q0 = np.array(home if q0 is None else q0, dtype=float)

        self._lock = threading.Lock()
        # Held while MuJoCo steps; a viewer takes it around ``sync()`` so it never reads a half-step.
        self.physics_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._connected = False

    @property
    def model(self) -> "mujoco.MjModel":
        return self._model

    @property
    def data(self) -> "mujoco.MjData":
        return self._data

    @property
    def threaded(self) -> bool:
        return self._thread is not None

    # ----------------------------------------------------------------- contract

    def manifest(self) -> Dict:
        return dict(self._manifest)

    def connect(self, *, threaded: bool = False) -> None:
        """Load the model and reset. ``threaded=True`` runs the control loop on its own thread."""
        urdf, ee = G.ARMS[self._arm_key]
        self._model = mujoco.MjModel.from_xml_string(G.build_mjcf(urdf, ee, armature=0.0))
        self._data = mujoco.MjData(self._model)
        self._substeps = max(1, round(self.dt / self._model.opt.timestep))
        self._data.qpos[: self.dof] = self._q0
        mujoco.mj_forward(self._model, self._data)

        idle = np.full(self._manifest["command_len"], np.nan)
        self._latest_cmd = idle
        depth = int(self._manifest["command_dispatch_cycles"]) + 1
        self._cmd_queue = deque([idle] * depth, maxlen=depth)
        self._q_ref = self._q0.copy()
        self._qd_ref = np.zeros(self.dof)
        self._tau = np.zeros(self.dof)
        self._fault = FAULT_NONE
        self._fault_epoch = 0  # bumped by clear_faults() so an in-flight cycle cannot re-latch a cleared fault
        self._time = 0.0
        self._cycles = 0
        first = self._make_state()
        depth_s = int(self._manifest["state_staleness_cycles"]) + 1
        self._state_queue = deque([first] * depth_s, maxlen=depth_s)
        self._connected = True
        if threaded:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="sim-driver", daemon=True)
            self._thread.start()

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._connected = False

    def write_command(self, buf) -> None:
        """Latch a command buffer (``CoreCommand`` layout). Never steps physics."""
        buf = np.asarray(buf, dtype=float)
        if buf.size < self._manifest["command_len"]:
            raise ValueError(f"command buffer too small: {buf.size} < {self._manifest['command_len']}")
        with self._lock:
            self._latest_cmd = buf[: self._manifest["command_len"]].copy()

    def read_state(self, buf) -> None:
        """Fill ``buf`` (``CoreState`` layout) with the most recent completed cycle's state."""
        n = self._manifest["state_len"]
        if len(buf) < n:
            raise ValueError(f"state buffer too small: {len(buf)} < {n}")
        with self._lock:
            state = self._state_queue[0]
        buf[:n] = state

    def clear_faults(self) -> None:
        """Clear the latched fault and drop the command that caused it (latched and in flight).

        Without the drop, the rejected command would be applied again on the next cycle and
        re-raise the fault; the client must send a new command.
        """
        idle = self.idle_command()
        with self._lock:
            self._fault = FAULT_NONE
            self._fault_epoch += 1
            self._latest_cmd = idle
            for i in range(len(self._cmd_queue)):
                self._cmd_queue[i] = idle

    def idle_command(self) -> np.ndarray:
        """A command buffer with every field unset (NaN)."""
        return np.full(self._manifest["command_len"], np.nan)

    # ---------------------------------------------------------------- stepping

    def step(self, n: int = 1) -> None:
        """Run ``n`` control cycles synchronously (only when not threaded)."""
        if self._thread is not None:
            raise RuntimeError("step() is for unthreaded use; the control thread is running")
        for _ in range(n):
            self._cycle()

    def _run(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            self._cycle()
            if self.realtime:
                next_t += self.dt
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()

    # ------------------------------------------------------------- the cycle

    def _field(self, cmd: np.ndarray, name: str) -> np.ndarray:
        off, n = self._cmd_layout[name]
        return cmd[off : off + n]

    def _cycle(self) -> None:
        with self._lock:
            self._cmd_queue.append(self._latest_cmd)
            cmd = self._cmd_queue[0]
            fault = self._fault
            epoch = self._fault_epoch

        d = self._data
        q = d.qpos[: self.dof].copy()
        qd = d.qvel[: self.dof].copy()

        if fault == FAULT_NONE:
            fault = self._update_reference(cmd)
        if fault != FAULT_NONE:
            # latched fault: hold the reference (brake it to rest within the acceleration limit,
            # otherwise a stale ``qd_ref`` keeps feeding the PD loop a velocity the arm cannot follow)
            self._advance(np.zeros(self.dof))

        g = np.array(self._math.gravity(q.tolist()))
        m_diag = np.diag(np.array(self._math.mass_matrix(q.tolist())))
        e, ed = self._q_ref - q, self._qd_ref - qd
        tau = g + m_diag * (self._wn**2 * e + 2.0 * self._zeta * self._wn * ed)
        tau = np.clip(tau, -self._effort, self._effort)
        d.ctrl[: self.dof] = tau
        with self.physics_lock:
            for _ in range(self._substeps):
                mujoco.mj_step(self._model, d)

        self._tau = tau
        self._time += self.dt
        self._cycles += 1
        with self._lock:
            if self._fault_epoch == epoch:  # a clear_faults() during this cycle wins over its stale result
                self._fault = fault
            self._state_queue.append(self._make_state())

    def _update_reference(self, cmd: np.ndarray) -> float:
        """Advance ``q_ref`` / ``qd_ref`` for the applied command; return the fault code."""
        if any(np.isfinite(self._field(cmd, f)).any() for f in _UNSUPPORTED_COMMAND_FIELDS):
            return FAULT_UNSUPPORTED

        pose = self._field(cmd, "target_pose")[:7]
        tq = self._field(cmd, "target_q")[: self.dof]
        tqd = self._field(cmd, "target_qd")[: self.dof]
        target: Optional[np.ndarray] = None

        if np.isfinite(pose).all():  # cartesian_pose: generated IK on the current branch
            target_or_fault = self._solve_pose(pose)
            if isinstance(target_or_fault, float):
                return target_or_fault
            target = target_or_fault
        elif np.isfinite(tq).all():  # joint_pos
            target = np.array(tq)
        elif np.isfinite(tqd).all():  # joint_vel: accelerate the reference toward the velocity
            n = self.dof
            v_des = np.clip(tqd, -self._vmax[:n], self._vmax[:n])
            self._advance(v_des)
            return FAULT_NONE
        else:  # nothing commanded: hold
            self._advance(np.zeros(self.dof))
            return FAULT_NONE

        if (target < self._qmin[: self.dof] - 1e-9).any() or (target > self._qmax[: self.dof] + 1e-9).any():
            return FAULT_OUT_OF_LIMITS
        n = self.dof
        err = target - self._q_ref
        # speed that still lets the joint brake to a stop at the target
        v_des = np.sign(err) * np.minimum(self._vmax[:n], np.sqrt(2.0 * self._amax[:n] * np.abs(err)))
        self._advance(v_des)
        crossed = (target - self._q_ref) * err < 0.0  # would pass the target this cycle
        self._q_ref[crossed] = target[crossed]
        self._qd_ref[crossed] = 0.0
        return FAULT_NONE

    def _advance(self, v_des: np.ndarray) -> None:
        """Move ``qd_ref`` toward ``v_des`` within the acceleration limit, then integrate."""
        n = self.dof
        dv = np.clip(v_des - self._qd_ref, -self._amax[:n] * self.dt, self._amax[:n] * self.dt)
        self._qd_ref = self._qd_ref + dv
        q_new = self._q_ref + self._qd_ref * self.dt
        hit = (q_new < self._qmin[:n]) | (q_new > self._qmax[:n])
        q_new = np.clip(q_new, self._qmin[:n], self._qmax[:n])
        self._qd_ref[hit] = 0.0
        self._q_ref = q_new

    def _solve_pose(self, pose: np.ndarray):
        """Joint target for ``pose``, or a fault code."""
        if np.linalg.norm(pose[3:7]) < 1e-9:  # not a rotation; scipy would raise inside the control thread
            return FAULT_IK_NO_SOLUTION
        seed = self._q_ref.tolist()
        q = self._math.ik_nearest(pose.tolist(), seed, self._ik_step)
        if q is not None:
            return np.array(q)
        T = np.eye(4)
        T[:3, :3] = R.from_quat(pose[3:7]).as_matrix()
        T[:3, 3] = pose[:3]
        if not self._math.solve(T.tolist(), respect_limits=True):
            return FAULT_IK_NO_SOLUTION
        return FAULT_IK_STEP_TOO_LARGE

    def _make_state(self) -> np.ndarray:
        buf = np.full(self._manifest["state_len"], np.nan)

        def put(name: str, values) -> None:
            off, n = self._state_layout[name]
            v = np.atleast_1d(np.asarray(values, dtype=float))
            buf[off : off + v.size] = v

        d = self._data
        q = d.qpos[: self.dof]
        put("q", q)
        put("qd", d.qvel[: self.dof])
        put("torques", self._tau)
        put("tcp_pose", self._math.fk_pose(q.tolist()))
        put("timestamp", self._time)
        put("robot_mode", ROBOT_MODE_RUNNING)
        put("fault_code", self._fault)
        return buf
