"""SimDriver: a simulated arm behind the dexhal driver contract, backed by MuJoCo.

It stands in for a real driver. A client calls ``write_command`` / ``read_state``
with the same flat ``f64`` buffers a real driver uses (layouts and lengths come
from the compiled ``dexforge_<oem>`` bindings, never hard-coded here). All arm
math is our generated code (IK, FK, gravity, mass matrix); only the hardware side
is replaced by MuJoCo. See ``tools/SIM_SPEC.md``.

Each control cycle (``1 / control_freq_hz``):

1. the latest command is pushed through a ``command_dispatch_cycles`` delay queue;
2. the applied command's ``control_mode`` selects the mode: cartesian pose
   (generated IK, nearest to the current branch), joint position, joint velocity,
   or idle (hold). Modes the manifest does not offer, unknown codes and
   non-finite targets are rejected at ``write_command``, like the mock driver;
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
FAULT_SIM_ERROR = 4.0  # the control loop raised (for example physics diverged); the loop has stopped

ROBOT_MODE_RUNNING = 7.0  # UR convention

# arm -> (bindings module, MJCF-builder arm key, initial joints)
_ARMS = {
    "ur5": ("dexforge_ur", "ur5", [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]),
    "ur3e": ("dexforge_ur", "ur3e", [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]),
    # SO-ARM101: five joints (the gripper jaw is not simulated); ready pose, tool pointing forward
    "so101": ("dexforge_lerobot", "so101", [0.0, -0.3, 0.5, -0.2, 0.0]),
}

# PD natural frequency (rad/s) of the tracking controller, per arm; 30 unless listed.
_DEFAULT_WN = {"so101": 20.0}

# Arms whose tracking controller uses the full mass matrix (computed torque) instead of its diagonal.
# Diagonal-mass gains ignore the inertial coupling between joints, which is mild for the UR arms at
# their 500 Hz but is not for the light SO-101 at 100 Hz: with them 8 of 30 random workspace targets
# never settled at 8 rad/s (5 of 30 even at 4 rad/s), while computed torque settled all of them at
# every natural frequency tried up to 30 rad/s (tools/tests/test_sim_so101.py keeps a regression).
_FULL_MASS_CONTROL = {"so101"}

# (position m, orientation rad) a Cartesian command may miss by and still be executed, per arm. Only
# arms that cannot reach every pose need it: the five-joint SO-101 reaches a 5-D subset of SE(3), so a
# pose from a camera, a teleop device or a typed command is almost never exactly reachable. The sim then
# moves to the nearest reachable configuration (``ik_project``) and reports what it reached in
# ``tcp_pose``; a command that misses by more is rejected like an unreachable one. These numbers are a
# simulation default, not a property of the hardware. Arms not listed (the UR arms reach every pose in
# their workspace) keep the exact behaviour.
_DEFAULT_IK_TOLERANCE = {"so101": (0.005, 0.1)}

# command mode -> the target field it needs (finite), for the modes this sim executes
_MODE_TARGET = {"joint_pos": "target_q", "joint_vel": "target_qd", "cartesian_pose": "target_pose"}


class SimDriver:
    """Simulated driver. Use ``connect()``, then ``write_command`` / ``read_state``."""

    def __init__(
        self,
        arm: str = "ur5",
        *,
        realtime: bool = False,
        wn: Optional[float] = None,
        zeta: float = 1.0,
        ik_max_joint_step: Optional[float] = None,
        q0: Optional[np.ndarray] = None,
        ik_tolerance="default",
        full_mass: Optional[bool] = None,
    ) -> None:
        if arm not in _ARMS:
            raise ValueError(f"unknown arm {arm!r}; supported: {sorted(_ARMS)}")
        mod_name, self._arm_key, home = _ARMS[arm]
        self._mod = importlib.import_module(mod_name)
        self._math = getattr(self._mod, arm)
        self.arm = arm
        self.realtime = realtime
        self._wn, self._zeta = (_DEFAULT_WN.get(arm, 30.0) if wn is None else wn), zeta
        self._full_mass = (arm in _FULL_MASS_CONTROL) if full_mass is None else bool(full_mass)

        self._manifest: Dict = self._math.manifest()
        m = self._manifest
        self.dof = int(m["dof"])
        self.dt = 1.0 / float(m["control_freq_hz"])
        self._state_layout = self._mod.state_layout()
        self._cmd_layout = self._mod.command_layout()
        self._modes = {code: name for name, code in self._math.control_modes().items()}  # code -> name
        self._supported = {c for c in self._modes if self._math.supports_mode(c)}  # fixed by the manifest
        self._qmin = np.array(m["joint_limits_min"])
        self._qmax = np.array(m["joint_limits_max"])
        self._vmax = np.array(m["velocity_limits"])
        self._amax = np.array(m["acceleration_limits"])
        if not (np.isfinite(self._amax[: self.dof]).all() and (self._amax[: self.dof] > 0).all()):
            raise ValueError(f"manifest acceleration_limits must be finite and positive, got {self._amax[: self.dof]}")
        self._effort = np.array(m["effort_limits"])
        # (position m, orientation rad), or None for the exact behaviour
        tol = _DEFAULT_IK_TOLERANCE.get(arm) if ik_tolerance == "default" else ik_tolerance
        if tol is not None:
            tol = (float(tol[0]), float(tol[1]))
            if not (all(np.isfinite(tol)) and min(tol) >= 0.0):
                raise ValueError(f"ik_tolerance must be two finite non-negative numbers, got {tol}")
            if not hasattr(self._math, "ik_project"):
                raise ValueError(f"{arm} has no nearest-reachable IK (ik_project); ik_tolerance must be None")
        self._ik_tol = tol
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
        self._sim_error = False

    @property
    def model(self) -> "mujoco.MjModel":
        return self._model

    @property
    def data(self) -> "mujoco.MjData":
        return self._data

    @property
    def ik_tolerance(self):
        """(position m, orientation rad) a Cartesian command may miss by, or ``None`` for exact only."""
        return self._ik_tol

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
        first = self._make_state()
        depth_s = int(self._manifest["state_staleness_cycles"]) + 1
        self._state_queue = deque([first] * depth_s, maxlen=depth_s)
        self._connected = True
        self._sim_error = False
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
        """Latch a command buffer (``CoreCommand`` layout). Never steps physics.

        Raises ``ValueError`` for an unknown ``control_mode`` or a non-finite target of the
        selected mode, and ``NotImplementedError`` for a mode the manifest does not offer
        (the contract's ``Fault`` / ``NotSupported``).
        """
        buf = np.asarray(buf, dtype=float)
        n = self._manifest["command_len"]
        if buf.size < n:
            raise ValueError(f"command buffer too small: {buf.size} < {n}")
        buf = buf[:n].copy()
        self._mode_name(buf)  # validates
        with self._lock:
            self._latest_cmd = buf

    def _mode_name(self, cmd: np.ndarray) -> str:
        """Decode and validate ``control_mode``; return its name (``idle`` for NaN)."""
        code = float(self._field(cmd, "control_mode")[0])
        if math.isnan(code):
            return "idle"
        name = self._modes.get(code)
        if name is None:
            raise ValueError(f"unknown control_mode {code!r}")
        if code not in self._supported:
            raise NotImplementedError(f"control mode {name!r} is not supported by the manifest")
        if name not in _MODE_TARGET:
            raise NotImplementedError(f"control mode {name!r} is not implemented by the simulator")
        target = self._field(cmd, _MODE_TARGET[name])
        if not np.isfinite(target[: 7 if name == "cartesian_pose" else self.dof]).all():
            raise ValueError(f"{_MODE_TARGET[name]} must be finite in control mode {name!r}")
        return name

    def read_state(self, buf) -> None:
        """Fill ``buf`` (``CoreState`` layout) with the most recent completed cycle's state."""
        n = self._manifest["state_len"]
        if len(buf) < n:
            raise ValueError(f"state buffer too small: {len(buf)} < {n}")
        if not self._connected:
            raise RuntimeError("SimDriver is not connected; call connect() first")
        with self._lock:
            state = self._state_queue[0]
        buf[:n] = state
        if self._sim_error:  # the loop died: report it instead of a healthy, frozen state
            off, _ = self._state_layout["fault_code"]
            buf[off] = FAULT_SIM_ERROR

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
        if not self._connected:
            raise RuntimeError("SimDriver is not connected; call connect() first")
        if self._thread is not None:
            raise RuntimeError("step() is for unthreaded use; the control thread is running")
        for _ in range(n):
            self._cycle()

    def _run(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            try:
                self._cycle()
            except Exception:  # a dead thread must not look like a healthy, frozen arm
                import traceback

                traceback.print_exc()
                self._sim_error = True
                return
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
        mass = np.array(self._math.mass_matrix(q.tolist()))
        e, ed = self._q_ref - q, self._qd_ref - qd
        accel = self._wn**2 * e + 2.0 * self._zeta * self._wn * ed
        # diagonal-mass gains (the UR arms), or computed torque with the full mass matrix, which
        # decouples the joints so the gains do not depend on the configuration's inertial coupling
        tau = g + (mass @ accel if self._full_mass else np.diag(mass) * accel)
        tau = np.clip(tau, -self._effort, self._effort)
        d.ctrl[: self.dof] = tau
        with self.physics_lock:
            for _ in range(self._substeps):
                mujoco.mj_step(self._model, d)

        self._tau = tau
        self._time += self.dt
        with self._lock:
            if self._fault_epoch == epoch:  # a clear_faults() during this cycle wins over its stale result
                self._fault = fault
            self._state_queue.append(self._make_state())

    def _update_reference(self, cmd: np.ndarray) -> float:
        """Advance ``q_ref`` / ``qd_ref`` for the applied command; return the fault code."""
        mode = self._mode_name(cmd)
        n = self.dof
        target: Optional[np.ndarray] = None

        if mode == "cartesian_pose":  # generated IK on the current branch
            target_or_fault = self._solve_pose(self._field(cmd, "target_pose")[:7])
            if isinstance(target_or_fault, float):
                return target_or_fault
            target = target_or_fault
        elif mode == "joint_pos":
            target = np.array(self._field(cmd, "target_q")[:n])
        elif mode == "joint_vel":  # accelerate the reference toward the velocity
            v_des = np.clip(self._field(cmd, "target_qd")[:n], -self._vmax[:n], self._vmax[:n])
            self._advance(v_des)
            return FAULT_NONE
        else:  # idle: hold
            self._advance(np.zeros(n))
            return FAULT_NONE

        if (target < self._qmin[: self.dof] - 1e-9).any() or (target > self._qmax[: self.dof] + 1e-9).any():
            return FAULT_OUT_OF_LIMITS
        n = self.dof
        err = target - self._q_ref
        # Fastest speed from which the joint can still stop at the target when its speed may
        # fall by at most amax*dt per cycle (discrete-time braking), capped at vmax.
        a_dt = self._amax[:n] * self.dt
        k = 0.5 * (1.0 + np.sqrt(1.0 + 8.0 * np.abs(err) / (a_dt * self.dt)))
        v_des = np.sign(err) * np.minimum(self._vmax[:n], a_dt * (k - 1.0))
        # Arrive: within one cycle's worth of motion and slow enough to stop in one step. Decided
        # before advancing and used instead of the advance, so the speed change this cycle is
        # the stop itself, at most amax*dt.
        arrived = (np.abs(err) <= a_dt * self.dt) & (np.abs(self._qd_ref) <= a_dt)
        self._advance(v_des)
        self._q_ref[arrived] = target[arrived]
        self._qd_ref[arrived] = 0.0
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
        if self._math.solve(T.tolist(), respect_limits=True):
            return FAULT_IK_STEP_TOO_LARGE  # reachable exactly, but only on a branch too far from here
        if self._ik_tol is not None:
            nearest = self._math.ik_project(pose.tolist(), seed)
            if nearest is not None:
                q, dp, da = nearest
                if dp <= self._ik_tol[0] and da <= self._ik_tol[1]:
                    if np.abs(np.array(q) - np.array(seed)).max() > self._ik_step:
                        return FAULT_IK_STEP_TOO_LARGE
                    return np.array(q)
        return FAULT_IK_NO_SOLUTION

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
