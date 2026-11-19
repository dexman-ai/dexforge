# MuJoCo simulator for DexForge arms: spec (draft)

Status: draft for review. UR5 first; other arms follow the same pattern.

Note on naming: the arm is the **UR5e**, kept under the name `ur5` (the arm name
in the code, manifest and bindings is unchanged).

## Goal

Run an arm in simulation through the same driver contract a real arm uses.
A client starts, picks UR5, and interacts with it. All kinematics and
dynamics come from our **generated** code (IK, FK, Jacobian, gravity, mass
matrix, Coriolis). Only the part that would talk to the real robot is replaced:
it talks to MuJoCo instead.

## Layout

```
client / dexhal consumer
        |  write_command(buf) / read_state(buf)      <- dexhal ArmDriver contract
        v
SimDriver   (Python, standalone)                     <- this component
   |  math: fk / ik / jacobian / gravity / mass / coriolis
   |        from the generated Rust, via pyo3 bindings (dexforge_<oem>)
   |  I/O : MuJoCo (mujoco Python package)
   v
MuJoCo model of the arm (URDF-derived, our UR5 meshes)
```

- The simulator is a standalone Python component. It does not link Rust code
  itself; it only calls the generated math through the existing bindings and
  speaks the driver contract's buffer layout.
- `dexhal/mock` stays as it is (a simple test double). MuJoCo does not go into
  `dexhal` core.
- It can be moved to a Rust crate later without changing the boundary below.

## Boundary (the contract the SimDriver honors)

Mirrors `dexhal::math::ArmDriver`:

| Call | Behavior |
|---|---|
| `manifest()` | Arm manifest (`repos/dexforge-<oem>/rust/manifest.yaml`): DOF, control modes, rate, limits |
| `connect()` / `disconnect()` | Load / release the MuJoCo model |
| `write_command(buf)` | Accept a flat `f64` buffer laid out as `CoreCommand` (`MAX_DOF = 8` slots per joint array); unset fields are NaN |
| `read_state(buf)` | Fill a flat `f64` buffer laid out as `CoreState` (`q`, `qd`, `torques`, `tcp_pose`, ...) |
| `arm()` | The host model: generated `fk`, `ik`, `jacobian`, `gravity`, `mass_matrix`, `coriolis` |

Buffer offsets must come from the contract (exposed by the bindings), not be
hard-coded in the simulator.

## Behavior

Per control cycle (manifest `control_freq_hz`, 500 Hz for UR5):

1. Take the latest command. Supported modes, first release: `joint_pos`,
   `joint_vel`, `cartesian_pose`.
   - `cartesian_pose`: solve with the **generated IK**, nearest to the current
     joints, then run as `joint_pos`. On failure see *IK fault handling*.
2. Slew a reference toward the target within `limits.velocity` and
   `limits.acceleration` (accelerate, cruise, brake to arrive), so a step command
   does not become a torque spike.
3. Compute actuator commands. Torque = PD tracking + gravity compensation from
   the **generated dynamics** (`gravity(q)`), written as direct joint torques
   (not MuJoCo position actuators), clamped to `limits.effort`. The PD gains are
   scaled by the diagonal of the generated `mass_matrix(q)` (natural frequency
   30 rad/s, damping ratio 1): fixed gains cannot be stable at 500 Hz for both
   the shoulder and the light wrist. The generated dynamics are on the path, not
   just MuJoCo's own.
4. Step MuJoCo for the cycle (two 1 ms physics steps).
5. Fill the state: `q`, `qd`, applied joint torques, TCP pose from the
   generated `fk`, timestamp, mode and fault fields. Unsupported fields (wrench,
   gripper, I/O) are NaN.
6. Latency: apply the manifest's `command_dispatch_cycles` and
   `state_staleness_cycles` as delay queues, as the mock does.

### IK fault handling

- **Continuity bound:** set through the model's `apply_ik_max_joint_step(step)`
  (rad per IK call). Default: the Rust model's own default, which is what the sim
  uses. `SimDriver(ik_max_joint_step=...)` is an explicit opt-in override, used by
  tests to provoke the fault and available for tuning; it is never on by default,
  so the sim is a faithful stand-in unless asked otherwise.
- **On failure** (no solution, or the best solution exceeds the bound): the
  command is rejected, the previous joint target is held, and
  `state.fault_code` is set to a non-zero code. Define the codes in one table
  in the sim (1 = IK no solution, 2 = IK step exceeds bound, 3 = command
  outside limits); `0.0` means healthy.
- **Recovery:** `clear_faults()` resets `fault_code` to `0.0`. A fault is also
  cleared by the next valid command only if the driver contract says so;
  until that is confirmed, require an explicit `clear_faults()`.
- A rejected command never moves the arm and never leaves the host model out
  of sync with MuJoCo.

### Who owns time

- The sim owns the clock: a background thread runs the control cycle at
  `control_freq_hz` and substeps MuJoCo inside each cycle (for example 2 ms
  cycle, 1 ms physics step).
- `write_command` only latches the latest command (as a real driver does) and
  `read_state` returns the most recent completed cycle's state. Neither call
  steps physics.
- Real-time pacing is optional (`--realtime`, default on with the viewer). Tests
  run with pacing off, or step explicitly, so they run faster than real time.
- The latency delay queues (`command_dispatch_cycles`, `state_staleness_cycles`)
  are counted in cycles, so they behave the same with pacing on or off.

## Model

- Built from our own URDF (`ur5.urdf`), which is the **UR5e imported from MuJoCo
  Menagerie** (`tools/import_menagerie_ur5e.py`), with Menagerie's meshes. The
  generated FK matches Menagerie's own MuJoCo model to ~1e-16 (position and
  orientation), so sim and solver agree by construction. Menagerie rounds some
  dimensions (0.163 vs 0.1625 m shoulder, 0.1 vs 0.0997 / 0.0996 m wrist, 0.392 vs
  0.3922 m forearm): the model follows Menagerie, not the datasheet, and a real arm
  carries its own calibration anyway.
- Menagerie's actuators and 0.1 joint armature are **not** used: torques are
  commanded directly. Set `armature=0` when cross-checking `mass_matrix` against
  MuJoCo (the generated dynamics have no rotor inertia).
- Needs link masses and inertias (the URDF has them, from Menagerie), joint limits, and
  position actuators. Gravity on.
- The model's FK is checked against the generated `fk` for random joints
  (existing check in `tools/visualize_ik.py`); a mismatch is a hard error.
- Frame: the solver's base is the UR *base* frame (Menagerie's world frame, x
  pointing away from the cable). The MuJoCo model hangs the arm off
  `base_link_inertia`, a fixed 180° yaw of `base_link`, as `ur_description` does.
  Any remaining fitted offset is printed, not hidden.

## Usage

```
mjpython tools/sim.py --arm ur5              # viewer (macOS needs mjpython)
python   tools/sim.py --arm ur5 --headless   # no window; demo prints tracking errors
python   tools/sim.py --arm ur5 --render out.png
python   tools/sim.py --arm ur5 --no-demo    # arm just holds against gravity
```

`sim.py` starts a `SimDriver` and a demo client that talks to it only through
the contract buffers (`write_command` / `read_state`): joint waypoints, then
cartesian poses solved by the generated IK. The existing `tools/visualize_ik.py`
remains the kinematics-only viewer.

## Checks

- Round trip: command a joint target, step the sim, state converges within
  tolerance; tracking error reported.
- Cartesian command: commanded TCP pose vs achieved pose.
- Gravity compensation: arm holds still at rest within tolerance.
- Contract: buffer lengths match `manifest.state_len()` / `command_len()`.
- **Dynamics cross-check (gravity):** at rest, for random `q`, the generated
  `gravity(q)` must match MuJoCo's own bias force (`qfrc_bias` with `qvel = 0`)
  within tolerance. A mismatch means the MJCF inertials or the generated
  dynamics are wrong, and it is reported with the worst joint and error. Add the
  same check for `mass_matrix` against MuJoCo's `mj_fullM`.
- IK faults: an unreachable pose sets `fault_code`, holds the arm, and
  `clear_faults()` recovers.
- Timing: with pacing off, N cycles produce N states; delay queues shift
  commands and states by exactly the manifest cycle counts.

## Status

Done:
1. Bindings expose the dynamics, manifest and buffer layout (`dexforge_ur`).
2. MJCF generator: `tools/urdf_to_mjcf.py` builds the model from `ur5.urdf` with
   inertials, joint limits, gravity, direct-torque motors and a `tcp` site.
   `test_mjcf_dynamics.py` checks generated gravity, mass matrix, Coriolis and FK
   against MuJoCo, and that gravity compensation holds the arm still.

3. `SimDriver` (`tools/sim_driver.py`): control loop, command/state buffers, the
   three control modes, IK fault handling, latency queues, optional control
   thread. `test_sim_driver.py` covers gravity hold, joint-position round trip,
   velocity mode, cartesian command vs achieved pose, IK faults (no solution,
   step too large, out of limits, unsupported field), latching until
   `clear_faults()`, and the latency shift.
4. Bindings `fk_pose` and `ik_nearest` (the contract's IK with its continuity
   bound, default pi/2).

5. `tools/sim.py`: viewer, headless and render modes, and the demo client.

To do: replay the Rust IK reference cases through `SimDriver`, a streamed
trajectory test, and a client that drives it through the real dexhal `ArmDriver`
path.

## Decisions

- **Standalone Python component** (not a Rust plugin). The buffer-layout
  boundary stays the same if it is ported later. If DexZero turns out to load
  only Rust drivers, add a thin shim or a shared-memory bridge.
- **Rate:** the manifest's `control_freq_hz` is the control rate (500 Hz for
  UR5); MuJoCo substeps inside each cycle.
- **Torque model:** PD tracking + generated gravity compensation as direct
  torques. Computed-torque control (mass matrix, Coriolis) only when a mode
  needs it. The gravity cross-check above guards the dynamics.
- **Control modes, first release:** `joint_pos`, `joint_vel`, `cartesian_pose`.
  No torque mode until these pass their checks.
- **Arms:** UR5 first. Piper next, after deciding whether its MJCF is generated
  from our new-revision URDF or Menagerie's old-revision model is patched.
  Rizon stays kinematic-only (no inertials).

## Open questions

- Confirm with the owner of the UR5 driver what the real driver sends (position
  or velocity targets) and whether a valid command clears a fault.
- Whether DexZero needs a Rust plugin after all (see Decisions).
