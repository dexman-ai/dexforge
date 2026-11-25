# MuJoCo simulator for DexForge arms: spec (draft)

Status: draft for review. UR5 first; other arms follow the same pattern.

Note on naming: the arm is the **UR5e**, kept under the name `ur5` (the arm name
in the code, manifest and bindings is unchanged).

The **UR3e** (`--arm ur3e`, `dexforge_ur.ur3e`) follows the same pattern. Both arms are imported from
Universal Robots' own description by `tools/import_ur_description.py` (`--arm ur5|ur3e`), and their IK
solve blocks are ported from the ssik artifacts by `tools/port_ur_ik_solver.py`.

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

1. Take the latest command. The mode is the explicit `control_mode` field (as in
   the dexhal contract: NaN = idle/hold, 1 = `joint_pos`, 2 = `joint_vel`,
   4 = `cartesian_pose`), never inferred from which targets are finite. Supported
   in the first release: those three. `write_command` rejects an unknown code,
   a mode the manifest does not offer, and a non-finite target for the selected
   mode, like the mock driver (`ValueError` / `NotImplementedError` here). Other
   command fields (gains, gripper, IO, feedforward) are ignored.
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
  in the sim (1 = IK no solution, 2 = IK step exceeds bound, 3 = target outside
  limits); `0.0` means healthy.
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

- Built from our own URDF (`ur5.urdf`), which is **Universal Robots' own UR5e description**
  (`ur_description`), imported by `tools/import_ur_description.py` at a pinned commit and
  with UR's meshes. Kinematics (0.1625, 0.425, 0.3922, 0.1333, 0.0997, 0.0996 m), link masses,
  centres of mass, inertia tensors and joint limits are all UR's. The script's float-noise snap
  (largest change ~2e-10) and the base-frame choice are documented in its docstring.
- UR does not publish joint acceleration limits, an actuator model, rotor inertia, or joint
  damping and friction, so none of those are in the model: the sim's 15 rad/s2 acceleration
  limit is a placeholder, motors are our own direct-torque ones, and armature is 0. Set
  `armature=0` when cross-checking `mass_matrix` against MuJoCo (the generated dynamics have
  no rotor inertia).
- Needs link masses and inertias (the URDF has them, from UR), joint limits, and
  one direct-torque motor per joint (ctrlrange = effort limit). Gravity on.
- The model's FK is checked against the generated `fk` for random joints
  (existing check in `tools/visualize_ik.py`); a mismatch is a hard error.
- Frame: the solver's base is the UR controller *base* frame (UR's `base`, which is `base_link`
  turned 180 deg about Z; q = 0 puts the TCP at x = -0.8172 m). Our `base_link` is that frame, so
  poses match what the controller reports. The MuJoCo model hangs the arm off
  `base_link_inertia`, which has the same orientation.

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
1. Bindings expose the dynamics, manifest, control modes and buffer layouts
   (`dexforge_ur`, built from `tools/sim_bindings`; sim-only, since the OEM crates
   no longer ship Python bindings).
2. MJCF generator: `tools/urdf_to_mjcf.py` builds the model from `ur5.urdf` with
   inertials, joint limits, gravity, direct-torque motors and a `tcp` site.
   `tools/tests/test_mjcf_dynamics.py` checks generated gravity, mass matrix, Coriolis and FK
   against MuJoCo, and that gravity compensation holds the arm still.

3. `SimDriver` (`tools/sim_driver.py`): control loop, command/state buffers, the
   three control modes selected by `control_mode`, IK fault handling, latency queues, optional control
   thread. `tools/tests/test_sim_driver.py` covers gravity hold, joint-position
   round trip, velocity mode, explicit mode selection, idle hold, cartesian command
   vs achieved pose, command rejection (unknown / unsupported mode, non-finite
   target), IK faults (no solution, step too large, out of limits), latching until
   `clear_faults()`, streamed trajectories, and the latency shift.
4. Bindings `fk_pose` and `ik_nearest` (the contract's IK with its continuity
   bound, default pi/2).

5. `tools/sim.py`: viewer, headless and render modes, and the demo client.

Streamed-trajectory tests are done (in `test_sim_driver.py`).

To do: a client that drives `SimDriver` through the real dexhal `ArmDriver` path.
Replaying the Rust IK reference cases was dropped on purpose: they are single
poses, not what a robot client sends.

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
- **SO-ARM101** (`--arm so101`, vendor crate `repos/dexforge-lerobot`, bindings `tools/sim_bindings_lerobot`,
  Python module `dexforge_lerobot`): a *five-joint* arm (the gripper jaw is folded into the end link and not
  simulated). Differences from the UR arms, all deliberate:
  - **DOF 5.** Its closed-form IK comes from a 6R embedding (the arm plus a zero-offset virtual joint, see the
    crate README). The arm reaches only a 5-D subset of poses, so a Cartesian command is executed on the exact
    solution when there is one, and otherwise on the *nearest reachable configuration* (`ik_project`) provided it
    misses the commanded pose by no more than `SimDriver(ik_tolerance=(position m, orientation rad))`, default
    (5 mm, 0.1 rad) for this arm (`sim_driver._DEFAULT_IK_TOLERANCE`; a simulation default, not a hardware
    property). A command that misses by more raises `FAULT_IK_NO_SOLUTION` and holds, as before. `ik_tolerance=None`
    restores exact-only behaviour. The achieved pose is in `tcp_pose`; the sim does not fault for a miss that is
    within tolerance. The UR arms are unaffected (they reach every pose in their workspace).
  - **100 Hz manifest** (assumed, not a measured controller rate). Tracking uses computed torque with the full mass
    matrix (`sim_driver._FULL_MASS_CONTROL`) at 20 rad/s (`_DEFAULT_WN`). The UR arms' diagonal-mass PD does not work
    here: with it 8 of 30 random workspace targets never settled at 8 rad/s (5 of 30 even at 4), because it ignores the
    coupling between joints. Computed torque settled all 30 at every frequency tried up to 30 rad/s.
  - **Manifest limits are placeholders**: velocity / effort are the CAD export's 10 rad/s and 10 N*m, not STS3215
    ratings, and acceleration is an assumed 20 rad/s^2.
  - Meshes: one decimated STL per link from SO-ARM100 (Apache-2.0), `tools/meshes/so101/`.
  Tests: `tools/tests/test_sim_so101.py` (bindings, generated dynamics vs MuJoCo, SimDriver).

## Open questions

- Confirm with the owner of the UR5 driver what the real driver sends (position
  or velocity targets) and whether a valid command clears a fault.
- Whether DexZero needs a Rust plugin after all (see Decisions).
