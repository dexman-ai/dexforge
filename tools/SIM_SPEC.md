# MuJoCo simulator for DexForge arms: spec (draft)

Status: draft for review. UR5 first; other arms follow the same pattern.

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
     joints, then run as `joint_pos`. If IK has no solution or jumps more than
     the continuity bound, keep the last target and report a fault.
2. Compute actuator commands. Torque = PD tracking + gravity compensation from
   the **generated dynamics** (`gravity(q)`), so the generated dynamics are on
   the path, not just MuJoCo's own.
3. Step MuJoCo for the cycle.
4. Fill the state: `q`, `qd`, applied joint torques, TCP pose from the
   generated `fk`, timestamp, mode and fault fields. Unsupported fields (wrench,
   gripper, I/O) are NaN.
5. Latency: apply the manifest's `command_dispatch_cycles` and
   `state_staleness_cycles` as delay queues, as the mock does.

## Model

- Built from our own URDF (`ur5.urdf`) with our UR5 meshes. Not Menagerie's UR5e,
  whose kinematics differ from our UR5.
- Needs link masses and inertias (the URDF has them for UR5), joint limits, and
  position actuators. Gravity on.
- The model's FK is checked against the generated `fk` for random joints
  (existing check in `tools/visualize_ik.py`); a mismatch is a hard error.
- Frame offsets (UR5's base is rotated 180° about Z in the solver) are fitted
  and printed, not hidden.

## Usage

```
python tools/sim.py --arm ur5            # start the sim, open the viewer
```

A client then writes commands / reads state through the driver buffers. The
existing `tools/visualize_ik.py` remains the kinematics-only viewer.

## Checks

- Round trip: command a joint target, step the sim, state converges within
  tolerance; tracking error reported.
- Cartesian command: commanded TCP pose vs achieved pose.
- Gravity compensation: arm holds still at rest within tolerance.
- Contract: buffer lengths match `manifest.state_len()` / `command_len()`.

## Needed before implementing

1. Bindings expose the dynamics and the buffer layout (today only `fk` and
   `solve`): `jacobian`, `gravity`, `mass_matrix`, `coriolis`, manifest, field
   offsets.
2. MJCF generator with inertia and actuators (today kinematics only, no mass).
3. The simulator loop and the checks above.

## Open questions

- Does DexZero want this as a simulated driver plugin (would need Rust), or is
  it a standalone tool? Python is the default until that is answered.
- Which control modes beyond the first three, and is 500 Hz the right sim rate?
- Torque model: PD + gravity compensation is a guess; confirm with the owner of
  the UR5 driver.
- Other arms: dynamics are wired only for UR5, Piper and Z1. Rizon has no
  inertials, so it could only run kinematically.
