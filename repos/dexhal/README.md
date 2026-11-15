# dexhal

The contract every dexforge arm is surfaced through, the math all drivers
share, and the tests for both.

`#![no_std]`, `#![forbid(unsafe_code)]`, no allocation, every loop bounded.
Minimum Rust version: 1.82.

## What lives where

dexhal contains no arm-specific code. An arm's forward kinematics, Jacobian,
inverse kinematics and dynamics are generated into its driver crate; dexhal
fixes their shape and supplies what is identical for every arm.

| In the driver crate | In dexhal |
| :-- | :-- |
| `ik_solver.rs`: `fk`, `solve` — the arm's own IK, including FK verification, refinement, de-duplication, joint limits and seed ordering | The trait the driver implements: `ArmModel` |
| `generated_kinematics.rs`: `jacobian` | TCP re-referencing, tool-frame quantities, pseudo-inverse |
| `generated_dynamics.rs`: `gravity`, `mass_matrix`, `coriolis`, `inverse_dynamics` of the arm's own links | Payload contribution to each of them |
| `arm.rs`: `impl ArmModel for <Arm>` delegating to the three files above | `Arm`: the model plus runtime TCP, payload, gravity and calibration |
| `manifest.yaml`, `connect`, `disconnect`, `read_state`, `write_command` | `ArmManifest`, the buffer layouts, `ArmDriver` |
| Tests against the real solver | Tests of the contract and the shared math, on a stand-in arm |

`repos/dexforge-ur` is the reference for the driver side.

## Modules

| Module | Contents |
| :-- | :-- |
| `math::model` | `ArmModel<DOF, MAX>`, `Pose` — what a driver implements |
| `math::arm` | `Arm<A, DOF, MAX>` — one physical arm |
| `math::contract` | `ArmDriver`, `ArmMath`, `ArmManifest`, `CoreState` / `CoreCommand` and their views, `CalibrationData` — what a kernel sees |
| `math::spatial` | Jacobian re-referencing, `J̇ q̇`, pseudo-inverse, base ↔ tool frame helpers |
| `math::payload` | `Payload` and its gravity / inertia / Coriolis / inverse-dynamics terms |
| `math::chain` | `KinematicChain` — per-robot calibrated kinematics, DH import |
| `math::identify` | `ChainIdentifier` — calibration from tracked motion |
| `math::ssik_core`, `kinematics`, `postprocess`, `tolerances` | Numeric kernels used by the drivers' IK solvers |

## Implementing a driver's model

```rust
use dexhal::math::model::{ArmModel, Pose};
use dexhal::math::postprocess::JointLimit;
use dexhal::math::tolerances::TolerancePolicy;

pub struct Ur5;

impl ArmModel<6, 16> for Ur5 {
    const NAME: &'static str = "ur5";
    const JOINT_LIMITS: [JointLimit; 6] = /* from the URDF */;
    const HAS_DYNAMICS: bool = true;

    fn flange_fk(q: &[f64; 6]) -> Pose { ik_solver::fk(q) }
    fn flange_jacobian(q: &[f64; 6]) -> [[f64; 6]; 6] { generated_kinematics::jacobian(q) }
    fn flange_solve(/* target, max_solutions, q_seed, respect_limits, … */) -> [Option<[f64; 6]>; 16] {
        ik_solver::solve(/* … */)
    }

    fn flange_gravity(q: &[f64; 6], g: &[f64; 3]) -> [f64; 6] { generated_dynamics::gravity(q, g) }
    fn flange_mass_matrix(q: &[f64; 6]) -> [[f64; 6]; 6] { generated_dynamics::mass_matrix(q) }
    fn flange_coriolis(q: &[f64; 6], qd: &[f64; 6]) -> [f64; 6] { generated_dynamics::coriolis(q, qd) }
    fn flange_inverse_dynamics(q: &[f64; 6], qd: &[f64; 6], qdd: &[f64; 6], g: &[f64; 3]) -> [f64; 6] {
        generated_dynamics::inverse_dynamics(q, qd, qdd, g)
    }
}
```

Only the three kinematic functions are required. The dynamics hooks default
to zero and are ignored unless `HAS_DYNAMICS` is set; dexhal never assembles
one hook from the others.

dexhal relies on two properties of `flange_solve`: with a seed the solutions
are ordered nearest-first, and with `respect_limits` they lie inside the arm's
limits. All dexforge solvers already behave this way.

## Using an arm

```rust
use dexhal::math::arm::Arm;
use dexhal::math::payload::Payload;
use dexhal::math::tolerances::TolerancePolicy;

let mut arm = Arm::<Ur5, 6, 16>::new();
arm.set_tcp(&tcp)?;                               // flange -> TCP
arm.set_payload(Payload::new(2.5, [0.0, 0.0, 0.06]))?;
arm.set_gravity([0.0, 0.0, -9.80665])?;           // base frame; wall / ceiling mounts

let pose = arm.fk(&q);                            // base -> TCP
let jac = arm.jacobian(&q);                       // [Jv; Jw] at the TCP, base axes
let jac_tool = arm.jacobian_tool(&q);             // same, TCP axes
let all = arm.ik(&target, None, None, true, true, TolerancePolicy::DEFAULT, 15);
let q_next = arm.ik_nearest(&target, &q, TolerancePolicy::DEFAULT);
let dynamics = arm.dynamics(&q, &qd);             // M, C·qd, G incl. payload
let tau = arm.inverse_dynamics(&q, &qd, &qdd);    // incl. payload
```

Setters validate their input and leave the arm unchanged on error.

**Inverse kinematics.** `ik` retargets the TCP and returns the driver's
solutions. `ik_nearest` is the servo form: it asks the driver for the one
solution nearest the seed, moves each joint to the 2π turn nearest the seed
when the limits allow, and returns `None` if any joint would still lie more
than `ik_max_joint_step` (default π/2, `set_ik_max_joint_step`) from the seed.
Near a singularity or a joint limit the nearest remaining solution is on
another branch; the bound turns that into "no solution" instead of a jump.

**Dynamics.** `dynamics` and `inverse_dynamics` return `None` unless the
driver declares `HAS_DYNAMICS`. The payload terms (`payload_gravity`,
`payload_mass_matrix`, `payload_coriolis`) are always available; they are
closed-form in the flange Jacobian. `Payload` carries an optional rotational
inertia (`Payload::new(..).with_inertia(..)`).

**Tool frame.** `math::spatial` moves wrenches and diagonal stiffness /
damping between base and tool axes (`wrench_base_to_tool`,
`diagonal_gain_tool_to_base`, …).

## Calibration

Generated drivers bake in the nominal geometry. A per-robot calibration is a
`KinematicChain<DOF>` — fixed transforms, joint axes and joint zero offsets —
which is a superset of standard DH, modified DH and URDF joint origins.

```rust
use dexhal::math::chain::{apply_dh_deltas, DhConvention, KinematicChain};

// Nominal DH + the per-robot deltas from the vendor's calibration file.
let rows = apply_dh_deltas(&NOMINAL_DH, &deltas);
let chain = KinematicChain::from_dh(DhConvention::Standard, &rows)?
    .with_base(&dh_base_in_driver_base)?          // only if the driver's frames
    .with_flange(&dh_flange_in_driver_flange)?;   // differ from the DH ones
arm.set_calibration(chain)?;
```

Reading a vendor file (for example UR's `calibration.conf`) and knowing the
arm's nominal DH table and frame adapters is the driver's job. DH rows with
`d` terms of hundreds of metres that cancel across near-parallel axes, as UR
files contain, keep full precision.

What changes once a calibration is installed, and what it costs:

- **FK / Jacobian** come from the chain: one fixed `DOF` loop.
- **`ik`** takes the driver's full nominal solution set, polishes each
  solution onto the calibrated chain with at most `CHAIN_IK_MAX_ITERS` (12)
  damped Newton steps, re-verifies it against the caller's
  `policy.subproblem_numerical`, re-applies the limits, de-duplicates and then
  truncates to `max_solutions`.
- **`ik_nearest`** polishes only the one solution the driver returned.
- **Dynamics** of the arm's own links stay nominal; payload terms use the
  calibrated flange.
- **Uncalibrated arms** take the generated path and pay nothing.

`set_calibration` compares the chain with the nominal model at six fixed
configurations and rejects it (`ModelError::ExcessiveDeviation`) beyond
`CALIBRATION_MAX_DEVIATION`. This catches a chain expressed in the wrong base
or flange frame.

`Arm::fk_residual(q, reported_tcp)` — and `ArmDriver::calibration_residual`
on a state buffer — measure the distance between the host model and the TCP
pose the hardware reports. A residual above the robot's repeatability means
the two disagree on calibration or TCP.

Limitation: a target reachable only by the calibrated geometry (within about a
millimetre of the workspace boundary) has no nominal solution to start from
and returns none.

### Arms without vendor calibration

`math::identify::ChainIdentifier` fits a chain to tracked motion: move the arm
through a few dozen configurations while an external tracker measures a
flange-mounted target (full pose or position only).

```rust
// tracker -> base and flange -> target, as well as they are known
let initial = nominal_chain.with_base(&tracker_to_base)?.with_flange(&flange_to_target)?;
let mut ident = ChainIdentifier::new(initial)
    .with_rotation_weight(0.1)                    // metres per radian
    .with_outlier_gate(0.01);                     // skip samples off by > 1 cm
for _ in 0..PASSES {                              // 3-6 passes
    for (q, measured) in samples { ident.add_sample(q, measured); }
    let report = ident.step();                    // RMS, rejected, observable
}
```

The caller owns the samples; the identifier accumulates normal equations, so
it allocates nothing. It is a commissioning tool (≈ 23 kB of state), not a
control-loop routine. An external reference is required — joint encoders alone
cannot observe link geometry.

The fitted chain is a tracker → target map. It is *not* in the driver's
frames, so it cannot be handed to `set_calibration` as is, and removing the
initial guesses of tracker → base and flange → target leaves their errors
inside the chain. To get a chain that is exact in the driver's frames, measure
those two transforms independently and hold them with `fix_base()` /
`fix_flange()`. The report's `observable` count tells whether the motion
excited every joint (`4·DOF + 6` for pose samples). See the module docs.

## Driver contract

`math::contract` is the driver contract the DexZero kernel consumes (`dexzero.md`,
`compliance.md`):

| Role | dexhal item |
| :-- | :-- |
| Version & modes | `MAX_DOF` (8), `ContractVersion`, `CONTRACT_VERSION` (0.2.0), `control_mode` |
| Static capabilities, read once at boot | `ArmManifest` (from `manifest.yaml`) |
| Driver → kernel buffer, every cycle | `CoreState` (56 × f64) + aux tail, `CoreStateView` / `CoreStateViewMut` |
| Kernel → driver buffer, every cycle | `CoreCommand` (96 × f64), `CoreCommandView` / `CoreCommandViewMut` |
| Host-side model & dynamics | `ArmMath` (`Send`), `DynamicsState`, `CalibrationData` (95 × f64) |
| Driver trait | `ArmDriver` (`Send`) |

A hardware driver implements `manifest`, `connect`, `disconnect`, `read_state`,
`write_command` and returns its `Arm` from `arm()` / `arm_mut()`. Everything
else is provided: FK, Jacobian (base and tool axes), pseudo-inverse, IK,
dynamics, inverse dynamics, and the configuration plane.

In DexZero, the real-time executor thread owns `Box<dyn ArmDriver>` and the
planner thread owns an independent `Box<dyn ArmMath>` (`Arm<Model, DOF, MAX>`)
so trajectory scrubbing and IK never contend with the control loop.

### Manifest

The manifest is a data file next to the driver, loaded once at start-up:

```rust
static MANIFEST: &str = include_str!("../manifest.yaml");
let manifest = ArmManifest::from_text::<Ur5, 6, 16>(MANIFEST)?;
```

[`examples/manifest.yaml`](examples/manifest.yaml) documents every key; the
format is a strict subset of YAML and unknown keys are errors. Name, DOF,
position limits and model capabilities come from the `ArmModel`. Velocity,
acceleration and effort limits have no default: the file (or
`ArmManifest::for_model` with a `JointRatings`) must give a finite, positive
value per joint. `limits.jerk` is optional (`NAN` when omitted so the kernel
applies its acceleration / ramp-time fallback), and `has_static_inertials`
defaults to `ArmModel::HAS_DYNAMICS` (`false` disables `HostImpedance` when an
arm obscures URDF inertials or relies on proprietary drive friction models).

### Buffers

`CoreState` (56 × `f64`) and `CoreCommand` (96 × `f64`) are `#[repr(C)]`
all-`f64` structs whose layout is the wire layout. Cartesian wire fields
(`tcp_pose`, `tcp_twist`, `wrench`, `target_pose`, `target_twist`,
`target_wrench`) are at the TCP origin in base axes; `CoreCommand::task_frame`
carries the compliance task frame for delegated native compliance, and
`CoreState` carries `timestamp` (or monotonic sample counter for dynamic
$k$-deep latency alignment), `robot_mode`, `fault_code`, and
`native_reflex_fired`. The data plane works in place through views:

```rust
fn read_state(&mut self, buffer: &mut [f64]) -> Result<(), ArmError> {
    let mut state = CoreStateViewMut::new(buffer)?;
    state.clear();                                // everything NAN = "not provided"
    state.q()[..6].copy_from_slice(&self.q);
    *state.in_contact() = 0.0;
    state.tail().copy_from_slice(&self.motor_temps);   // aux fields
    Ok(())
}
```

Slot offsets come from the struct definitions (`offset_of!`), so there is no
`unsafe` and no copy. `CoreState::read_from` / `write_to` remain for code that
wants an owned value.

### Configuration plane

| `ArmDriver` method | Driver hook | Result |
| :-- | :-- | :-- |
| `set_payload` | `set_payload_native` | `Applied::Native` or `Applied::Host` |
| `set_tcp_offset` | `set_tcp_offset_native` | same |
| `set_gravity` | `set_gravity_native` | same |
| `set_calibration(&CalibrationData)`, `clear_calibration` | — | host model only |
| `set_ik_max_joint_step` | — | host model only |

A change is validated, offered to the controller through the hook, then
applied to the host model, so the two cannot drift apart; a controller fault
leaves the host model untouched. `CalibrationData` is a `KinematicChain` in
flat, `MAX_DOF`-padded form (95 × f64).

`ArmMath` is the DOF-erased form of `Arm` behind `arm()`. Its methods never
share a name with `Arm`'s own: queries with a typed twin end in `_padded`,
configuration methods are `apply_*`.

## Tests

```
cargo test -p dexhal
```

The tests use a made-up 6R arm (`src/math/testing.rs`) that implements
`ArmModel` the way a driver does. Real arms are tested in their own
repositories against their real solvers.

For integration tests of dexhal clients, `mock/rust` (`dexforge-mock-core`,
vendor `"mock"`) provides a `MockDriver` with no-op `connect`/`disconnect`, a
command queue whose depth is the manifest latency, and state computed from the
oldest queued command. The crate ships a client manifest (`MANIFEST`) and a
simple six-R arm model (`MockArm`, NAME `"mock-6r"`), so `MockDriver::default_arm()`
works without any extra setup. Test it with `cargo test -p dexforge-mock-core`.

Open work is tracked in [`workitems.md`](workitems.md).
