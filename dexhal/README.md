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
| Version & modes | `MAX_DOF` (8), `ContractVersion`, `CONTRACT_VERSION` (0.4.0), `ActiveMode` |
| Static capabilities, read once at boot | `ArmManifest` (from `manifest.yaml`) |
| Driver → kernel buffer, every cycle | `CoreState` (56 × f64) + aux tail, `CoreStateView` / `CoreStateViewMut` |
| Kernel → driver buffer, every cycle | `CoreCommand` (109 × f64, no tail), `CoreCommandView` / `CoreCommandViewMut` |
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
applies its acceleration / ramp-time fallback). `has_static_inertials`
defaults to `ArmModel::HAS_DYNAMICS` (`false` disables Route 3 `HostImpedance`
when an arm obscures URDF inertials or relies on proprietary drive friction
models), `has_residual_inertials` defaults to `has_static_inertials` (allowing
Route 6 `HostAdmittanceTorque` torque-residual wrench estimation to stay
enabled when `has_static_inertials: false`), and both are validated against
`ArmModel::HAS_DYNAMICS` (setting either to `true` when `!HAS_DYNAMICS` is
rejected). `dynamics_source` (`model` default, `vendor` / `aux`, or `learned`) declares
whether per-tick control routes use `ArmModel` SymPy dynamics, controller
dynamics (`gravity_torque`, `coriolis_torque`, `mass_matrix`) passed through in
the `CoreState` `aux` tail (`ArmManifest::aux_dynamics`), or a learned torque
observer (below). `has_torque_observer` is not a manifest key: the driver sets
it when it installs an observer, and it also enables `has_residual_dynamics()`.

### Learned torque observer (FACTR 2)

`math::torque_observer::LstmObserver` is a `no_std`, allocation-free LSTM
(≤ 2 layers × 64, optional 64-wide MLP head, sliding window ≤ 50 or a carried
stream) that predicts the free-space link torque from `[q, q̇, q_cmd − q]`,
`[q, q̇, q̈_ref]` or `[q, q̇]`. Weights are a `.dexnext` file
([`specs/factr2.md`](specs/factr2.md) §2.4) decoded in place from a
`&'static [u8]`, so they never cross the plugin boundary:

```rust
static NEXT: &[u8] = include_bytes!("ur5.dexnext");
arm.set_torque_observer(NEXT)?; // ModelError::InvalidObserver if malformed
manifest.has_torque_observer = true;
```

`Arm::step_free_torque(q, q̇, q̈_ref, q_cmd, control_freq_hz)` returns
$\hat{\tau}_{\text{free}} = \hat{\tau}_{\text{nn}} (+\,\tau_{\text{model}}$ in
`FrictionResidual` mode$) + \tau_{\text{payload}}(P_{\text{runtime}}) -
\tau_{\text{payload}}(P_{\text{train}})$. The network is fixed to FACTR 2's
configuration (2 × 128 LSTM, 256 MLP head, stateless 50-sample window at
100 Hz): a window is evaluated every `round(control_freq_hz / 100)` ticks,
its work spread over the period, and held in between (`None` for the first
`stride − 1` ticks after a reset);
`ArmDriver::step_free_torque` falls back to the model or vendor `aux` dynamics
when no observer is installed. Call `reset_torque_observer()` on connect,
fault clear, freedrive toggles and reference snaps. `set_gravity` rejects a
gravity that differs from the `FullDynamics` training gravity.

Weights are trained with `codegen/next` (`pip install .[next]` for PyTorch)
from a free-space log recorded by `dexzero` (spec §3.1; for `FrictionResidual`
the log carries the driver's own `inverse_dynamics` as `tau_model`):
`train_next.py` trains and exports with a PyTorch/NumPy parity check, and
`make_fixtures.py` regenerates the Rust parity fixture
`tests/fixtures/next/parity.expected`.

### Buffers

`CoreState` (56 × `f64`) and `CoreCommand` (109 × `f64`) are `#[repr(C)]`
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

`write_command` is one `match` on `CoreCommandView::active_mode()`
(`ActiveMode`, carried in `CoreCommand::control_mode`; `NAN` = `Idle`, which
the driver answers with the hardware's hold or zero-damped command). Each mode
names the fields the driver forwards, so precedence between finite targets is
never guessed; peripheral fields (digital / analog outputs, gripper, reflex
thresholds) apply whenever they are finite. There is no command-side tail:
vendor features outside the layout go through
`handle_vendor_command(name, payload, response)`, which writes its reply into
the caller's buffer and returns its length.

`ArmDriver::external_wrench(state)` turns a reported `CoreState::wrench` into
the external wrench at the TCP in base axes, using `sensing.wrench_frame` and
removing the configured payload weight unless the sensor is already
gravity-compensated; `ArmManifest::link_torques` applies
`sensing.torque_constants` to raw motor currents.

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

## Driver implementation tiers: Minimal vs. Advanced

A hardware driver is a pass-through between the wire buffers (`CoreState`,
`CoreCommand`) and the robot's transport (CAN, EtherCAT, UDP, RTDE, vendor SDK).
All kinematics, dynamics, compliance integration, force conditioning, latency
compensation, and trajectory scrubbing live in `dexhal` and the DexZero kernel.
Because the kernel inspects `ArmManifest` at boot and automatically falls back
to its host model whenever a hardware capability is absent or a `CoreState`
slot is left `NAN`, a **minimal pass-through driver** (for example, a raw CAN
bus reading encoder counts and motor phase currents) can participate in full
Cartesian motion, compliant force control, and protective reflexes without
implementing any control theory or coordinate transforms in the driver.

An **advanced controller driver** (for example, Franka `libfranka`, Flexiv RDK,
or Universal Robots RTDE + URScript `force_mode`) can additionally declare the
loops its hardware controller closes natively so the kernel delegates those
loops directly.

### Knob-by-knob comparison (Minimal vs. Advanced alternatives)

| Knob / Capability | Minimal driver (pass-through) | Advanced driver (controller-delegated) | What the kernel does |
| :-- | :-- | :-- | :-- |
| **Kinematics (`tcp_pose`, `tcp_twist`)** | Leave `CoreState::tcp_pose` and `tcp_twist` as `NAN`. Report only `q` and `qd`. | Populate `tcp_pose` and `tcp_twist` at the TCP origin in base axes if the controller already computes them. | Computes `fk(q_actual)` and `J(q_actual) · qd_actual` from the host `ArmModel` whenever `NAN`; uses hardware `tcp_pose` for `calibration_residual` self-checks when present. |
| **Motion command mode (`control_modes`)** | Implement a single joint mode: `joint_pos`, `joint_vel`, or `joint_torque`. | Expose multiple modes (`joint_pos`, `joint_vel`, `joint_torque`, `cartesian_pose`, `cartesian_twist`, `cartesian_wrench`) and switch on `CoreCommand::control_mode` (`native.realtime_mode_switch`). | Selects the best available output mode at boot and streams dense joint or Cartesian references at `control_freq_hz`. |
| **Joint acceleration (`target_qdd`)** | Ignore `target_qdd`. | Forward `target_qdd` to the controller's feedforward acceleration input (e.g., `servoj` / `speedj` / CAN feedforward). | Always evaluates `reference.qdd` from the quintic / reactive generator and uses it in host inverse dynamics. |
| **Joint torque sensing (`sensing.torque`, `CoreState::torques`)** | Read motor phase current $i$ and write link-side effort $\tau = k_t \cdot n_{\text{gear}} \cdot i$ (a single constant multiply per joint, or declare `sensing.torque_constants` in `manifest.yaml` and pass raw motor current). Includes gravity and payload. | Report link-side torque sensor readings in the positive joint direction (including gravity and payload). | Runs the torque-residual observer (Route 6): $r = \tau_{\text{model}}(q_{\text{actual}}, \dot{q}_{\text{actual}}, \ddot{q}_{\text{ref}}) - \tau_{\text{measured}}$ and projects $F_{\text{ext}} = (J^T)^+ r$ onto the TCP. |
| **Wrist F/T sensing (`sensing.wrench`, `CoreState::wrench`)** | Copy the 6 raw sensor floats `[Fx, Fy, Fz, Tx, Ty, Tz]` straight into `CoreState::wrench`; set `sensing.wrench_frame: flange` (or `tcp`) and `sensing.wrench_is_gravity_compensated: false` in `manifest.yaml`. No `fk(q)` in the driver. | Report wrench already transformed to the TCP origin in base axes (`sensing.wrench_frame: base`), or set `sensing.wrench_is_gravity_compensated: true` if the controller subtracts gravity internally. | Rotates the wrench into the compliance frame, shifts moments to the TCP, subtracts configured payload gravity and inertial load, applies low-pass filtering + deadband, and handles `tare_wrench()`. |
| **Compliance & force control (`native.*_impedance`, `native.*_admittance`)** | Set all `native.*` compliance flags to `false`. | Set `native.cartesian_impedance`, `native.cartesian_admittance`, or `native.joint_impedance` (MIT 5-tuple `(q_ref, qd_ref, kp, kd, tau_ff)`) to `true` and forward `task_frame`, `task_stiffness_k`, `task_damping_b`, `task_mass_m`, `target_wrench`, or `joint_stiffness_k` / `joint_damping_b`. | Selects one of the 7 compliance routes (`compliance.md` §2–3). For minimal drivers, closes the loop on the host via `HostImpedance` (Route 3), `HostAdmittanceWrench` (Route 5), or `HostAdmittanceTorque` (Route 6). |
| **Static link inertials & runtime dynamics (`has_static_inertials`, `has_residual_inertials`, `dynamics_source`)** | Leave at defaults (`has_static_inertials` and `has_residual_inertials` default to `ArmModel::HAS_DYNAMICS`; `dynamics_source: model`). Both flags must be `false` when `ArmModel::HAS_DYNAMICS` is `false`. | Set `has_static_inertials: false` when URDF inertials are obscured (e.g., Flexiv) or dominated by harmonic-drive friction (e.g., UR); optionally keep `has_residual_inertials: true` if static URDF dynamics are still usable for Route 6 torque-residual wrench estimation, or set `dynamics_source: vendor` and pass through `gravity_torque` (`DOF`), `coriolis_torque` (`DOF`), and `mass_matrix` (`DOF * DOF`) in `CoreState` `aux`, or set `dynamics_source: learned` and install a FACTR 2 `.dexnext` torque observer (`Arm::set_torque_observer`, `has_torque_observer`). | Uses `has_impedance_dynamics()` to gate Route 3 (`HostImpedance`) and `has_residual_dynamics()` to gate Route 6 (`HostAdmittanceTorque`). When `dynamics_source: vendor`, per-tick control routes read $(M, C\dot{q}, G)$ from `aux` (`ArmManifest::aux_dynamics`, plus host payload when `native.payload == false`), while planning and simulation continue to query `ArmMath` at arbitrary poses. |
| **Gravity compensation in `JOINT_TORQUE` mode** | Set `native.gravity_compensation: false`. Driver forwards `target_torques` straight to the actuators. | Set `native.gravity_compensation: true` if the controller's torque mode already adds link gravity/friction internally (e.g., `libfranka`, Flexiv RDK). | Includes $\tau_{\text{model}}(q_{\text{actual}}, \dot{q}_{\text{actual}})$ in `target_torques` when `false`; sends only task/impedance torque (plus host payload gravity if `Applied::Host`) when `true`. |
| **7-DOF null-space posture** | Leave `native.nullspace_posture: false`. | Set `native.nullspace_posture: true` and read `CoreCommand::target_q` alongside `target_pose` during `NativeCartesianImpedance` / `NativeCartesianAdmittance`. | On host routes (Routes 3–6), projects $\tau_{\text{null}}$ or commands all 7 joints directly. On native Cartesian routes, populates `target_q = reference.q` when `native.nullspace_posture` is `true`. |
| **Protective reflexes (`native.contact_detection`, `native.reflexes`)** | Set both to `false`; leave `in_contact` and `native_reflex_fired` as `NAN`. | Report `in_contact` (`0.0`/`1.0`), accept `reflex_threshold` and `reflex_action_code` (`STOP`/`YIELD`/`RETRACT`), and set `CoreState::native_reflex_fired = 1.0` when a controller reflex or script triggers. | Monitors the conditioned host wrench (from sensor or motor-current residual) every cycle and triggers `STOP`, `YIELD`, or `RETRACT` on the host; coordinates with `native_reflex_fired` when delegated. |
| **Tool, payload & gravity (`native.payload`, `native.tcp_offset`, `set_*_native`)** | Leave `set_payload_native`, `set_tcp_offset_native`, and `set_gravity_native` at their default `Err(ArmError::NotSupported)`. | Implement the `set_*_native` hooks to inform the controller's internal kinematics/gravity model and return `Ok(())` (`Applied::Native`). | Applies the change to the host `Arm` model in both cases; when `Applied::Host`, compensates TCP offset, payload dynamics, and gravity direction entirely on the host. |
| **Latency & timing (`CoreState::timestamp`, `latency.*_cycles`)** | Declare static `state_staleness_cycles` and `command_dispatch_cycles` in `manifest.yaml`; write a packet counter or bus timestamp to `CoreState::timestamp` if available, else `NAN`. | Stamp `CoreState::timestamp` with the hardware clock (s) or monotonic sample counter on every packet. | Aligns delayed measurements against its $K$-deep reference history ring using `timestamp` (falling back to static `state_staleness_cycles` when `NAN`) and rolls the state forward to $(q_{\text{actual}}, \dot{q}_{\text{actual}})$. |
| **Watchdog / command stop** | No software timer thread in the driver. Configure the motor drive's hardware watchdog register in `connect()` if present; on `CoreCommand::IDLE` (all `NAN`), send a static hold or zero-damped command (`kp = 0, kd = kd_safe, tau = 0`). | Configure the controller's native stream watchdog (e.g., RTDE / `libfranka` timeout) in `connect()` and report timeout faults in `CoreState::fault_code`. | Streams a jerk-limited `stop()` whenever a client disconnects or a command expires; relies on the hardware watchdog only if the kernel process itself dies. |
| **Lifecycle & faults (`connect`, `disconnect`, `clear_faults`)** | `connect()` opens the bus and enables drives; `disconnect()` disables drives and closes the bus; report drive faults in `CoreState::fault_code`. No separate `enable`/`disable` state machine. | Same 3 lifecycle hooks (`connect`, `disconnect`, `clear_faults`), plus detailed `CoreState::robot_mode` and `fault_code` reporting from the controller. | Latches faults from `CoreState::fault_code` or `Err(ArmError::Fault(..))`, rejects new motion until `clear_faults()`, and re-seeds $q_{\text{ref}} \leftarrow q_{\text{actual}}$ upon recovery. |
| **Limits (`limits.jerk`, `compliance.*`, `limits.cartesian_*`, `gripper.*`)** | Supply only the required `limits.velocity`, `limits.acceleration`, and `limits.effort`. Omit optional jerk, Cartesian, compliance, and gripper limits (`NAN`). | Populate `limits.jerk`, `limits.cartesian_velocity`, `limits.cartesian_acceleration`, `compliance.*` ($K_{\max}, D_{\min}, M_{\min}, F_{\max}$), and `gripper.*` from vendor specs. | Derives missing `limits.jerk` from acceleration and ramp time, and derives missing compliance clamps from loop latency $T_d$ and `effort_limits`. |
| **Peripherals & extensions (`aux`, `handle_vendor_command`)** | `AuxSchema::EMPTY` (no state tail); default `handle_vendor_command` (`Err(ArmError::NotSupported)`). No command-side tail. | Declare `aux` fields in `manifest.yaml` (e.g., joint temperatures, bus voltage, or streamed $M, C, G$) and write them into `state.tail()`; implement `handle_vendor_command` with caller-supplied response slice. | Publishes `aux` fields by label in shared memory (`meta` + `state` channels) and forwards `vendor_command` over `ctl_req` / `ctl_rsp`. |

---

### Implementing a Minimal Driver

A minimal driver is a pure pass-through: it owns no control loop, no kinematic
or dynamic math, no background threads, and no mode-switching state machine.
Even so, when paired with the generated `ArmModel` from `dexforge` (which
supplies `fk`, `jacobian`, `solve`, and SymPy `gravity` / `mass_matrix` /
`coriolis` / `inverse_dynamics` from the arm's URDF), a minimal driver gets
**full DexZero functionality** on the host.

#### What a minimal driver provides

1. **`manifest.yaml`** with only the required keys:
   - `driver`, `version`, `control_freq_hz`
   - `control_modes`: typically `[joint_pos]`, `[joint_torque]`, or `[joint_pos, joint_torque]`
   - `latency.state_staleness_cycles` and `latency.command_dispatch_cycles`
   - `limits.velocity`, `limits.acceleration`, `limits.effort` (from the URDF / datasheet)
   - `sensing.torque: true` if motor currents or joint torques are readable, or `sensing.wrench: true` if a 6-axis F/T sensor is attached
   - All `native.*` flags omitted (default `false`)
2. **`connect(&mut self)` and `disconnect(&mut self)`**:
   - Open the hardware bus (CAN, serial, UDP, EtherCAT), arm the drive's
     built-in hardware watchdog register if it has one, and enable the
     actuators in `connect()`.
   - Disable the actuators and close the bus in `disconnect()`.
3. **`read_state(&mut self, state_buffer: &mut [f64])`**:
   - Call `state.clear()` (sets all 56 slots of `CoreState` to `NAN`).
   - Copy joint encoder angles into `state.q()[..DOF]` and joint velocities
     into `state.qd()[..DOF]`.
   - If motor phase currents $i$ are available, write link-side effort
     $\tau = k_t \cdot n_{\text{gear}} \cdot i$ into `state.torques()[..DOF]`
     (or raw sensor floats into `state.wrench()`).
   - Write `state.fault_code()` (`0.0` when healthy) and `state.timestamp()` if
     the bus provides a packet counter or timestamp.
   - Leave `tcp_pose`, `tcp_twist`, and every unmeasured slot as `NAN`.
4. **`write_command(&mut self, cmd_buffer: &[f64])`**:
   - Read the slice for the supported mode (`cmd.target_q()[..DOF]`,
     `cmd.target_qd()[..DOF]`, or `cmd.target_torques()[..DOF]`, or the MIT
     5-tuple if `native.joint_impedance: true`) and put it on the wire.
   - If the command is `CoreCommand::IDLE` (all `NAN`), send a static hold or
     zero-damped packet.

#### Expected functionality of a minimal driver in DexZero

Because the host model fills in everything the hardware omits:

- **Cartesian & joint motion (`move()`, `speed()`):** Full Pipeline 1A
  (geometry-first), Pipeline 1B (time-first), and Pipeline 2 (velocity
  streaming) with quintic jerk verification, chord-error bisection, singularity
  braking, workspace bounds, and constraint funnels.
- **Force-based control & compliance from motor current alone:**
  - If the minimal driver supports `joint_torque` (`control_mode::JOINT_TORQUE`)
    and its URDF provides inertials (`has_static_inertials: true`), the kernel
    selects **Route 3 (`HostImpedance`)**: it evaluates $J(q_{\text{actual}})$
    and $\tau_{\text{model}}(q_{\text{actual}}, \dot{q}_{\text{actual}}, \ddot{q}_{\text{ref}})$
    on the host and commands $\tau = J^T (K e + D \dot{e} + F_{\text{virt}}) + \tau_{\text{null}} + \tau_{\text{model}}$.
  - If the minimal driver only supports `joint_pos` or `joint_vel`, reports
    motor-current torques (`sensing.torque: true`), and provides usable
    residual dynamics (`has_residual_inertials: true` or
    `dynamics_source: vendor`), the kernel selects
    **Route 6 (`HostAdmittanceTorque`)**: it estimates the external TCP wrench
    from the torque residual $r = \tau_{\text{model}} - \tau_{\text{measured}}$
    via $F_{\text{ext}} = (J^T)^+ r$, integrates virtual mass-damper-spring
    admittance on the host, solves seeded IK, and streams displaced `target_q`.
  - If a raw wrist F/T sensor is wired into `CoreState::wrench`
    (`sensing.wrench: true`), the kernel selects **Route 5 (`HostAdmittanceWrench`)**,
    rotates and tares the wrench on the host, and integrates admittance in
    position mode.
- **Protective reflexes (`STOP`, `YIELD`, `RETRACT`):** The kernel monitors the
  host-conditioned wrench (from the F/T sensor or motor-current torque residual)
  every cycle and executes `STOP`, `YIELD` (freedrive), or path `RETRACT`
  entirely on the host.
- **Runtime tool, payload, gravity & calibration:** Handled on the host model
  (`Applied::Host`) with zero driver code.

---

### Implementing an Advanced Driver

An advanced driver wraps a robot controller that runs its own internal high-rate
servo, impedance/admittance loop, kinematics, or safety supervisor (such as
Franka `libfranka`, Flexiv RDK, Universal Robots RTDE + URScript, or smart
MIT-mode actuators with onboard PD loops).

#### What an advanced driver adds

1. **Rich `manifest.yaml` declarations:**
   - Multiple `control_modes` (e.g., `[joint_pos, joint_vel, joint_torque, cartesian_pose, cartesian_twist, cartesian_wrench]`).
   - Native compliance capabilities:
     - `native.cartesian_impedance: true` (Route 1) or `native.cartesian_admittance: true` (Route 2): the driver reads `target_pose` (or `target_q`/`target_qd`), `target_twist`, `task_frame: [f64; 7]`, diagonal `task_stiffness_k`, `task_damping_b`, `task_mass_m`, and `target_wrench` ($F_{\text{virt}}$) from `CoreCommandView` and streams them to the controller's native compliance API (e.g., URScript `force_mode`, Franka Cartesian impedance).
     - `native.joint_impedance: true` (Route 4, MIT mode): the driver reads the 5-tuple `(target_q, target_qd, joint_stiffness_k, joint_damping_b, target_torques)` and packs it into the actuator's MIT frame.
   - Inertial and gravity flags:
     - `has_static_inertials: false` when the vendor obscures URDF link masses/inertias or when unmodelled harmonic-drive friction makes static URDF open-loop torque impedance (Route 3) inaccurate. Set `has_residual_inertials: true` when static URDF dynamics are still accurate enough for torque-residual wrench estimation (Route 6), or set `dynamics_source: vendor` to pass through the controller's internal `gravity_torque` (`DOF`), `coriolis_torque` (`DOF`), and `mass_matrix` (`DOF * DOF`) in the `CoreState` `aux` tail for per-tick control and shared memory.
     - `native.gravity_compensation: true` when the controller's `joint_torque` mode automatically adds link gravity and friction compensation.
   - Explicit controller compliance and Cartesian bounds: `limits.cartesian_velocity`, `limits.cartesian_acceleration`, and `compliance.*` ($K_{\max}, D_{\min}, M_{\min}, F_{\max}$).
2. **Hardware state & reflex reporting in `read_state`:**
   - Populate `tcp_pose` and `tcp_twist` (at the TCP origin in base axes) from the controller's internal model so `calibration_residual` can verify host-vs-controller agreement.
   - Populate `timestamp` with the controller's hardware clock or packet counter so the kernel's $K$-deep latency compensation tracks variable network/RTDE jitter cycle by cycle.
   - Populate `robot_mode`, `fault_code`, `in_contact`, and `native_reflex_fired` (`1.0` when a controller protective reflex or URScript contract break occurs).
   - Populate `state.tail()` with any `aux` fields declared in `manifest.yaml`.
3. **Native configuration & reflex delegation:**
   - Implement `set_payload_native`, `set_tcp_offset_native`, and `set_gravity_native` so the controller's internal gravity compensation and force-mode frames stay synchronized with the kernel (`Applied::Native`).
   - When `native.reflexes: true`, forward `reflex_threshold` and `reflex_action_code` to the controller so contact stops or yields happen inside the controller's microsecond loop, and set `CoreState::native_reflex_fired = 1.0` when triggered.
   - Implement `clear_faults` to reset protective stops on the controller, and `handle_vendor_command` for vendor-specific actions outside the portable contract.

#### Expected functionality of an advanced driver in DexZero

- **Controller-rate stiffness and passivity (Routes 1, 2, 4):** Compliance is
  closed inside the hardware controller or actuator drive (often at 1–10 kHz),
  giving higher stable stiffness $K_{\max}$ and passive behavior against stiff
  surfaces without round-trip bus delay.
- **Seamless host fallback (`(primary_route, fallback_host_route)`):** Even on
  an advanced driver, the kernel precomputes `fallback_host_route` at boot and
  keeps the host `Arm` model synchronized on every `set_payload` /
  `set_tcp_offset` / `set_gravity` call. If the user requests a capability the
  native controller does not cover (such as path `RETRACT` after a native
  reflex stop, or an arbitrary compliance frame on a controller that does not
  accept `task_frame`), the kernel handles it on the host automatically.

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

## Loading drivers at run time (`dexhal-plugin`)

`plugin/` (`dexhal-plugin`) is a stable C ABI around `ArmDriver` and `ArmMath`,
so a host can load a driver it was not compiled with. A driver crate builds as
a `cdylib`, implements `PluginDriver` (`create(config)` + `create_math()`) and calls
`export_plugin!(MyDriver<…>)`, which exports `dexhal_plugin_entry`. The host
loads it with `Plugin::load(path)` (ABI version 3 and contract major / minor are checked),
then calls `create_driver(config)` and `create_math()` to get a
`LoadedDriver: ArmDriver` and a `LoadedMath: ArmMath`. The mock driver is the
reference export (`mock/rust/src/plugin.rs`); its tests in
`mock/rust/tests/plugin.rs` check plugin-vs-direct parity and load
`dexforge_mock_core` dynamically (`cargo build --release -p dexforge-mock-core`
first).

Open work is tracked in [`workitems.md`](workitems.md).
