# dexhal

The common abstraction every dexforge arm is surfaced through: forward and
inverse kinematics, Jacobian, dynamics, runtime configuration (TCP, payload,
gravity, per-robot calibration), a capability manifest and a control API.

`#![no_std]`, `#![forbid(unsafe_code)]`, no allocation, every loop bounded.

The contract surface is [`src/math/arm.rs`](src/math/arm.rs). It has three
layers:

| Layer | Type | Who implements it | What it is |
| :-- | :-- | :-- | :-- |
| Model | `ArmModel<DOF, MAX>` | driver crate (generated) | Nominal, stateless flange FK / Jacobian / analytic IK / dynamics, plus `NAME`, `JOINT_LIMITS`, `HAS_DYNAMICS` |
| Arm | `Arm<A, DOF, MAX>` | dexhal | One physical arm: a model plus its TCP, payload, gravity vector and calibration |
| Contract | `ArmMath`, `ArmDriver`, `ArmManifest`, `CoreState`, `CoreCommand` | dexhal (math) + hardware driver (I/O) | `MAX_DOF`-padded, object-safe form consumed by the DexZero kernel |

`ArmModel` is unchanged for existing drivers: the three new associated
constants have defaults.

## What the OEM supplies and what they get

OEMs own their driver. The driver depends on dexhal only, and dexhal owns the
whole driver contract.

| OEM provides | Produced by | Lands in |
| :-- | :-- | :-- |
| URDF | dexforge codegen + coding agent | `ArmModel` impl: FK, Jacobian, IK, dynamics, joint limits, nominal chain |
| Driver manifest (rates, modes, latencies, sensing, limits) | hand-written or generated data | `ArmManifest` |
| Hardware API or protocol spec | coding agent | `connect`, `disconnect`, `read_state`, `write_command` |

Everything else — TCP, payload, calibration, host fallbacks, buffer layouts —
is provided by dexhal.

This is the target workflow; parts of it are not in place yet. See
[`workitems.md`](workitems.md).

## Using an arm

```rust
use dexhal::math::arm::{Arm, Payload, Pose};
use dexhal::math::tolerances::TolerancePolicy;

let mut arm = Arm::<Ur5, 6, 16>::new();          // Ur5: the driver's ArmModel
arm.set_tcp(&tcp)?;                               // flange -> TCP
arm.set_payload(Payload::new(2.5, [0.0, 0.0, 0.06]))?;
arm.set_gravity([0.0, 0.0, -9.80665])?;           // base frame; wall/ceiling mounts

let pose = arm.fk(&q);                            // base -> TCP
let jac = arm.jacobian(&q);                       // [Jv; Jw] at the TCP
let q_next = arm.ik_nearest(&target, &q, TolerancePolicy::DEFAULT);
let all = arm.ik(&target, None, None, true, true, TolerancePolicy::DEFAULT, 15);
let dynamics = arm.dynamics(&q, &qd);             // M, C·qd, G incl. payload
```

Setters validate their input and leave the arm unchanged on error.
`dynamics` returns `None` unless the model declares `HAS_DYNAMICS`; the
payload-only terms (`payload_gravity`, `payload_mass_matrix`,
`payload_coriolis`) are always available. `set_payload` also accepts a
`RigidPayload`, which adds a rotational inertia tensor
(`RigidPayload::new(..).with_inertia(..)`); `Payload` itself is unchanged.

## Calibration

Generated drivers bake in the nominal geometry. A per-robot calibration is a
`KinematicChain<DOF>` — fixed transforms, joint axes and joint zero offsets —
which is a superset of standard DH, modified DH and URDF joint origins.

```rust
use dexhal::math::arm::{apply_dh_deltas, DhConvention, KinematicChain};

// UR: nominal DH + the deltas from the robot's calibration.conf
let rows = apply_dh_deltas(&UR5_NOMINAL_DH, &deltas);
let chain = KinematicChain::from_dh(DhConvention::Standard, &rows)?
    .with_base(&dh_base_in_driver_base)?          // only if the driver's frames
    .with_flange(&dh_flange_in_driver_flange)?;   // differ from the DH ones
arm.set_calibration(chain)?;
```

What changes once a calibration is installed, and what it costs:

- **FK / Jacobian** come from the chain: one fixed `DOF` loop.
- **IK** runs the nominal analytic solver, then polishes each solution onto the
  calibrated chain with at most `CHAIN_IK_MAX_ITERS` (12) damped Newton steps
  and re-verifies it against `CALIBRATED_FK_ATOL` (1e-5).
- **Dynamics** of the arm's own links stay nominal; payload terms use the
  calibrated flange.
- **Uncalibrated arms** take the generated path and pay nothing.

`set_calibration` compares the chain with the nominal model at six fixed
configurations and rejects it (`ModelError::ExcessiveDeviation`) beyond
`CALIBRATION_MAX_DEVIATION`. This catches a chain expressed in the wrong base
or flange frame — for example UR's DH base, which is rotated by π about z
relative to the URDF `base_link`.

Limitation: a target reachable only by the calibrated geometry (within about a
millimetre of the workspace boundary) has no nominal seed and returns no
solution.

### Arms without vendor calibration

The same chain can be identified from tracked motion with
`math::identify::ChainIdentifier`: move the arm through a few dozen
configurations while an external tracker measures a flange-mounted target
(full pose or position only), then

```rust
let mut ident = ChainIdentifier::new(initial_chain);
for _ in 0..PASSES {                              // 3-6 passes
    for (q, measured) in samples { ident.add_sample(q, measured); }
    ident.step();                                 // reports the pass RMS
}
arm.set_calibration(*ident.chain())?;
```

The caller owns the samples; the identifier accumulates normal equations, so
it allocates nothing. It is a commissioning tool (≈ 23 kB of state), not a
control-loop routine. An external reference is required — joint encoders alone
cannot observe link geometry. See the module docs for frames and observability.

## Driver contract

`arm.rs` mirrors DexZero's `docs/specs/driver_contract.md`:

| DexZero spec | dexhal |
| :-- | :-- |
| `MAX_DOF`, `ContractVersion`, `ControlMode` | `MAX_DOF`, `ContractVersion`, `control_mode` |
| `DriverManifest` | `ArmManifest` (`ArmManifest::for_model` fills name, DOF, limits, model capabilities) |
| `CoreDriverState`, `CoreDriverCommand` | `CoreState` (52 × f64), `CoreCommand` (89 × f64) |
| `DynamicsState` | `DynamicsState` |
| `DexArmDriver` | `ArmDriver` |

A hardware driver implements `manifest`, `connect`, `disconnect`, `read_state`,
`write_command` and returns its `Arm` from `arm()` / `arm_mut()`. FK, Jacobian,
IK, dynamics and the payload / TCP fallbacks are then provided.

Deliberate differences from the spec, proposed for adoption there:

- **Errors** are an `ArmError` enum rather than `&'static str`, so the kernel
  can match on `NotSupported` instead of comparing strings.
- **`set_payload` / `set_tcp_offset`** return `Applied::Native` or
  `Applied::Host`. The driver overrides only `set_payload_native` /
  `set_tcp_offset_native`; the provided method validates, offers the change to
  the controller and updates the host model, so the two cannot drift apart. The
  manifest also states `native_payload` / `native_tcp_offset` up front.
- **Aux schema** is a `&'static [AuxField]` instead of `Vec<String>` /
  `Vec<u32>`.
- **`dynamics`** already includes the payload, and **`inverse_kinematics`**
  returns the solution on the seed's 2π branch.
- **No `unsafe`**: `CoreState` / `CoreCommand` are `#[repr(C)]` all-`f64`, so a
  consumer may overlay them on its buffer; dexhal itself offers the copying
  `read_from` / `write_to`.
- **Calibration and gravity** are not in the spec's control plane. Calibration
  is typed by `DOF`, so a driver installs it on its own `Arm`; a kernel-facing
  entry point would need a `MAX_DOF`-padded chain format.
