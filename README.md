# dexforge

**Generate real-time robot control for any robot arm (6DOF and 7DOF) in pure Rust.**

**Under-100 µs kinematics, under-10 µs dynamics, hardware drivers, 500–1000 Hz robot control loops, zero-allocation, bounded time, cross-platform.**

![1000Hz](https://img.shields.io/badge/-1000_Hz-blue) ![analytical IK](https://img.shields.io/badge/-analytical_IK-blue)  ![no alloc](https://img.shields.io/badge/-no_alloc-blue)  ![bounded WCET](https://img.shields.io/badge/-bounded_WCET-blue)   ![no_std](https://img.shields.io/badge/-no__std-blue)  ![forbid(unsafe__code)](https://img.shields.io/badge/-forbid(unsafe__code)-blue)  


---

## Why `dexforge`?

Most robotics stacks force a choice between heavyweight C++ libraries with runtime allocators (`Pinocchio`, `MoveIt`, `Orocos`) and bespoke embedded firmware that lacks analytical IK, payload compensation, or kinematic calibration. **`dexforge` compiles analytical IK and symbolic Lagrangian dynamics directly into standalone `#![no_std]` Rust crates behind a single zero-copy hardware contract.**

| Capability | What You Get |
| :--- | :--- |
| **Hard-Real-Time by Construction** | `#![no_std]`, `#![forbid(unsafe_code)]`, zero heap allocation (`no Vec`, `Box`, or dynamic dispatch on the hot path), fixed stack arrays, and strict iteration ceilings on every numerical routine. |
| **Analytical IK** |  [ssik](https://github.com/personalrobotics/ssik) artifacts translated to Rust, with closed-form 6-DOF (`SP1`–`SP6`, Raghavan-Roth) and seed-locked 7-DOF solvers with seed-ordered solutions, joint-limit filtering, and continuity-guarded servoing. |
| **Branch-Free Symbolic Dynamics** | Flat `sympy.cse()` forward kinematics, geometric Jacobians (base & tool frames), and full Lagrangian dynamics: $M(q)$, $C(q,\dot{q})\dot{q}$, $g(q)$, and $\tau = M\ddot{q} + C\dot{q} + g$. |
| **Live Runtime Arm State** | Update TCP offsets, rigid tool payloads (mass, CoM, rotational inertia), gravity vectors (wall/ceiling mounts), and per-robot DH/motion calibration at runtime, **without regenerating code**. |
| **Dumb-to-Smart Driver Contract** | `#[repr(C)]` zero-copy wire views and C-ABI shared-library plugins . Minimal CAN pass-through drivers get full host-side compliance and reflexes automatically. |
| **Dual Ground-Truth Parity** | Every driver is verified in pure `cargo test` against **ssik** and **Pinocchio**. No Python or C++ needed at build or test time. |
| **Agent-Driven Driver Factory** | Point an autonomous coding agent at [`codegen/new_driver.md`](codegen/new_driver.md) with a URDF, `manifest.yaml`, and OEM SDK docs to compile and verify a new robot driver crate. |

---

## Representative Arms & Real-Time Benchmarks

Each OEM lives in `repos/dexforge-<oem>/rust` and implements [`dexhal::math::ArmModel`](dexhal/README.md) and [`ArmDriver`](dexhal/README.md). Below is a representative sample across the [`ssik`](https://github.com/personalrobotics/ssik) kinematic solver families:

IK cells are **mean (max) µs / coverage %**, where coverage is the share of targets for which the call returned an FK-verified solution.

| Arm | Crate | DOF | `ssik` Solver Family | `MAX_SOL` | `ik_all` | `ik_one` (`Fast`) | `dls_ik` | Dynamics (µs) | Inv. Dynamics (µs) |
| :--- | :--- | :---: | :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| **UR5** | `dexforge-ur-core` | 6 | Closed-form 6R (`ikgeo.three_parallel`, SP6) | 16 | 11.05 (53.10) / 100.0% | 1.87 (10.07) / 100.0% | 1.92 (14.14) / 100.0% | 1.62 | 1.62 |
| **UR3e** | `dexforge-ur-core` | 6 | Closed-form 6R (`ikgeo.three_parallel`, SP6) | 16 | 13.40 (60.69) / 100.0% | 1.99 (6.85) / 100.0% | 2.03 (59.03) / 100.0% | 1.56 | 1.56 |
| **Unitree Z1** | `dexforge-unitree-core` | 6 | Closed-form 6R (`ikgeo.three_parallel`, SP6) | 16 | 16.58 (42.37) / 100.0% | 1.69 (6.45) / 100.0% | 1.70 (28.92) / 100.0% | 1.85 | 1.86 |
| **AgileX Piper** | `dexforge-agilex-core` | 6 | General 6R (`ikgeo.general_6r`, Raghavan-Roth) | 16 | 47.17 (156.26) / 100.0% | 2.42 (7.14) / 100.0% | 2.40 (10.12) / 100.0% | 3.80 | 3.85 |
| **Franka Panda** | `dexforge-franka-core` | 7 | Joint-lock 7R (`jointlock.seven_r`, SP6 inner) | 128 | 169.88 (231.09) / 99.4% | 2.50 (26.15) / 100.0% | 2.39 (17.12) / 99.3% | 6.27 | 6.27 |
| **UFactory xArm7** | `dexforge-ufactory-core` | 7 | Joint-lock 7R (`jointlock.seven_r`, SP6 inner) | 128 | 155.62 (215.21) / 99.5% | 2.49 (36.86) / 99.8% | 2.35 (48.71) / 99.5% | 8.32 | 8.34 |
| **Flexiv Rizon 4** | `dexforge-flexiv-core` | 7 | Joint-lock 7R (`jointlock.seven_r`, RR inner) | 256 | 622.18 (719.25) / 100.0% | 3.14 (84.93) / 99.8% | 2.75 (11.07) / 99.3% | 5.20 | 5.20 |
| **Flexiv Rizon 10** | `dexforge-flexiv-core` | 7 | Joint-lock 7R (`jointlock.seven_r`, RR inner) | 256 | 624.10 (728.84) / 100.0% | 2.59 (80.23) / 99.8% | 2.27 (11.02) / 99.3% | 5.06 | 5.07 |

* **Reference Mock Arm:** [`dexforge-mock-core`](dexhal/mock/rust) (`"mock-6r"`) provides a complete `MockArm`, `MockDriver` with configurable queue latency, and C-ABI plugin export for hardware-free integration testing.
* **Reproduce Benchmarks:** Wall-clock timings are measured in `--release` via the shared harness in [`dexhal/perf/arm_timing.rs`](dexhal/perf/arm_timing.rs) over 2,000 deterministic random in-limit configurations, with `fk(q)` as the target and a seed jittered 0.05 rad from `q` (`ik_all` is seeded too). **Max** is robust rather than a single sample: the 10 slowest targets are re-timed 100× back to back and the largest of those 10 averages is reported. 7-DOF `ik_one` visits only the seed lock and the two nearest lock samples, so its coverage can differ from the full `ik_all` sweep in either direction. `ik_one` and `dls_ik` coverage counts only solutions within `ik_max_joint_step` (π/2 per joint) of the seed; 7-DOF `ik_one` also bounds the lock joint to π/8.
  ```powershell
  cargo test --release --manifest-path repos\dexforge-<oem>\rust\Cargo.toml --test perf -- --ignored --nocapture --test-threads=1
  ```

---

## Architecture in 3 Layers

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ 1. codegen/ ,  Agent-Driven OEM Driver Generator                            │
│    URDF + manifest.yaml + OEM SDK ──► SymPy CSE + ssik + Pinocchio Golden  │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 2. repos/dexforge-<oem>/ ,  Standalone #![no_std] OEM Driver Crates         │
│    ik_solver.rs · generated_kinematics.rs · generated_dynamics.rs          │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 3. dexhal/ ,  Shared #![no_std] Hardware Contract, Math Kernels & C-ABI     │
│    Arm<Model> (TCP, Payload, Gravity, Calibration) · ArmDriver · Spatial   │
└────────────────────────────────────────────────────────────────────────────┘
```

1. **[`dexhal`](dexhal/README.md) ,  The Hardware Abstraction & Math Contract:**
   Owns the single `#![no_std]` driver contract (`ArmModel`, `Arm`, `ArmDriver`, `ArmMath`, `ArmManifest`, zero-copy `CoreState` / `CoreCommand` wire views, `CalibrationData`), the C-ABI shared-library loader (`dexhal-plugin`), and all shared numerical kernels. Contains zero arm-specific code.
2. **[`codegen`](codegen/README.md) ,  Agent-Driven OEM Driver Generator:**
   Prompts, templates, and deterministic Python compilers (`ssik` artifact builder/inspector, SymPy CSE symbolic FK/Jacobian/Lagrangian dynamics compiler, and `ssik` / Pinocchio ground-truth dataset generators) that turn an OEM's URDF, `manifest.yaml`, and hardware API spec into a verified crate.
3. **`repos/dexforge-*` ,  Pre-Generated OEM Driver Crates:**
   Ready-to-link `#![no_std]` Rust driver crates that depend directly on `dexhal` without vendoring or duplicating shared math.

---

## Distinguishing Features (In Depth)

### 1. Hard-Real-Time Guarantees & Lock-Free Thread Separation
- **Strict `#![no_std]` & `#![forbid(unsafe_code)]`:** Every math kernel and generated solver uses fixed-size stack arrays (`MAX_DOF = 8`) with zero heap allocation and hardcoded iteration ceilings (`CHAIN_IK_MAX_ITERS = 12`, bounded polynomial root Polishing) for deterministic worst-case execution time (WCET).
- **Decoupled Executor & Planner Instances:** Both `ArmDriver` and the DOF-erased `ArmMath` trait are `Send`. A real-time 1 kHz control thread can own `Box<dyn ArmDriver>` exclusively while trajectory planners, scrubbers, and simulators own independent `Box<dyn ArmMath>` (`Arm<Model, DOF, MAX>`) instances, eliminating lock contention on the hot path.

### 2. Analytical IK, Continuity Guards & Seed-Locked 7-DOF Redundancy
- **Closed-Form & Companion-Matrix Solvers:** Translates [`ssik`](https://github.com/personalrobotics/ssik) subproblems (`SP1`–`SP6` and deterministic Raghavan-Roth companion-matrix eigensolvers) into straight-line Rust with FK verification, Levenberg-Marquardt refinement, de-duplication, joint-limit filtering, and nearest-to-seed sorting.
- **Branch-Jump Rejection (`ik_one`):** For high-rate servoing (the contract uses `IkKind::Fast`), `ik_one` unwraps each joint to the $2\pi$ turn closest to `q_seed` (within joint limits) and discards every proposed solution (checked after polishing) in which any joint would step further than `ik_max_joint_step` (default and maximum $\pi/2$; on 7-DOF arms the redundant joint is further limited to $\pi/8$), returning `None` if none passes. Near singularities or workspace boundaries where the nearest valid branch flips posture, the continuity bound cleanly signals "no local solution" instead of commanding a violent branch jump.
- **Posture-Preserving Seeded 7-DOF IK:** `ik_all` sweeps all 16 fixed samples of the redundant joint. `ik_one` never sweeps: `IkKind::Fast` tries an unlocked DLS from the seed, then a polish with the redundant joint held at the seed, then the two lock samples nearest the seed; `IkKind::Near` holds the redundant joint at the seed value throughout (seed polish, the two nearest samples, then a longer masked DLS from the seed). Each stage returns the first solution it finds, so tracking does not drift along the null space or pay for the sweep.

### 3. Branch-Free Symbolic Dynamics & Analytic Payload Compensation
- **Flat `sympy.cse()` Rigid-Body Dynamics:** Compiles URDF link inertials into branch-free Rust functions for gravity torques $g(q)$, symmetric mass matrix $M(q)$, Coriolis/centrifugal vector $C(q,\dot{q})\dot{q}$, and RNEA-equivalent inverse dynamics $\tau = M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q)$ in **1.2–8.3 µs**.
- **Closed-Form Runtime `Payload` & Arbitrary Mount Gravity:** `Arm::set_payload` attaches a rigid tool payload (mass, center of mass, and optional 3×3 rotational inertia tensor). Payload contributions to gravity, mass matrix, inverse dynamics, and **analytic Coriolis** (computed directly from $\dot{J}\dot{q}$ without finite differences) are evaluated in closed form at the flange and summed automatically. `Arm::set_gravity` supports arbitrary base-frame gravity vectors for wall, ceiling, or mobile mounts.
- **Flexible Inertial & Vendor Dynamics Routing:** `manifest.yaml` decouples `has_static_inertials` (open-loop torque impedance) from `has_residual_inertials` (motor-current torque-residual wrench estimation $r = \tau_{\text{model}} - \tau_{\text{measured}}$), and supports `dynamics_source: vendor` to stream controller-computed $(M, C, g)$ matrices through the `CoreState` `aux` tail when an OEM obscures URDF inertials or relies on internal harmonic-drive friction models.

### 4. Per-Robot Kinematic Calibration & Tracked-Motion Identification
- **Universal `KinematicChain<DOF>`:** Represents per-robot calibrated geometry (fixed link transforms, joint axes, and joint zero offsets) as a strict superset of Standard DH, Modified DH, and URDF joint origins, preserving full floating-point precision even when vendor DH tables (such as Universal Robots `calibration.conf`) contain hundreds-of-metres $d$-offsets that cancel across near-parallel axes.
- **Zero-Cost When Uncalibrated, Bounded Polish When Calibrated:** Uncalibrated arms take the fast generated code path directly. Installing a calibration via `arm.set_calibration(chain)` validates the chain against the nominal model across six test postures (`CALIBRATION_MAX_DEVIATION`) to catch frame mismatches, switches FK/Jacobian to the calibrated chain, and polishes nominal analytical IK roots onto the calibrated geometry in at most 12 damped Newton steps before re-verifying closure tolerances and joint limits.
- **Hardware Self-Check (`fk_residual` / `calibration_residual`):** Measures the SE(3) residual between the host model's TCP and the `tcp_pose` reported by the hardware controller in `CoreState` to detect unapplied calibration files or mismatched TCP configs at boot.
- **Zero-Allocation Commissioning (`ChainIdentifier`):** For arms without factory calibration files, `math::identify::ChainIdentifier` fits a `KinematicChain` directly from external 3D/6D tracker samples by accumulating normal equations in fixed stack memory (~23 kB, zero heap allocation), complete with rotation weighting, outlier gating, and rank/observability reporting (`4·DOF + 6`).

### 5. Complete Operational-Space & Spatial Linear Algebra Kernel
- **Stateless `#![no_std]` Control Primitives (`math::spatial` & `math::chain`):** Supplies every linear-algebra building block required by operational-space, impedance, and admittance controllers so downstream kernels need zero `nalgebra` imports:
  - Exact `rotation_exp`, antipodal-safe `rotation_error` / `pose_error`, `se3_lerp`, and `se3_distance`.
  - Standard and weighted damped least-squares (`damped_least_squares`, `weighted_damped_least_squares`).
  - Khatib's dynamically consistent generalized inverse $\bar{J} = M^{-1} J^T (J M^{-1} J^T)^{-1}$ (`dynamically_consistent_pseudo_inverse`), null-space torque projector $N^T \tau_0 = (I - J^T \bar{J}^T)\tau_0$ (`nullspace_project_torque`), and Cholesky/LDLT mass-matrix solver (`solve_mass_matrix`).
- **Tool-Frame Transforms & Automatic Wrench Conditioning:** Transforms Jacobians (`jacobian_tool`), twists, wrenches (`wrench_base_to_tool`), and diagonal stiffness/damping gains (`diagonal_gain_tool_to_base`) between base and TCP axes. `ArmDriver::external_wrench` automatically rotates raw wrist F/T sensor readings (`sensing.wrench_frame: base | flange | tcp`), shifts moments to the TCP, and subtracts configured payload weight unless the sensor is already gravity-compensated.

### 6. Zero-Copy Wire Contract, Pass-Through Drivers & C-ABI Plugins
- **Safe Offset-Based Wire Views (`CoreState` & `CoreCommand`):** The cyclic data plane is two homogeneous `#[repr(C)]` `f64` buffers, `CoreState` (`56 × f64` + declared `aux` tail) and `CoreCommand` (`109 × f64`), accessed in place via `CoreStateView(Mut)` and `CoreCommandView(Mut)` using compile-time `offset_of!` indices (zero `unsafe`, zero copies). Every unmeasured or uncommanded slot is `NAN`.
- **Single-Match `ActiveMode` Dispatch:** `CoreCommand` carries an explicit `control_mode` tag (`ActiveMode`: joint pos/vel/torque/impedance, Cartesian pose/twist/wrench/impedance/admittance, or `NAN` = `Idle` hold) plus feedforward `target_qdd`, compliance `task_frame`, 7-DOF `target_q` null-space posture, digital I/O bitmasks (53-bit exact), analog I/O, and gripper width/force bounds. `write_command` is a single `match` statement with no mode-transition state machine in the driver.
- **Minimal Pass-Through vs. Advanced Controller Delegation:**
  - A **minimal driver** (e.g., raw CAN/EtherCAT reading encoder angles $q, \dot{q}$ and motor currents $i$, leaving `tcp_pose` as `NAN` and `native.*` flags `false`) automatically gets full host-side Cartesian motion, gravity/payload compensation, open-loop torque impedance, motor-current torque-residual admittance ($F_{\text{ext}} = (J^T)^+ r$), and protective contact reflexes (`STOP`, `YIELD`, `RETRACT`).
  - An **advanced driver** (e.g., Franka `libfranka`, Flexiv RDK, UR RTDE + `force_mode`, or MIT-mode actuators) declares its native loops in `manifest.yaml` (`native.cartesian_impedance`, `native.joint_impedance`, `native.gravity_compensation`, `native.reflexes`) so the kernel delegates 1–10 kHz stiffness and reflexes directly to the hardware while keeping a precomputed host fallback route synchronized.
- **Dynamic C-ABI Plugin Boundary (`dexhal-plugin`):** Wraps any `ArmDriver` as a `cdylib` (`export_plugin!`) with a `#[repr(C)]` manifest descriptor, `ContractVersion` / ABI handshake, panic containment across FFI, parameterized `create_driver(config)`, and an independent `create_math()` factory.

---

## Installation

### Rust

Add [`dexhal`](dexhal) and the target OEM driver crate to your `Cargo.toml` (minimum Rust version: **1.82**):

```toml
[dependencies]
dexhal = { path = "dexhal" }
dexforge-ur-core = { path = "repos/dexforge-ur/rust", default-features = false }
```

Each driver crate enables the `std` feature by default (for host socket / SDK I/O in `driver.rs`); set `default-features = false` for pure `#![no_std]` kinematics and dynamics.

### Python (Codegen Tooling)

To install the [`codegen/`](codegen/README.md) package and all offline codegen dependencies (`ssik`, `sympy`, `numpy`, `pin`, `pytest`) from a clone:

```sh
pip install .
```

---

## Usage

### Rust (`dexhal::math::Arm` & `ArmDriver`)

Use `dexhal::math::Arm<Model, DOF, MAX>` for stateless or stateful kinematics, IK, and dynamics (including runtime TCP offsets, payloads, gravity, and calibration), or `Driver::new()` for the full `ArmDriver` contract:

```rust
use dexforge_ur_core::{Ur5, Ur5Driver};
use dexhal::math::{Arm, ArmDriver, Payload, Pose, TolerancePolicy};

// 1. Host-side kinematics, IK, and dynamics with runtime TCP, payload & gravity
let mut arm = Arm::<Ur5, 6, 16>::new();

let mut tcp = Pose::identity();
tcp[(2, 3)] = 0.12; // 12 cm tool along flange Z
arm.set_tcp(&tcp)?;
arm.set_payload(Payload::new(2.5, [0.0, 0.0, 0.06]))?;
arm.set_gravity([0.0, 0.0, -9.80665])?;                      // wall / ceiling mounts supported

let q = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0];
let qd = [0.0; 6];
let qdd = [0.0; 6];

let pose = arm.fk(&q);                                       // base -> TCP SE(3) pose
let jac = arm.jacobian(&q);                                  // 6x6 geometric Jacobian (base axes)
let jac_tool = arm.jacobian_tool(&q);                        // 6x6 geometric Jacobian (TCP axes)
let q_next = arm.ik_one(&pose, &q, IkKind::Fast, TolerancePolicy::DEFAULT); // continuity-guarded servo IK
let dyn_state = arm.dynamics(&q, &qd).unwrap();              // M(q), C(q,qd)*qd, G(q) incl. payload
let tau = arm.inverse_dynamics(&q, &qd, &qdd).unwrap();      // M*qdd + C*qd + G incl. payload

// 2. Full hardware driver contract (manifest + zero-copy state/command views)
let driver = Ur5Driver::new()?;
assert_eq!(driver.manifest().name, "ur5");
```

See [`dexhal/README.md`](dexhal/README.md) for the complete `Arm`, `ArmDriver`, `CoreStateView` / `CoreCommandView`, `manifest.yaml`, minimal-vs-advanced driver tiers, and per-robot calibration documentation.

### Generating a New OEM Driver

New drivers are generated from three OEM inputs, **1. `<arm>.urdf`**, **2. `manifest.yaml`**, and **3. OEM hardware API / protocol docs**, using an autonomous coding agent pointed at [`codegen/new_driver.md`](codegen/new_driver.md). See [`codegen/README.md`](codegen/README.md) for the complete specification, directory layout, and generator CLI reference.

---

## Testing

```sh
# Run dexhal contract, spatial math, calibration & plugin tests
cargo test --manifest-path dexhal/Cargo.toml

# Run an OEM driver's IK, ArmModel, manifest, and Pinocchio dynamics parity tests
cargo test --manifest-path repos/dexforge-ur/Cargo.toml

# Verify ssik dataset version alignment and run codegen unit tests
python codegen/ik_test_data.py --check
pytest codegen/tests -v
```

---

## Documentation Index

- **[`dexhal/README.md`](dexhal/README.md):** Driver contract (`ArmModel`, `Arm`, `ArmDriver`, `ArmManifest`, `CoreState` / `CoreCommand`), minimal vs. advanced driver tiers, payload math, kinematic calibration, and `dexhal-plugin`.
- **[`dexhal/workitems.md`](dexhal/workitems.md):** Architectural decisions and completed contract capabilities across `dexhal`, `codegen`, and `DexZero`.
- **[`dexhal/examples/manifest.yaml`](dexhal/examples/manifest.yaml):** Annotated reference for the driver manifest schema.
- **[`codegen/README.md`](codegen/README.md):** Agent-driven codegen architecture, deterministic generators, and target vendor repository structure.
- **[`codegen/new_driver.md`](codegen/new_driver.md) & [`codegen/translate_ssik_artifact.md`](codegen/translate_ssik_artifact.md):** Master and IK translation prompts for generating new OEM drivers.
- **[`codegen/upgrade_ssik_version.md`](codegen/upgrade_ssik_version.md):** Runbook for upgrading the locked `ssik` version in [`pyproject.toml`](pyproject.toml).
