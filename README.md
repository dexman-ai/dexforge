# dexforge

Deterministic, zero-allocation, `#![no_std]` robot arm kinematics, dynamics, and hardware driver crates for real-time control loops (500–1000 Hz).

`dexforge` combines three layers in one repository:

1. **[`dexhal`](dexhal/README.md) — The Hardware Abstraction & Math Contract:**
   Defines the single `#![no_std]` driver contract (`ArmModel`, `Arm`, `ArmDriver`, `ArmManifest`, zero-copy `CoreState` / `CoreCommand` wire views, `CalibrationData`) and the shared numerical kernels (`ssik_core`, `kinematics`, `postprocess`, `tolerances`, `spatial`, `payload`, `chain`, `identify`).
2. **[`codegen`](codegen/README.md) — Agent-Driven OEM Driver Generator:**
   Prompts, templates, and deterministic Python compilers (`ssik` artifact builder/inspector, SymPy CSE symbolic FK/Jacobian/Lagrangian dynamics compiler, and `ssik` / Pinocchio ground-truth dataset generators) that enable an autonomous coding agent to turn an OEM's URDF, `manifest.yaml`, and hardware API specification into a verified driver crate.
3. **`repos/dexforge-*` — Pre-Generated OEM Driver Crates:**
   Ready-to-link `#![no_std]` Rust driver crates (with optional PyO3 Python bindings) implementing `dexhal::math::ArmModel` and `dexhal::math::ArmDriver`.

---

## Differentiating Features

- **Hard-Real-Time Guarantees (`#![no_std]`, `#![forbid(unsafe_code)]`):**
  Zero heap allocation (`no Vec`, `Box`, or dynamic dispatch on the hot path), fixed-size stack arrays throughout, and hardcoded iteration ceilings on every numerical routine for bounded worst-case execution time.
- **Analytical IK + Branch-Free Symbolic Dynamics:**
  Analytical inverse kinematics translated from [`ssik`](https://github.com/personalrobotics/ssik) (`SP1`–`SP6` subproblems, joint-locked 7-DOF sweeps, and deterministic Raghavan-Roth companion-matrix solvers) paired with flat, branch-free `sympy.cse()` forward kinematics, geometric Jacobians, and Lagrangian inverse dynamics ($M(q)$, $C(q,\dot{q})\dot{q}$, $g(q)$, $\tau = M\ddot{q} + C\dot{q} + g$).
- **Complete Stateful Arm Model in `dexhal`:**
  Runtime TCP offsets, rigid tool payload compensation (gravity, inertia, analytic Coriolis, and inverse dynamics), continuity-guarded servo IK (`ik_nearest`), tool-frame wrench/stiffness transforms, and per-robot kinematic calibration (`KinematicChain` DH import and `ChainIdentifier` motion fitting) without regenerating driver code.
- **Dual Ground-Truth Verification:**
  Every arm driver is verified in pure Rust `cargo test` against 500 `ssik` IK reference poses (`test_data.bin`) and 1,000 Pinocchio CRBA/RNEA dynamics states (`dynamics_test_data.bin`), without needing Python or Pinocchio at build or test time.

---

## Supported Arms

All OEM crates live under `repos/dexforge-<oem>/rust` and link directly against [`dexhal`](dexhal/README.md):

| Vendor Crate | Arm (`ArmModel::NAME`) | Model / Driver Types | DOF | IK Solver Family | `MAX_SOLUTIONS` | Dynamics (`HAS_DYNAMICS`) |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: |
| `dexforge-ur-core` (`repos/dexforge-ur`) | `ur5` | `Ur5` / `Ur5Driver` | 6 | Closed-form 6R (`ikgeo.three_parallel`, SP6) | 16 | Yes (Pinocchio-verified) |
| `dexforge-franka-core` (`repos/dexforge-franka`) | `franka_panda` | `FrankaPanda` / `FrankaPandaDriver` | 7 | Joint-lock 7R (`jointlock.seven_r`, SP6 inner) | 128 | Yes (Pinocchio-verified) |
| `dexforge-ufactory-core` (`repos/dexforge-ufactory`) | `xarm7` | `Xarm7` / `Xarm7Driver` | 7 | Joint-lock 7R (`jointlock.seven_r`, SP6 inner) | 128 | Yes (Pinocchio-verified) |
| `dexforge-agilex-core` (`repos/dexforge-agilex`) | `piper` | `Piper` / `PiperDriver` | 6 | General 6R (`ikgeo.general_6r`, Raghavan-Roth) | 16 | Yes (Pinocchio-verified) |
| `dexforge-flexiv-core` (`repos/dexforge-flexiv`) | `rizon4` | `Rizon4` / `Rizon4Driver` | 7 | Joint-lock 7R (`jointlock.seven_r`, RR inner) | 256 | Yes (Pinocchio-verified) |
| `dexforge-flexiv-core` (`repos/dexforge-flexiv`) | `rizon10` | `Rizon10` / `Rizon10Driver` | 7 | Joint-lock 7R (`jointlock.seven_r`, RR inner) | 256 | Yes (Pinocchio-verified) |
| `dexforge-unitree-core` (`repos/dexforge-unitree`) | `z1` | `Z1` / `Z1Driver` | 6 | Closed-form 6R (`ikgeo.three_parallel`, SP6) | 16 | Yes (Pinocchio-verified) |

In addition, [`dexforge-mock-core`](dexhal/mock/rust) (`dexhal/mock/rust`) provides a reference `MockDriver` and `MockArm` (`"mock-6r"`) for integration testing of `dexhal` clients.

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

### Python (Optional Bindings & Codegen Tooling)

To build an OEM's Python bindings (`dexforge_<oem>`):

```sh
cd repos/dexforge-ur/python
pip install maturin
maturin develop --release
```

To install the Python dependencies used by the offline [`codegen/`](codegen/README.md) scripts:

```sh
pip install $(grep ssik SSIK_VERSION.txt) sympy numpy pin pytest
```

---

## Usage

### Rust (`dexhal::math::Arm` & `ArmDriver`)

Use `dexhal::math::Arm<Model, DOF, MAX>` for stateless or stateful kinematics, IK, and dynamics (including runtime TCP offsets, payloads, and calibration), or `Driver::new()` for the full `ArmDriver` contract:

```rust
use dexforge_ur_core::{Ur5, Ur5Driver};
use dexhal::math::{Arm, ArmDriver, Payload, Pose, TolerancePolicy};

// 1. Host-side kinematics, IK, and dynamics with runtime TCP & payload
let mut arm = Arm::<Ur5, 6, 16>::new();

let mut tcp = Pose::identity();
tcp[(2, 3)] = 0.12; // 12 cm tool along flange Z
arm.set_tcp(&tcp)?;
arm.set_payload(Payload::new(2.5, [0.0, 0.0, 0.06]))?;

let q = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0];
let qd = [0.0; 6];
let qdd = [0.0; 6];

let pose = arm.fk(&q);                                       // base -> TCP SE(3) pose
let jac = arm.jacobian(&q);                                  // 6x6 geometric Jacobian (base axes)
let q_next = arm.ik_nearest(&pose, &q, TolerancePolicy::DEFAULT); // continuity-guarded servo IK
let dyn_state = arm.dynamics(&q, &qd).unwrap();              // M(q), C(q,qd)*qd, G(q) incl. payload
let tau = arm.inverse_dynamics(&q, &qd, &qdd).unwrap();      // M*qdd + C*qd + G incl. payload

// 2. Full hardware driver contract (manifest + zero-copy state/command views)
let driver = Ur5Driver::new()?;
assert_eq!(driver.manifest().name, "ur5");
```

See [`dexhal/README.md`](dexhal/README.md) for the complete `Arm`, `ArmDriver`, `CoreStateView` / `CoreCommandView`, `manifest.yaml`, and per-robot calibration documentation.

### Python

```python
import numpy as np
import dexforge_ur

q = [0.0] * 6
T = dexforge_ur.ur5.fk(q)                                  # 4x4 SE(3) flange pose
solutions = dexforge_ur.ur5.solve(np.asarray(T), respect_limits=True)
```

### Generating a New OEM Driver

New drivers are generated from an OEM's URDF, `manifest.yaml` values, and hardware API documentation using an autonomous coding agent pointed at [`codegen/new_driver.md`](codegen/new_driver.md). See [`codegen/README.md`](codegen/README.md) for the complete specification, directory layout, and generator CLI reference.

---

## Testing

```sh
# Run dexhal contract & math tests
cargo test --manifest-path dexhal/Cargo.toml

# Run an OEM driver's IK, ArmModel, manifest, and Pinocchio dynamics parity tests
cargo test --manifest-path repos/dexforge-ur/Cargo.toml

# Verify ssik dataset version alignment and run codegen unit tests
python codegen/ik_test_data.py --check
pytest codegen/tests -v

# Run shared Python ssik parity tests (after `maturin develop --release`)
pytest repos/dexforge-*/python/tests -v
```

---

## Documentation Index

- **[`dexhal/README.md`](dexhal/README.md):** Driver contract (`ArmModel`, `Arm`, `ArmDriver`, `ArmManifest`, `CoreState` / `CoreCommand`), payload math, and kinematic calibration.
- **[`dexhal/examples/manifest.yaml`](dexhal/examples/manifest.yaml):** Annotated reference for the driver manifest schema.
- **[`codegen/README.md`](codegen/README.md):** Agent-driven codegen architecture, deterministic generators, and target vendor repository structure.
- **[`codegen/new_driver.md`](codegen/new_driver.md) & [`codegen/translate_ssik_artifact.md`](codegen/translate_ssik_artifact.md):** Master and IK translation prompts for generating new OEM drivers.
- **[`codegen/upgrade_ssik_version.md`](codegen/upgrade_ssik_version.md):** Runbook for upgrading the locked `ssik` version in [`SSIK_VERSION.txt`](SSIK_VERSION.txt).

