# `codegen` — Agent-Driven OEM Driver Generation

`codegen/` contains the prompts, templates, and deterministic Python tools that enable an autonomous coding agent (invoked via standard CLI/IDE tools such as Claude CLI or GitHub Copilot CLI) to generate a complete, verified OEM robot driver crate from hardware specifications.

---

## Design Goals & Principles

1. **Agent-First Orchestration (No Custom LLM Wrapper)**
   - The previous `dexforge_build.py` design—where a Python CLI invoked an OpenAI-compatible chat endpoint in a retry loop and stitched raw text blocks together—is retired.
   - Coding agent sessions are launched directly through standard agent tooling (Claude CLI, Copilot CLI, VS Code autopilot, etc.) pointed at the master prompt [`codegen/new_driver.md`](new_driver.md).
   - Everything in `codegen/` exists to give that coding agent clear instructions, reference examples, code templates, and deterministic scripts it can run and verify autonomously using its own file and shell tools.

2. **Link Against `dexhal` — Never Duplicate Shared Code**
   - Every generated vendor repository and driver crate depends directly on [`dexhal`](../dexhal/README.md) (`dexhal = { ... }`).
   - Generated crates **do not** depend on `dexforge` and **never** vendor or copy `dexhal`'s shared math or contract modules (`ssik_core`, `kinematics`, `postprocess`, `tolerances`, `spatial`, `payload`, `chain`, `identify`, `model`, `arm`, `contract`).
   - `dexhal` owns the driver contract (`ArmModel`, `Arm`, `ArmDriver`, `ArmManifest`, `CoreState` / `CoreCommand` zero-copy views, `CalibrationData`) and shared `#![no_std]` math kernels; each OEM driver crate owns only its arm-specific generated kinematics/IK/dynamics and hardware I/O implementation.

3. **Strict Separation Between Agent Reasoning and Deterministic Scripts**
   - **Coding Agent Responsibilities:**
     1. Translating the straight-line algebraic blocks of an arm's `ssik` Python IK solver into `#![no_std]`, zero-allocation, bounded-loop Rust (`ik_solver.rs`) wired to `dexhal::math` kernels.
     2. Adapting per-arm helper scripts (`generate_code.py`, `generate_test_data.py`) from the kinematic-family templates when solver-specific constant baking or refinement is required.
     3. Implementing the `ArmDriver` hardware I/O methods (`connect`, `disconnect`, `read_state`, `write_command`, and optional native configuration hooks) from the OEM hardware API / protocol documentation and `manifest.yaml`.
   - **Deterministic Script Responsibilities (No Agent Translation):**
     1. Generating the `ssik` Python artifact (`<arm>_ik.py`) from the URDF and detecting the kinematic solver family.
     2. Compiling symbolic forward kinematics, geometric Jacobian, and Lagrangian rigid-body dynamics (`gravity`, `mass_matrix`, `coriolis`, `inverse_dynamics`) via SymPy (`generate_dynamics.py`).
     3. Generating the `ArmModel` implementation (`arm.rs`), nominal `KinematicChain` constants, and crate scaffolding (`Cargo.toml`, `lib.rs`, `mod.rs`) from templates.
     4. Generating binary ground-truth datasets and Rust test harnesses for both analytical IK (`ik_test_data.py` via `ssik`) and rigid-body dynamics (`generate_dynamics_test_data.py` via Pinocchio).

---

## OEM Inputs

Producing a new driver requires three inputs from the OEM (see [`dexhal/README.md`](../dexhal/README.md)):

| Input | Purpose |
| :--- | :--- |
| **1. URDF (`<arm>.urdf`) + `--base` / `--ee` link names** | Defines the nominal kinematic chain (joint axes, origins, types), mechanical position/velocity/effort limits, and per-link `<inertial>` properties (mass, CoM, inertia tensor). |
| **2. Driver Manifest (`manifest.yaml`)** | Declares static driver capabilities, control loop rate, dispatch/staleness latency cycles, mandatory per-joint velocity/acceleration/effort ratings (and optional jerk limits), supported control modes, native compliance flags, I/O counts, and auxiliary state fields. Validated against `dexhal::math::ArmManifest::from_text` (see [`dexhal/examples/manifest.yaml`](../dexhal/examples/manifest.yaml)). |
| **3. OEM Hardware API / Protocol Specification** | Vendor SDK headers, wire protocol specification, or register map used by the agent to implement `ArmDriver::{connect, disconnect, read_state, write_command}`, optional native hooks (`set_payload_native`, `set_tcp_offset_native`, `set_gravity_native`), and optional vendor calibration file parsing. |

---

## Target Vendor Repository Structure

Each OEM lives in its own repository under `repos/dexforge-<oem>/` (with [`repos/dexforge-ur`](../repos/dexforge-ur) as the reference partition and [`dexhal/mock/rust`](../dexhal/mock/rust) as the reference `ArmDriver` implementation):

```text
repos/dexforge-<oem>/
├── Cargo.toml
├── README.md
├── rust/
│   ├── Cargo.toml                              # Depends on dexhal, libm, nalgebra (no_std)
│   ├── manifest.yaml                           # Validated by ArmManifest::from_text
│   └── src/
│       ├── lib.rs                              # Re-exports driver(s) and ArmModel(s)
│       └── drivers/
│           ├── mod.rs
│           └── <arm>/
│               ├── mod.rs                      # Declares arm, ik_solver, generated_*, driver, parity tests
│               ├── arm.rs                      # impl dexhal::math::ArmModel<DOF, MAX> for <Arm>
│               ├── driver.rs                   # impl dexhal::math::ArmDriver for <Arm>Driver
│               ├── ik_solver.rs                # Agent-translated #![no_std] analytical IK + unit tests
│               ├── consts.rs / pq.rs           # (Optional) Baked sub-chain / polynomial constants
│               ├── generated_kinematics.rs     # SymPy-generated symbolic FK & geometric Jacobian
│               ├── generated_dynamics.rs       # SymPy-generated gravity, mass_matrix, coriolis, inverse_dynamics
│               ├── test_data.rs                # #![no_std] loader for IK reference test cases
│               ├── test_data.bin               # Binary ssik IK ground-truth dataset (500 poses)
│               ├── dynamics_parity.rs          # Pinocchio vs. SymPy dynamics parity test suite
│               ├── dynamics_test_data.rs       # #![no_std] loader for Pinocchio golden dataset
│               ├── dynamics_test_data.bin      # Binary Pinocchio RNEA/CRBA ground-truth dataset (1,000 states)
│               ├── <arm>.urdf                  # Source URDF (provenance)
│               └── codegen/
│                   ├── <arm>_ik.py             # Source ssik Python artifact (provenance)
│                   ├── generate_code.py        # (Optional) Emits ../consts.rs or ../pq.rs
│                   └── generate_test_data.py   # Emits ../test_data.bin and ../test_data.rs
└── python/                                     # (Optional) PyO3 bindings & shared ssik parity test suite
```

All generated Rust in `drivers/<arm>/` (except host I/O in `driver.rs` when using OS sockets/SDKs) must be `#![no_std]`, `#![forbid(unsafe_code)]`, heap-free, and bounded-iteration.

---

## Step-by-Step Autonomous Agent Workflow

The master agent prompt guides the coding agent through four sequential phases to build and verify a new driver crate:

### Step 1: Generate and Inspect the `ssik` Artifact
1. Verify the Python environment matches the locked `ssik` version in [`SSIK_VERSION.txt`](../SSIK_VERSION.txt).
2. Run the `ssik` artifact builder against `<arm>.urdf` with `--base <base_link>` and `--ee <ee_link>` to produce `src/drivers/<arm>/codegen/<arm>_ik.py`:
   ```sh
   python codegen/ssik_artifact.py build --urdf <arm>.urdf --base <base_link> --ee <ee_link> --out <arm>_ik.py
   ```
   (`codegen/scaffold.py` runs the same build when no `--artifact` is given and copies the artifact and URDF into the driver directory.)
3. Inspect the generated `<arm>_ik.py` (`python codegen/ssik_artifact.py inspect <arm>_ik.py`) to determine:
   - `DOF` (6 or 7) and `MAX_SOLUTIONS` (16 for 6-DOF, 128 for 7-DOF joint-lock with an SP6 inner solve, 256 with a Raghavan-Roth inner solve).
   - The kinematic solver family and its matching reference example in the repo (detected from the artifact's `SOLVER_NAME` and inner solver):
     - **Category 1 — Closed-form 6R (`ikgeo` SP6: three-parallel / spherical wrist):** reference `ur5` (`repos/dexforge-ur`)
     - **Category 2 — General 6R (`ikgeo.general_6r`, Raghavan-Roth):** reference `piper` (`repos/dexforge-agilex`)
     - **Category 3 — Joint-lock 7-DOF (`jointlock.*`, `SP6` inner):** reference `franka_panda` (`repos/dexforge-franka`); `xarm7` (`repos/dexforge-ufactory`) is an alternate
     - **Category 4 — Joint-lock 7-DOF (`jointlock.*`, Raghavan-Roth inner, e.g. Flexiv Rizon, Kassow):** reference `rizon4` (`repos/dexforge-flexiv`)

### Step 2: Translate Analytical IK and Hardware I/O to Rust
1. **Analytical IK (`ik_solver.rs`):**
   - Following [`translate_ssik_artifact.md`](translate_ssik_artifact.md) and the selected reference arm's `ik_solver.rs`, translate the Python solver's constants and straight-line algebraic blocks into `#![no_std]` Rust.
   - Import and call shared numerical kernels directly from `dexhal::math` (`ssik_core`, `kinematics`, `postprocess`, `tolerances`). Never reimplement subproblem solvers, root finders, LM refinement, joint wrapping, or seed sorting.
   - Ensure `solve()` respects joint limits when `respect_limits == true`, orders solutions nearest-first when `q_seed` is provided, and guards all `acos`/`asin`/`sqrt` domain boundaries without panicking. Joint-lock 7-DOF solvers hold the locked joint at the seed's value when `q_seed` is provided (`seed_locked` in the reference arms).
2. **Hardware Driver (`driver.rs` & `manifest.yaml`):**
   - Place and validate `manifest.yaml` so `ArmManifest::from_text::<Model, DOF, MAX>(MANIFEST)` succeeds.
   - Implement `dexhal::math::ArmDriver` (`manifest`, `connect`, `disconnect`, `read_state`, `write_command`, `arm`, `arm_mut`) against the OEM hardware API documentation using zero-copy `CoreStateViewMut` and `CoreCommandView` buffer views.

### Step 3: Instantiate Per-Arm Codegen Scripts from Templates
`codegen/scaffold.py` renders the templates in `codegen/templates/` (vendor crate skeleton, `mod.rs`, `driver.rs` stub, `manifest.yaml` draft with URDF velocity / effort limits, `codegen/generate_test_data.py`, and `arm.rs`). Then, using the selected reference arm's `codegen/` directory:
1. **`generate_code.py`** *(when required by the solver family)*:
   - For 7-DOF joint-lock arms (`franka_panda`, `xarm7`, `rizon4`, `rizon10`) or Raghavan-Roth / polynomial arms (`piper`), instantiate `src/drivers/<arm>/codegen/generate_code.py` to precompute and bake locked/reversed 6R sub-chain geometry or polynomial coefficient matrices into `../consts.rs` or `../pq.rs`.
2. **`generate_test_data.py`** *(required for all arms)*:
   - Instantiate `src/drivers/<arm>/codegen/generate_test_data.py` to sample 500 reachable SE(3) target poses from `<arm>_ik.py`, apply matching pose polish where applicable, and call `codegen.ik_test_data.write_ik_test_data` to emit `../test_data.bin` and `../test_data.rs`.

### Step 4: Run All Generators and Verify Parity
The coding agent executes the deterministic generators in order and runs the test suites:

1. **Bake Solver Constants (if applicable):**
   ```sh
   python repos/dexforge-<oem>/rust/src/drivers/<arm>/codegen/generate_code.py
   ```
   Produces `consts.rs` / `pq.rs`.

2. **Compile Symbolic Kinematics & Dynamics (SymPy):**
   ```sh
   python codegen/generate_dynamics.py \
       --artifact repos/dexforge-<oem>/rust/src/drivers/<arm>/codegen/<arm>_ik.py \
       --urdf repos/dexforge-<oem>/rust/src/drivers/<arm>/<arm>.urdf \
       --base <base_link> --ee <ee_link> \
       --output repos/dexforge-<oem>/rust/src/drivers/<arm>/generated_dynamics.rs
   ```
   Produces `generated_dynamics.rs` (`DOF`, `fk`, `jacobian`, `gravity`, `mass_matrix`, `coriolis`, `inverse_dynamics`) and re-exports `DOF`, `fk`, and `jacobian` in `generated_kinematics.rs`.

3. **Generate `arm.rs` (`ArmModel` Implementation):**
   ```sh
   python codegen/scaffold.py --oem <oem> --arm <arm> \
       --urdf repos/dexforge-<oem>/rust/src/drivers/<arm>/<arm>.urdf \
       --base <base_link> --ee <ee_link> --only arm.rs
   ```
   Emits `src/drivers/<arm>/arm.rs` implementing `dexhal::math::ArmModel<DOF, MAX>` plus the nominal `KinematicChain` constants (`nominal_chain()`):
   - `const NAME: &'static str = "<arm>";`
   - `const JOINT_LIMITS: [JointLimit; DOF]` sourced from the URDF / `ssik` artifact.
   - `const HAS_DYNAMICS: bool = true;` (when the URDF provides valid `<inertial>` data).
   - `flange_fk`, `flange_jacobian`, `flange_solve`, `flange_gravity`, `flange_mass_matrix`, `flange_coriolis`, and `flange_inverse_dynamics`.

4. **Generate `ssik` IK Ground-Truth Test Data:**
   ```sh
   python repos/dexforge-<oem>/rust/src/drivers/<arm>/codegen/generate_test_data.py
   ```
   Produces `test_data.bin` and `test_data.rs` (500 reachable poses verified against Python `ssik`).

5. **Generate Pinocchio Dynamics Ground-Truth Test Data:**
   ```sh
   python codegen/generate_dynamics_test_data.py \
       --urdf repos/dexforge-<oem>/rust/src/drivers/<arm>/<arm>.urdf \
       --base <base_link> --ee <ee_link> \
       --output-dir repos/dexforge-<oem>/rust/src/drivers/<arm> \
       --arm <arm> --samples 1000 --seed 0
   ```
   Produces `dynamics_test_data.bin`, `dynamics_test_data.rs`, and `dynamics_parity.rs` (1,000 random states comparing SymPy Lagrangian dynamics against Pinocchio CRBA/RNEA).

6. **Build and Run All Tests:**
   ```sh
   cargo test --manifest-path repos/dexforge-<oem>/Cargo.toml
   ```
   The agent diagnoses any compilation errors, FK/IK tolerance mismatches, or Pinocchio dynamics parity failures, updates the generated driver files, and re-runs `cargo test` until the crate builds cleanly and all tests pass.

---

## Generator Technical Details

### Symbolic Kinematics & Dynamics (`generate_dynamics.py`)

`codegen/generate_dynamics.py` is an offline symbolic compiler:
- **Kinematics from the `ssik` artifact:** It sources the product-of-exponentials chain (`_JOINT_T_LEFTS`, `_JOINT_AXES`, `_JOINT_T_RIGHTS`) directly from `<arm>_ik.py`, derives closed-form forward kinematics (`fk(q: &[f64; N]) -> [[f64; 4]; 4]`) and the geometric Jacobian (`jacobian(q: &[f64; N]) -> [[f64; N]; 6]`, rows `[Jv; Jw]`), and emits branch-free `sympy.cse()` expressions using `libm::*` transcendentals and `core::f64::consts::PI`.
- **Manipulator dynamics from the URDF:** When `--urdf <file>` carries `<inertial>` blocks for the actuated links, the compiler aligns the URDF base frame to the `ssik` base frame (`R_base`) and derives the Lagrangian rigid-body dynamics:

| Function | Returns | Meaning |
| :--- | :--- | :--- |
| `gravity(q, g_world)` | `[f64; N]` | Holding torque $\tau = \partial U / \partial q$ for $U = -\sum m_i (g \cdot p_{\text{com},i})$ |
| `mass_matrix(q)` | `[[f64; N]; N]` | Symmetric joint-space inertia $M = \sum (m_i J_{v,i}^T J_{v,i} + J_{w,i}^T R_i I_i R_i^T J_{w,i})$ |
| `coriolis(q, qd)` | `[f64; N]` | Velocity-product torque $C(q, \dot{q})\dot{q}$ from the Christoffel symbols of the first kind |
| `inverse_dynamics(q, qd, qdd, g_world)` | `[f64; N]` | Feed-forward torque $\tau = M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q)$ |

Writing to `generated_dynamics.rs` also emits the sibling `generated_kinematics.rs` (`pub use super::generated_dynamics::{fk, jacobian, DOF};`) so the CSE expressions are compiled once.

### Pinocchio Dynamics Ground Truth (`generate_dynamics_test_data.py`)

`codegen/generate_dynamics_test_data.py` is a one-time generator that builds a restricted Pinocchio model for the exact `base_link -> ee_link` serial chain (stripping world pre-transforms and gripper finger branches), samples 1,000 random `(q, qd, qdd)` states within joint limits, and evaluates:
- `computeGeneralizedGravity` $\rightarrow$ `g(q)`
- `crba` (symmetrized) $\rightarrow$ `M(q)`
- `computeCoriolisMatrix(q, qd) @ qd` $\rightarrow$ `C(q, qd) qd`
- `rnea(q, qd, qdd)` $\rightarrow$ `tau`

It writes `dynamics_test_data.bin`, `dynamics_test_data.rs`, and `dynamics_parity.rs` into the driver directory. Day-to-day `cargo test` runs in pure Rust without requiring Pinocchio.

### Deterministic Spatial Bypass for General 6R (`Raghavan-Roth`)

In Python `ssik`, general 6R solvers (`ikgeo.general_6r`) fall back to a randomized Möbius reparameterization and generalized eigenvalue solve when the equilibrated leading matrix is ill-conditioned (`cond > 1e10`). Because real-time `#![no_std]` drivers forbid RNG and unbounded solvers, translated Rust drivers (e.g. `piper`) replace the Möbius fallback with a **deterministic spatial bypass**: when the unperturbed companion-matrix pass is ill-conditioned or yields no closing in-limit root, the target pose is nudged by 10 µm along fixed tool-frame axes (`PERTURBATIONS`), solved algebraically, and snapped back onto the exact unperturbed target via bounded Levenberg-Marquardt refinement (`lm_refine6`).

### `ssik` Version Locking

All generated IK artifacts and `test_data.bin` files are locked to the `ssik` version in [`SSIK_VERSION.txt`](../SSIK_VERSION.txt):
```sh
pip install $(grep ssik SSIK_VERSION.txt)
python codegen/ik_test_data.py --check
```
See [`upgrade_ssik_version.md`](upgrade_ssik_version.md) when upgrading `SSIK_VERSION.txt`.

---

## Contents of `codegen/`

| File | Role in the New Design |
| :--- | :--- |
| `README.md` | This architecture and workflow specification. |
| `new_driver.md` | Master agent prompt: inputs, hard rules, and the four-step workflow with exact commands. |
| `translate_ssik_artifact.md` | Prompt instructions for translating a `ssik` Python IK solver into `#![no_std]` Rust against `dexhal::math`. |
| `ur5_dynamics_template.md` | Reference checklist and conventions for wiring symbolic dynamics and Pinocchio parity tests. |
| `upgrade_ssik_version.md` | Runbook for upgrading the locked `ssik` version in `SSIK_VERSION.txt` and regenerating arm test data. |
| `ssik_artifact.py` | Deterministic `ssik` artifact builder / inspector: `build` (URDF $\rightarrow$ `<arm>_ik.py`), `inspect` (DOF, `MAX_SOLUTIONS`, joints, limits, solver family + reference arm), and the artifact $\rightarrow$ `KinematicChain` mapping. |
| `scaffold.py` | Deterministic scaffolder: vendor crate skeleton, arm `mod.rs` / `driver.rs` / `manifest.yaml` / `codegen/generate_test_data.py` from `templates/`, and the always-regenerated `arm.rs` (`ArmModel`). |
| `templates/` | Crate and script templates rendered by `scaffold.py` (`vendor/`, `arm/`). |
| `generate_dynamics.py` | Deterministic SymPy compiler: `ssik` artifact + URDF `<inertial>` $\rightarrow$ `#![no_std]` Rust (`fk`, `jacobian`, `gravity`, `mass_matrix`, `coriolis`, `inverse_dynamics`) in `generated_dynamics.rs`, plus the `generated_kinematics.rs` re-export. |
| `generate_dynamics_test_data.py` | Deterministic Pinocchio ground-truth generator: URDF $\rightarrow$ `dynamics_test_data.bin`, `dynamics_test_data.rs`, `dynamics_parity.rs`. |
| `ik_test_data.py` | Shared binary packer/unpacker, `#![no_std]` Rust loader emitter, and monorepo arm-discovery helpers for `ssik` IK reference datasets (`test_data.bin`, `test_data.rs`). |
| `_shared.py` | Pure-Python pose/joint/tolerance conversion helpers for optional Python bindings. |
