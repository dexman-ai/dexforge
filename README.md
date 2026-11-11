# dexforge

Zero-allocation, `#![no_std]` analytical rigid-body kinematics **and dynamics**
translated from the Python
[`ssik`](https://github.com/personalrobotics/ssik) prebuilt artifacts (inverse
kinematics) and a symbolic SymPy compiler (forward kinematics, Jacobian and
inverse dynamics).

The crate is a mathematically faithful port targeting hard real-time control
loops (e.g. a 1 kHz servo thread): there is **no heap allocation**, every data
structure is a fixed-size stack array, and every iterative routine carries a
hardcoded iteration ceiling so worst-case execution time is bounded.

## Rust interface

### Universal Robots UR5 (6-DOF)

```rust
use nalgebra::SMatrix;
use dexforge_examples::drivers::ur5::ik_solver::{fk, solve, TolerancePolicy};

let target_tf: SMatrix<f64, 4, 4> = /* SE(3) pose of ee_link in base_link */;
let solutions: [Option<[f64; 6]>; 16] =
    solve(&target_tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
for q in solutions.iter().flatten() {
    // q = [q1, q2, q3, q4, q5, q6]
}
```

`solve` mirrors `ur5_ik.solve` with the same argument list. The UR5 has a
tier-0 closed form (SP6 + SP1+SP3+SP1 chain) and yields at most 8 solutions;
unused slots are `None`.

### Franka Emika Panda (7-DOF)

```rust
use nalgebra::SMatrix;
use dexforge_examples::drivers::franka_panda::ik_solver::{fk, solve, TolerancePolicy};

let target_tf: SMatrix<f64, 4, 4> = /* SE(3) pose of panda_hand in panda_link0 */;
let solutions: [Option<[f64; 7]>; 128] =
    solve(&target_tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
for q in solutions.iter().flatten() {
    // q = [q1, q2, q3, q4, q5, q6, q7]
}
```

The Panda is kinematically redundant, so `franka_panda_ik.solve` locks joint 4
across a fixed set of samples and solves the resulting 6R sub-problem. The full
discrete solution set is therefore much larger than a 6-DOF arm's: the output is
a `[Option<[f64; 7]>; 128]` stack array (unused slots are `None`).

### AgileX Piper (6-DOF)

```rust
use nalgebra::SMatrix;
use dexforge_examples::drivers::piper::ik_solver::{fk, solve, TolerancePolicy};

let target_tf: SMatrix<f64, 4, 4> = /* SE(3) pose of link6 in base_link */;
let solutions: [Option<[f64; 6]>; 16] =
    solve(&target_tf, None, None, false, false, TolerancePolicy::DEFAULT, 15);
for q in solutions.iter().flatten() {
    // q = [q1, q2, q3, q4, q5, q6]
}
```

The Piper is a generic (non-Pieper) 6R arm, so it is solved with the numeric
Raghavan-Roth elimination followed by the Manocha-Canny `24x24` companion
eigenvalue route in [`math::ssik_core::rr_real_roots`], then back-substitution.

The Python reference falls back to a randomised Möbius reparameterisation +
generalised eigenvalue solve when the equilibrated leading matrix is
ill-conditioned. That path is non-deterministic (RNG) and is not ported.
Instead, ill-conditioned targets are handled by a *deterministic spatial
bypass*: the target pose is nudged 10 µm along a small set of pre-defined
perturbation directions (`PERTURBATIONS`), re-solved through the standard
bounded algebraic route, and the resulting roots are snapped back onto the
original target with bounded Levenberg-Marquardt. This keeps worst-case
execution time bounded and reproducible across the entire reachable workspace.

### Runtime TCP & grasped payload

The end-effector frame (TCP) and the inertial properties of a grasped tool can
be changed **at runtime**, without regenerating any per-arm FK / Jacobian /
dynamics code. Both attach rigidly to the flange (the last actuated link), so a
single constant transform plus the already-generated geometric Jacobian is all
that is needed. The helpers are `#![no_std]`, allocate nothing, run in bounded
time and are fully deterministic — the same constraints as the rest of the
crate.

The recommended entry point is the generic [`math::arm`] facade: every arm is a
zero-sized marker (`Ur5`, `Piper`, `FrankaPanda`, `Xarm7`, `Rizon4`, `Rizon10`)
implementing the [`Arm`] trait, so a single call shape — `fk`, `ik`, `jacobian`,
`inverse_jacobian`, `gravity`, `mass_matrix`, `coriolis` — works for every arm
and takes the runtime TCP / payload as plain arguments.

```rust
use dexforge_examples::math::arm::{Arm, Xarm7, Payload};
use dexforge_examples::math::tcp::Pose;
use dexforge_examples::math::tolerances::TolerancePolicy;

// A tool whose control point sits 12 cm along the flange z axis.
let mut tcp = Pose::identity();
tcp[(2, 3)] = 0.12;

let q = [0.0; 7];
let tcp_pose = Xarm7::fk(&q, &tcp);        // base -> TCP forward kinematics
let j = Xarm7::jacobian(&q, &tcp);         // geometric Jacobian at the TCP
let jinv = Xarm7::inverse_jacobian(&q, &tcp); // Option<DOF×6 pseudo-inverse>

// Inverse kinematics for a desired *TCP* pose (retargeted to the flange):
let tcp_target = Pose::identity();
let solutions = Xarm7::ik(&tcp_target, &tcp, None, None, false, true,
                          TolerancePolicy::DEFAULT, 15);

// Dynamics of a 2.5 kg grasped tool (mass + centre of mass in the flange
// frame). Add each elementwise to the arm's own torque / inertia.
let payload = Payload::new(2.5, [0.0, 0.0, 0.06]);
let qd = [0.1; 7];
let tau_g = Xarm7::gravity(&q, &payload, &[0.0, 0.0, -9.81]); // holding torque
let m_add = Xarm7::mass_matrix(&q, &payload);                 // ΔM(q)
let tau_c = Xarm7::coriolis(&q, &qd, &payload);              // C(q,qd) qd
```

`gravity` returns the exact additional holding torque
`tau_j = -m (g · ∂p_com/∂q_j)` (the same convention the offline dynamics codegen
uses), `mass_matrix` returns the payload's point-mass contribution to the
joint-space inertia `M(q)`, and `coriolis` returns its velocity-product torque
`C(q, qd) qd`. The Coriolis term is required because a grasped tool makes `M(q)`
configuration-dependent — in particular **when the tool centre of mass is off
the end-effector axis**, where the coupling is strongest. Because a runtime tool
is scoped to *mass* and *centre of mass*, no rotational inertia tensor is
required.

The lower-level primitives the facade is built on live in [`math::tcp`]
(`fk_with_tcp`, `flange_target_for_tcp`, `payload_gravity_torque`,
`payload_inertia_matrix`, `flange_point_jacobian`) and can be used directly with
the per-arm `ik_solver`/`generated_kinematics` functions when finer control is
needed.

## Python interface

Each OEM Python package wraps its Rust solvers via [pyo3](https://pyo3.rs).
Arms are exposed as submodules, mirroring the `ssik.prebuilt.<arm>_ik` module
structure:

```python
import numpy as np
import dexforge_ur

T = np.eye(4)   # 4×4 SE(3) target pose (numpy array or list-of-lists)
q = [0.0] * 6  # joint angles in radians

solutions = dexforge_ur.ur5.solve(T, respect_limits=False)
for q_sol in solutions:
    # q_sol is a list of floats, one per joint
    pass

T_fk = dexforge_ur.ur5.fk(q)  # returns a 4×4 list-of-lists
```

Package names are OEM-specific:

- `dexforge_ur.ur5`
- `dexforge_agilex.piper`
- `dexforge_franka.franka_panda`
- `dexforge_ufactory.xarm7`
- `dexforge_flexiv.rizon4`
- `dexforge_flexiv.rizon10`

Each `solve` accepts the same optional keyword arguments as the Python ssik
artifact: `max_solutions`, `q_seed`, `respect_limits`, `allow_refinement`,
`fk_atol`, `dedup_atol`, and `refinement_max_iters`.

## Driver compiler CLI (`dexforge build`)

`dexforge` is an **end-to-end rigid-body driver compiler**: point `dexforge
build` at a URDF and it translates the arm's full **kinematics and dynamics**
into a *stand-alone* Rust crate written **outside** this repository. It calls an
LLM to translate the analytical IK and runs the symbolic compiler
(`codegen/generate_dynamics.py`) to derive the forward kinematics, geometric
Jacobian and — from the URDF's inertial data — gravity / mass-matrix / Coriolis /
inverse-dynamics. The emitted crate is fully self-contained: it copies the shared
math kernels and the per-arm driver sources into the target directory and
generates its own `Cargo.toml` + `lib.rs`, so it builds and runs **without** the
dexforge repository or the `dexforge` Python package. This lets an OEM use
dexforge as a one-shot URDF compiler for their hardware and vendor the resulting
minimal, zero-allocation `#![no_std]` driver straight into firmware.

```sh
pip install dexforge

# End-to-end: compile a kinematics + dynamics driver from a URDF.
dexforge build --urdf my_arm.urdf --base base_link --ee tool0 -o ~/my_arm_ik
cd ~/my_arm_ik && cargo build --release
```

The URDF path drives the LLM-assisted generator (see *Adding a new arm* below):
it builds a `ssik` artifact from the URDF, translates the IK to `#![no_std]`
Rust, builds and tests it, then compiles the symbolic kinematics + dynamics with
`codegen/generate_dynamics.py` and exports the result as the stand-alone crate.
Set `DEXFORGE_LLM_KEY` / `DEXFORGE_LLM_BASE_URL` / `DEXFORGE_LLM_MODEL` (or the
`--llm-key` / `--llm-url` / `--llm-model` flags) for your LLM backend.

The generated crate re-exports the arm's entry points at its root, so a consumer
just calls `my_arm_ik::fk(&q)` / `my_arm_ik::solve(&tf, …)` for kinematics and
reaches the full symbolic kinematics + dynamics suite via
`my_arm_ik::generated_kinematics::{fk, jacobian, gravity, mass_matrix, coriolis,
inverse_dynamics}` (the dynamics functions are emitted when the URDF carries
`<inertial>` data). Useful flags:

```sh
dexforge build --urdf my_arm.urdf --base base_link --ee tool0 \
    -o ~/my_arm_ik --list       # preview the file list, write nothing
dexforge build --urdf my_arm.urdf --base base_link --ee tool0 \
    -o ~/my_arm_ik --no-verify  # skip the standalone cargo build check
```

By default the crate is verified by running `cargo build --release` inside it.
`--output` must be outside the dexforge repository (the whole point is a
self-contained driver).

## Adding a new arm

The `codegen/ik_generator.py` script automates translation of a `ssik`
prebuilt artifact into a `#![no_std]` Rust driver, including the
machine-generated auxiliary files (`consts.rs`, `pq.rs`, `test_data.rs`). It
calls an LLM API, builds the crate, runs the tests, and feeds any failures
back to the LLM for self-correction.

```sh
# From a ssik prebuilt artifact name (looks up ssik.prebuilt):
python codegen/ik_generator.py --artifact ur5_ik --output-dir out/

# From a local artifact .py file:
python codegen/ik_generator.py --artifact /path/to/my_arm_ik.py --output-dir out/

# Generate a fresh artifact from a URDF, then translate:
python codegen/ik_generator.py --urdf robot.urdf --base base_link --ee ee_link \
    --output-dir out/
```

Set `DEXFORGE_LLM_KEY`, `DEXFORGE_LLM_BASE_URL`, and `DEXFORGE_LLM_MODEL` for
your LLM backend (defaults work with local OpenAI-compatible servers).

After the Rust driver is accepted:

1. In `repos/dexforge-<oem>/python/src/lib.rs`, add `#[pyfunction]` + `#[pyo3(name = "fk")]` /
   `#[pyo3(name = "solve")]` wrappers (copying an existing arm as a template),
    then create a submodule block in the OEM module function (for example,
    `dexforge_ur()`) and call `m.add_submodule`.

### Symbolic dynamics & kinematics code generation

`codegen/generate_dynamics.py` is a one-time symbolic compiler: it sources the
kinematic chain **directly from the ssik artifact** (the same `*_ik.py` prebuilt
the IK pipeline produces — no URDF parsing), derives **exact** closed-form
forward kinematics and the geometric Jacobian with SymPy, and emits pure,
zero-dependency `#![no_std]` Rust. The ssik artifact encodes the chain as a
product-of-exponentials (`_JOINT_T_LEFTS` / `_JOINT_AXES` / `_JOINT_T_RIGHTS`),
so — like the per-driver helper scripts under `src/drivers/<arm>/codegen/` —
this generator must run **after** the ssik artifact is available. The degrees of
freedom `N` are read from the artifact and used to size the generated fixed
arrays — there is no heap allocation, no branching, and no loops (the body is a
flat list of `sympy.cse()` temporaries). Every transcendental is rewritten as a
`libm::*` call and `pi` maps to `core::f64::consts::PI`, so the output
cross-compiles cleanly for any ARM target.

```sh
pip install ssik sympy numpy
# From a ssik prebuilt artifact name (looks up ssik.prebuilt.<name>):
python codegen/generate_dynamics.py --artifact ur5_ik

# From a local artifact .py file, writing to a driver module:
python codegen/generate_dynamics.py --artifact /path/to/my_arm_ik.py \
    --output repos/dexforge-<oem>/rust/src/drivers/<arm>/generated_dynamics.rs
```

The emitted module exposes `fk(q: &[f64; N]) -> [[f64; 4]; 4]` (the 4×4
base→end-effector SE(3) pose, matching the existing per-arm `fk` drivers) and
`jacobian(q: &[f64; N]) -> [[f64; N]; 6]` (the `[Jv; Jw]` geometric Jacobian).
Every arm driver's public `fk` (`src/drivers/<arm>/ik_solver.rs`) delegates to
its generated, branch-free `generated_kinematics::fk`
(`src/drivers/<arm>/generated_kinematics.rs`) instead of the old loop over the
baked chain constants. The arms whose algebraic solver still needs a chain FK on
its bit-exact hot path (the `jointlock.seven_r` 7R arms reuse it for refinement)
keep that internal FK; only the public `fk` is rerouted to the generated module.

#### Inverse dynamics (gravity, mass matrix, Coriolis)

The ssik artifact carries only the kinematic chain, so the manipulator
**dynamics** are derived from the *physical* properties in a URDF. Pass
`--urdf <file>` (whose actuated-joint order matches the artifact) and the
generator additionally emits, via the energy (Lagrangian) method:

| Function | Returns | Meaning |
| --- | --- | --- |
| `gravity(q)` | `[f64; N]` | holding torque `τ = ∂U/∂q` for `U = −Σ mᵢ (g·p_comᵢ)` |
| `mass_matrix(q)` | `[[f64; N]; N]` | symmetric inertia `M = Σ (mᵢ Jvᵢᵀ Jvᵢ + Jwᵢᵀ Rᵢ Iᵢ Rᵢᵀ Jwᵢ)` |
| `coriolis(q, qd)` | `[f64; N]` | `C(q, q̇) q̇` from the Christoffel symbols of the first kind |
| `inverse_dynamics(q, qd, qdd)` | `[f64; N]` | MIT-mode feed-forward `τ = M q̈ + C q̇ + g` |

```sh
# Kinematics from the artifact, masses/inertias from a URDF.
# Runtime gravity is an argument to the emitted gravity/inverse_dynamics
# functions (default test vector is (0, 0, −9.81)):
python codegen/generate_dynamics.py --artifact ur5_ik \
    --urdf repos/dexforge-ur/rust/src/drivers/ur5/ur5.urdf \
    --output repos/dexforge-ur/rust/src/drivers/ur5/generated_dynamics.rs
```

`dexforge build` runs this symbolic compiler as part of the end-to-end URDF
flow: after the LLM produces the IK driver, it invokes
`codegen/generate_dynamics.py` against the driver's ssik artifact to emit
`generated_kinematics.rs`, passing the source URDF so the inverse-dynamics suite
(gravity / mass-matrix / Coriolis / inverse-dynamics) is baked whenever the URDF
carries `<inertial>` blocks. A URDF that is a single serial chain of exactly `N`
actuated joints, each with an `<inertial>` block, yields the full dynamics; one
without inertial data (or with branching gripper joints) yields FK/Jacobian only.
The all-revolute path covers every arm currently in the repo; prismatic joints
contribute to the linear Jacobian only. The derivation is validated against an
independent closed-form planar model in
`codegen/tests/test_generate_dynamics.py`.


## Development

### Dependencies

Only [`nalgebra`](https://nalgebra.org) (with `serde` and `macros` disabled, in
`no_std` + `libm` mode) and [`libm`](https://crates.io/crates/libm) for `f64`
transcendentals.

The Python bindings add [`pyo3`](https://pyo3.rs) as a dependency, isolated to
each OEM Python crate under `repos/dexforge-*/python`.

### Building and testing

Each OEM Python package is built independently with maturin from
`repos/dexforge-*/python`.

To build and install an OEM Python extension (example: UR):

```sh
cd repos/dexforge-ur/python
pip install maturin
maturin develop --release
```

### Python tests

Shared ssik-derived tests are now centralized in `ssik_shared_tests/`.

- **Shared test source**: `ssik_shared_tests/`
- **OEM execution points**: `repos/dexforge-*/python/tests/test_shared_suite.py`
- **OEM-local wrappers/manifests**: `repos/dexforge-*/python/tests/ssik_shim/`

```sh
pip install pytest scipy numpy ssik hypothesis
pytest repos/dexforge-*/python/tests/
```

### Regenerating the test data

IK reference datasets (ssik):

```sh
pip install $(grep ssik SSIK_VERSION.txt)
python repos/dexforge-ur/rust/src/drivers/ur5/codegen/generate_test_data.py
# …analogous scripts under other OEM driver `codegen/` dirs
```

### Pinocchio dynamics ground truth (UR5)

`codegen/generate_dynamics_test_data.py` is a **one-time** generator. It loads
the same URDF the driver was built from, samples 10 000 random `(q, qd, qdd)`
states, and asks Pinocchio for gravity torque, mass matrix, Coriolis torque and
inverse-dynamics torque. Results are saved as
`repos/dexforge-ur/rust/src/drivers/ur5/dynamics_test_data.bin` plus a thin
Rust loader `dynamics_test_data.rs`. After that, `cargo test` does **not** need
Pinocchio.

```sh
pip install pin numpy
python codegen/generate_dynamics_test_data.py \
    --urdf repos/dexforge-ur/rust/src/drivers/ur5/ur5.urdf \
    --output-dir repos/dexforge-ur/rust/src/drivers/ur5 \
    --arm ur5 --samples 10000 --seed 0
cd repos/dexforge-ur && cargo test -p dexforge-ur-core dynamics_parity
```

**Conventions** (check these before trusting a mismatch): joint order is the
URDF/Pinocchio actuated order printed at generation time; gravity is the same
base-frame vector on both sides (default `[0, 0, -9.81]`). See the module docs
in `dynamics_test_data.rs` and the script docstring.

Full UR5 wiring + regenerate recipe (template for the next arm):
[`codegen/ur5_dynamics_template.md`](codegen/ur5_dynamics_template.md).

### ssik Version Locking

dexforge uses a **locked version** of the `ssik` library (`personalrobotics/ssik`)
to ensure that newly translated artifacts are compatible with the already-translated
core math functions in `src/math/ssik_core.rs` and `src/math/kinematics.rs`.

```sh
# Check the pinned version:
cat SSIK_VERSION.txt

# Install it before running the IK generator or regenerating test data:
pip install $(grep ssik SSIK_VERSION.txt)
```

Upgrading to a new ssik version requires careful inspection of changes and
regeneration of affected artifacts. See [codegen/upgrade_ssik_version.md](codegen/upgrade_ssik_version.md) for
detailed instructions.
