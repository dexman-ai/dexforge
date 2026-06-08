# dexforge

Zero-allocation, `#![no_std]` analytical inverse kinematics translated from the
Python [`ssik`](https://github.com/personalrobotics/ssik) prebuilt artifacts.

The crate is a mathematically faithful port targeting hard real-time control
loops (e.g. a 1 kHz servo thread): there is **no heap allocation**, every data
structure is a fixed-size stack array, and every iterative routine carries a
hardcoded iteration ceiling so worst-case execution time is bounded.

## Rust interface

### Universal Robots UR5 (6-DOF)

```rust
use nalgebra::SMatrix;
use ssik_rust::drivers::ur5::ik_solver::{fk, solve, TolerancePolicy};

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
use ssik_rust::drivers::franka_panda::ik_solver::{fk, solve, TolerancePolicy};

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
use ssik_rust::drivers::piper::ik_solver::{fk, solve, TolerancePolicy};

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

## Python interface

`dexforge-py` wraps the Rust solvers via [pyo3](https://pyo3.rs). Each arm is
exposed as a submodule, mirroring the `ssik.prebuilt.<arm>_ik` module
structure:

```python
import numpy as np
import dexforge_py

T = np.eye(4)   # 4×4 SE(3) target pose (numpy array or list-of-lists)
q = [0.0] * 6  # joint angles in radians

solutions = dexforge_py.ur5.solve(T, respect_limits=False)
for q_sol in solutions:
    # q_sol is a list of floats, one per joint
    pass

T_fk = dexforge_py.ur5.fk(q)  # returns a 4×4 list-of-lists
```

Supported arm submodules: `dexforge_py.ur5`, `dexforge_py.piper`,
`dexforge_py.franka_panda`, `dexforge_py.xarm7`, `dexforge_py.rizon4`,
`dexforge_py.rizon10`.

Each `solve` accepts the same optional keyword arguments as the Python ssik
artifact: `max_solutions`, `q_seed`, `respect_limits`, `allow_refinement`,
`fk_atol`, `dedup_atol`, and `refinement_max_iters`.

## Code generation CLI

The whole inverse-kinematics and dynamics code-generation pipeline is wrapped by
a single `dexforge` command:

```sh
pip install dexforge
dexforge build
```

`dexforge build` regenerates every committed Rust driver under
`src/drivers/<arm>/` by running the `codegen/` generators in dependency order
(baked constants → symbolic FK/Jacobian → regression datasets) for all supported
arms. Useful flags:

```sh
dexforge build --list            # print the plan without running anything
dexforge build --arm ur5         # regenerate a single arm (repeatable)
dexforge build --stage kinematics  # only re-run one stage (repeatable)
```

The generators require the *locked* `ssik` version plus `sympy`/`numpy`
(`pip install $(grep ssik SSIK_VERSION.txt) sympy numpy`) and write back into the
checked-out source tree, so run the command from a dexforge clone (install with
`pip install -e .` when iterating). The individual generators are documented
below.

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

1. In `dexforge-py/src/lib.rs`, add `#[pyfunction]` + `#[pyo3(name = "fk")]` /
   `#[pyo3(name = "solve")]` wrappers (copying an existing arm as a template),
   then create a submodule block in `dexforge_py()` and call `m.add_submodule`.

### Symbolic dynamics & kinematics code generation

`codegen/generate_dynamics.py` is a one-time symbolic compiler: it sources the
kinematic chain **directly from the ssik artifact** (the same `*_ik.py` prebuilt
the IK pipeline produces — no URDF parsing), derives **exact** closed-form
forward kinematics and the geometric Jacobian with SymPy, and emits pure,
zero-dependency `#![no_std]` Rust. The ssik artifact encodes the chain as a
product-of-exponentials (`_JOINT_T_LEFTS` / `_JOINT_AXES` / `_JOINT_T_RIGHTS`),
so — like `generate_<arm>_consts.py` — this generator must run **after** the ssik
artifact is available. The degrees of freedom `N` are read from the artifact and
used to size the generated fixed arrays — there is no heap allocation, no
branching, and no loops (the body is a flat list of `sympy.cse()` temporaries).
Every transcendental is rewritten as a `libm::*` call and `pi` maps to
`core::f64::consts::PI`, so the output cross-compiles cleanly for any ARM target.

```sh
pip install ssik sympy numpy
# From a ssik prebuilt artifact name (looks up ssik.prebuilt.<name>):
python codegen/generate_dynamics.py --artifact ur5_ik

# From a local artifact .py file, writing to a driver module:
python codegen/generate_dynamics.py --artifact /path/to/my_arm_ik.py \
    --output src/drivers/<arm>/dynamics.rs
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
# Kinematics from the artifact, masses/inertias from a URDF; default gravity
# is (0, 0, −9.81), override with --gravity GX GY GZ:
python codegen/generate_dynamics.py --artifact ur5_ik \
    --urdf src/drivers/ur5/ur5_inertial.urdf \
    --gravity 0 0 -9.81 \
    --output src/drivers/ur5/dynamics.rs
```

Like FK/Jacobian, every dynamics function is a flat `sympy.cse()` body of `libm`
arithmetic — no heap, no branches, no loops. The derivation is validated against
an independent closed-form planar model in
`codegen/tests/test_generate_dynamics.py`. The path is all-revolute (every arm
currently in the repo); prismatic joints contribute to the linear Jacobian only.

The dynamics path is wired and tested but **not yet baked into a driver**: it
needs a URDF that is a single serial chain of exactly `N` actuated joints, each
carrying an `<inertial>` block. None of the URDFs currently in `src/drivers/`
qualify — `rizon4`/`rizon10` carry no `<inertial>` blocks, and the
`franka_panda`/`piper`/`xarm7` URDFs branch into gripper joints — so only the
FK/Jacobian `generated_kinematics.rs` is baked per arm. Drop in an arm-only
inertial URDF to emit the inverse-dynamics suite.


## Development

### Dependencies

Only [`nalgebra`](https://nalgebra.org) (with `serde` and `macros` disabled, in
`no_std` + `libm` mode) and [`libm`](https://crates.io/crates/libm) for `f64`
transcendentals.

The Python bindings add [`pyo3`](https://pyo3.rs) as a dependency, isolated to
the `dexforge-py` crate.

### Building and testing

```sh
cargo test                                             # host tests, incl. 500-pose dataset
cargo build --release --target thumbv7em-none-eabihf   # embedded no_std build
```

To build and install the Python extension:

```sh
cd dexforge-py
pip install maturin
maturin build --release          # produces a wheel in ../target/wheels/
pip install ../target/wheels/dexforge_py-*.whl
```

### Python tests

The `dexforge-py/tests/` directory contains three test layers:

- **`test_dexforge_py.py`** — cross-validates the native `dexforge_py`
  functions against the reference `ssik.prebuilt` artifacts side by side,
  using random FK-seeded poses. This is the primary correctness gate for
  dexforge.

- **`test_prebuilt_sanity.py`** and **`test_prebuilt_uniform_fuzz.py`** — these
  are verbatim copies from the `ssik` test suite (referenced from
  `dexforge-py/tests/README.md`). They run the ssik reference solver and serve
  as a baseline: if they fail, the reference is broken, not dexforge. The
  Hypothesis strategies in `_hypothesis_strategies.py` are also copied verbatim
  from ssik; `_hypothesis_strategies_dexforge.py` contains dexforge-specific
  variants.

```sh
pip install pytest scipy numpy ssik hypothesis
pytest dexforge-py/tests/
```

### Regenerating the test data

```sh
pip install $(grep ssik SSIK_VERSION.txt)
python codegen/generate_test_data.py
```

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
