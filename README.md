# dexforge

Zero-allocation, `#![no_std]` analytical inverse kinematics translated from the
Python [`ssik`](https://github.com/personalrobotics/ssik) prebuilt artifacts.

The crate is a mathematically faithful port targeting hard real-time control
loops (e.g. a 1 kHz servo thread): there is **no heap allocation**, every data
structure is a fixed-size stack array, and every iterative routine carries a
hardcoded iteration ceiling so worst-case execution time is bounded.

## Layout

| Path | Contents |
| --- | --- |
| `src/math/ssik_core.rs` | Generic bounded-iteration numeric kernels: companion-matrix eigenvalues via a Francis double-shift QR (real Schur) and the `numpy.roots`-equivalent `quartic_roots`. |
| `src/drivers/ur5/ik_solver.rs` | Universal Robots **UR5** closed-form solver (`compute_ik`), ported from `ssik/prebuilt/ur5_ik.py` (solver `ikgeo.three_parallel`). Includes the SP6 subproblem, the SP1+SP3+SP1 chain, FK verification, and the `#[cfg(test)]` reference test. |
| `src/drivers/ur5/test_data.rs` | Auto-generated 500-pose reference dataset (poses with >1 solution). |
| `src/drivers/franka_panda/ik_solver.rs` | Franka Emika **Panda** (7-DOF) closed-form solver (`compute_ik`), ported from `ssik/prebuilt/franka_panda_ik.py` (solver `jointlock.seven_r`). Locks joint 4 across 16 samples, solves the residual 6R sub-chain (SP1/3/4/5 subproblems), FK-verifies, and refines each root with bounded Levenberg-Marquardt. |
| `src/drivers/franka_panda/consts.rs` | Auto-generated baked spatial constants (lock samples, screw axes, link transforms). |
| `src/drivers/franka_panda/test_data.rs` | Auto-generated 500-pose reference dataset (poses with >1 solution). |
| `src/drivers/piper/ik_solver.rs` | AgileX **Piper** (6-DOF) solver (`solve`, `fk`), ported from `ssik/prebuilt/piper_ik.py` (solver `ikgeo.general_6r`, tier 2). Numeric Raghavan-Roth elimination + Manocha-Canny `24x24` companion-eigenvalue route, back-substitution, FK verification, and the `#[cfg(test)]` reference test. |
| `src/drivers/piper/consts.rs` | Auto-generated baked geometry (joint axes/transforms, DH parameters, frame offsets, joint limits). |
| `src/drivers/piper/pq.rs` | Machine-generated `(P_sin, P_cos, P_one, Q)` Raghavan-Roth matrix builder. |
| `src/drivers/piper/test_data.rs` | Auto-generated 500-pose reference dataset (poses with >1 solution; singular poses flagged for FK-residual comparison). |
| `scripts/generate_test_data.py` | Regenerates `test_data.rs` from the reference Python solver. |
| `scripts/generate_franka_consts.py` | Regenerates the Franka `consts.rs` from the Python artifact. |
| `scripts/generate_franka_test_data.py` | Regenerates the Franka `test_data.rs` from the reference Python solver. |
| `scripts/generate_piper_pq.py` | Regenerates the Piper `pq.rs` matrix builder from the Python artifact. |
| `scripts/generate_piper_test_data.py` | Regenerates the Piper `test_data.rs` from the reference Python solver. |
| `examples/piper_solve.rs` | Host harness that runs the Piper solver on stdin poses (used by the Piper data generator). |
| `dexforge-py/` | Secondary crate: Python bindings (pyo3) wrapping `dexforge`. |

## Rust interface

```rust
use nalgebra::SMatrix;
use ssik_rust::drivers::ur5::ik_solver::compute_ik;

let target_tf: SMatrix<f64, 4, 4> = /* SE(3) pose of ee_link in base_link */;
let solutions: [Option<[f64; 6]>; 16] = compute_ik(&target_tf);
for q in solutions.iter().flatten() {
    // q = [q1, q2, q3, q4, q5, q6]
}
```

`compute_ik` reproduces `ur5_ik.solve(T, respect_limits=False)`: the raw
geometric solution set, FK-verified to `1e-5` and de-duplicated modulo `2*pi`.
The UR5 yields at most 8 solutions; unused slots are `None`.

### Franka Emika Panda (7-DOF)

```rust
use nalgebra::SMatrix;
use ssik_rust::drivers::franka_panda::ik_solver::compute_ik;

let target_tf: SMatrix<f64, 4, 4> = /* SE(3) pose of panda_hand in panda_link0 */;
let solutions: [Option<[f64; 7]>; 128] = compute_ik(&target_tf);
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
For deterministic, bounded real-time execution the randomised Möbius /
generalised-eigenvalue fallback used by the Python reference for ill-conditioned
poses is intentionally omitted; equilibration covers the reachable workspace.

## Python interface

```python
import numpy as np
import dexforge_py

T = np.eye(4)  # 4x4 SE(3) target pose
solutions = dexforge_py.solve("ur5", T, respect_limits=False)
for q in solutions:
    # q = [q1, q2, q3, q4, q5, q6]  (radians)
    pass
```

### Building the Python extension

```sh
cd dexforge-py
pip install maturin
maturin build --release          # produces a wheel in ../target/wheels/
pip install ../target/wheels/dexforge_py-*.whl
```

### Running the Python tests

```sh
pip install pytest scipy numpy
pytest dexforge-py/tests/
```

## Dependencies

Only [`nalgebra`](https://nalgebra.org) (with `serde` and `macros` disabled, in
`no_std` + `libm` mode) and [`libm`](https://crates.io/crates/libm) for `f64`
transcendentals.

The Python bindings add [`pyo3`](https://pyo3.rs) as a dependency, isolated to
the `dexforge-py` crate.

## Building and testing

```sh
cargo test                                             # host tests, incl. 500-pose dataset
cargo build --release --target thumbv7em-none-eabihf   # embedded no_std build
```

## Regenerating the test data

```sh
pip install ssik
python scripts/generate_test_data.py
```

## ssik Version Locking

dexforge uses a **locked version** of the `ssik` library (`personalrobotics/ssik`)
to ensure that newly translated artifacts are compatible with the already-translated
core math functions in `src/math/ssik_core.rs` and `src/math/kinematics.rs`.

### Checking the locked version

```sh
cat SSIK_VERSION.txt
```

### Installing the locked version for development

Before running the IK generator or regenerating test data:

```sh
pip install $(grep ssik SSIK_VERSION.txt)
```

### Upgrading the ssik version

Upgrading to a new ssik version requires careful inspection of changes and
regeneration of affected artifacts. See `codegen/upgrade_ssik_version.md` for
detailed instructions.
