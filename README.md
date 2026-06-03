# ssik-rust

Zero-allocation, `#![no_std]` analytical inverse kinematics translated from the
Python [`ssik`](https://github.com/personalrobotics/ssik) prebuilt artifacts.

The crate is a mathematically faithful port targeting hard real-time control
loops (e.g. a 1 kHz servo thread): there is **no heap allocation**, every data
structure is a fixed-size stack array, and every iterative routine carries a
hardcoded iteration ceiling so worst-case execution time is bounded.

## Layout

| Path | Contents |
| --- | --- |
| `src/math/ssik_core.rs` | Arm-agnostic numeric kernels shared by every driver: companion-matrix eigenvalues via a Francis double-shift QR (real Schur), the `numpy.roots`-equivalent `quartic_roots`, vector primitives (`dot3`/`cross3`/`rotate`/`wrap_to_pi`) and the full **SP6** subproblem (`sp6_solve`). |
| `src/math/kinematics.rs` | Generic, const-DOF forward-kinematics chain (`fk_chain`/`fk_residual`) and the wrap-modulo dedup helper (`q_close_wrap`) used by all drivers. |
| `src/drivers/ur5/ik_solver.rs` | Universal Robots **UR5** closed-form solver (`compute_ik`), ported from `ssik/prebuilt/ur5_ik.py` (solver `ikgeo.three_parallel`). Uses the shared SP6 + FK core; keeps only the baked constants and the SP1/SP3 closed forms. |
| `src/drivers/z1/ik_solver.rs` | Unitree **Z1** closed-form solver, ported from `ssik/prebuilt/z1_ik.py` (solver `ikgeo.three_parallel`, same UR-class structure as the UR5). |
| `src/drivers/puma560/ik_solver.rs` | **PUMA 560** closed-form solver, ported from `ssik/prebuilt/puma560_ik.py` (solver `ikgeo.spherical_two_parallel`). Fully inlined SP4/SP3/SP1 chain — no SP6. |
| `src/drivers/*/test_data.rs` | Auto-generated 500-pose reference datasets per arm (poses with >1 solution). |
| `scripts/generate_test_data.py` | Multi-arm generator; regenerates each `test_data.rs` from the reference Python solver. Run with no arguments for all arms, or pass arm names (e.g. `python scripts/generate_test_data.py puma560`). |
| `ssik-py/` | Secondary crate: Python bindings (pyo3) wrapping `ssik-rust`. |

## Supported arms and scope

Every `ssik` prebuilt artifact targets one of four solver families. This crate
ports the **closed-form** families, which translate 1-to-1 into bounded,
heap-free Rust:

| Arm | Solver family | Status |
| --- | --- | --- |
| UR5 | `ikgeo.three_parallel` | ✅ ported & verified |
| Z1 | `ikgeo.three_parallel` | ✅ ported & verified |
| PUMA 560 | `ikgeo.spherical_two_parallel` | ✅ ported & verified |

The remaining `ssik` artifacts are **out of scope** for this crate because
their solvers are not closed form and would require unbounded, allocation-heavy
numeric infrastructure incompatible with the `#![no_std]`, hard-real-time
goals above:

- **`ikgeo.general_6r`** (jaco2, xarm6, piper, fanuc, yam, big_yam): the generic
  Raghavan–Roth pipeline — a 24×24 generalized eigenproblem, a pseudo-inverse,
  and a degree-16 univariate solve — with data-dependent iteration counts.
- **`seven_r.*` / `jointlock.seven_r`** (iiwa14, gen3, fr3, panda, xarm7,
  rizon4/10, kassow): 7-DOF redundancy resolution requiring a 1-D self-motion
  search plus large cached symbolic coefficient blobs.

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
The UR5 yields at most 8 solutions; unused slots are `None`. The same applies to
the `z1` and `puma560` drivers under `ssik_rust::drivers::{z1, puma560}`.

## Python interface

```python
from scipy.spatial.transform import Rotation
import ssik_py

rot = Rotation.from_euler("xyz", [0.1, 0.2, 0.3])
solutions = ssik_py.compute_ik(x=0.3, y=0.1, z=0.5, rotation=rot, arm="ur5")
for q in solutions:
    # q = [q1, q2, q3, q4, q5, q6]  (radians)
    pass
```

Supported `arm` values are `"ur5"`, `"z1"` and `"puma560"`.

### Building the Python extension

```sh
cd ssik-py
pip install maturin
maturin build --release          # produces a wheel in ../target/wheels/
pip install ../target/wheels/ssik_py-*.whl
```

### Running the Python tests

```sh
pip install pytest scipy numpy
pytest ssik-py/tests/
```

## Dependencies

Only [`nalgebra`](https://nalgebra.org) (with `serde` and `macros` disabled, in
`no_std` + `libm` mode) and [`libm`](https://crates.io/crates/libm) for `f64`
transcendentals.

The Python bindings add [`pyo3`](https://pyo3.rs) as a dependency, isolated to
the `ssik-py` crate.

## Building and testing

```sh
cargo test                                             # host tests, incl. 500-pose dataset
cargo build --release --target thumbv7em-none-eabihf   # embedded no_std build
```

## Regenerating the test data

```sh
pip install ssik
python scripts/generate_test_data.py           # all arms
python scripts/generate_test_data.py puma560    # a single arm
```
