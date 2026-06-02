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
| `src/math/ssik_core.rs` | Generic bounded-iteration numeric kernels: companion-matrix eigenvalues via a Francis double-shift QR (real Schur) and the `numpy.roots`-equivalent `quartic_roots`. |
| `src/drivers/ur5/ik_solver.rs` | Universal Robots **UR5** closed-form solver (`compute_ik`), ported from `ssik/prebuilt/ur5_ik.py` (solver `ikgeo.three_parallel`). Includes the SP6 subproblem, the SP1+SP3+SP1 chain, FK verification, and the `#[cfg(test)]` reference test. |
| `src/drivers/ur5/test_data.rs` | Auto-generated 500-pose reference dataset (poses with >1 solution). |
| `scripts/generate_test_data.py` | Regenerates `test_data.rs` from the reference Python solver. |
| `ssik-py/` | Secondary crate: Python bindings (pyo3) wrapping `ssik-rust`. |

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
python scripts/generate_test_data.py
```
