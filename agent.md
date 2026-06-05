# agent.md: DexForge (ssik-rust) Coding Standards & Requirements

This document captures the core tenets, architectural principles, and requirements for any AI agent contributing Rust code or translations to the DexForge repository (`ssik-rust`).

## Mission

**ssik-rust** is a zero-allocation, `#![no_std]` analytical inverse kinematics (IK) solver library that translates Python [ssik](https://github.com/personalrobotics/ssik) prebuilt IK artifacts into deterministic, heap-free Rust code suitable for hard real-time robotic control loops (1 kHz servo threads, embedded targets).

Each contribution must preserve **three core invariants**:
1. **Zero allocation**: No `Vec`, `Box`, `BTreeMap`, or any use of the Rust `std` heap.
2. **Deterministic execution time**: Every iterative routine has a hardcoded iteration ceiling; worst-case execution time is bounded and predictable for real-time scheduling.
3. **Mathematical faithfulness**: Results are bit-identical (or within floating-point tolerance) to the Python reference implementation.

---

## Architecture Overview

### Crate Structure

```
ssik-rust/
├── src/
│   ├── lib.rs                    # Library entry point, #![no_std] declaration
│   ├── math/
│   │   ├── ssik_core.rs          # Generic numeric kernels (QR, eigenvalues, roots, dot/cross, rotations)
│   │   └── kinematics.rs         # 4×4 homogeneous utils, LM refinement loop, pose residuals
│   └── drivers/
│       ├── ur5/                  # UR5 (6-DOF, SP6 + SP1/3/1 chain)
│       ├── franka_panda/         # Franka Panda (7-DOF, jointlock.seven_r)
│       ├── piper/                # AgileX Piper (6-DOF, Raghavan-Roth)
│       └── xarm7/                # UFactory xArm7 (7-DOF, jointlock.seven_r)
├── scripts/                      # Python helpers: code generation and test data
├── examples/                     # Harnesses (e.g., piper_solve.rs for stdin filtering)
└── dexforge-py/                  # Secondary crate: pyo3 bindings wrapping ssik-rust
```

### Module Responsibilities

#### `src/math/ssik_core.rs` — Generic Numeric Kernels
- **3-vector primitives**: `dot3`, `cross3`, `norm3`, `add3`, `sub3`, `scale3`, `rotate`, `rotation_matrix`.
- **3×3 matrix operations**: `mat3_mul`, `mat3_transpose`, `mat3_vec`, `mat3t_vec`.
- **Polynomial root-finding**: `quartic_roots` (companion-matrix via Francis QR algorithm), `cubic_roots`.
- **Subproblem solvers**: `sp1_solve`, `sp3_solve`, `sp4_solve`, `sp5_solve`, `sp6_solve` (IK-Geo).
- **Raghavan-Roth general 6R elimination**: `rr_real_roots`, `rr_left_null_space`, `rr_pseudo_inverse`.
- **Angle wrapping**: `wrap_to_pi` (modulo 2π deduplication).

**Key constraint**: All routines are allocation-free, use fixed-size types (arrays or `nalgebra::SMatrix`), and carry bounded-iteration loops.

#### `src/math/kinematics.rs` — Arm-Agnostic Kinematics
- **4×4 homogeneous utilities**: `eye_with`, `rot_homogeneous`, `mat4_mul`.
- **2×2 matrix helpers**: `mat2_ata`, `vec2_mat2`.
- **Pose residual norm**: `norm12` (Frobenius-norm of residual transformation).
- **Rigid-transform type**: `Transform` struct, `IDENTITY`, `compose`, `se3_inv`, `frob_diff`.
- **Generic bounded Levenberg-Marquardt refinement**: `lm_refine<const DOF, F>` (const-generic over degrees of freedom, polymorphic in the residual functor).
- **Backward-compat wrapper**: `lm_refine6` (calls `lm_refine::<6, _>`).

**Key constraint**: Refinement loop has hardcoded maximum iterations (`LM_MAX_ITERS = 15`), initial and final damping factors, and convergence threshold.

#### Per-Arm Drivers (`src/drivers/{arm}/ik_solver.rs`)

Each driver implements:
- **`fk(q: &[f64; DOF]) -> Transform`**: Forward kinematics using baked constants.
- **`solve(tf: &SMatrix<f64, 4, 4>, ...) -> [Option<[f64; DOF]>; MAX_SOLUTIONS]`**: Main IK solver.
  - Arguments: target pose, max_solutions (optional cap), q_seed (nearest-first sorting), respect_limits, allow_refinement, tolerance_policy, refinement_max_iters.
- **`pub const DOF: usize`**: Degrees of freedom.
- **`pub const MAX_SOLUTIONS: usize`**: Upper bound on output array size.
- **Solver metadata constants**: `SOLVER_NAME`, `SOLVER_TIER`, `EXPECTED_MS_MEDIAN`, `FLOP_BUDGET`, `DISPATCH_REASON`.
- **Kinematic constants**: `BASE_LINK`, `EE_LINK`, `T_HOME`, `JOINT_LIMITS`, `JOINT_AXES`, `T_LEFTS`, `T_RIGHTS`.
- **Tolerance policy struct**: `TolerancePolicy { fk_atol, dedup_atol }` with `DEFAULT` constant.
- **`#[cfg(test)]` test module**: Includes reference dataset via `test_data.rs`, validates FK closure and joint-space agreement to tolerance.

**Key patterns**:
- Each driver mirrors the structure of its Python ssik artifact.
- Subproblem kernels are called from `ssik_core`; no reimplementation of complex numerics.
- FK verification is done after each algebraic solve: `frob_diff(compute_fk(&q), &target) < policy.fk_atol`.
- De-duplication is done modulo `2*pi` using `wrap_to_pi`.

---

## Core Tenets for Rust Code

### 1. `#![no_std]` Compliance

Every `.rs` file in `src/` must be compatible with `#![no_std]`:
- **No `std` imports** (except in `#[cfg(test)]` blocks where `extern crate std` is allowed).
- **No heap allocation**: No `Vec`, `Box`, `BTreeMap`, `String` (unless in tests).
- **All data structures** are fixed-size stack arrays: `[T; N]`, `nalgebra::SMatrix`, or small tuples.
- **All loops** carry hardcoded iteration ceilings.

**Verify with**:
```bash
cargo build --release --target thumbv7em-none-eabihf
```

### 2. Deterministic Execution Time

Every numerical loop must have a **hardcoded maximum iteration count**:

```rust
const MAX_ITER: usize = 15;  // IK-Geo or LM refinement
for _ in 0..MAX_ITER {
    // bounded loop body
}
```

- **No unbounded recursion**, no dynamic branching based on runtime convergence.
- **No randomized algorithms** (e.g., random eigenvalue fallbacks). If the Python reference uses Möbius or randomized fallback for ill-conditioned poses, use deterministic perturbation + `lm_refine` instead.
- **No external crate dependencies** beyond `nalgebra` (with `libm` feature for transcendentals) and `libm` itself.

### 3. Type Safety & Fixed-Size Semantics

- **3D vectors**: Use `[f64; 3]` or `nalgebra::SVector<f64, 3>` (prefer array literals for simplicity).
- **4×4 matrices**: Use `nalgebra::SMatrix<f64, 4, 4>` or `[[f64; 4]; 4]` (prefer `SMatrix` for composition via `mat4_mul`).
- **Configuration output**: Always `[Option<[f64; DOF]>; MAX_SOLUTIONS]` (fixed-size, no `Vec`).
- **Angle arrays**: `[f64; DOF]` for joint configurations.

### 4. Numerical Stability & Tolerance Management

- **FK verification tolerance**: Default `1e-5` Frobenius norm (baked as `NUM_TOL`).
- **De-duplication tolerance**: Default `1e-3` per joint modulo `2*pi` (baked as `DEDUP_TOL`).
- **Degeneracy threshold**: `1e-12` (singularity detection).
- **Feasibility threshold**: `1e-8` or `1e-9` (algebraic block constraint checking).
- **Floating-point guard**: Use `1e-15` or machine epsilon for near-zero comparisons (avoid exact `== 0.0`).

**Never hardcode tolerances inline**; declare them as `const` at the module level and reference them by name.

### 5. Representation of Angles

- **All angles in radians**.
- **Joint limits**: Respect and enforce via `respect_limits` flag (check before returning solution).
- **Wrap-to-pi de-duplication**: Use `wrap_to_pi(angle)` to canonicalize angles modulo `2*pi` before comparing two solutions.

Example:
```rust
fn q_eq_mod(a: &[f64; 6], b: &[f64; 6], tol: f64) -> bool {
    a.iter().zip(b.iter()).all(|(ai, bi)| {
        (wrap_to_pi(ai - bi)).abs() < tol
    })
}
```

### 6. Forward Kinematics (FK)

- Every solution candidate must be FK-verified before returning.
- FK uses **product-of-exponentials** with baked screw axes and link transforms.
- Baked constants (axes, transforms, limits) are typically generated by Python helper scripts (`generate_<arm>_consts.py`).
- Never hardcode FK transformations inline; use baked constants.

Example pattern (UR5):
```rust
const JOINT_AXES: [[f64; 3]; 6] = [ /* from Python */ ];
const T_LEFTS: [[[f64; 4]; 4]; 6] = [ /* from Python */ ];

fn fk(q: &[f64; 6]) -> SMatrix<f64, 4, 4> {
    let mut T = IDENTITY;
    for i in 0..6 {
        T = mat4_mul(&T, &rot_homogeneous(&JOINT_AXES[i], q[i]));
        T = mat4_mul(&T, &T_LEFTS[i]);
    }
    T
}
```

### 7. Testing & Validation

Every driver must include a `#[cfg(test)]` module:
- **Include reference dataset** via `#[path = "test_data.rs"]` macro.
- **Test FK closure** at home pose (q = [0; DOF]).
- **Test round-trip**: FK of the home pose should recover the home transform.
- **Test reference dataset match**: For each pose in `test_data.rs`, solve and assert joint-space agreement to `1e-5`.
- **Test respect_limits flag**: Verify out-of-range solutions are dropped.
- **Test max_solutions cap**: Verify the output array respects any user-provided cap.
- **Test q_seed sorting**: Verify nearest-first ordering when seed is provided.

**Run tests**:
```bash
cargo test
```

### 8. Documentation

- **Module-level docs** (`//!`): Explain the arm's kinematics, solver name/tier, equivalence to Python reference.
- **Function docs** (`///`): Describe argument types, return semantics, tolerance parameters.
- **Constant docs**: Include `SOLVER_NAME`, `SOLVER_TIER`, `EXPECTED_MS_MEDIAN`, `FLOP_BUDGET`, `DISPATCH_REASON` with exact mirroring of Python artifact metadata.
- **Inline comments**: Only for complex numerical blocks or non-obvious variable roles.

Example:
```rust
/// Analytical inverse kinematics for the **Franka Emika Panda** (7-DOF).
///
/// This is a deterministic, `#![no_std]`, zero-allocation translation of the
/// Python `ssik` prebuilt artifact `ssik/prebuilt/franka_panda_ik.py`
/// (solver `jointlock.seven_r`, tier 1) into Rust.
///
/// ## Algorithm
///
/// The Franka is a redundant 7R arm...
```

### 9. Code Style & Formatting

- **Line length**: Aim for ≤100 characters for readability (not strict).
- **Indentation**: 4 spaces (Rust standard).
- **Naming**: `snake_case` for functions/variables, `SCREAMING_SNAKE_CASE` for constants.
- **Comments**: Use existing comments as style reference; avoid over-commenting.
- **Float literals**: Write `2.0_f64` or `1e-5` explicitly for clarity; avoid ambiguity.

**Format with**:
```bash
cargo fmt
```

---

## Core Tenets for Python-to-Rust Translations

### 1. Reference Implementation Selection

Before writing Rust code, **study the existing translated reference** that most closely matches the target arm's kinematic family:

| Arm | Solver Type | Reference |
|-----|-------------|-----------|
| UR5 | SP6 + SP1/3/1 chain | `src/drivers/ur5/ik_solver.rs` |
| Franka Panda (7R) | Jointlock 7R (locks joint 4, solves 6R subproblem) | `src/drivers/franka_panda/ik_solver.rs` |
| AgileX Piper (6R) | Raghavan-Roth elimination + Manocha-Canny eigenvalue | `src/drivers/piper/ik_solver.rs` |
| UFactory xArm7 (7R) | Jointlock 7R (same as Franka) | `src/drivers/xarm7/ik_solver.rs` |

**Mirror the structure, constants layout, and test pattern exactly from the closest match.**

### 2. The `ssik_core` Boundary (CRITICAL)

**DO NOT reimplement complex numerical kernels.** The heavy mathematical lifting is extracted into hand-maintained libraries:
- `src/math/ssik_core.rs` — bounded-iteration eigenvalue (QR), root-finding, vector/matrix ops.
- `src/math/kinematics.rs` — homogeneous transformations, LM refinement.

**Your responsibility**:
1. Extract **arm-specific constants** (link lengths, axes, limits, home pose) from the Python artifact.
2. Extract **straight-line algebraic blocks** (scalar arithmetic, transcendental function chains).
3. **Wire them** to the static `ssik_core` and `kinematics` functions.

**Kernel mapping** (critical):

| Python | Rust | Location |
|--------|------|----------|
| `_sp6_runtime.solve(...)` | `sp6_solve(...)` | `ssik_core` |
| `sp1_solve(...)` | `sp1_solve(...)` | `ssik_core` |
| `sp3_solve(...)` | `sp3_solve(...)` | `ssik_core` |
| `sp4_solve(...)` | `sp4_solve(...)` | `ssik_core` |
| `sp5_solve(...)` | `sp5_solve(...)` | `ssik_core` |
| `numpy.roots(...)` (degree ≤ 4) | `quartic_roots(...)` or `cubic_roots(...)` | `ssik_core` |
| `eliminate_q0_q1(...)` | `rr_real_roots(...)` | `ssik_core` |
| `pinv(...)` fallback | **Deterministic perturbation + `lm_refine`** (Möbius is non-deterministic, omit it) | `kinematics` |
| `lm_refine(...)` | `lm_refine::<DOF, _>(...)` or `lm_refine6(...)` | `kinematics` |

### 3. Translation Execution Steps

#### Step 1: Extract Constants
Extract all hardcoded spatial dimensions, link lengths, joint limits, and baseline offsets from the Python script. Emit them as Rust `const` arrays:
- DOF, MAX_SOLUTIONS
- Screw axes: `const JOINT_AXES: [[f64; 3]; DOF] = [...]`
- Transforms: `const T_LEFTS: [[[f64; 4]; 4]; DOF] = [...]`, `const T_RIGHTS: ...`
- Joint limits: `const JOINT_LIMITS: [[f64; 2]; DOF] = [...]`
- Home pose: `const T_HOME: [[f64; 4]; 4] = [...]`

**Do NOT "simplify" geometry or zero out floating-point noise.** Preserve exact literals from Python (e.g., `6.123233995736766e-17` is intentional, not a bug).

#### Step 2: Lower the Algebraic Block
Transpile the straight-line scalar arithmetic blocks (e.g., `th14_x*`, `q6_x*`, `dinr_*`, SP3 boundary branches):
- `math.sin(x)` → `x.sin()`
- `math.cos(x)` → `x.cos()`
- `math.atan2(y, x)` → `y.atan2(x)`
- `math.asin(x)` → `x.asin()` (check `-1 ≤ x ≤ 1` to avoid `NaN`)
- `math.acos(x)` → `x.acos()` (check `-1 ≤ x ≤ 1` to avoid `NaN`)
- `math.sqrt(x)` → `x.sqrt()` (check `x ≥ 0` to avoid `NaN`)
- `math.atan(x)` → `x.atan()`

**Preserve exact variable assignment ordering** to prevent floating-point accumulation drift.

#### Step 3: Define the Artifact Interface
Generate a Rust interface matching the Python artifact and the Rust reference template:

```rust
pub fn solve(
    tf: &SMatrix<f64, 4, 4>,
    max_solutions: Option<usize>,
    q_seed: Option<&[f64; DOF]>,
    respect_limits: bool,
    allow_refinement: bool,
    policy: TolerancePolicy,
    refinement_max_iters: usize,
) -> [Option<[f64; DOF]>; MAX_SOLUTIONS]

pub fn fk(q: &[f64; DOF]) -> SMatrix<f64, 4, 4>
```

#### Step 4: Handle NaN-Producing Operations
Operations that could result in `NaN` (e.g., `acos` or `asin` receiving an argument outside `[-1.0, 1.0]`) must be caught and handled explicitly. Do **NOT panic**; instead, `continue` to the next potential solution root or skip the candidate:

```rust
let c = (val).max(-1.0).min(1.0);  // clamp to [-1, 1] to avoid acos NaN
let theta = c.acos();
```

or

```rust
if !(val >= -1.0 && val <= 1.0) {
    continue;  // skip this root
}
let theta = val.asin();
```

### 4. Reference Dataset & Python Helper Scripts

Generate Python helper scripts that regenerate the auxiliary Rust files:

| Script | Generates | Required? |
|--------|-----------|-----------|
| `generate_<arm>_consts.py` | `src/drivers/<arm>/consts.rs` | If arm uses precomputed constants (jointlock 7-DOF arms) |
| `generate_<arm>_pq.py` | `src/drivers/<arm>/pq.rs` | If arm uses Raghavan-Roth PQ matrices |
| `generate_<arm>_test_data.py` | `src/drivers/<arm>/test_data.rs` | Always required |

Each script must:
1. Import the relevant `ssik.prebuilt.<arm>_ik` artifact.
2. Write Rust source to `src/drivers/<arm>/<filename>.rs`.
3. Compute output path as `os.path.join(os.path.dirname(__file__), "..", "src", "drivers", "<arm>", "<filename>.rs")`.
4. Print confirmation: `"wrote src/drivers/<arm>/<filename>.rs"`.

**Example test_data generation**:
```python
# scripts/generate_<arm>_test_data.py
import numpy as np
from ssik.prebuilt import <arm>_ik
import os

solver = <arm>_ik.IKSolver()
poses = generate_diverse_poses()  # or load from reference dataset

cases = []
for T in poses:
    sols = solver.solve(T, respect_limits=False)
    cases.append((T, sols))

# Write to Rust struct and array
output = f"""
pub const CASES: &[([f64; 16], [[f64; {DOF}]; MAX_SOLUTIONS])] = &[
    ...
];
"""
```

### 5. Testing Requirements

Generate a `#[cfg(test)]` module in `ik_solver.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SMatrix;
    use std::vec::Vec;

    fn tf_from_row_major(a: &[f64; 16]) -> SMatrix<f64, 4, 4> {
        // convert from test_data row-major format
    }

    fn q_eq_mod(a: &[f64; DOF], b: &[f64; DOF], tol: f64) -> bool {
        a.iter().zip(b.iter()).all(|(ai, bi)| {
            (wrap_to_pi(ai - bi)).abs() < tol
        })
    }

    #[test]
    fn fk_home_pose() {
        let T_computed = fk(&[0.0; DOF]);
        // check close to T_HOME
    }

    #[test]
    fn home_pose_round_trips() {
        let T_home = tf_from_row_major(&T_HOME);
        let sols = solve(&T_home, None, None, false, false, TolerancePolicy::DEFAULT, 15);
        let q0 = sols[0].expect("home pose should have solution");
        assert!(q_eq_mod(&q0, &[0.0; DOF], 1e-5));
    }

    #[test]
    fn matches_ssik_reference_dataset() {
        for (T_raw, expected_sols) in test_data::CASES.iter() {
            let T = tf_from_row_major(T_raw);
            let sols = solve(&T, None, None, false, false, TolerancePolicy::DEFAULT, 15);
            // Check that all expected_sols (modulo dedup) are in sols
        }
    }
}
```

**Tolerance strategy**:
- For **strict (non-singular)** poses: Joint-space agreement `< 1e-5`.
- For **singular** poses: FK residual `< 1e-5` AND joint-space agreement `< 1e-3` (relaxed due to redundancy).
- Use `q_eq_mod` with `wrap_to_pi` for modulo-2π comparisons.

### 6. Handling Singular & Near-Singular Poses

For 7R arms with full-revolution locked joints (e.g., xArm7 joint 5, limits ±6.2832 ≈ ±2π):
- The lock-sample endpoint (`-6.2832`) is ~2π-equivalent to interior sample `0.0` but differs from `-2π` by machine epsilon.
- The surviving 2π-representative after `wrap_to_pi` dedup is machine-epsilon-noise-dependent.
- **Never require bit-for-bit match on singular poses.**
- Instead: Always test **sound closure** (FK residual `< 1e-5`) and allow **reference match** (deterministic kernel match) only for **non-singular (equal-count)** poses.

**Test design**:
- Mark poses as `singular: bool` in test data.
- For singular poses: Check FK closure only.
- For non-singular poses: Check both FK closure AND joint-space agreement.

---

## Example: Complete Translation Workflow

### Input: Python Artifact `my_arm_ik.py`
```python
# Python ssik artifact
DOF = 6
MAX_SOLUTIONS = 16
SOLVER_NAME = "my_solver"
SOLVER_TIER = 2

# ... geometry ...
JOINT_AXES = np.array([[...], ...])
T_LEFTS = np.array([[...], ...])

def solve(T, respect_limits=False, ...):
    solutions = []
    for th_sample in range(16):
        # algebraic solve...
        q = [...]
        if verify_fk(q, T):
            solutions.append(q)
    return solutions

def fk(q):
    T = np.eye(4)
    for i in range(DOF):
        T = T @ exp_so3(q[i] * JOINT_AXES[i]) @ T_LEFTS[i]
    return T
```

### Output: Rust Driver `src/drivers/my_arm/ik_solver.rs`
```rust
//! My Arm IK solver, translated from ssik prebuilt artifact.

use nalgebra::SMatrix;
use crate::math::ssik_core::{...};
use crate::math::kinematics::{...};

pub const DOF: usize = 6;
pub const MAX_SOLUTIONS: usize = 16;
pub const SOLVER_NAME: &str = "my_solver";
pub const SOLVER_TIER: u32 = 2;

const JOINT_AXES: [[f64; 3]; 6] = [ /* from Python */ ];
const T_LEFTS: [[[f64; 4]; 4]; 6] = [ /* from Python */ ];

pub fn fk(q: &[f64; 6]) -> SMatrix<f64, 4, 4> {
    // ... implementation ...
}

pub fn solve(
    tf: &SMatrix<f64, 4, 4>,
    max_solutions: Option<usize>,
    q_seed: Option<&[f64; 6]>,
    respect_limits: bool,
    allow_refinement: bool,
    policy: TolerancePolicy,
    refinement_max_iters: usize,
) -> [Option<[f64; 6]>; 16] {
    // ... implementation ...
}

#[cfg(test)]
mod tests {
    // ... test implementation ...
}
```

### Output: Python Helper `scripts/generate_my_arm_consts.py`
```python
#!/usr/bin/env python3
"""Generate consts.rs for my_arm from ssik artifact."""

import numpy as np
from ssik.prebuilt import my_arm_ik
import os

# Extract constants and write Rust code...
output = f"""
// Generated by scripts/generate_my_arm_consts.py
pub const JOINT_AXES: [[f64; 3]; 6] = {json.dumps(...)};
...
"""

output_path = os.path.join(os.path.dirname(__file__), "..", "src", "drivers", "my_arm", "consts.rs")
with open(output_path, 'w') as f:
    f.write(output)
print(f"wrote {output_path}")
```

---

## Build & Test Verification

### Cargo Commands
```bash
# Host tests (includes 500-pose reference dataset)
cargo test

# Release build with optimizations
cargo build --release

# Embedded (no_std) build
cargo build --release --target thumbv7em-none-eabihf

# Code formatting
cargo fmt

# Linting
cargo clippy
```

### Python Tests (if using dexforge-py bindings)
```bash
cd dexforge-py
maturin build --release
pip install ../target/wheels/dexforge_py-*.whl
pytest tests/
```

---

## Important References & Links

- **Python ssik repository**: https://github.com/personalrobotics/ssik
- **nalgebra documentation**: https://nalgebra.org
- **Rust no_std guide**: https://docs.rust-embedded.org/book/
- **GitHub Issues**:
  - [Issue #36](https://github.com/dexman-ai/dexforge/issues/36) — Xarm7 IK translation
  - [Issue #12](https://github.com/dexman-ai/dexforge/issues/12) — (Reference for core tenets)
- **Key scripts**:
  - `codegen/ik_generator.py` — Automated LLM-driven translation pipeline
  - `codegen/translate_ssik_artifact.md` — Detailed translation prompt template

---

## Checklist for Contributions

Before submitting a PR with new Rust IK code or translations:

- [ ] Code compiles with `cargo build --release`.
- [ ] Code compiles with `cargo build --release --target thumbv7em-none-eabihf` (no_std).
- [ ] All tests pass: `cargo test`.
- [ ] No `unsafe`, no `Vec`/`Box`/`std` heap allocation outside `#[cfg(test)]`.
- [ ] All loops have hardcoded iteration ceilings.
- [ ] FK verification is done for every solution candidate.
- [ ] De-duplication uses `wrap_to_pi` modulo 2π.
- [ ] Reference dataset in `test_data.rs` matches expected solutions to tolerance.
- [ ] Documentation includes SOLVER_NAME, SOLVER_TIER, EXPECTED_MS_MEDIAN, FLOP_BUDGET.
- [ ] Python helper scripts (`generate_<arm>_*.py`) are provided and tested.
- [ ] Code follows Rust style: `cargo fmt --check` passes.
- [ ] No hardcoded tolerances; all thresholds use named constants.
- [ ] Numerical domain checks (acos, asin, sqrt) prevent NaN.

---

## Emergency Contacts & Escalation

- **Blocked on numerical stability**: Check `ssik_core.rs` routines and their test tolerances.
- **Blocked on no_std compliance**: Ensure no `extern crate std` outside `#[cfg(test)]`, no Vec, no Box.
- **Blocked on reference matching**: Review test data generation script (`generate_<arm>_test_data.py`); may need to run Python solver on real hardware for custom arms.
- **Blocked on performance**: Profile with `cargo bench` (if benchmarking infrastructure is set up); check loop iteration counts and constant extraction.

---

**Last updated**: 2026-06-05  
**Maintained by**: DexForge Contributors
