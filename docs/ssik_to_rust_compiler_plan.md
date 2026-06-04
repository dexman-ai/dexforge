# `ssik_to_rust_compiler.py` — Feasibility Analysis & Implementation Plan

> **Status: PLAN / REVIEW GATE.** This document is the deliverable requested by
> the issue *"Write an AST Compiler to translate ssik python artifacts to
> Rust"*, which asks us to **"determine the feasibility of the approach … and
> generate a detailed plan … If feasibility is limited or conditioned on
> additional decisions, explain the limitations and the required decisions and
> ask for a review before proceeding with coding."**
>
> The conclusion below is that a *pure* `ast.NodeVisitor` line-by-line
> transpiler — as literally specified in the issue — **cannot** produce a
> correct, zero-dependency, `#![no_std]` Rust solver, because the assumptions
> the issue makes about the shape of an `ssik` artifact do not hold. A
> **scoped, AST-assisted code generator** *is* feasible and is the recommended
> path. Several decisions are required before coding; they are listed in
> §5 and we are explicitly requesting review on them.

---

## 1. TL;DR verdict

| Issue assumption | Reality (verified against `personalrobotics/ssik@ef29a4f`) | Impact |
| --- | --- | --- |
| The artifact "consists entirely of flat, math-heavy linear algebra and trigonometry." | It is mostly that *for the per-arm algebraic block*, but the **hard numeric kernels are not in the file** — they are `import`ed from the `ssik` runtime. | A visitor over the artifact AST never *sees* the core algorithm. |
| "The `ssik` Python output utilizes a polynomial companion matrix to extract eigenvalues" *in the file*. | No artifact builds a companion matrix or calls `numpy.roots`/`eig`. UR5 calls `ssik.subproblems.sp6.solve(...)`; Piper calls `ssik.solvers.ikgeo._raghavan_roth.*` and `numpy.linalg.pinv`. | The "hook the companion coefficients into a static solver" step has nothing in the AST to hook. |
| `numpy.dot` / `numpy.eye` / `numpy.zeros` map 1:1 to `nalgebra`. | Artifacts use `np.asarray`, `np.linalg.norm`, `np.linalg.pinv`, `np.cross`, broadcasting, and **slice assignment** (`J[:3, i] = ...`, `rot[:3,:3] = ...`). | These have no mechanical 1:1 `nalgebra` translation. |
| The solver is deterministic and bounded. | The tier-2 path uses a **randomised Möbius / generalised-eigenvalue fallback** (`solve_x2_roots_mobius`) and `np.linalg.pinv`. | Directly violates the DexZero real-time constraints; cannot be transpiled verbatim. |
| Generated Rust should "match the artifacts already translated (UR5, Franka, Piper)." | Those Rust files are **hand-restructured ports**, not the output of any mechanical transpile (see §4). | A transpiler that targets their structure must replicate human design decisions. |

**Recommendation:** build an **AST-assisted code generator** (Option A, §4) that
extracts the per-arm *algebraic block* and *baked constants* from an artifact
and emits them into Rust against a **hand-maintained, reusable `#![no_std]`
kernel library** (the already-existing `src/math/ssik_core.rs` +
`src/math/kinematics.rs`). Do **not** attempt a general Python→Rust transpiler.

---

## 2. Evidence: what an `ssik` artifact actually contains

Source examined: `personalrobotics/ssik`, `src/ssik/prebuilt/ur5_ik.py`,
`piper_ik.py`, `franka_panda_ik.py` (commit `ef29a4f`).

### 2.1 Imports pull the real algorithm in from the runtime

`ur5_ik.py`:

```python
import math
import numpy as np
import cython
from ssik.subproblems import sp6 as _sp6_runtime
from ssik._kinbody import Joint, KinBody, Link
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.refinement import lm_refine as _lm_refine
from ssik.postprocess import nearest_to_seed, respect_limits, wrap_to_limits
from ssik.subproblems._rotation import rotation_matrix as _rotation_matrix
```

`piper_ik.py` additionally:

```python
from ssik.solvers.ikgeo._raghavan_roth import (
    eliminate_q0_q1, weierstrass_eliminate_trig, build_m_matrix,
    solve_x2_roots_mobius, _back_substitute_inner, _fk_dh,
)
```

The artifact body then **calls** these opaque functions:

```python
# ur5_ik.py — the (q1,q5) pair (which is the quartic / SP6 solve) is a runtime call:
theta15_solutions, _ = _sp6_runtime.solve(h_sp, k_sp, p_sp, 0.10915, 0.0)

# piper_ik.py — RR elimination + eigen-root extraction are runtime calls:
e_sin, e_cos, e_one = _ssik_eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
e_quad, e_lin, e_const = _ssik_weierstrass(e_sin, e_cos, e_one)
m_quad, m_lin, m_const = _ssik_build_m_matrix(e_quad, e_lin, e_const)
roots, eigvecs = _ssik_solve_x2_roots_mobius(m_quad, m_lin, m_const)
q_pinv = np.linalg.pinv(q_mat).astype(np.float64)
```

A `SSIKToRustVisitor(ast.NodeVisitor)` parsing the artifact sees only the
*names* `_sp6_runtime.solve`, `solve_x2_roots_mobius`, `np.linalg.pinv` — not
their bodies. The companion-matrix eigenvalue routine the issue wants to "hook"
lives in **`ssik.subproblems` / `ssik.solvers.ikgeo`**, which are separate
modules (and partly Cython/LAPACK-backed). They are *not* parseable as part of
the artifact AST.

### 2.2 The artifact uses numpy semantics that don't map 1:1

From `ur5_ik.py` `_spatial_jacobian` / `_fk`:

```python
rot[:3, :3] = _rotation_matrix(_JOINT_AXES[i], float(q[i]))   # slice assignment
J = np.zeros((6, n)); J[:3, i] = np.cross(p_i, z_i); J[3:, i] = z_i
residual = float(np.linalg.norm(T_check - T))                 # Frobenius norm
axis_unit = _JOINT_AXES[i] / np.linalg.norm(_JOINT_AXES[i])    # broadcast divide
```

`visit_Call` rules for `np.dot`/`np.eye`/`np.zeros` (as the issue lists) do not
cover slice-assignment, `np.linalg.*`, `np.cross`, `np.asarray`, broadcasting,
or `.astype(...)`. Each needs real semantic handling, not a name swap.

### 2.3 Non-determinism conflicts with the stated real-time constraints

`solve_x2_roots_mobius` applies a **randomised Möbius transform** to dodge
ill-conditioned generalised eigenproblems, and `np.linalg.pinv` is an SVD. The
existing Piper Rust port **deliberately omits** the Möbius fallback for exactly
this reason (README: *"the randomised Möbius / generalised-eigenvalue fallback
… is intentionally omitted"*). A verbatim transpile would either reproduce the
non-determinism (violating DexZero) or silently diverge from the Python
reference. This is a **design decision a transpiler cannot make on its own.**

### 2.4 Python dynamic constructs in the body

`solve()` and `_solve_algebraic()` use: keyword-only args with defaults
(`*, max_solutions=None, ...`), `list`/`tuple` accumulation
(`candidates.append([...])`), tuple unpacking in `for q1, q5 in ...`,
`min/max/abs`, `math.pi % (2*pi)` wrapping, `dataclass` results (`Solution`),
list comprehensions, and exception-free LS fallbacks. Emitting correct,
heap-free Rust for these requires per-pattern understanding, not generic node
translation.

---

## 3. Why the literal "pure AST transpiler" cannot meet the goals

1. **The core algorithm is out of scope of the AST.** SP6, Raghavan-Roth
   elimination, the companion-matrix eigenvalue solve, `lm_refine`, and the
   postprocessors live in imported runtime modules. The transpiler would have
   to *also* parse and transpile `ssik.subproblems`, `ssik.solvers.ikgeo`,
   `ssik.refinement`, `ssik.postprocess` — i.e. a large fraction of the whole
   `ssik` package, much of it Cython/LAPACK — which is far beyond "flat math in
   one file."
2. **numpy ≠ nalgebra by name substitution.** Slice assignment, `linalg.norm`,
   `linalg.pinv`, `cross`, broadcasting and `asarray` need semantic lowering.
3. **Determinism is a human design choice.** The reference deliberately uses a
   randomised fallback; the Rust target deliberately forbids it. No visitor can
   resolve that trade-off.
4. **Type inference is undecidable in general.** "Infer when a literal is a
   float" is easy for `5.0`; deciding whether `q[i]` is `f64`, whether a name is
   a `SMatrix<4,4>` vs `SVector<3>` vs scalar, and whether `a * b` is scalar
   mul vs matrix mul, requires whole-program type propagation that Python's
   `ast` module does not provide.
5. **The existing ports prove the point.** `src/drivers/ur5/ik_solver.rs` (739
   lines), `piper/ik_solver.rs` (890), `franka_panda/ik_solver.rs` (1004) and
   the shared `src/math/ssik_core.rs` (1913) are clearly **hand-authored**:
   they share generic kernels, replace numpy with fixed-size arrays, swap the
   Möbius fallback for an equilibrated deterministic route, and carry bespoke
   doc-comments. They are not, and could not be, the byte-output of a mechanical
   transpiler.

---

## 4. What *is* feasible

### Option A — AST-assisted per-arm code generator (RECOMMENDED)

Keep the genuinely reusable, hard, numeric kernels as a **hand-maintained Rust
library** (already present: `src/math/ssik_core.rs`, `src/math/kinematics.rs`).
Use Python's `ast` only to **extract and lower the per-arm, side-effect-free
pieces** that *are* flat math, and emit them into a driver template:

1. **Constant extraction (high confidence).** Parse the module-level
   assignments (`_JOINT_AXES`, `_JOINT_T_LEFTS`, `_JOINT_T_RIGHTS`,
   `_JOINT_LIMITS`, `DOF`, `T_HOME`, `BASE_LINK`, …) and emit Rust `const`s.
   This is what `scripts/generate_franka_consts.py` and
   `scripts/generate_piper_pq.py` already do by hand; an AST pass generalises
   it. **Deterministic, testable, immediately valuable.**
2. **Algebraic-block lowering (medium confidence).** Within `_solve_algebraic`,
   transpile *only* the straight-line scalar arithmetic blocks (the
   `th14_x*`, `q6_x*`, `dinr_*`, `d_inner_*`, SP3 branch, `q2`/`q4`
   closed forms) — these genuinely are `math.sin/cos/atan2/sqrt` + `+ - * /`.
   Map `math.sin(x)`→`x.sin()`, `math.atan2(y,x)`→`y.atan2(x)`,
   `a**2`→`a*a`/`a.powi(2)`, and format every float literal as `_f64`.
3. **Runtime calls become typed calls into the Rust kernel library.** Wherever
   the artifact calls `_sp6_runtime.solve`, `eliminate_q0_q1`,
   `solve_x2_roots_mobius`, `lm_refine`, etc., the generator emits a call to the
   **corresponding hand-written Rust kernel** (`ssik_core::sp6_solve`,
   `ssik_core::rr_real_roots`, `kinematics::lm_refine6`, …). The mapping is a
   small, explicit, reviewed dispatch table — *not* a transpile of those
   bodies.
4. **Orchestration (`solve` dedup/limits/seed) stays templated.** Emit the
   fixed `[Option<[f64; N]>; CAP]` accumulation, FK verification, wrap-to-pi
   dedup and limit filtering from a parameterised Rust template (these are
   structurally identical across arms — compare the three existing drivers).

The generated artifact then matches the existing driver interface, e.g. UR5's
`compute_ik(&SMatrix<f64,4,4>) -> [Option<[f64;6]>;16]` and Piper's
`solve(&T, max_solutions, q_seed, respect_limits, allow_refinement, policy,
refinement_max_iters)`.

### Option B — full Python→Rust transpiler (NOT recommended)

Would require transpiling numpy + a large part of the `ssik` runtime (incl.
Cython/LAPACK kernels and a randomised solver), reimplementing whole-program
type inference, and still producing non-deterministic code that violates the
real-time constraints. High effort, low correctness, unbounded scope.

---

## 5. Decisions required before coding (please review)

1. **Scope of `ssik_to_rust_compiler.py`.** Approve **Option A** (AST-assisted
   generator that targets the existing hand-written kernels), or insist on the
   literal Option B transpiler despite §3? *Recommendation: Option A.*
2. **New arm or re-derive existing?** Should the generator first reproduce one
   already-ported arm **byte-for-byte comparable** to its committed Rust (regression
   anchor), or target a *new* arm (e.g. `z1_ik`, `puma560_ik`) to prove value?
   *Recommendation: reproduce UR5 first as a golden anchor, then a new arm.*
3. **Determinism policy for tier-2 arms.** Confirm the generator must **omit the
   Möbius/`pinv` fallback** and route through the deterministic
   `ssik_core::rr_real_roots`, accepting the documented edge-case divergence
   (as the current Piper port does). *Recommendation: yes, omit; match current
   behavior.*
4. **Source of truth for kernels.** Confirm the kernel library
   (`sp6_solve`, `rr_*`, `hessenberg_eigenvalues`, `lm_refine6`, FK) stays
   **hand-maintained** and the generator only *calls* it. *Recommendation: yes.*
5. **Where the `ssik` source comes from at generate-time.** The generator needs
   the artifact `.py` files. Confirm the workflow is `pip install ssik` then
   point the script at `ssik/prebuilt/<arm>_ik.py` (consistent with the
   existing `scripts/generate_*` provenance scripts).
6. **Output location & naming.** Confirm emitted file path/name (the issue says
   `ik_solver.rs`; the repo lays drivers out as
   `src/drivers/<arm>/ik_solver.rs` + `consts.rs`). *Recommendation: follow the
   existing per-arm directory layout.*
7. **Validation bar.** Confirm "correct" means: generated Rust **compiles under
   `#![no_std]`**, and reproduces `solve(T, respect_limits=False)` on the
   500-pose dataset within the existing tolerances (FK `1e-5`, dedup `1e-3`) —
   reusing the `scripts/generate_*_test_data.py` harness.

---

## 6. Phased plan for Option A (once §5 is approved)

- **Phase 0 — Harness & fixtures.** Add a `pip install ssik` dev step; pick the
  golden arm (UR5). Wire generated output through the existing
  `scripts/generate_test_data.py` 500-pose comparison.
- **Phase 1 — `ssik_to_rust_compiler.py` skeleton.** `ast.parse()` the
  artifact; `__main__` taking `<input.py> <output.rs>` via `sys.argv`;
  `SSIKToRustVisitor(ast.NodeVisitor)` accumulating a Rust string;
  float-literal formatting helper (`fmt_f64`).
- **Phase 2 — Constant emission.** `visit_Assign` for module-level baked
  constants → Rust `const` arrays (`[[f64;4];4]`, axes, limits, `DOF`,
  `T_HOME`). Validate byte-comparable against the committed `consts.rs`.
- **Phase 3 — Algebraic scalar lowering.** `visit_Assign`/`visit_BinOp`/
  `visit_Call`/`visit_Subscript`/`visit_Compare`/`visit_If` for the
  straight-line trig/algebra blocks and the SP3 boundary branch (clamp `acos`
  domain, *continue* instead of panic). Emit `let`/`let mut` bindings.
- **Phase 4 — Kernel dispatch table.** Map runtime calls
  (`_sp6_runtime.solve`, RR builders, `lm_refine`, postprocess) to the
  hand-written `ssik_core`/`kinematics` Rust functions. This table is reviewed
  by hand, not generated.
- **Phase 5 — `solve` orchestration template.** Emit the fixed-size candidate
  array, FK verification, dedup, limits, and seed ordering to match the
  existing driver signatures.
- **Phase 6 — Validation & docs.** `cargo test` (host) +
  `cargo build --release --target thumbv7em-none-eabihf` (no_std/no-heap);
  500-pose regression; update README provenance row.

---

## 7. Interfaces the generated Rust must match (from existing drivers)

- UR5: `compute_ik(&SMatrix<f64,4,4>) -> [Option<[f64;6]>;16]`
  (reproduces `ur5_ik.solve(T, respect_limits=False)`).
- Piper: `solve(&T, max_solutions, q_seed, respect_limits, allow_refinement,
  policy: TolerancePolicy, refinement_max_iters) -> [Option<[f64;6]>;16]`,
  plus `fk`, `TolerancePolicy`, `JOINT_LIMITS`, `DOF`.
- Franka (7R): `[Option<[f64;7]>;128]` output, joint-4 lock over 16 samples.
- Hard constraints (all arms): `#![no_std]`, `#![forbid(unsafe_code)]`, zero
  heap (no `Vec`/`Box`), every loop with a hardcoded iteration ceiling,
  `nalgebra` static `SMatrix`/`SVector`, `libm` transcendentals.

---

## 8. Risks

- **Codegen drift vs. hand-written drivers.** Mitigate by making UR5 the golden
  anchor and diffing generated output against the committed file.
- **Per-arm idiosyncrasies.** Tier-0 (UR5) is clean; tier-2 (Piper/JACO) carry
  the Möbius/`pinv` non-determinism — covered by Decision §5.3.
- **`ssik` API churn.** Pin the `ssik` version used for generation (as the
  existing provenance scripts implicitly do).
