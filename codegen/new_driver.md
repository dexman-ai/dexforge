# Master prompt — generate a dexhal OEM driver crate

You are a coding agent producing (or extending) the vendor crate
`repos/dexforge-<oem>/` for one arm `<arm>`. Work autonomously with your own
file and shell tools; run every command from the repository root. You are done
only when `cargo test --manifest-path repos/dexforge-<oem>/Cargo.toml` passes.

Read first: [`README.md`](README.md) (architecture and target layout),
[`dexhal/README.md`](../dexhal/README.md) (driver contract),
[`dexhal/mock/rust`](../dexhal/mock/rust) (reference `ArmDriver`).

## Inputs (from the user)

| Input | Use |
| :--- | :--- |
| `--oem`, `--arm` | Crate `repos/dexforge-<oem>`, module `drivers/<arm>` (snake_case). |
| URDF + `--base` / `--ee` links | Kinematic chain, limits, `<inertial>` data. |
| OEM data sheet / manifest values | `control_freq_hz`, `control_modes`, latency cycles, acceleration (and jerk) limits, sensing / native capabilities. |
| OEM hardware API / protocol spec | `ArmDriver::{connect, disconnect, read_state, write_command}` and optional native hooks. |

Ask for any missing input rather than inventing hardware facts. Never edit
`dexhal` and never copy its math into the vendor crate.

## Hard rules

- The crate depends on `dexhal` (path dependency), `libm` and `nalgebra`
  (`default-features = false`, `libm`). It never depends on `dexforge`.
- Everything under `drivers/<arm>/` is `#![no_std]`-compatible,
  `#![forbid(unsafe_code)]`, heap-free and bounded-iteration. Host I/O in
  `driver.rs` may use `std` only behind `#[cfg(feature = "std")]`.
- Use the shared kernels from `dexhal::math` (`ssik_core` subproblems,
  `kinematics`, `postprocess`, `tolerances`, LM refinement). Never reimplement
  subproblem solvers, root finders, refinement, joint wrapping or seed sorting.
- No finite-difference Jacobians anywhere in `ik_solver.rs` (refinement, DLS,
  or helpers). Use the arm's `generated_kinematics::jacobian` with
  `kinematics::residual_columns` and `lm_refine_masked_analytic` /
  `dls_ik`. See `translate_ssik_artifact.md` section 2.
- Do not hand-edit generated files: `arm.rs`, `generated_*.rs`, `test_data.*`,
  `dynamics_*`, `consts.rs` / `pq.rs`. Fix the generator inputs and rerun.
- The existing driver repos (`repos/dexforge-*`) predate dexhal and carry their
  own math copies; their `ik_solver.rs`, `generate_code.py`,
  `generate_test_data.py`, IK tests and dynamics tests are valid references for
  the *algorithm*, but rewrite their `crate::math::...` imports to `dexhal::math`.

## Step 1 — ssik artifact and family

```sh
python codegen/ik_test_data.py --check          # installed ssik == pyproject.toml
python codegen/ssik_artifact.py build --urdf <urdf> --base <base> --ee <ee> --out /tmp/<arm>_ik.py
python codegen/ssik_artifact.py inspect /tmp/<arm>_ik.py --arm <arm>
```

`inspect` reports `DOF`, `MAX_SOLUTIONS`, joint names/limits and the solver
family with its reference arm:

| Category | ssik solver | Reference | `MAX_SOLUTIONS` | Baked constants |
| :--- | :--- | :--- | :--- | :--- |
| 1 — closed-form 6R (`ikgeo` SP6) | `ikgeo.three_parallel`, `ikgeo.spherical_*` | `ur5` (`repos/dexforge-ur`) | 16 | — |
| 2 — general 6R (`ikgeo.general_6r`) | `ikgeo.general_6r` | `piper` (`repos/dexforge-agilex`) | 16 | `pq.rs` |
| 3 — joint-lock 7-DOF, SP6 inner | `jointlock.*` | `franka_panda` (`repos/dexforge-franka`); also `xarm7` | 128 | `consts.rs` |
| 4 — joint-lock 7-DOF, Raghavan-Roth inner | `jointlock.*` (`raghavan_roth` inner) | `rizon4` (`repos/dexforge-flexiv`) | 256 | `consts.rs`, `pq.rs` |

`seven_r.srs*` solvers (iiwa-like) have no in-repo reference yet; `inspect`
maps them to category 3 and says so — confirm `MAX_SOLUTIONS` against the
artifact before relying on it.

## Step 2 — scaffold (deterministic)

```sh
python codegen/scaffold.py --oem <oem> --arm <arm> --urdf <urdf> --base <base> --ee <ee> \
    [--artifact /tmp/<arm>_ik.py]
```

Creates the vendor crate if missing, copies `<arm>.urdf` and
`codegen/<arm>_ik.py` into `drivers/<arm>/` (building the artifact when
`--artifact` is omitted), registers the arm in `drivers/mod.rs` / `lib.rs`, and
writes `mod.rs`, `driver.rs` (stub), `manifest.yaml` (draft),
`codegen/generate_test_data.py` and `rust/tests/perf.rs`. The perf file is
created once with the shared `dexhal/perf/arm_timing.rs` harness; each scaffolded
arm appends its own `<arm>_arm_timing` test, skipped if already present. It always regenerates `arm.rs`: the
`ArmModel` impl (`NAME`, `JOINT_LIMITS`, `HAS_DYNAMICS`, every `flange_*` hook)
and `nominal_chain()`. Existing files are kept unless `--force`; use
`--only arm.rs` after changing the URDF or artifact. Follow the printed
"next steps".

## Step 3 — agent translation

1. **`ik_solver.rs`.** Translate `codegen/<arm>_ik.py` following
   [`translate_ssik_artifact.md`](translate_ssik_artifact.md) and the reference
   arm's `ik_solver.rs`. Required signature (called by `arm.rs`):

   ```rust
   pub fn solve(
       target: &dexhal::math::Pose,
       max_solutions: Option<usize>,
       q_seed: Option<&[f64; DOF]>,
       respect_limits: bool,
       allow_refinement: bool,
       policy: dexhal::math::TolerancePolicy,
       refinement_max_iters: usize,
   ) -> [Option<[f64; DOF]>; MAX_SOLUTIONS]
   ```

   Respect joint limits when asked, order nearest-first when `q_seed` is given,
   guard every `acos`/`asin`/`sqrt` domain edge, and include the reference arm's
   IK tests against `test_data.rs` (`#[cfg(test)] #[path = "test_data.rs"] mod test_data;`).
   Joint-lock 7-DOF arms (categories 3–4) must hold the locked joint at the
   seed's value on a seeded solve. Copy the reference arm's seeded early exit:
   visit lock samples nearest-first to `q_seed[LOCKED_JOINT]`, polish each
   sample's candidates with `seed_locked` as they are collected, and stop at the
   first sample that closes. Polish the seed itself only once (first
   `seed_locked` call). If no sample closes, return the full sweep's result
   without re-polishing it. Copy
   the reference arm's `seeded_solve_keeps_the_seed_locked_joint` test too.
   For 7-DOF joint-lock arms, `ik_nearest` must provide the `Firm` and
   `Relaxed` flavors (selected by an extra `JointLockFlavor` argument) described
   in [`translate_ssik_artifact.md`](translate_ssik_artifact.md) §4 item 7.
   Firm keeps the redundant joint at the seed value in all three stages;
   Relaxed tries full-joint DLS first and then the sampled-lock state. Leave
   the configurable joint-step continuity check to the owning `Arm`; do not
   hard-code the default π/2 bound in the generated solver. Do not generate a
   capped or FLOP-budgeted `ik_nearest`. For 6-DOF closed-form arms, use the
   nearest-first single-solution early exit in
   [`translate_ssik_artifact.md`](translate_ssik_artifact.md) §4 item 8 (no lock,
   no flavors).
2. **`codegen/generate_code.py`** (categories 2–4): adapt the reference arm's
   script so it reads `codegen/<arm>_ik.py` and writes `../consts.rs` / `../pq.rs`;
   add `pub mod consts;` / `pub mod pq;` to `mod.rs` if the scaffold did not.
3. **`codegen/generate_test_data.py`**: the scaffolded script samples 500
   multi-solution poses. If the reference arm's script polishes solutions,
   splits strict / singular cases or enables refinement, mirror that.
4. **`manifest.yaml`**: replace every `TODO` from the OEM data; keep
   `driver: <arm>` and the scaffolded `version`. Check against
   [`dexhal/examples/manifest.yaml`](../dexhal/examples/manifest.yaml).
5. **`driver.rs`**: implement `ArmDriver` from the OEM protocol (see the mock
   driver): reject wrong buffer lengths with `ArmError`, fill `CoreState` via
   the zero-copy views, honour `manifest().control_modes`, and keep sockets /
   SDK calls behind `feature = "std"`.

## Step 4 — generate and verify

```sh
D=repos/dexforge-<oem>/rust/src/drivers/<arm>
python $D/codegen/generate_code.py                     # categories 2-4 only
python codegen/generate_dynamics.py --artifact $D/codegen/<arm>_ik.py --urdf $D/<arm>.urdf \
    --base <base> --ee <ee> --arm <arm> --output $D/generated_dynamics.rs
python $D/codegen/generate_test_data.py
python codegen/generate_dynamics_test_data.py --urdf $D/<arm>.urdf --base <base> --ee <ee> \
    --output-dir $D --arm <arm> --samples 1000 --seed 0  # only when HAS_DYNAMICS
cargo test --manifest-path repos/dexforge-<oem>/Cargo.toml
```

`generate_dynamics.py` writes `generated_dynamics.rs` and the
`generated_kinematics.rs` re-export. Without `<inertial>` data the scaffold sets
`HAS_DYNAMICS = false`: still run `generate_dynamics.py` (FK/Jacobian), skip the
Pinocchio step.

Iterate on compile errors, FK/IK tolerance failures and dynamics-parity failures
by fixing `ik_solver.rs`, `driver.rs`, `manifest.yaml` or the generator inputs
— never by loosening tolerances or editing generated files — until every test
passes. Finish with a summary of the files written, the family, and any OEM
facts you had to assume.
