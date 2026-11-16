# Upgrading the Locked `ssik` Version

## Overview

This runbook describes how to upgrade the locked `ssik` version in [`SSIK_VERSION.txt`](../SSIK_VERSION.txt). The `ssik` library (`personalrobotics/ssik`) is the source of truth for analytical inverse kinematics solvers; dexforge translates those solvers into `#![no_std]` Rust inside each OEM driver crate (`repos/dexforge-<oem>/rust/src/drivers/<arm>/`) backed by shared numerical kernels in [`dexhal/src/math/`](../dexhal/src/math).

When upgrading `ssik`:
1. **Prepare** an isolated environment with the target `ssik` version and update `SSIK_VERSION.txt`.
2. **Inspect** changes in `ssik` since the previously locked version.
3. **Analyze and update** the shared math kernels in `dexhal/src/math/` if solver algorithms or tolerances changed.
4. **Diff** each arm's generated `<arm>_ik.py` artifact and retranslate `ik_solver.rs` (or refresh `arm.rs` / dynamics) if the artifact changed.
5. **Regenerate** baked solver constants (`consts.rs` / `pq.rs`) and `ssik` IK reference datasets (`test_data.bin`, `test_data.rs`) for every arm.
6. **Verify** with `ik_test_data.py --check`, `cargo test` across `dexhal` and all OEM crates, and the Python test suite where present.

---

## Step 1: Preparation

### 1.1 Read the current locked version
```bash
cat SSIK_VERSION.txt
```
This shows the currently pinned version (e.g., `ssik==1.2.3`).

### 1.2 Install the target `ssik` version and update `SSIK_VERSION.txt`
Use a virtual environment so you can test the upgrade cleanly:
```bash
python -m venv /tmp/ssik-upgrade
source /tmp/ssik-upgrade/bin/activate
pip install ssik==<NEW_VERSION> numpy
```

Update `SSIK_VERSION.txt` now so `codegen/ssik_artifact.py` and `codegen/ik_test_data.py` (which enforce that the installed `ssik.__version__` matches `SSIK_VERSION.txt`) accept the newly installed version:
```bash
echo "ssik==<NEW_VERSION>" > SSIK_VERSION.txt
```

---

## Step 2: Inspect `ssik` Changes

### 2.1 Review the changelog / commit range
```bash
git clone https://github.com/personalrobotics/ssik.git /tmp/ssik-repo
cd /tmp/ssik-repo
git log --oneline <OLD_TAG>..<NEW_TAG>
```

### 2.2 Check core solver and kinematics modules
Inspect changes under:
- `ssik/solvers/` — Subproblem solvers (`sp6_runtime`), Raghavan-Roth elimination, joint-lock schedules.
- `ssik/kinematics/` — Product-of-exponentials chain extraction, chain reversal, LM refinement, and residual calculation.
- `ssik/core/tolerances.py` — Default numerical and deduplication tolerances.
- `ssik/prebuilt/` — Prebuilt artifacts (reflects solver codegen changes).

---

## Step 3: Analyze Impact on `dexhal::math` Kernels

### 3.1 Map `ssik` modules to `dexhal/src/math/`

| Python (`ssik`) | Rust (`dexhal::math`) | File |
| :--- | :--- | :--- |
| `ssik.solvers.sp6_runtime` (`solve`, subproblems, polynomial roots) | `ssik_core::{sp1_solve, sp3_solve, sp4_solve, sp5_solve, sp6_solve, quartic_roots, cubic_roots, rr_real_roots, wrap_to_pi}` | `dexhal/src/math/ssik_core.rs` |
| `ssik.kinematics` (FK, pose residuals, Levenberg-Marquardt refinement) | `kinematics::{Transform, lm_refine, lm_refine6, mat4_frob_diff, norm12}` | `dexhal/src/math/kinematics.rs` |
| Post-processing (joint limits, modulo-$2\pi$ dedup, seed ordering) | `postprocess::*` | `dexhal/src/math/postprocess.rs` |
| `ssik.core.tolerances.TolerancePolicy` | `tolerances::TolerancePolicy` | `dexhal/src/math/tolerances.rs` |

### 3.2 Update and test `dexhal` if needed
- If a subproblem algorithm, root finder, or tolerance default changed in `ssik`, update the corresponding module in `dexhal/src/math/` while keeping `#![no_std]`, `#![forbid(unsafe_code)]`, zero heap allocation, and bounded loop ceilings.
- Run the `dexhal` test suite:
  ```bash
  cargo test --manifest-path dexhal/Cargo.toml
  ```

---

## Step 4: Analyze Impact on Arm Drivers

### 4.1 Rebuild and diff `<arm>_ik.py` artifacts
For each arm in the repository, rebuild its artifact from its committed URDF and compare against the provenance artifact (or `ssik.prebuilt.<arm>_ik` if `codegen/<arm>_ik.py` is not yet committed):

```bash
D=repos/dexforge-ur/rust/src/drivers/ur5
python codegen/ssik_artifact.py build --urdf $D/ur5.urdf --base base_link --ee ee_link \
    --out /tmp/upgrade/ur5_ik.py
diff -u $D/codegen/ur5_ik.py /tmp/upgrade/ur5_ik.py
```

If `codegen/<arm>_ik.py` is not present in a legacy driver directory, compare against `ssik.prebuilt`:
```bash
python -c "import inspect, ssik.prebuilt.ur5_ik as m; print(inspect.getsource(m), end='')" > /tmp/upgrade/ur5_prebuilt.py
diff -u /tmp/upgrade/ur5_prebuilt.py /tmp/upgrade/ur5_ik.py
```

Repeat for all committed arms (`ur5`, `franka_panda`, `xarm7`, `piper`, `rizon4`, `rizon10`, `z1`). You can inspect any artifact's `BASE_LINK`, `EE_LINK`, and solver family with:
```bash
python codegen/ssik_artifact.py inspect /tmp/upgrade/<arm>_ik.py --arm <arm>
```

### 4.2 Act on artifact diffs
- **No change or comment/metadata-only change:** Copy the updated `<arm>_ik.py` to `drivers/<arm>/codegen/<arm>_ik.py` and proceed to Step 5.
- **Chain constant or joint limit change (`_JOINT_AXES`, `_JOINT_T_LEFTS`, `_JOINT_T_RIGHTS`, `_JOINT_LIMITS`):**
  1. Copy `<arm>_ik.py` to `drivers/<arm>/codegen/<arm>_ik.py`.
  2. Regenerate `arm.rs` (`python codegen/scaffold.py --oem <oem> --arm <arm> --urdf $D/<arm>.urdf --base <base> --ee <ee> --only arm.rs`).
  3. Recompile symbolic kinematics/dynamics (`python codegen/generate_dynamics.py --artifact $D/codegen/<arm>_ik.py --urdf $D/<arm>.urdf --base <base> --ee <ee> --arm <arm> --output $D/generated_dynamics.rs`).
- **Solver algorithm or algebraic block change:**
  - Retranslate `ik_solver.rs` with a coding agent following [`codegen/new_driver.md`](new_driver.md) (Steps 3–4) and [`codegen/translate_ssik_artifact.md`](translate_ssik_artifact.md).

---

## Step 5: Regenerate Baked Constants and IK Test Data

### 5.1 Regenerate baked constants (`consts.rs` / `pq.rs`)
For arms in categories 2–4 (general-6R and 7-DOF joint-lock):
```bash
python repos/dexforge-franka/rust/src/drivers/franka_panda/codegen/generate_code.py
python repos/dexforge-ufactory/rust/src/drivers/xarm7/codegen/generate_code.py
python repos/dexforge-agilex/rust/src/drivers/piper/codegen/generate_code.py
python repos/dexforge-flexiv/rust/src/drivers/rizon4/codegen/generate_code.py
python repos/dexforge-flexiv/rust/src/drivers/rizon10/codegen/generate_code.py
```

### 5.2 Regenerate `ssik` IK reference datasets (`test_data.bin` & `test_data.rs`)
Every arm's `test_data.bin` records the `ssik` version header and is checked against `SSIK_VERSION.txt`:
```bash
python repos/dexforge-ur/rust/src/drivers/ur5/codegen/generate_test_data.py
python repos/dexforge-franka/rust/src/drivers/franka_panda/codegen/generate_test_data.py
python repos/dexforge-ufactory/rust/src/drivers/xarm7/codegen/generate_test_data.py
python repos/dexforge-agilex/rust/src/drivers/piper/codegen/generate_test_data.py
python repos/dexforge-flexiv/rust/src/drivers/rizon4/codegen/generate_test_data.py
python repos/dexforge-flexiv/rust/src/drivers/rizon10/codegen/generate_test_data.py
python repos/dexforge-unitree/rust/src/drivers/z1/codegen/generate_test_data.py
```

### 5.3 Verify dataset version consistency
```bash
python codegen/ik_test_data.py --check
```
This confirms every arm's `test_data.bin` carries the version in `SSIK_VERSION.txt` and a non-empty case list.

---

## Step 6: Run Tests

### 6.1 Run Rust tests (`dexhal` + OEM driver crates)
```bash
cargo test --manifest-path dexhal/Cargo.toml
for toml in repos/dexforge-*/Cargo.toml; do
    cargo test --manifest-path "$toml"
done
```

### 6.2 Run `codegen` unit tests
```bash
pytest codegen/tests -v
```

### 6.3 Rebuild and test Python bindings (for OEM repos that include `python/`)
```bash
for py_dir in repos/dexforge-*/python; do
    (cd "$py_dir" && maturin develop --release)
done
pytest repos/dexforge-*/python/tests -v
```

---

## Troubleshooting

- **`RuntimeError: Installed ssik version (...) does not match SSIK_VERSION.txt (...)`**
  Ensure `SSIK_VERSION.txt` was updated to `<NEW_VERSION>` before running `ssik_artifact.py`, `generate_test_data.py`, or `ik_test_data.py --check`.
- **IK reference tests fail after regenerating `test_data.bin`**
  Do not loosen test tolerances (`1e-5` FK closure / strict joint match, `1e-3` singular joint dedup). Diff the old and new `<arm>_ik.py` artifacts to identify which subproblem, lock-sample schedule, or algebraic coefficient changed, and update `ik_solver.rs` (or `dexhal/src/math/`) to match.
- **Crate fails `#![no_std]` or embedded target check**
  Verify no `std` imports or heap allocations (`Vec`, `Box`) were introduced outside `#[cfg(test)]`:
  ```bash
  cargo build --manifest-path dexhal/Cargo.toml --release --target thumbv7em-none-eabihf
  ```

---

## Checklist

- [ ] Updated `SSIK_VERSION.txt` to `ssik==<NEW_VERSION>`
- [ ] Reviewed `ssik` changelog and diffed solver/kinematics modules
- [ ] Updated `dexhal/src/math/` kernels if needed (`cargo test --manifest-path dexhal/Cargo.toml` passes)
- [ ] Diffed `<arm>_ik.py` for each arm and updated `ik_solver.rs` / `arm.rs` / `generated_dynamics.rs` if needed
- [ ] Regenerated `consts.rs` / `pq.rs` where applicable (`generate_code.py`)
- [ ] Regenerated `test_data.bin` and `test_data.rs` for all arms (`generate_test_data.py`)
- [ ] `python codegen/ik_test_data.py --check` passes
- [ ] `cargo test` passes for `dexhal` and all `repos/dexforge-*` crates
- [ ] `pytest codegen/tests -v` passes
