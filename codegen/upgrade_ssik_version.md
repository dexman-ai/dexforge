# Upgrading the Locked ssik Version

## Overview

This document provides detailed instructions for upgrading the locked `ssik` version used by dexforge. The ssik library (`personalrobotics/ssik`) is the source of truth for arm-specific inverse kinematics solvers; dexforge translates these solvers from Python to Rust.

When upgrading ssik, you must:
1. **Inspect** changes in ssik since the previously locked version
2. **Analyze** how those changes affect the arm-agnostic math (in `src/math/ssik_core.rs` and `src/math/kinematics.rs`)
3. **Analyze** how those changes affect the already-translated arm drivers
4. **Update** the translated math if needed
5. **Update** or regenerate the arm drivers if needed
6. **Regenerate** test data for each arm
7. **Run tests** to verify the upgrade

## Step 1: Preparation

### 1.1 Determine the target ssik version
Identify the ssik release or commit you want to upgrade to. Check the [ssik GitHub releases](https://github.com/personalrobotics/ssik/releases) or the ssik PyPI page.

### 1.2 Read the current locked version
```bash
cat SSIK_VERSION.txt
```
This shows the currently locked version (e.g., `ssik==0.8.3`).

### 1.3 Create a safe environment
Avoid polluting your main Python environment. Use a virtual environment or conda environment to test:
```bash
python -m venv /tmp/ssik-upgrade
source /tmp/ssik-upgrade/bin/activate
pip install ssik==<NEW_VERSION>
```

## Step 2: Inspect ssik Changes

### 2.1 Review the changelog
Compare the release notes or commit log between the old and new versions:
```bash
# Clone ssik (if not already cloned)
git clone https://github.com/personalrobotics/ssik.git /tmp/ssik-repo
cd /tmp/ssik-repo
git log --oneline <OLD_TAG>..<NEW_TAG>
```

### 2.2 Inspect changes to core math
Look for changes in these directories:
- `ssik/solvers/` — Solver algorithms (affects translation)
- `ssik/kinematics/` — Kinematic chain utilities
- `ssik/prebuilt/` — Prebuilt artifacts (may reflect solver algorithm updates)

### 2.3 Check for breaking changes
**CRITICAL:** Look for any of these in the changelog:
- Changes to `sp6_solve`, `quartic_roots`, or root-finding algorithms
- Changes to `kinematics` module (FK, IK residual computation)
- Changes to joint-locking or coordinate-frame transformations
- Changes to numeric tolerance defaults
- Changes to the prebuilt artifact API signatures

## Step 3: Analyze Impact on ssik_core.rs and kinematics.rs

### 3.1 Identify affected kernels
Review the changes identified in Step 2.2 and map them to the core Rust files:

| Python Location | Rust Equivalent |
|---|---|
| `ssik.solvers.sp6_runtime.solve` | `ssik_core::sp6_solve` |
| `ssik.solvers.sp6_runtime` (polynomial roots) | `ssik_core::quartic_roots`, `ssik_core::rr_real_roots` |
| `ssik.kinematics.fk_chain` | `kinematics::fk_chain` |
| `ssik.kinematics.fk_residual` | `kinematics::fk_residual` |
| `ssik.kinematics.lm_refine` | `kinematics::lm_refine` |

### 3.2 Review current implementations
Examine the current Rust implementations in:
- `src/math/ssik_core.rs` (750 lines, Francis QR + quartic roots + RR elimination)
- `src/math/kinematics.rs` (350 lines, generic FK/IK chains + Levenberg-Marquardt)

Check docstrings for version or algorithm notes.

### 3.3 Determine if updates are needed
Compare the new ssik implementation against the current Rust translation:
- If the algorithm **changed**: You must update the Rust code to match the new algorithm.
- If only **numerical constants changed** (thresholds, iteration limits): Update those values in Rust.
- If **no relevant changes**: Skip to Step 4.

### 3.4 Update the Rust code if needed
If changes are required:
1. Create a test file (e.g., `src/math/ssik_core_upgrade_test.rs`) to compare old vs. new behavior
2. Update the Rust implementation to match the new ssik code
3. Run `cargo test` to verify the numeric kernels still produce correct results
4. Document the changes in the function docstrings (e.g., "Updated in ssik 0.9.0 to handle...").

## Step 4: Analyze Impact on Arm Drivers

Each arm driver is a translation of a specific ssik prebuilt artifact. The prebuilt artifacts may change when:
- The underlying solver algorithm changes (e.g., `sp6_solve`, `jointlock.seven_r`)
- The solver parameters change (e.g., lock-joint samples, tolerance thresholds)
- New arms are added to ssik (not relevant to existing drivers)

### 4.1 Check if prebuilt artifacts changed
For each currently-translated arm, generate the new artifact and compare:
```bash
# Install the new ssik version in the virtual environment
pip install ssik==<NEW_VERSION>

# Generate fresh artifact for each arm
python codegen/ik_generator.py --artifact ur5_ik --dump-context --output-dir /tmp/upgrade/
python codegen/ik_generator.py --artifact franka_panda_ik --dump-context --output-dir /tmp/upgrade/
python codegen/ik_generator.py --artifact xarm7_ik --dump-context --output-dir /tmp/upgrade/
python codegen/ik_generator.py --artifact piper_ik --dump-context --output-dir /tmp/upgrade/
```

Then diff the generated Python artifacts:
```bash
diff /tmp/upgrade/ur5_ik.py <(python -c "import ssik.prebuilt.ur5_ik as m; import inspect; print(inspect.getsource(m))")
```

### 4.2 Categorize the changes
- **Minor changes** (e.g., docstring updates, variable renames): No action needed.
- **Algorithmic changes** (e.g., lock samples, tolerance thresholds, root-finding logic): Requires regeneration.
- **Signature changes** (e.g., new parameters, return type changes): Likely breaking; needs manual review.

### 4.3 Regenerate artifacts if needed
If the prebuilt artifacts changed significantly:
1. Regenerate the test data for each arm:
   ```bash
   pip install ssik==<NEW_VERSION>  # Ensure new version is active
   python ur5/codegen/generate_test_data.py
   python franka_panda/codegen/generate_test_data.py
   python xarm7/codegen/generate_test_data.py
   python piper/codegen/generate_test_data.py
   python rizon4/codegen/generate_test_data.py
   python rizon10/codegen/generate_test_data.py
   ```

2. Regenerate constants for arms that use them (7-DOF jointlock arms):
   ```bash
   python franka_panda/codegen/generate_franka_consts.py
   python xarm7/codegen/generate_xarm7_consts.py
   python piper/codegen/generate_code.py
   python rizon4/codegen/generate_code.py
   python rizon10/codegen/generate_code.py
   ```

3. If the Rust drivers must be regenerated due to major algorithmic changes, use the ik_generator with the LLM:
   ```bash
   # For each arm, if needed:
   python codegen/ik_generator.py --artifact ur5_ik --output-dir out/
   ```

## Step 5: Update Test Data and Constants

### 5.1 Regenerate test data
Test data is generated from the Python artifacts and is used to validate the Rust translations:
```bash
cd /tmp/ssik-upgrade  # Virtual environment with new ssik version
pip install numpy scipy

# Generate fresh test data
cd /path/to/dexforge
python ur5/codegen/generate_test_data.py
python franka_panda/codegen/generate_test_data.py
python xarm7/codegen/generate_test_data.py
python piper/codegen/generate_test_data.py
```

### 5.2 Regenerate constants (if applicable)
For arms that use precomputed constants (7-DOF jointlock arms):
```bash
python franka_panda/codegen/generate_franka_consts.py
python xarm7/codegen/generate_xarm7_consts.py
python piper/codegen/generate_code.py
```

### 5.3 Verify test data integrity
Check that test data was regenerated successfully:
```bash
git status src/drivers/*/test_data.rs src/drivers/*/consts.rs src/drivers/*/pq.rs
# Should show modified files if changes are needed.
```

## Step 6: Run Tests

### 6.1 Build the Rust crate
```bash
cd /path/to/dexforge
cargo test --lib
```

Verify:
- All unit tests pass
- No compiler warnings related to the upgraded code

### 6.2 Test host binaries (if applicable)
```bash
cargo build --example piper_solve --release
```

### 6.3 Test embedded compilation (arm-eabihf)
dexforge is a `#![no_std]` crate; verify it compiles for embedded targets:
```bash
cargo build --release --target thumbv7em-none-eabihf
```

### 6.4 Smoke tests on reference data
If you regenerated test data or arm drivers, spot-check a few poses:
```bash
# For Piper (has hosted binary):
cat <<EOF | cargo run --example piper_solve --release
0.5 0.5 1.0 [[1 0 0 0.1] [0 1 0 0.2] [0 0 1 0.3] [0 0 0 1]]
EOF
```

## Step 7: Update the Locked Version

Once all tests pass:

### 7.1 Update SSIK_VERSION.txt
```bash
echo "ssik==<NEW_VERSION>" > SSIK_VERSION.txt
```

### 7.2 Commit the changes
```bash
git add -A
git commit -m "Upgrade ssik version to <NEW_VERSION>

- Updated ssik_core.rs (if changes needed)
- Updated kinematics.rs (if changes needed)
- Regenerated test data for all arms
- Regenerated constants (if needed)
- All tests passing"
```

### 7.3 Create a PR if needed
Ensure the following are included in the PR:
- Updated `SSIK_VERSION.txt`
- Updated `src/math/ssik_core.rs` (if changed)
- Updated `src/math/kinematics.rs` (if changed)
- Regenerated `src/drivers/*/test_data.rs`
- Regenerated `src/drivers/*/consts.rs` and `src/drivers/*/pq.rs` (if applicable)
- Summary of changes in the commit message

## Troubleshooting

### The new ssik generates a different artifact
**Cause:** The underlying solver algorithm or parameters changed.
**Action:** 
1. If the change is minor (e.g., numerical thresholds), regenerate test data and constants.
2. If the change is major (e.g., new solver or algorithm), regenerate the Rust driver using `ik_generator.py`.

### Tests fail after upgrade
**Cause:** Numeric precision or algorithm differences between old and new ssik.
**Action:**
1. Review the test failure output.
2. Check if tolerance thresholds changed in ssik (common after version upgrades).
3. Update tolerance constants in the failing driver (search for `1e-5`, `1e-4` in the driver).
4. Regenerate test data with the new ssik version.
5. Update the Rust driver if needed and re-run tests.

### The Rust crate fails to compile for embedded targets
**Cause:** New numeric or allocation patterns introduced during upgrade.
**Action:**
1. Verify `#![no_std]` is still respected (check for `use std::*` or heap allocation).
2. Check for new `libm` or `nalgebra` version requirements.
3. Review the upgrade diff carefully for hidden dependencies.

## Checklist

Before finalizing an ssik upgrade, ensure:

- [ ] Read ssik changelog and identified affected components
- [ ] Reviewed ssik source code changes in core math
- [ ] Updated `ssik_core.rs` (if needed)
- [ ] Updated `kinematics.rs` (if needed)
- [ ] Regenerated test data for all arms
- [ ] Regenerated constants for applicable arms
- [ ] All unit tests pass (`cargo test --lib`)
- [ ] Embedded target compiles (`cargo build --release --target thumbv7em-none-eabihf`)
- [ ] Smoke tests on reference data (if applicable)
- [ ] Updated `SSIK_VERSION.txt` to the new version
- [ ] Committed all changes with a clear commit message
- [ ] Created a PR summarizing the upgrade

## Questions?

If you encounter issues during an ssik upgrade:
1. Check the ssik GitHub issues and discussions
2. Review the memory facts about ssik porting in the codebase
3. Consult the `agent.md` documentation for architecture and translation tenets
4. Ask the team for guidance if the upgrade requires significant changes
