# dexforge-py Testing

This directory contains the test suite for the `dexforge_py` Python bindings.

## Test Structure

The tests are **directly from [ssik](https://github.com/personalrobotics/ssik)** and run unmodified against dexforge_py via a compatibility layer. This ensures dexforge's Rust IK implementations maintain fidelity to the original Python ssik artifacts.

### Core Test Files

- **`test_prebuilt_sanity.py`** - Sanity checks for every arm driver (copied from ssik)
  - Verifies that `solve()` and `fk()` functions are callable
  - Tests that FK-seeded targets return at least one IK solution
  - Ensures FK closure within 1e-6 (smoke test threshold)
  - ssik reference: `tests/test_prebuilt_sanity.py` (#132)

- **`test_prebuilt_uniform_fuzz.py`** - Property-based tests with Hypothesis (copied from ssik)
  - 500-pose uniform sweep across non-singular configurations
  - Separate tests for 6R arms (UR5, PiPER) and 7R arms (Franka Panda)
  - Verifies FK residual within per-arm documented ceiling
  - ssik reference: `tests/test_prebuilt_uniform_fuzz.py` (#267)

### Compatibility Layer

The ssik tests expect to import from `ssik.prebuilt.<arm>_ik` and work with `Solution` objects. dexforge_py provides a compatibility shim:

- **`ssik/`** - Package structure mimicking ssik's public API
  - `ssik/core/solution.py` - Solution dataclass (q, fk_residual, refinement_used)
  - `ssik/core/tolerances.py` - TolerancePolicy for test parameterization
  - `ssik/prebuilt/_manifest.py` - load_manifest() providing arm metadata
  - `ssik/prebuilt/{ur5,piper,franka_panda}_ik.py` - Per-arm wrappers
    - Each wrapper provides solve() and fk() wrapping dexforge_py calls
    - Converts plain list returns to Solution objects with FK residuals
    - Exposes all ssik constants (DOF, T_HOME, SOLVER_NAME, etc.)

This approach allows zero modification to the upstream ssik tests while validating the Rust implementations.

### Supporting Files

- **`conftest.py`** - Shared fixtures and configuration
  - Contains original ARMS_METADATA from initial test implementation
  - Not used by ssik tests (they use _manifest.py instead)
  - Kept for backward compatibility and potential future extensions

- **`_hypothesis_strategies.py`** - Hypothesis strategies (copied from ssik)
  - `non_singular_q6r()` - Generates 6R joint vectors avoiding singularities
  - `non_singular_q7r()` - Generates 7R joint vectors avoiding singularities
  - Filters axes prone to solver-level degeneracies

## Running Tests

### All Tests

```bash
pytest dexforge-py/tests/ -v
```

### Sanity Tests Only

```bash
pytest dexforge-py/tests/test_prebuilt_sanity.py -v
```

### Uniform Fuzz Tests

```bash
pytest dexforge-py/tests/test_prebuilt_uniform_fuzz.py -v
```

## Adding a New Arm

When a new arm driver is implemented in the Rust crate and exposed in `dexforge_py`:

1. Create a new wrapper in `ssik/prebuilt/<arm>_ik.py` following the existing pattern
   - Implement solve() and fk() wrapping dexforge_py calls
   - Define all ssik constants (DOF, T_HOME, SOLVER_NAME, BASE_LINK, EE_LINK, etc.)
   - Wrap returned solutions in Solution objects with FK residuals

2. Add an entry to `_ARMS` dict in `ssik/prebuilt/_manifest.py` with:
   - name, display_name, dof
   - sample_q - A known in-limits joint configuration  
   - fk_ceiling_fuzz - Maximum FK residual under fuzzing (from ssik MANIFEST.toml)
   - known_gaps - Optional xfail reason if solver has known limitations

3. Tests will automatically discover and parametrize over the new arm
   - No edits needed in test files themselves

4. Rebuild the wheel and reinstall:
   ```bash
   cd dexforge-py && maturin build --release
   pip3 install --force-reinstall target/wheels/dexforge_py-*.whl
   ```

## Known Issues

- **PiPER**: Has a known RR coverage gap (ssik#286) where some poses return 0 solutions. The uniform fuzz test is xfailed until the upstream issue is resolved.

## Test Philosophy

These tests validate that:

1. The Rust IK implementations produce FK-correct solutions
2. FK residuals stay within documented per-arm ceilings
3. The dexforge_py API is compatible with ssik prebuilt artifacts
4. Edge cases (singularities, refinement policies) behave as expected

The tests do NOT assert:

- Exact joint-space seed recovery (q-space drift is acceptable if T-space is correct)
- Specific solution counts (deterministic solvers may return subsets)
- Cross-platform bit-exact reproducibility (floating-point noise varies)

The bulletproof contract: **every returned IK candidate must FK-close to the target within the arm's documented precision floor**.

## Maintenance

When ssik tests are updated upstream:

1. Copy the latest versions from [ssik/tests](https://github.com/personalrobotics/ssik/tree/main/tests):
   - `test_prebuilt_sanity.py`
   - `test_prebuilt_uniform_fuzz.py`
   - `_hypothesis_strategies.py`

2. If imports fail, check for new ssik dependencies and add compatibility shims as needed

3. Update `_manifest.py` if ssik adds new arm metadata fields referenced by tests

The compatibility layer is intentionally minimal - only wrapping the dexforge_py API surface needed by the ssik tests.
