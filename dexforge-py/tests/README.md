# dexforge-py Testing

This directory contains the test suite for the `dexforge_py` Python bindings.

## Test Structure

The tests are organized to mirror the [ssik](https://github.com/personalrobotics/ssik) prebuilt artifact test patterns:

### Core Test Files

- **`test_dexforge_sanity.py`** - Sanity checks for every arm driver
  - Verifies that `solve()` and `fk()` functions are callable
  - Tests that FK-seeded targets return at least one IK solution
  - Ensures FK closure within 1e-6 (smoke test threshold)
  - Mirrors ssik's `test_prebuilt_sanity.py` (#132)

- **`test_dexforge_uniform_fuzz.py`** - Property-based tests with Hypothesis
  - 500-pose uniform sweep across non-singular configurations
  - Separate tests for 6R arms (UR5, PiPER) and 7R arms (Franka Panda)
  - Verifies FK residual within per-arm documented ceiling
  - Mirrors ssik's `test_prebuilt_uniform_fuzz.py` (#267)

- **`test_dexforge_py.py`** - Cross-validation and edge cases
  - Cross-validates dexforge_py against ssik prebuilt artifacts
  - Tests edge cases (unsupported arms, unreachable poses, etc.)
  - Performance benchmarking (with `--perf` flag)

### Supporting Files

- **`conftest.py`** - Shared fixtures and configuration
  - `ARMS_METADATA` registry with per-arm metadata (DOF, sample_q, fk_ceiling_fuzz)
  - CLI options (`--arm`, `--n`, `--perf`)

- **`_hypothesis_strategies.py`** - Hypothesis strategies
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
pytest dexforge-py/tests/test_dexforge_sanity.py -v
```

### Uniform Fuzz Tests

```bash
pytest dexforge-py/tests/test_dexforge_uniform_fuzz.py -v
```

### Specific Arm

```bash
pytest dexforge-py/tests/ -v --arm=ur5
```

### Custom Sample Size

```bash
pytest dexforge-py/tests/ -v --n=100
```

### With Performance Metrics

```bash
pytest dexforge-py/tests/test_dexforge_py.py -v --perf
```

## Adding a New Arm

When a new arm driver is implemented in the Rust crate and exposed in `dexforge_py`:

1. Add an entry to `ARMS_METADATA` in `conftest.py` with:
   - `dof` - Degrees of freedom
   - `sample_q` - A known in-limits joint configuration
   - `fk_ceiling_fuzz` - Maximum FK residual under fuzzing (from ssik MANIFEST.toml)

2. Tests will automatically pick up the new arm and parametrize over it

3. No edits needed in test files themselves

## Known Issues

- **PiPER**: Has a known RR coverage gap (ssik#286) where some poses return 0 solutions. The uniform fuzz test is xfailed until the upstream issue is resolved.

## Test Philosophy

These tests follow the ssik pattern of validating that:

1. The API surface matches the prebuilt artifact interface
2. FK-seeded targets are solvable (sanity check)
3. Every returned IK solution FK-closes within the documented ceiling (correctness)
4. The solver handles edge cases gracefully (robustness)

The tests do NOT assert:

- Exact joint-space seed recovery (q-space drift is acceptable if T-space is correct)
- Specific solution counts (deterministic solvers may return subsets)
- Cross-platform bit-exact reproducibility (floating-point noise varies)

Instead, the bulletproof contract is: **every returned IK candidate must FK-close to the target within the arm's documented precision floor**.
