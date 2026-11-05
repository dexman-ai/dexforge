# ssik Shared Tests

This directory is the canonical home for the ssik-derived shared tests used across OEM repos.

## Purpose

The files here are copied/adapted from upstream ssik tests and are executed from each OEM repository via thin delegate test files.

- `test_prebuilt_sanity.py`
- `test_prebuilt_uniform_fuzz.py`
- `_hypothesis_strategies.py`

The tests import `ssik.prebuilt.<arm>_ik`, `ssik.prebuilt._manifest`, and `ssik.core.*`.
Those modules are provided per OEM repo by local test shims under:

- `repos/dexforge-*/python/tests/ssik_shim/ssik/...`

## Execution Model

Each OEM repo has one delegate file under `repos/dexforge-*/python/tests/`:

- `test_shared_suite.py`

This delegate imports the shared suites in this directory so the same tests run
everywhere while arm wrappers remain OEM-local.

## How To Run

Example for one OEM repo:

```bash
pytest repos/dexforge-agilex/python/tests -v
```

Run all OEM repos:

```bash
pytest repos/dexforge-*/python/tests -v
```

## Source Of Truth

- Shared ssik-style prebuilt tests: this directory.
- Shared native API/parity contracts: this directory (`test_native_contracts.py`).
- OEM compatibility wrappers and manifests: each OEM repo under `repos/dexforge-*/python/tests/ssik_shim/`.
- Optional perf benchmarking script: `ssik_shared_tests/perf_test.py`.

## Maintenance

When syncing with upstream ssik:

1. Update files in this directory from upstream.
2. Keep imports compatible with package-local layout (`from ._hypothesis_strategies import ...`).
3. Ensure OEM shims still satisfy the expected `ssik.prebuilt.*` public surface.
