# Flexiv Rizon IK Solvers

This directory contains Python passthrough wrappers for the Flexiv Rizon 4 and Rizon 10 IK solvers.

## Implementation Approach

Unlike other dexforge IK drivers that are implemented in Rust and wrapped with Python bindings, the Rizon drivers are **Python passthrough wrappers** that delegate to the ssik library's implementation.

### Why Passthrough?

The Flexiv Rizon robots (both Rizon 4 and Rizon 10) use the `husty_pfurner.general_6r` solver, which is a tier 3 solver that requires:
- Cached Raghavan-Roth symbolic derivation blobs
- Polynomial matrix eigenvalue solvers
- Compiled extensions (e.g., `ssik._pencil`)

This is significantly more complex than the existing dexforge implementations:
- **Tier 0** (UR5): Closed-form analytical solution
- **Tier 1** (Franka, xArm7): jointlock.seven_r with spherical/spherical_two_parallel
- **Tier 2** (Piper): ikgeo.general_6r using standard Raghavan-Roth elimination
- **Tier 3** (Rizon 4/10): jointlock.seven_r with husty_pfurner.general_6r

A full Rust implementation would require porting the entire husty_pfurner solver stack, which is beyond the scope of a simple artifact translation.

## API Compatibility

The passthrough wrappers maintain full API compatibility with other dexforge drivers:

```python
from ssik.prebuilt import rizon4_ik, rizon10_ik

# Forward kinematics
T = rizon4_ik.fk(q)

# Inverse kinematics
solutions = rizon4_ik.solve(
    T_target,
    max_solutions=8,
    respect_limits=True,
    allow_refinement=False
)
```

## Technical Details

### Lazy Import Implementation

The wrappers use a lazy import mechanism to avoid circular dependencies between the dexforge-py ssik compatibility layer and the system ssik package:

1. On first use of `fk()` or `solve()`, the wrapper:
   - Filters out dexforge-related paths from `sys.path`
   - Removes dexforge-py ssik modules from `sys.modules`
   - Imports from the system ssik package
   - Saves references to the system modules
   - Restores `sys.path` (but leaves system ssik in `sys.modules`)

2. Subsequent calls use the cached references

### Robot Specifications

**Flexiv Rizon 4:**
- DOF: 7
- Solver: jointlock.seven_r → husty_pfurner.general_6r
- Locked joint: joint 2
- Base link: `base_link`
- End-effector link: `flange`

**Flexiv Rizon 10:**
- DOF: 7
- Solver: jointlock.seven_r → husty_pfurner.general_6r
- Locked joint: joint 2
- Base link: `base_link`
- End-effector link: `flange`
- Longer reach than Rizon 4, different joint limits

## Dependencies

Requires `ssik` to be installed:
```bash
pip install ssik
```

Note: The repository is locked to ssik 0.8.3 (see `SSIK_VERSION.txt`), but the Rizon artifacts were added in ssik 1.2.4+. The wrappers require a newer version of ssik to be installed in the system Python environment.

## Future Work

A native Rust implementation is planned for a future release and would require:
1. Porting `husty_pfurner.general_6r` solver to Rust
2. Implementing polynomial matrix eigenvalue solver
3. Pre-computing and embedding RR blob constants
4. Testing against the ssik reference implementation

For now, the Python passthrough provides full functionality while maintaining API compatibility.
