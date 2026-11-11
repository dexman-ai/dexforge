# UR5 dynamics template

Golden path for getting an arm’s symbolic dynamics wired and proven against
Pinocchio. Use this when porting the next OEM arm.

## What “done” looks like

1. `generated_dynamics.rs` exposes `gravity`, `mass_matrix`, `coriolis`, `inverse_dynamics`.
2. The driver compiles those kernels (`pub mod generated_dynamics` in `mod.rs`).
3. `ArmModel` overrides `flange_gravity` / `flange_mass_matrix` / `flange_coriolis`.
4. Pinocchio golden data + `dynamics_parity` tests pass under `cargo test`.

Pinocchio is **only** needed to regenerate the golden file. Day-to-day tests are pure Rust.

## UR5 paths (reference)

| Role | Path |
| --- | --- |
| URDF (has `<inertial>` blocks) | `repos/dexforge-ur/rust/src/drivers/ur5/ur5.urdf` |
| Symbolic kernels | `…/ur5/generated_dynamics.rs` |
| Pinocchio blob + loader | `…/ur5/dynamics_test_data.bin`, `dynamics_test_data.rs` |
| Parity tests | `…/ur5/dynamics_parity.rs` |
| Wiring | `…/ur5/mod.rs`, `…/ur5/arm.rs` |

Base / EE used by the generators: `base_link` → `ee_link`.

## Regenerate symbolic dynamics

```sh
pip install ssik sympy numpy
python codegen/generate_dynamics.py --artifact ur5_ik \
    --urdf repos/dexforge-ur/rust/src/drivers/ur5/ur5.urdf \
    --output repos/dexforge-ur/rust/src/drivers/ur5/generated_dynamics.rs
```

## Regenerate Pinocchio ground truth

```sh
pip install pin numpy
python codegen/generate_dynamics_test_data.py \
    --urdf repos/dexforge-ur/rust/src/drivers/ur5/ur5.urdf \
    --output-dir repos/dexforge-ur/rust/src/drivers/ur5 \
    --arm ur5 --samples 10000 --seed 0
```

Pinocchio conventions baked into that script:

- **Mass matrix:** `crba`, then symmetrize (Pinocchio fills only the upper triangle).
- **Coriolis:** `C(q,qd) qd` via `computeCoriolisMatrix` (not raw `nonLinearEffects`).
- **Inverse dynamics:** `rnea(q, qd, qdd)`.
- **Gravity:** `model.gravity.linear = [0, 0, -9.81]` (must match dexforge’s `g_world` argument).

## Run the parity suite

```sh
cd repos/dexforge-ur && cargo test -p dexforge-ur-core dynamics_parity
```

Expect three tests: conventions, four-term Pinocchio match, and `τ = M q̈ + C q̇ + g`.

`dexforge build` (LLM or `--from-existing`) runs the same Pinocchio golden +
`dynamics_parity` setup automatically before `cargo test --lib`, so a successful
end-to-end build already includes this gate.

## Driver wiring checklist (copy for the next arm)

1. Confirm the arm URDF has real `<inertial>` masses on every moving link.
2. Emit `generated_dynamics.rs` (command above, swap arm/URDF/artifact).
3. In `mod.rs`: `pub mod generated_dynamics;` and `#[cfg(test)] mod dynamics_parity;`.
4. In `arm.rs`: override the three `flange_*` methods to call `generated_dynamics::{gravity, mass_matrix, coriolis}`.
5. Generate `dynamics_test_data.{bin,rs}`; keep `dynamics_parity.rs` next to them.
6. Run `cargo test … dynamics_parity`.

## Before trusting a mismatch

- Joint order matches the URDF actuated order (UR5: shoulder_pan → … → wrist_3).
- Same URDF for SymPy codegen and Pinocchio.
- Same gravity vector on both sides.
- Python ≤ 3.13 for Pinocchio / codegen tooling in this repo.
