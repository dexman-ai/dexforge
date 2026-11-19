# sim bindings

Sim-only Python bindings of the generated arm math, used by `tools/sim_driver.py`.
They are **not** part of the OEM crates (upstream removed the per-driver Python
bindings; nothing in `repos/` needs Python at build or test time).

```
cd tools/sim_bindings && maturin develop --release     # installs `dexforge_ur`
python -m pytest tools/tests -q
```

Exposes `dexforge_ur.ur5` (`fk`, `solve`, `fk_pose`, `ik_nearest`, `jacobian`,
`gravity`, `mass_matrix`, `coriolis`, `inverse_dynamics`, `manifest`,
`standard_gravity`) and the buffer layouts `state_layout()` / `command_layout()`.
