# sim bindings (LeRobot)

Sim-only Python bindings of the generated SO-ARM101 math, used by `tools/sim_driver.py`. Same code as
`tools/sim_bindings` (shared `common.rs`), built for this vendor's arms. Not part of the OEM crates.

```
cd tools/sim_bindings_lerobot && maturin develop --release     # installs `dexforge_lerobot`
python -m pytest tools/tests -q
```

Exposes `dexforge_lerobot.so101` (`fk`, `solve`, `fk_pose`, `ik_nearest`, `jacobian`, `gravity`,
`mass_matrix`, `coriolis`, `inverse_dynamics`, `manifest`, `standard_gravity`) and the buffer layouts
`state_layout()` / `command_layout()`. The arm is the five-joint SO-ARM101 (gripper excluded).
