# dexforge-agilex

Scaffolded OEM repository for AgileX.

Layout:
- `rust/` - no_std Rust crate for the Piper driver
- `python/` - Python bindings crate for `dexforge_agilex`

Standalone OEM package layout for in-workspace builds.
## Piper kinematic revision

AgileX ships two Piper DH/URDF revisions. Firmware before S-V1.6-3 is the
**old** revision. S-V1.6-3 and later is the **new** revision, which shifts the
joint 2 and joint 3 frames by 2° (a joint-zero offset: same hardware, different
zero position). This crate targets the **new (S-V1.6-3 and later)** revision,
which is AgileX's current default.

`piper.urdf` is the new revision. It differs from the old one in three values:

| Field | Old | New |
|---|---|---|
| joint1 upper limit | 2.168 | 2.618 |
| joint2 origin pitch | -0.10095 | -0.1359 |
| joint3 origin yaw | -1.759 | -1.7939 |

The IK solver, dynamics and reference data are all generated from it. The
`piper_description_old.urdf` / `piper_description.urdf` pair in
`agilexrobotics/Piper_ros` (`src/piper_description/urdf`, branch `noetic`) is the
source of truth. MuJoCo Menagerie's `agilex_piper` is the old revision.

To switch revision, edit those three values in `piper.urdf`, then run from the
repo root (`D=repos/dexforge-agilex/rust/src/drivers/piper`):

```sh
python codegen/ssik_artifact.py build --urdf $D/piper.urdf --base base_link --ee link6 --out $D/codegen/piper_ik.py
python codegen/scaffold.py --oem agilex --arm piper --urdf $D/piper.urdf --base base_link --ee link6 --only arm.rs
python $D/codegen/generate_code.py
python codegen/generate_dynamics.py --artifact $D/codegen/piper_ik.py --urdf $D/piper.urdf --base base_link --ee link6 --arm piper --output $D/generated_dynamics.rs
python codegen/generate_dynamics_test_data.py --urdf $D/piper.urdf --base base_link --ee link6 --output-dir $D --arm piper --samples 1000 --seed 0
(cd repos/dexforge-agilex/python && maturin develop --release)
python $D/codegen/generate_test_data.py
```
