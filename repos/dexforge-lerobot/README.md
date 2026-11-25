# dexforge-lerobot

SO-ARM101 (Hugging Face [LeRobot](https://github.com/huggingface/lerobot) /
[TheRobotStudio SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)) for
[`dexhal`](../dexhal/README.md): generated kinematics, closed-form IK and dynamics, a five-joint
`ArmModel`, and a manifest. The crate in `rust/` links against `dexhal`.

```sh
cargo test --manifest-path repos/dexforge-lerobot/Cargo.toml
```

## Layout

| Module | What it is |
| :--- | :--- |
| `so101` | **The arm to use.** Five joints (`shoulder_pan` .. `wrist_roll`; the gripper jaw is not part of it). `So101: ArmModel<5, 16>`, `So101Driver`, `manifest.yaml`. Hand-written, in `model.rs` / `arm.rs` / `driver.rs`. |
| `drivers/so101_6r` | The arm plus one zero-offset *virtual* wrist joint, as a 6R chain. Scaffolded and generated like every other arm here (ssik artifact, SymPy kinematics and dynamics, IK translation, ssik and Pinocchio reference data). Internal: use `so101`. |

### Why a virtual joint

The ssik / dexforge pipeline handles full 6R and 7R chains only, and a five-joint arm has no
ssik artifact. A zero-offset revolute joint appended after the wrist roll (axis perpendicular to
the roll) makes the chain 6R with a closed-form (`ikgeo.three_parallel`) solution, and at `q6 = 0`
it is *exactly* the physical arm. `so101::model` evaluates everything at `q6 = 0` (the leading
5x5 block of `M`, the first five entries of `g`, `C qd` and `tau`) and solves IK by keeping the
6R solutions whose virtual joint is ~0, then polishing the five real joints with it pinned.

### Five joints cannot reach every pose

A five-joint arm reaches only a 5-D subset of SE(3), and a pose from a camera, a teleop device or
a typed command is almost never in it. Two entry points, both in `so101::model`:

* `solve` returns every *exact* solution, like the other arms: the tolerance policy decides what
  counts (`subproblem_numerical`, default `1e-5`, i.e. about 0.01 mm). It returns nothing for a
  pose farther than that from the reachable set. A caller that can live with millimetres passes a
  looser policy; this admits near-reachable targets (about 97% of targets 1 mm / 0.6 deg off, against
  about 1.5% at the default) while the default stays strict.
* `solve_nearest` returns the **closest reachable configuration** and its position / orientation
  error, so the caller decides what is close enough (weighted least squares from the closed-form
  branches, the seed, and a fixed set of fallback starts; deterministic and bounded). It is separate
  from `ArmModel::flange_solve` on purpose: that contract promises solutions that close. Python:
  `dexforge_lerobot.so101.ik_project(pose, q_seed)`. Near a joint limit the fit is clamped into the
  limits and the reported error says so; it does not search along the limit.

The simulator uses `ik_project` for Cartesian commands (see `tools/SIM_SPEC.md`), with a documented
default tolerance.

## Provenance of the model

`so101.urdf` and `so101_6r.urdf` are written by `tools/import_so101_description.py` from the
upstream `so101_new_calib.urdf` (Apache-2.0, pinned commit; see `THIRD_PARTY_NOTICES.md`), with
three deliberate changes: rounding noise in the joint origins is snapped (at most 3.7e-6 rad),
the gripper jaw's mass is folded into `gripper_link`, and the massless `gripper_frame_link` loses
its all-zero inertial. The tests compare against Pinocchio on the **unmodified** upstream URDF;
the measured effect of those changes is <= 1.6e-5 on FK and <= 4.3e-6 N*m on the dynamics.

## The real arm

Two Python scripts in `tools/`, both run with **LeRobot's interpreter** (the dexforge bindings are
found in dexforge's `.venv`; nothing is installed into LeRobot's environment):

* `so101_probe.py` is read-only: servo scan and status, calibration, and live joint angles in the
  convention this model uses. It never writes a register.
* `so101_bridge.py` plans moves with this model and sends them through LeRobot. It is a **dry run
  unless `--live`**; targets must lie inside the model's limits, moves are capped (15 deg per joint,
  25 deg for a Cartesian move, 30 mm), motion is a minimum-jerk path at <= 15 deg/s with LeRobot's
  `max_relative_target` (a float) on every command, a lagging joint stops the run, the gripper is
  never commanded, and torque is never switched off by the script (a limp arm falls; use `release`).

LeRobot's zero and signs for the five joints match this model (checked by moving each joint by
hand: lift + is forward, pan + is clockwise from above, wrist_flex + is toward the front, wrist_roll
+ is clockwise seen from the tip; upright is lift ~0 / elbow ~-90). The real servos reach a few
degrees past the model's limits (pan to ~120, wrist_flex to ~100); the bridge clamps to the model's.
Cartesian moves hold the orientation, which a five-joint arm can only do at a cost in wrist and
elbow motion, and are ill-conditioned near the fully upright pose (a 20 mm drop from straight up asks
for a 74 deg elbow swing and is refused): start Cartesian tests from a bent pose.

## What is not done

* **Hardware I/O.** `So101Driver` has the manifest and buffer contract, but `connect`,
  `read_state` and `write_command` return `ArmError::NotSupported`. The real arm is Feetech
  STS3215 servos on a serial bus, normally driven from Python by LeRobot; no Rust driver exists.
  The MuJoCo simulator (`tools/sim.py --arm so101`) implements the same contract instead.
  A Python bridge to the real arm exists instead: `tools/so101_bridge.py` (below).
* **Manifest values.** `limits.velocity` / `limits.effort` are the URDF's CAD placeholders
  (10 rad/s, 10 N*m), not STS3215 ratings; the control rate, latency and acceleration limit are
  assumed simulation defaults. All are marked in `rust/manifest.yaml`. Replace them before any
  hardware use.
* **Calibration.** Joint zero offsets follow LeRobot's `new_calib` convention (`lerobot-calibrate`
  writes the arm's own); nothing here measures your arm's actual kinematics, and there is no camera /
  hand-eye calibration.
* **The bridge has only been dry-run.** No `--live` move has been made yet; the manifest limits above
  are not the servos' real ones.
* `nominal_chain()` for kinematic calibration is not provided for the five-joint arm.
* `solve_nearest` is an unconstrained fit followed by a clamp, not a joint-limit-constrained
  optimum, and it takes up to `FALLBACK_STARTS` (24) extra bounded fits when no exact solution is
  found (about 0.4 ms in the simulator's Python binding, 1.3 ms worst case measured).

## Regenerating

```sh
python tools/import_so101_description.py /path/to/SO-ARM100        # URDFs + meshes
D=repos/dexforge-lerobot/rust/src/drivers/so101_6r
python codegen/ssik_artifact.py build --urdf $D/so101_6r.urdf --base base_link --ee gripper_frame_link --out $D/codegen/so101_6r_ik.py
python $D/codegen/transpile_ik_solver.py $D/codegen/so101_6r_ik.py $D/ik_solver.rs
python codegen/generate_dynamics.py --artifact $D/codegen/so101_6r_ik.py --urdf $D/so101_6r.urdf --base base_link --ee gripper_frame_link --arm so101_6r --output $D/generated_dynamics.rs
python $D/codegen/generate_test_data.py
python codegen/generate_dynamics_test_data.py --urdf $D/so101_6r.urdf --base base_link --ee gripper_frame_link --output-dir $D --arm so101_6r --samples 1000 --seed 0
python codegen/scaffold.py --oem lerobot --arm so101_6r --urdf $D/so101_6r.urdf --base base_link --ee gripper_frame_link --only arm.rs
# five-joint arm: Pinocchio parity of the facade, and the independent reference on the unmodified URDF
S=repos/dexforge-lerobot/rust/src/so101
python codegen/generate_dynamics_test_data.py --urdf $S/so101.urdf --base base_link --ee gripper_frame_link --output-dir $S --arm so101 --samples 1000 --seed 0
python $S/codegen/generate_reference_data.py
```
