# Third-party notices

## ssik (IK algorithm translation)

Parts of `dexhal` IK solvers are translated from ssik.

- Source: https://github.com/personalrobotics/ssik
- License: BSD-3-Clause
- Copyright (c) 2026, Siddhartha Srinivasa

## Upstream URDF models

The kinematic and inertial models in `repos/*/rust/src/drivers/*/*.urdf` derive from these upstream packages.

| Driver | Upstream | License |
| :--- | :--- | :--- |
| `franka_panda` | franka_description (frankarobotics) | Apache-2.0 |
| `ur5` | Universal_Robots_ROS2_Description (UniversalRobots) | BSD-3-Clause |
| `ur3e` | Universal_Robots_ROS2_Description (UniversalRobots), `ur_type:=ur3e`; meshes in `tools/meshes/ur3e/` | BSD-3-Clause |
| `z1` | unitree_ros | BSD-3-Clause |
| `piper` | piper_ros (agilexrobotics) | MIT |
| `rizon4`, `rizon10` | flexiv_description (flexivrobotics) | Apache-2.0 |
| `xarm7` | xArm-Developer/xarm_ros (UFACTORY, Copyright 2018) | BSD-3-Clause |
