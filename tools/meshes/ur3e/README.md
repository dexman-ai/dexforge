# UR3e meshes

Collision meshes of the Universal Robots UR3e from UR's own
[Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)
(`meshes/ur3e/collision/*.stl`), written by `tools/import_ur_description.py --arm ur3e`, which also
writes `ur3e.urdf` from UR's xacro. The checked-out commit is recorded in the URDF header.
Meshes: BSD 3-Clause, Universal Robots A/S (see `LICENSE`).

UR's per-link placement of each mesh is baked into the vertices, so the URDF mesh elements have
no origin. The simplified collision meshes are used for display too (UR's visual `.dae` files are
large and MuJoCo cannot load them). File names match the URDF mesh names (`base`, `shoulder`,
`upperarm`, `forearm`, `wrist1`, `wrist2`, `wrist3`).
