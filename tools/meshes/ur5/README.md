# UR5e meshes

Collision meshes of the Universal Robots UR5e from UR's own
[Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)
(`meshes/ur5e/collision/*.stl`), written by `tools/import_ur_description.py`, which also
writes `ur5.urdf` from UR's xacro. The checked-out commit is recorded in the URDF header.
The arm keeps the name `ur5` in code. Meshes: BSD 3-Clause, Universal Robots A/S (see `LICENSE`).

UR's per-link placement of each mesh is baked into the vertices, so the URDF mesh elements have
no origin. The simplified collision meshes are used for display too: UR's visual `.dae` files
are about 9 MB and MuJoCo cannot load them.

They are used by `tools/visualize_ik.py` and the simulator. File names match the URDF mesh
names (`base`, `shoulder`, `upperarm`, `forearm`, `wrist1`, `wrist2`, `wrist3`).
