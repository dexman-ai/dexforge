# UR5e meshes

Visual meshes of the Universal Robots UR5e from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(`universal_robots_ur5e`), merged into one STL per link by
`tools/import_menagerie_ur5e.py`, which also writes `ur5.urdf` from Menagerie's
`ur5e.xml`. The arm keeps the name `ur5` in code. Meshes: BSD 3-Clause, Copyright 2018 ROS Industrial Consortium (see `LICENSE`).

They are used by `tools/visualize_ik.py`. File names match the URDF mesh names
(`base`, `shoulder`, `upperarm`, `forearm`, `wrist1`, `wrist2`, `wrist3`).
