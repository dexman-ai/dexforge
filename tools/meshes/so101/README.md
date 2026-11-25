# SO-ARM101 meshes

Meshes of the SO-ARM101 (Hugging Face LeRobot / TheRobotStudio) from the
[SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) repository
(`Simulation/SO101/assets/*.stl`, pinned commit in `tools/import_so101_description.py`),
license Apache-2.0 (see `LICENSE`).

Written by `tools/import_so101_description.py`, which also writes `so101.urdf` / `so101_6r.urdf`.
Modified from upstream: each link's visual STLs (several per link, each with its own placement)
are placed by their URDF visual origins, merged into one STL per link (the gripper jaw is merged
into `gripper_link.stl`, at its closed pose) and decimated by vertex clustering (1.2 mm cell) to
keep the files small. The URDF mesh elements therefore have no origin. File names are the link
names: `base_link`, `shoulder_link`, `upper_arm_link`, `lower_arm_link`, `wrist_link`,
`gripper_link`.
