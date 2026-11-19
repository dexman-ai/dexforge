#!/usr/bin/env python3
"""Import the UR5e from MuJoCo Menagerie as the DexForge ``ur5`` URDF + meshes.

Menagerie ships an MJCF, not a URDF, and our codegen (IK, dynamics) is driven by
a URDF. This converts ``universal_robots_ur5e/ur5e.xml`` into
``repos/dexforge-ur/rust/src/drivers/ur5/ur5.urdf`` and merges its visual OBJ
parts into one STL per link under ``tools/meshes/ur5/``.

Kept the same on purpose, so nothing downstream changes shape: joint names,
``base_link``, ``ee_link`` and ``tool0`` (both the z-out flange frame, see below).
``base_link`` is the UR *base* frame (Menagerie's world frame, which is
what the controller reports TCP poses in, and what the solver has always used).
Menagerie's base body is yawed 180 deg about Z relative to it, so the arm hangs
off ``base_link_inertia`` (ROS ``ur_description`` convention), a fixed 180 deg
yaw child of ``base_link``.

Values that do not come from Menagerie (documented here, not guessed silently):
  * velocity limit: UR5e datasheet, 180 deg/s = pi rad/s on every joint.
  * joint position limits: +-2*pi, with the elbow at +-pi as in Menagerie
    (Menagerie writes 3.1415 / 6.28319; we use the exact values).
  * effort limits: Menagerie ``forcerange`` (150 N*m size-3 joints, 28 N*m size-1).

Menagerie rounds some UR5e dimensions (0.163 vs the datasheet 0.1625, 0.392 vs
0.3922, 0.1 vs 0.0997 / 0.0996). We keep Menagerie's numbers so the solver and a
MuJoCo model of the same arm agree exactly.

Usage::

    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie /tmp/menagerie
    python tools/import_menagerie_ur5e.py /tmp/menagerie/universal_robots_ur5e
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from lxml import etree
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
URDF_OUT = REPO / "repos/dexforge-ur/rust/src/drivers/ur5/ur5.urdf"
MESH_OUT = REPO / "tools/meshes/ur5"
MESH_REF = "tools/meshes/ur5"  # URDF mesh paths are relative to the repo root

# (URDF link, MJCF body, mesh stem, MJCF mesh prefix)
LINKS = [
    ("base_link_inertia", "base", "base", "base"),
    ("shoulder_link", "shoulder_link", "shoulder", "shoulder"),
    ("upper_arm_link", "upper_arm_link", "upperarm", "upperarm"),
    ("forearm_link", "forearm_link", "forearm", "forearm"),
    ("wrist_1_link", "wrist_1_link", "wrist1", "wrist1"),
    ("wrist_2_link", "wrist_2_link", "wrist2", "wrist2"),
    ("wrist_3_link", "wrist_3_link", "wrist3", "wrist3"),
]
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
LIMITS = {  # lower/upper (rad), effort (N*m), velocity (rad/s)
    "shoulder_pan_joint": (2 * math.pi, 150.0), "shoulder_lift_joint": (2 * math.pi, 150.0),
    "elbow_joint": (math.pi, 150.0), "wrist_1_joint": (2 * math.pi, 28.0),
    "wrist_2_joint": (2 * math.pi, 28.0), "wrist_3_joint": (2 * math.pi, 28.0),
}
VELOCITY = math.pi


def _floats(s: str | None, default: str) -> np.ndarray:
    return np.array([float(x) for x in (s if s is not None else default).split()])


def _quat(body_or_inertial) -> R:
    w, x, y, z = _floats(body_or_inertial.get("quat"), "1 0 0 0")
    return R.from_quat([x, y, z, w])  # scipy is (x, y, z, w); normalizes


def _n(v: float) -> str:
    v = 0.0 if abs(v) < 1e-15 else v
    return repr(float(v))


def _vec(v) -> str:
    return " ".join(_n(x) for x in v)


def _rpy(rot: R) -> np.ndarray:
    return rot.as_euler("xyz")  # URDF rpy == extrinsic xyz


def _link_xml(name: str, body, mesh_stem: str | None) -> str:
    inertial = body.find("inertial")
    mass = float(inertial.get("mass"))
    pos = _floats(inertial.get("pos"), "0 0 0")
    rot = _quat(inertial).as_matrix()
    inertia = rot @ np.diag(_floats(inertial.get("diaginertia"), "0 0 0")) @ rot.T
    geom = ""
    if mesh_stem:
        ref = f"{MESH_REF}/{mesh_stem}.stl"
        geom = (f'    <visual>\n      <geometry>\n        <mesh filename="{ref}"/>\n      </geometry>\n'
                f'    </visual>\n    <collision>\n      <geometry>\n        <mesh filename="{ref}"/>\n'
                f'      </geometry>\n    </collision>\n')
    return (f'  <link name="{name}">\n    <inertial>\n      <origin xyz="{_vec(pos)}" rpy="0.0 0.0 0.0"/>\n'
            f'      <mass value="{_n(mass)}"/>\n      <inertia ixx="{_n(inertia[0, 0])}" ixy="{_n(inertia[0, 1])}" '
            f'ixz="{_n(inertia[0, 2])}" iyy="{_n(inertia[1, 1])}" iyz="{_n(inertia[1, 2])}" izz="{_n(inertia[2, 2])}"/>\n'
            f'    </inertial>\n{geom}  </link>\n')


def _zero_link(name: str) -> str:
    return (f'  <link name="{name}">\n    <inertial>\n      <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>\n'
            f'      <mass value="0.0"/>\n      <inertia ixx="0.0" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/>\n'
            f'    </inertial>\n  </link>\n')


def _fixed(name: str, parent: str, child: str, xyz, rpy) -> str:
    return (f'  <joint name="{name}" type="fixed">\n    <parent link="{parent}"/>\n    <child link="{child}"/>\n'
            f'    <origin xyz="{_vec(xyz)}" rpy="{_vec(rpy)}"/>\n    <axis xyz="1.0 0.0 0.0"/>\n  </joint>\n')


def build_urdf(root) -> str:
    bodies = {b.get("name"): b for b in root.iter("body")}
    out = ['<robot name="ur5">\n']
    for link, body, stem, _ in LINKS:
        out.append(_link_xml(link, bodies[body], stem))
    out.append(_zero_link("base_link"))
    out.append(_zero_link("ee_link"))
    out.append(_zero_link("tool0"))
    out.append('  <link name="world"/>\n')

    for i, joint in enumerate(JOINTS):
        parent, child = LINKS[i][0], LINKS[i + 1][0]
        body = bodies[LINKS[i + 1][1]]
        jnt = body.find("joint")
        assert jnt.get("name") == joint, (jnt.get("name"), joint)
        axis = _floats(jnt.get("axis"), "0 1 0")  # MJCF default for the arm class
        lim, effort = LIMITS[joint]
        pos = _floats(body.get("pos"), "0 0 0")
        rpy = _rpy(_quat(body))
        out.append(
            f'  <joint name="{joint}" type="revolute">\n    <parent link="{parent}"/>\n    <child link="{child}"/>\n'
            f'    <origin xyz="{_vec(pos)}" rpy="{_vec(rpy)}"/>\n    <axis xyz="{_vec(axis)}"/>\n'
            f'    <limit effort="{_n(effort)}" velocity="{_n(VELOCITY)}" lower="{_n(-lim)}" upper="{_n(lim)}"/>\n'
            f'    <dynamics damping="0.0" friction="0.0"/>\n  </joint>\n')

    # Flange: Menagerie's attachment_site is the z-out tool frame (tool0). ee_link is the same
    # frame: the solver's TCP orientation has always been z along the wrist axis (verified
    # against the previous solver), so ee_link must not be the x-forward ROS ee frame.
    site = bodies["wrist_3_link"].find("site")
    flange = _floats(site.get("pos"), "0 0 0")
    tool_rpy = _rpy(_quat(site))
    out.append(_fixed("ee_fixed_joint", "wrist_3_link", "ee_link", flange, tool_rpy))
    out.append(_fixed("base_link-base_link_inertia_fixed_joint", "base_link", "base_link_inertia", [0, 0, 0],
                      [0.0, 0.0, math.pi]))
    out.append(_fixed("wrist_3_link-tool0_fixed_joint", "wrist_3_link", "tool0", flange, tool_rpy))
    out.append(_fixed("world_joint", "world", "base_link", [0, 0, 0], [0, 0, 0]))
    out.append("</robot>\n")
    return "".join(out)


def write_meshes(src: Path) -> None:
    import trimesh

    MESH_OUT.mkdir(parents=True, exist_ok=True)
    for _, _, stem, prefix in LINKS:
        parts = sorted((src / "assets").glob(f"{prefix}_*.obj")) or [src / "assets" / f"{prefix}.obj"]
        mesh = trimesh.util.concatenate([trimesh.load(p, force="mesh", process=False) for p in parts])
        mesh.export(MESH_OUT / f"{stem}.stl")
        print(f"  {stem}.stl  {len(mesh.faces)} faces from {len(parts)} part(s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("menagerie_ur5e", type=Path, help="mujoco_menagerie/universal_robots_ur5e")
    args = ap.parse_args()
    root = etree.parse(str(args.menagerie_ur5e / "ur5e.xml")).getroot()
    URDF_OUT.write_text(build_urdf(root))
    print(f"wrote {URDF_OUT.relative_to(REPO)}")
    write_meshes(args.menagerie_ur5e)


if __name__ == "__main__":
    main()
