#!/usr/bin/env python3
"""Import the manufacturer's UR5e description as the DexForge ``ur5`` URDF + meshes.

Source of truth: Universal Robots' own ROS 2 description
(https://github.com/UniversalRobots/Universal_Robots_ROS2_Description, BSD-3-Clause).
Its xacro is expanded for ``ur_type:=ur5e`` and rewritten into
``repos/dexforge-ur/rust/src/drivers/ur5/ur5.urdf`` plus one STL per link under
``tools/meshes/ur5/``. Everything physical comes from UR: kinematics, link masses,
centres of mass, inertia tensors, joint limits and meshes. Nothing is taken from MuJoCo
Menagerie.

Conventions kept so nothing downstream changes shape: the arm is still ``ur5``, with
``base_link``, ``base_link_inertia``, the six ``*_joint`` names, ``ee_link`` and ``tool0``.

Deliberate changes from UR's file (each one is checked, not silent):

* **Base frame.** UR's ``base_link`` is REP-103 (x forward); the controller reports poses in
  UR's ``base`` frame (``base_link`` yawed 180 deg about Z), where q = 0 puts the TCP at
  x = -0.817. This driver has always used the controller frame, so our ``base_link`` *is* UR's
  ``base``. ``base_link_inertia`` has the same orientation as ``base``, so the joint between
  them is the identity.
* **ee_link.** The flange frame ``wrist_3_link -> flange -> tool0`` (z out of the flange)
  composed into one fixed joint; ``ee_link`` and ``tool0`` are the same frame.
* **Inertia frames.** UR gives some inertial origins a rotation; the tensor is rotated into
  the link frame so every ``<inertial>`` has ``rpy = 0`` (what the MJCF generator and the
  dynamics codegen expect).
* **Meshes.** UR's collision STLs (simplified, a few hundred KB in total) are used for both
  display and collision. They carry a per-link placement in the URDF; it is baked into the
  vertices so the URDF mesh elements have no origin. The visual ``.dae`` files are not used:
  MuJoCo cannot load them and they are about 9 MB.
* **Float noise.** UR derives its model from DH parameters, which leaves a few rounded or
  residual values (for example a roll of 1.570796327, which is pi/2 truncated to 10 digits, and
  offsets of 2e-11 m). Rotation angles within 1e-9 of a multiple of pi/2 are set to that
  multiple, and lengths below 1e-9 m to 0. Nothing else is modified (not dimensions, masses,
  inertias or limits). Every change is printed, and none may exceed 1e-9. The IK artifact
  classifies axes as parallel or intersecting with tolerances, so this noise would otherwise
  leak into it.

UR does *not* provide: joint acceleration limits (their yaml says "not publicly available"),
an actuator or motor model, rotor inertia, joint damping or friction. Those are not in the URDF.

Usage::

    pip install xacro
    git clone https://github.com/UniversalRobots/Universal_Robots_ROS2_Description /tmp/ur_description
    python tools/import_ur_description.py /tmp/ur_description

The checked-out commit is recorded in the generated URDF.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from lxml import etree
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
URDF_OUT = REPO / "repos/dexforge-ur/rust/src/drivers/ur5/ur5.urdf"
MESH_OUT = REPO / "tools/meshes/ur5"
MESH_REF = "tools/meshes/ur5"  # URDF mesh paths are relative to the repo root
UR_TYPE = "ur5e"
SNAP_TOL = 1e-9  # largest value change the float-noise snap may make

# (URDF link, mesh stem in UR's meshes/<type>/collision/)
LINKS = [
    ("base_link_inertia", "base"),
    ("shoulder_link", "shoulder"),
    ("upper_arm_link", "upperarm"),
    ("forearm_link", "forearm"),
    ("wrist_1_link", "wrist1"),
    ("wrist_2_link", "wrist2"),
    ("wrist_3_link", "wrist3"),
]
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

_max_snap = 0.0
_snaps: dict[tuple[str, float, float], None] = {}  # (kind, before, after), insertion-ordered


def _snap(v: float, step: float = math.pi / 2) -> float:
    """Snap ``v`` to the nearest multiple of ``step`` if within SNAP_TOL (angles)."""
    global _max_snap
    k = round(v / step)
    d = abs(v - k * step)
    if d <= SNAP_TOL:
        _max_snap = max(_max_snap, d)
        if v != k * step:
            _snaps[("angle", float(v), float(k * step))] = None
        return k * step
    return v


def _snap_len(v: float) -> float:
    global _max_snap
    if abs(v) <= SNAP_TOL:
        _max_snap = max(_max_snap, abs(v))
        if v != 0.0:
            _snaps[("length", float(v), 0.0)] = None
        return 0.0
    return v


def _floats(s: str | None, default: str = "0 0 0") -> np.ndarray:
    return np.array([float(x) for x in (s if s else default).split()])


def _n(v: float) -> str:
    return repr(0.0 if abs(v) < 1e-15 else float(v))


def _vec(v) -> str:
    return " ".join(_n(x) for x in v)


def _rot(rpy) -> np.ndarray:
    return R.from_euler("xyz", rpy).as_matrix()  # URDF rpy == extrinsic xyz


def _origin(el) -> tuple[np.ndarray, np.ndarray]:
    o = el.find("origin")
    if o is None:
        return np.zeros(3), np.zeros(3)
    return _floats(o.get("xyz")), _floats(o.get("rpy"))


def expand_xacro(ur_description: Path) -> etree._Element:
    """Expand UR's xacro for ``ur_type`` without a ROS install.

    xacro resolves ``$(find ur_description)`` through ``ament_index_python``; a stub that maps
    that one package name to ``ur_description`` is all it needs.
    """
    pkgs = types.ModuleType("ament_index_python.packages")

    def get_package_share_directory(name: str) -> str:
        if name != "ur_description":
            raise KeyError(name)
        return str(ur_description)

    pkgs.get_package_share_directory = get_package_share_directory  # type: ignore[attr-defined]
    top = types.ModuleType("ament_index_python")
    top.packages = pkgs  # type: ignore[attr-defined]
    sys.modules["ament_index_python"] = top
    sys.modules["ament_index_python.packages"] = pkgs

    import xacro

    doc = xacro.process_file(
        str(ur_description / "urdf/ur.urdf.xacro"), mappings={"ur_type": UR_TYPE, "name": "ur"}
    )
    return etree.fromstring(doc.toxml().encode())


def _inertial_in_link_frame(link) -> tuple[float, np.ndarray, np.ndarray]:
    """(mass, centre of mass, 3x3 inertia) with the tensor rotated into the link frame."""
    i = link.find("inertial")
    mass = float(i.find("mass").get("value"))
    xyz, rpy = _origin(i)
    rpy = np.array([_snap(a) for a in rpy])
    t = i.find("inertia")
    ixx, iyy, izz = (float(t.get(k)) for k in ("ixx", "iyy", "izz"))
    ixy, ixz, iyz = (float(t.get(k)) for k in ("ixy", "ixz", "iyz"))
    tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    rot = _rot(rpy)
    return mass, xyz, rot @ tensor @ rot.T


def _link_xml(name: str, mass: float, com, tensor, mesh_stem: str | None) -> str:
    geom = ""
    if mesh_stem:
        ref = f"{MESH_REF}/{mesh_stem}.stl"
        geom = (f'    <visual>\n      <geometry>\n        <mesh filename="{ref}"/>\n      </geometry>\n'
                f'    </visual>\n    <collision>\n      <geometry>\n        <mesh filename="{ref}"/>\n'
                f'      </geometry>\n    </collision>\n')
    return (f'  <link name="{name}">\n    <inertial>\n      <origin xyz="{_vec(com)}" rpy="0.0 0.0 0.0"/>\n'
            f'      <mass value="{_n(mass)}"/>\n      <inertia ixx="{_n(tensor[0, 0])}" ixy="{_n(tensor[0, 1])}" '
            f'ixz="{_n(tensor[0, 2])}" iyy="{_n(tensor[1, 1])}" iyz="{_n(tensor[1, 2])}" izz="{_n(tensor[2, 2])}"/>\n'
            f'    </inertial>\n{geom}  </link>\n')


def _zero_link(name: str) -> str:
    return (f'  <link name="{name}">\n    <inertial>\n      <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>\n'
            f'      <mass value="0.0"/>\n      <inertia ixx="0.0" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/>\n'
            f'    </inertial>\n  </link>\n')


def _fixed(name: str, parent: str, child: str, xyz, rpy) -> str:
    return (f'  <joint name="{name}" type="fixed">\n    <parent link="{parent}"/>\n    <child link="{child}"/>\n'
            f'    <origin xyz="{_vec(xyz)}" rpy="{_vec(rpy)}"/>\n    <axis xyz="1.0 0.0 0.0"/>\n  </joint>\n')


def build_urdf(root, commit: str) -> str:
    links = {l.get("name"): l for l in root.findall("link")}
    joints = {j.get("name"): j for j in root.findall("joint")}
    out = [f'<!-- Generated by tools/import_ur_description.py from '
           f'UniversalRobots/Universal_Robots_ROS2_Description@{commit} (ur_type={UR_TYPE}). Do not edit. -->\n',
           '<robot name="ur5">\n']
    for name, stem in LINKS:
        mass, com, tensor = _inertial_in_link_frame(links[name])
        com = np.array([_snap_len(c) for c in com])
        out.append(_link_xml(name, mass, com, tensor, stem))
    for name in ("base_link", "ee_link", "tool0"):
        out.append(_zero_link(name))
    out.append('  <link name="world"/>\n')

    for i, jname in enumerate(JOINTS):
        j = joints[jname]
        assert j.find("parent").get("link") == LINKS[i][0], jname
        assert j.find("child").get("link") == LINKS[i + 1][0], jname
        xyz, rpy = _origin(j)
        xyz = np.array([_snap_len(c) for c in xyz])
        rpy = np.array([_snap(a) for a in rpy])
        axis = _floats(j.find("axis").get("xyz"))
        lim = j.find("limit")
        out.append(
            f'  <joint name="{jname}" type="revolute">\n    <parent link="{LINKS[i][0]}"/>\n'
            f'    <child link="{LINKS[i + 1][0]}"/>\n    <origin xyz="{_vec(xyz)}" rpy="{_vec(rpy)}"/>\n'
            f'    <axis xyz="{_vec(axis)}"/>\n'
            f'    <limit effort="{_n(float(lim.get("effort")))}" velocity="{_n(float(lim.get("velocity")))}" '
            f'lower="{_n(float(lim.get("lower")))}" upper="{_n(float(lim.get("upper")))}"/>\n'
            f'    <dynamics damping="0.0" friction="0.0"/>\n  </joint>\n')

    # Flange: wrist_3_link -> flange -> tool0 composed into one fixed joint (rotation only; UR's
    # flange and tool0 joints have zero translation).
    fl, tl = joints["wrist_3-flange"], joints["flange-tool0"]
    (fx, fr), (tx, tr) = _origin(fl), _origin(tl)
    assert not fx.any() and not tx.any(), "UR flange/tool0 joints are expected to be pure rotations"
    rot = _rot([_snap(a) for a in fr]) @ _rot([_snap(a) for a in tr])
    rot = np.where(np.abs(rot) < 1e-12, 0.0, np.round(rot, 12))  # exact 0 / +-1 entries
    tool_rpy = R.from_matrix(rot).as_euler("xyz")
    assert np.abs(_rot(tool_rpy) - rot).max() < 1e-12
    out.append(_fixed("ee_fixed_joint", "wrist_3_link", "ee_link", [0.0, 0.0, 0.0], tool_rpy))
    out.append(_fixed("wrist_3_link-tool0_fixed_joint", "wrist_3_link", "tool0", [0.0, 0.0, 0.0], tool_rpy))
    # base_link is UR's `base` (controller) frame, which has the orientation of base_link_inertia.
    out.append(_fixed("base_link-base_link_inertia_fixed_joint", "base_link", "base_link_inertia",
                      [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
    out.append(_fixed("world_joint", "world", "base_link", [0, 0, 0], [0, 0, 0]))
    out.append("</robot>\n")
    return "".join(out)


def write_meshes(root, ur_description: Path) -> None:
    import trimesh

    links = {l.get("name"): l for l in root.findall("link")}
    MESH_OUT.mkdir(parents=True, exist_ok=True)
    for old in MESH_OUT.glob("*.stl"):
        old.unlink()
    for name, stem in LINKS:
        src = ur_description / f"meshes/{UR_TYPE}/collision/{stem}.stl"
        mesh = trimesh.load(src, force="mesh", process=False)
        el = links[name].find("collision")
        xyz, rpy = _origin(el)
        T = np.eye(4)
        T[:3, :3] = _rot([_snap(a) for a in rpy])
        T[:3, 3] = [_snap_len(c) for c in xyz]
        mesh.apply_transform(T)  # bake the placement into the vertices
        mesh.export(MESH_OUT / f"{stem}.stl")
        print(f"  {stem}.stl  {len(mesh.faces)} faces")
    shutil.copyfile(ur_description / "LICENSE", MESH_OUT / "LICENSE")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ur_description", type=Path, help="checkout of Universal_Robots_ROS2_Description")
    args = ap.parse_args()
    ur = args.ur_description.resolve()
    commit = subprocess.run(["git", "-C", str(ur), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    root = expand_xacro(ur)
    URDF_OUT.write_text(build_urdf(root, commit))
    print(f"wrote {URDF_OUT.relative_to(REPO)}  (ur_description@{commit[:10]})")
    write_meshes(root, ur)
    print("float-noise snaps applied (the only values changed from UR's file):")
    for kind, before, after in _snaps:
        print(f"  {kind:6s} {before!r:>26} -> {after!r}")
    print(f"largest change: {_max_snap:.3e} (limit {SNAP_TOL:.0e})")


if __name__ == "__main__":
    main()
