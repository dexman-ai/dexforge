#!/usr/bin/env python3
"""Import the SO-ARM101 (Hugging Face LeRobot / TheRobotStudio) as DexForge URDFs + meshes.

Source of truth: ``Simulation/SO101/so101_new_calib.urdf`` and its STL assets in
https://github.com/TheRobotStudio/SO-ARM100 (Apache-2.0), pinned to commit
``PINNED_COMMIT``. The ``new_calib`` URDF matches LeRobot's newer calibration convention
(joint zero at the middle of each joint's range). Everything physical comes from upstream:
kinematics, link masses, centres of mass, inertia tensors, joint limits and meshes.

Three files are produced:

* ``repos/dexforge-lerobot/rust/src/so101/so101.urdf``: the five-joint arm
  (``shoulder_pan``, ``shoulder_lift``, ``elbow_flex``, ``wrist_flex``, ``wrist_roll``) with
  the gripper jaw folded into ``gripper_link``; the arm the simulator and driver expose.
* ``repos/dexforge-lerobot/rust/src/drivers/so101_6r/so101_6r.urdf``: the same chain plus one
  zero-offset *virtual* revolute joint ``virtual_wrist`` (axis x of ``gripper_link``, i.e.
  perpendicular to the wrist roll) before ``gripper_frame_link``. At q6 = 0 it is exactly the
  five-joint arm. The ssik / dexforge pipeline only handles full 6R and 7R chains; this
  embedding is what lets it generate the arm's closed-form IK (``ikgeo.three_parallel``).
* ``repos/dexforge-lerobot/rust/src/so101/codegen/so101_new_calib.urdf``: the
  unmodified upstream URDF, kept as the independent reference for the Pinocchio tests
  (which therefore include the effect of every change below).
* ``tools/meshes/so101/<link>.stl``: one decimated STL per link for the simulator.

Deliberate changes from upstream's file (each one measured by the tests, not silent):

* **Origin noise.** Upstream rounds rotations to 5 digits (``1.5708`` for pi/2, ``3.14159``
  for pi) and carries offsets of ~1e-9..1e-17 m. Joint-origin rpy angles within ``RPY_SNAP`` of
  a multiple of pi/2 are set to that multiple, and joint-origin lengths below ``XYZ_SNAP`` to 0.
  The ssik artifact classifies axes as parallel / intersecting with tolerances of 1e-8, and
  the symbolic dynamics compile (SymPy) is orders of magnitude slower on the noisy values.
  Effect on FK: at most a few micrometres and ~1e-5 in rotation-matrix entries; a test compares
  against Pinocchio on the *unmodified* URDF. The real 2.8 deg tilt of ``wrist_roll``
  (rpy y = 0.0486795) is kept.
* **Gripper jaw.** ``moving_jaw`` hangs off ``gripper_link`` as a side branch and would be
  ignored by the base-to-flange dynamics. Its mass and inertia (0.012 kg) are composed into
  ``gripper_link`` with the jaw at its q = 0 pose, and the jaw joint/link are removed.
* **Inertia frames.** Every ``<inertial>`` has ``rpy = 0`` (what the MJCF generator and the
  dynamics codegen expect); upstream's already are, and the composed gripper inertia is
  expressed in the link axes.
* **Frame link.** ``gripper_frame_link`` (the end-effector frame) is a pure frame: upstream gives
  it a mass of 1e-9 kg and an all-zero inertia tensor (not a physical inertia; MuJoCo rejects
  it) and a stray ``<origin>`` inside the ``<link>``. Both are removed.
* **Meshes.** All of a link's visual STLs (and the jaw's, for ``gripper_link``) are placed
  by their visual origins, merged into one STL per link and decimated by vertex clustering
  (``MESH_CELL``). The URDF mesh elements therefore have no origin. Collision elements are
  dropped (the simulator does not collide).

Not provided by upstream: joint acceleration limits, a motor / actuator model, joint damping
or friction. The URDF's ``effort`` and ``velocity`` limits (10 and 10) are placeholders from
the CAD export, not Feetech STS3215 ratings; this importer keeps them as they are.

Usage::

    git clone --depth 1 --filter=blob:none --sparse https://github.com/TheRobotStudio/SO-ARM100 /tmp/SO-ARM100
    git -C /tmp/SO-ARM100 sparse-checkout set Simulation/SO101
    python tools/import_so101_description.py /tmp/SO-ARM100
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
PINNED_COMMIT = "141aea7bcac2744d92998e30268c413202a56f96"
UPSTREAM = "https://github.com/TheRobotStudio/SO-ARM100"
SRC_URDF = "Simulation/SO101/so101_new_calib.urdf"

CRATE_SRC = REPO / "repos/dexforge-lerobot/rust/src"
DRIVERS = CRATE_SRC / "drivers"
OUT_5R = CRATE_SRC / "so101/so101.urdf"
OUT_6R = DRIVERS / "so101_6r/so101_6r.urdf"
OUT_REF = CRATE_SRC / "so101/codegen/so101_new_calib.urdf"
MESH_OUT = REPO / "tools/meshes/so101"
MESH_REF = "tools/meshes/so101"  # URDF mesh paths are relative to the repo root

RPY_SNAP = 1e-4  # rad: upstream rounds angles to 5 digits (pi/2 -> 1.5708 is off by 3.7e-6)
XYZ_SNAP = 1e-7  # m
MESH_CELL = 0.0012  # m: vertex-clustering cell for the simulator meshes
HALF_PI = math.pi / 2

JAW_JOINT = "gripper"  # upstream's name for the jaw joint
FLANGE_LINK = "gripper_link"
EE_LINK = "gripper_frame_link"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def vec(s: str | None, default: str = "0 0 0") -> np.ndarray:
    return np.array([float(x) for x in (s or default).split()])


def fmt(v) -> str:
    return " ".join(repr(float(x)) for x in v)


def rot(rpy) -> np.ndarray:
    return R.from_euler("xyz", rpy).as_matrix()  # URDF fixed-axis rpy == scipy extrinsic xyz


def transform(xyz, rpy) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot(rpy)
    t[:3, 3] = xyz
    return t


def inertial_of(link: ET.Element):
    """(mass, com xyz, 3x3 inertia about the COM, in the *link* axes)."""
    inr = link.find("inertial")
    o = inr.find("origin")
    rpy = vec(o.get("rpy") if o is not None else None)
    xyz = vec(o.get("xyz") if o is not None else None)
    a = inr.find("inertia")
    i = np.array(
        [
            [float(a.get("ixx")), float(a.get("ixy")), float(a.get("ixz"))],
            [float(a.get("ixy")), float(a.get("iyy")), float(a.get("iyz"))],
            [float(a.get("ixz")), float(a.get("iyz")), float(a.get("izz"))],
        ]
    )
    rm = rot(rpy)
    return float(inr.find("mass").get("value")), xyz, rm @ i @ rm.T


def snap_joint_origins(joints: dict[str, ET.Element]) -> int:
    n = 0
    for j in joints.values():
        o = j.find("origin")
        if o is None:
            continue
        rpy, xyz = vec(o.get("rpy")), vec(o.get("xyz"))
        k = np.round(rpy / HALF_PI)
        snapped = np.where(np.abs(rpy - k * HALF_PI) < RPY_SNAP, k * HALF_PI, rpy)
        nxyz = np.where(np.abs(xyz) < XYZ_SNAP, 0.0, xyz)
        change = max(np.abs(snapped - rpy).max(), np.abs(nxyz - xyz).max())
        if change > 0:
            n += 1
            print(f"  snap {j.get('name'):<20s} max change {change:.2e}")
        o.set("rpy", fmt(snapped))
        o.set("xyz", fmt(nxyz))
    return n


def fold_jaw(links: dict[str, ET.Element], joints: dict[str, ET.Element]) -> np.ndarray:
    """Compose the jaw's inertia into gripper_link; returns T(gripper_link <- jaw link)."""
    jj = joints[JAW_JOINT]
    o = jj.find("origin")
    t_gj = transform(vec(o.get("xyz")), vec(o.get("rpy")))
    flange, jaw = links[FLANGE_LINK], links[jj.find("child").get("link")]
    m_g, c_g, i_g = inertial_of(flange)
    m_j, c_j_local, i_j_local = inertial_of(jaw)
    r_gj = t_gj[:3, :3]
    c_j = t_gj[:3, 3] + r_gj @ c_j_local
    i_j = r_gj @ i_j_local @ r_gj.T
    m = m_g + m_j
    c = (m_g * c_g + m_j * c_j) / m

    def shift(i, mass, d):
        return i + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    i_tot = shift(i_g, m_g, c_g - c) + shift(i_j, m_j, c_j - c)
    inr = flange.find("inertial")
    inr.find("origin").set("xyz", fmt(c))
    inr.find("origin").set("rpy", "0 0 0")
    inr.find("mass").set("value", repr(float(m)))
    a = inr.find("inertia")
    for key, (r, cc) in dict(ixx=(0, 0), ixy=(0, 1), ixz=(0, 2), iyy=(1, 1), iyz=(1, 2), izz=(2, 2)).items():
        a.set(key, repr(float(i_tot[r, cc])))
    print(f"  folded jaw into {FLANGE_LINK}: {m_g:.6f} + {m_j:.6f} = {m:.6f} kg")
    return t_gj


def cluster_decimate(mesh, cell: float):
    """Vertex-clustering decimation: snap vertices to a grid, merge, drop degenerate faces."""
    import trimesh

    keys = np.round(mesh.vertices / cell).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    sums = np.zeros((len(uniq), 3))
    np.add.at(sums, inv, mesh.vertices)
    counts = np.bincount(inv, minlength=len(uniq))[:, None]
    verts = sums / counts
    faces = inv[mesh.faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[keep]
    out =trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    out.remove_unreferenced_vertices()
    return out


def build_link_meshes(src: Path, links: dict[str, ET.Element], jaw_link: str, t_gj: np.ndarray) -> dict[str, str]:
    """Merge each link's visuals into one STL (placement baked in). Returns link -> repo-relative path."""
    import trimesh

    if MESH_OUT.exists():
        for old in MESH_OUT.glob("*.stl"):
            old.unlink()
    MESH_OUT.mkdir(parents=True, exist_ok=True)
    refs: dict[str, str] = {}
    for name, link in links.items():
        if name == jaw_link:
            continue
        sources = [(v, np.eye(4)) for v in link.findall("visual")]
        if name == FLANGE_LINK:
            sources += [(v, t_gj) for v in links[jaw_link].findall("visual")]
        if not sources:
            continue
        parts = []
        for v, base in sources:
            o = v.find("origin")
            t = base @ transform(vec(o.get("xyz") if o is not None else None), vec(o.get("rpy") if o is not None else None))
            m = trimesh.load(src / "Simulation/SO101" / v.find("geometry/mesh").get("filename"), force="mesh", process=False)
            m.apply_transform(t)
            parts.append(m)
        merged = trimesh.util.concatenate(parts)
        faces_in = len(merged.faces)
        merged = cluster_decimate(merged, MESH_CELL)
        merged.export(MESH_OUT / f"{name}.stl")
        refs[name] = f"{MESH_REF}/{name}.stl"
        print(f"  {name}.stl  {faces_in} -> {len(merged.faces)} faces")
    return refs


def strip_visuals_and_collisions(links: dict[str, ET.Element]) -> None:
    for link in links.values():
        for tag in ("visual", "collision"):
            for e in link.findall(tag):
                link.remove(e)


def attach_meshes(links: dict[str, ET.Element], refs: dict[str, str]) -> None:
    for name, ref in refs.items():
        v = ET.SubElement(links[name], "visual")
        ET.SubElement(v, "origin", xyz="0 0 0", rpy="0 0 0")
        g = ET.SubElement(v, "geometry")
        ET.SubElement(g, "mesh", filename=ref)


def add_virtual_wrist(root: ET.Element, joints: dict[str, ET.Element], links: dict[str, ET.Element]) -> None:
    link = ET.Element("link", name="virtual_wrist_link")
    inr = ET.SubElement(link, "inertial")
    ET.SubElement(inr, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inr, "mass", value="1e-9")
    ET.SubElement(inr, "inertia", ixx="1e-12", ixy="0", ixz="0", iyy="1e-12", iyz="0", izz="1e-12")
    root.insert(list(root).index(links[EE_LINK]), link)
    vj = ET.Element("joint", name="virtual_wrist", type="revolute")
    ET.SubElement(vj, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(vj, "parent", link=FLANGE_LINK)
    ET.SubElement(vj, "child", link="virtual_wrist_link")
    ET.SubElement(vj, "axis", xyz="1 0 0")
    ET.SubElement(vj, "limit", effort="10", velocity="10", lower=repr(-math.pi), upper=repr(math.pi))
    ee_joint = joints["gripper_frame_joint"]
    root.insert(list(root).index(ee_joint), vj)
    ee_joint.find("parent").set("link", "virtual_wrist_link")


def header(kind: str, commit: str) -> str:
    return (
        f"Derived from {UPSTREAM} ({SRC_URDF}) at commit {commit}, Apache-2.0.\n"
        f"{kind}\n"
        "Generated by tools/import_so101_description.py; do not edit by hand."
    )


def write(tree: ET.ElementTree, path: Path, comment: str) -> None:
    root = tree.getroot()
    root.insert(0, ET.Comment("\n" + comment + "\n"))
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, xml_declaration=True, encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="checkout of TheRobotStudio/SO-ARM100 (Simulation/SO101 present)")
    args = ap.parse_args()
    src = args.source.resolve()
    urdf = src / SRC_URDF
    if not urdf.is_file():
        print(f"missing {urdf}", file=sys.stderr)
        return 2
    commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if commit != PINNED_COMMIT:
        print(f"WARNING: checkout is at {commit or 'unknown'}, pinned commit is {PINNED_COMMIT}", file=sys.stderr)

    OUT_REF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(urdf, OUT_REF)
    print(f"wrote {OUT_REF.relative_to(REPO)} (unmodified upstream, for the Pinocchio reference tests)")

    tree = ET.parse(urdf)
    root = tree.getroot()
    root.set("name", "so101")
    links = {l.get("name"): l for l in root.findall("link")}
    joints = {j.get("name"): j for j in root.findall("joint")}
    actuated = [n for n, j in joints.items() if j.get("type") == "revolute" and n != JAW_JOINT]
    assert set(actuated) == set(ARM_JOINTS), f"unexpected actuated joints {actuated}"

    print("snapping joint origins:")
    print(f"  {snap_joint_origins(joints)} origins changed")
    t_gj = fold_jaw(links, joints)
    jaw_link = joints[JAW_JOINT].find("child").get("link")

    print("meshes:")
    refs = build_link_meshes(src, links, jaw_link, t_gj)

    # remove the jaw link / joint (and the upstream <transmission> / <ros2_control> extras)
    root.remove(links.pop(jaw_link))
    root.remove(joints.pop(JAW_JOINT))
    for tag in ("transmission", "gazebo", "ros2_control"):
        for e in root.findall(tag):
            root.remove(e)
    for e in root.findall("material"):
        root.remove(e)

    strip_visuals_and_collisions(links)
    for stray in list(links[EE_LINK]):  # a pure frame: no inertial, no stray <origin>
        links[EE_LINK].remove(stray)
    attach_meshes(links, refs)
    note5 = (
        "Five-joint SO-ARM101: snapped joint origins, gripper jaw folded into gripper_link,\n"
        "meshes merged per link (tools/meshes/so101). Effort / velocity limits are upstream's CAD placeholders."
    )
    write(ET.ElementTree(root), OUT_5R, header(note5, commit))

    # 6R embedding: same chain without meshes, plus the virtual wrist joint
    tree6 = ET.parse(OUT_5R)
    root6 = tree6.getroot()
    for c in list(root6):
        if c.tag is ET.Comment:
            root6.remove(c)
    links6 = {l.get("name"): l for l in root6.findall("link")}
    joints6 = {j.get("name"): j for j in root6.findall("joint")}
    strip_visuals_and_collisions(links6)
    add_virtual_wrist(root6, joints6, links6)
    root6.set("name", "so101_6r")
    note6 = (
        "SO-ARM101 embedded as a 6R chain for the ssik / dexforge pipeline: the five-joint arm plus a\n"
        "zero-offset virtual revolute joint (virtual_wrist, axis x of gripper_link) before gripper_frame_link.\n"
        "At q6 = 0 this is exactly the five-joint arm."
    )
    write(ET.ElementTree(root6), OUT_6R, header(note6, commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
