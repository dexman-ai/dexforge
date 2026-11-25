#!/usr/bin/env python3
"""Generate a MuJoCo model (MJCF) of an arm from its driver URDF, with dynamics.

``tools/visualize_ik.py`` builds a kinematics-only model. The simulator needs the
physical one: link masses and inertias, joint limits, gravity, and joints driven
by **direct torque** (no built-in position controller), because the SimDriver
computes torques itself (PD + generated gravity compensation).

The URDF stays the single source of truth; this only translates it:

* links with a ``<inertial>`` become bodies with ``<inertial fullinertia=...>``;
* revolute joints become hinges with the URDF limits, damping and axis;
* one ``<motor>`` per hinge, ``ctrlrange = +-effort`` (the URDF effort limit);
* ``<visual>`` meshes (STL, repo-relative paths) become non-colliding visual geoms;
* the end-effector link gets a ``tcp`` site, used to read the TCP pose back.

Joint ``armature`` defaults to 0 so MuJoCo's mass matrix equals the generated
one (which has no rotor inertia). Pass ``--armature`` to model rotors.

Usage::

    python tools/urdf_to_mjcf.py --arm ur5 --out ur5.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional
from xml.sax.saxutils import quoteattr

import numpy as np
from lxml import etree
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
# arm -> (URDF path, end-effector link)
ARMS = {
    "ur5": (REPO / "repos/dexforge-ur/rust/src/drivers/ur5/ur5.urdf", "ee_link"),
    "ur3e": (REPO / "repos/dexforge-ur/rust/src/drivers/ur3e/ur3e.urdf", "ee_link"),
    "so101": (REPO / "repos/dexforge-lerobot/rust/src/so101/so101.urdf", "gripper_frame_link"),
}
# Same value as dexhal's STANDARD_GRAVITY (the Arm model) and the bindings' default;
# a test checks they agree.
GRAVITY = (0.0, 0.0, -9.80665)


def _vec(s: Optional[str], default: str = "0 0 0") -> np.ndarray:
    return np.array([float(x) for x in (s or default).split()])


def _fmt(v) -> str:
    return " ".join(f"{float(x):.12g}" for x in v)


class _Link:
    def __init__(self, el):
        self.name = el.get("name")
        self.inertial = el.find("inertial")
        vis = el.find("visual/geometry/mesh")
        self.mesh = vis.get("filename") if vis is not None else None
        self.children: List["_Joint"] = []


class _Joint:
    def __init__(self, el):
        self.name = el.get("name")
        self.type = el.get("type")
        self.parent = el.find("parent").get("link")
        self.child = el.find("child").get("link")
        o = el.find("origin")
        self.xyz = _vec(o.get("xyz") if o is not None else None)
        self.rpy = _vec(o.get("rpy") if o is not None else None)
        a = el.find("axis")
        self.axis = _vec(a.get("xyz") if a is not None else None, "1 0 0")
        self.limit = el.find("limit")
        d = el.find("dynamics")
        self.damping = float(d.get("damping", 0.0)) if d is not None else 0.0


def build_mjcf(urdf: Path, ee_link: str, armature: float = 0.0, timestep: float = 0.001) -> str:
    root = etree.parse(str(urdf)).getroot()
    links: Dict[str, _Link] = {l.get("name"): _Link(l) for l in root.findall("link")}
    joints = [_Joint(j) for j in root.findall("joint")]
    children = {j.child for j in joints}
    for j in joints:
        links[j.parent].children.append(j)
    top = [n for n in links if n not in children]
    if len(top) != 1:
        raise SystemExit(f"URDF must have exactly one root link, found {top}")
    # `world` (if any) is just a frame (and MuJoCo reserves the name); start at its child.
    start = top[0]
    if start == "world":
        start = links[start].children[0].child

    meshes: Dict[str, str] = {}
    motors: List[str] = []
    hinge_order: List[str] = []

    def mesh_name(path: str) -> str:
        p = Path(path)
        full = str(p if p.is_absolute() else REPO / p)
        # name by stem, but never let two different files share a name
        stem, k = p.stem, 1
        while meshes.get(stem, full) != full:
            k += 1
            stem = f"{p.stem}_{k}"
        meshes[stem] = full
        return stem

    def body_xml(link: _Link, joint: Optional[_Joint], depth: int) -> str:
        ind = "  " * depth
        attrs = f'name="{link.name}"'
        if joint is not None:
            quat = R.from_euler("xyz", joint.rpy).as_quat()  # x y z w
            attrs += f' pos="{_fmt(joint.xyz)}" quat="{_fmt([quat[3], *quat[:3]])}"'
        out = [f"{ind}<body {attrs}>"]
        if link.inertial is not None:
            m = float(link.inertial.find("mass").get("value"))
            if m > 0:
                o = link.inertial.find("origin")
                if o is not None and np.abs(_vec(o.get("rpy"))).max() > 1e-12:
                    raise SystemExit(
                        f"link {link.name!r}: a rotated <inertial> origin is not supported; "
                        "rotate the tensor into the link frame (tools/import_ur_description.py does)"
                    )
                i = link.inertial.find("inertia")
                full = [float(i.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
                out.append(
                    f'{ind}  <inertial pos="{_fmt(_vec(o.get("xyz") if o is not None else None))}" '
                    f'mass="{m:.12g}" fullinertia="{_fmt(full)}"/>'
                )
        if joint is not None and joint.type in ("revolute", "continuous"):
            attr = f'name="{joint.name}" type="hinge" axis="{_fmt(joint.axis)}"'
            if joint.type == "revolute" and joint.limit is not None:
                attr += f' range="{float(joint.limit.get("lower")):.12g} {float(joint.limit.get("upper")):.12g}"'
            attr += f' damping="{joint.damping:.12g}" armature="{armature:.12g}"'
            out.append(f"{ind}  <joint {attr}/>")
            hinge_order.append(joint.name)
            eff = float(joint.limit.get("effort")) if joint.limit is not None else 0.0
            ctrl = f' ctrllimited="true" ctrlrange="{-eff:.12g} {eff:.12g}"' if eff > 0 else ""
            motors.append(f'    <motor name="{joint.name}" joint="{joint.name}" gear="1"{ctrl}/>')
        if link.mesh:
            out.append(
                f'{ind}  <geom type="mesh" mesh="{mesh_name(link.mesh)}" contype="0" conaffinity="0" '
                f'group="2" rgba="0.7 0.7 0.75 1"/>'
            )
        if link.name == ee_link:
            out.append(f'{ind}  <site name="tcp" size="0.005" rgba="1 0 0 1"/>')
        for j in link.children:
            if j.type in ("fixed", "revolute", "continuous"):
                out.append(body_xml(links[j.child], j, depth + 1))
        out.append(f"{ind}</body>")
        return "\n".join(out)

    body = body_xml(links[start], None, 2)
    asset = "\n".join(f'    <mesh name="{n}" file={quoteattr(p)}/>' for n, p in meshes.items())
    return f"""<mujoco model="{root.get("name")}">
  <compiler angle="radian" inertiafromgeom="false" autolimits="true"/>
  <option timestep="{timestep}" gravity="{_fmt(GRAVITY)}" integrator="implicitfast"/>
  <asset>
{asset}
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.01" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>
{body}
  </worldbody>
  <actuator>
{chr(10).join(motors)}
  </actuator>
</mujoco>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="ur5", choices=sorted(ARMS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--armature", type=float, default=0.0, help="joint armature, kg*m^2 (default 0)")
    args = ap.parse_args()
    urdf, ee = ARMS[args.arm]
    args.out.write_text(build_mjcf(urdf, ee, args.armature))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
