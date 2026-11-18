#!/usr/bin/env python3
"""Visualize DexForge FK/IK for an arm in MuJoCo (no meshes, capsule geometry).

The arm is built directly from the driver's URDF kinematic chain, drawn in the
*solver's* base frame, and driven through the compiled ``dexforge_<oem>``
Python bindings.

Interactive (macOS needs ``mjpython``)::

    mjpython tools/visualize_ik.py --arm ur5

    keys:  N / Right  next IK solution      P / Left  previous solution
           R          new random reachable target
           double-click the target frame, then Ctrl+right-drag / Ctrl+left-drag
           to move/rotate it; IK is re-solved live.

Headless (PNG of every IK solution for one random target)::

    python tools/visualize_ik.py --arm ur5 --render out.png --seed 0

Checks (printed): MuJoCo FK vs. dexforge FK, and FK error of each IK solution.
"""

from __future__ import annotations

import argparse
import importlib
import re
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "codegen"))

import generate_dynamics as gd  # noqa: E402

try:
    import mujoco
except ImportError:  # pragma: no cover
    sys.exit("mujoco is required: pip install mujoco")

_CAPSULE_RADIUS = 0.022
_JOINT_RADIUS = 0.032
_AXIS_LEN = 0.08
_FK_TOL = 1e-4
_MESH_ROOT = _REPO_ROOT / "tools" / "meshes"

# arm -> (OEM repo directory, Python package)
_ARMS = {
    "ur5": ("dexforge-ur", "dexforge_ur"),
    "z1": ("dexforge-unitree", "dexforge_unitree"),
    "piper": ("dexforge-agilex", "dexforge_agilex"),
    "franka_panda": ("dexforge-franka", "dexforge_franka"),
    "xarm7": ("dexforge-ufactory", "dexforge_ufactory"),
    "rizon4": ("dexforge-flexiv", "dexforge_flexiv"),
    "rizon10": ("dexforge-flexiv", "dexforge_flexiv"),
}


@dataclass
class Arm:
    name: str
    chain: List["gd.UrdfJoint"]
    limits: List[Tuple[float, float]]
    fk: object
    solve: object
    base_link: str = ""
    # link name -> (stl path, pos, rpy) for links that have a local mesh
    meshes: dict = field(default_factory=dict)

    @property
    def dof(self) -> int:
        return len(self.limits)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_endpoints(driver_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    src = (driver_dir / "ik_solver.rs").read_text(encoding="utf-8")
    base = re.search(r'BASE_LINK:\s*&str\s*=\s*"([^"]+)"', src)
    ee = re.search(r'EE_LINK:\s*&str\s*=\s*"([^"]+)"', src)
    return (base.group(1) if base else None, ee.group(1) if ee else None)


def _joint_limits(urdf: Path) -> dict:
    import xml.etree.ElementTree as ET

    limits = {}
    for j in ET.parse(urdf).getroot().findall("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("lower") is not None:
            limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return limits


def _link_meshes(urdf: Path, name: str) -> dict:
    """Map URDF links to local ``tools/meshes/<arm>/<stem>.stl`` files, if present."""
    import xml.etree.ElementTree as ET

    out = {}
    for link in ET.parse(urdf).getroot().findall("link"):
        for kind in ("visual", "collision"):
            node = link.find(kind)
            mesh = node.find("geometry/mesh") if node is not None else None
            if mesh is None:
                continue
            stl = _MESH_ROOT / name / (Path(mesh.get("filename")).stem + ".stl")
            if not stl.is_file() or mesh.get("scale") not in (None, "1 1 1"):
                continue
            org = node.find("origin")
            xyz = gd._parse_triple(org.get("xyz") if org is not None else None, (0, 0, 0))
            rpy = gd._parse_triple(org.get("rpy") if org is not None else None, (0, 0, 0))
            out[link.get("name")] = (stl, xyz, rpy)
            break
    return out


def load_arm(name: str) -> Arm:
    if name not in _ARMS:
        raise SystemExit(f"unknown arm {name!r}; known: {', '.join(_ARMS)}")
    oem_dir, oem_pkg = _ARMS[name]
    driver_dir = _REPO_ROOT / "repos" / oem_dir / "rust" / "src" / "drivers" / name
    urdf = driver_dir / f"{name}.urdf"
    base, ee = _read_endpoints(driver_dir)
    chain = gd.order_urdf_chain(gd.parse_urdf_joints(urdf), base, ee)
    declared = _joint_limits(urdf)
    limits = [declared.get(jt.name, (-np.pi, np.pi)) for jt in chain if jt.actuated]
    try:
        mod = getattr(importlib.import_module(oem_pkg), name)
    except ImportError:
        raise SystemExit(
            f"cannot import {oem_pkg}.{name}; build it first: "
            f"cd repos/{oem_dir}/python && maturin develop --release"
        )
    return Arm(name, chain, limits, mod.fk, mod.solve, chain[0].parent, _link_meshes(urdf, name))


# --------------------------------------------------------------------------- #
# MuJoCo model
# --------------------------------------------------------------------------- #
def _rpy_to_rot(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp_, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp_], [0, 1, 0], [-sp_, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def urdf_fk(arm: Arm, q: np.ndarray) -> np.ndarray:
    """Forward kinematics of the URDF chain in the URDF base frame."""
    t = np.eye(4)
    qi = 0
    for jt in arm.chain:
        o = np.eye(4)
        o[:3, :3] = _rpy_to_rot(jt.origin_rpy)
        o[:3, 3] = jt.origin_xyz
        t = t @ o
        if jt.actuated:
            ax = np.asarray(gd._normalize_axis(jt.axis, jt.name))
            m = np.eye(4)
            if jt.jtype == "prismatic":
                m[:3, 3] = ax * q[qi]
            else:
                k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                m[:3, :3] = np.eye(3) + np.sin(q[qi]) * k + (1 - np.cos(q[qi])) * (k @ k)
            t = t @ m
            qi += 1
    return t


def _rotvec_axis(r: np.ndarray) -> np.ndarray:
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    n = np.linalg.norm(w)
    return w / n if n > 1e-9 else np.zeros(3)


def solver_frame_offsets(arm: Arm, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Fit constant ``M`` (base) and ``N`` (tool) with ``fk_solver = M @ fk_urdf @ N``.

    Prebuilt solvers can use a base/tool frame that differs from the raw URDF
    (e.g. UR5's base is rotated 180 degrees about Z). Exits if no constant
    ``M``/``N`` explains the solver, i.e. the URDF and driver are inconsistent.
    """
    qs = [sample_q(arm, rng) for _ in range(24)]
    fs = [np.asarray(arm.fk(q.tolist())) for q in qs]
    fu = [urdf_fk(arm, q) for q in qs]
    src, dst = [], []
    for i in range(1, len(qs)):
        a = _rotvec_axis(fu[i][:3, :3] @ fu[0][:3, :3].T)
        b = _rotvec_axis(fs[i][:3, :3] @ fs[0][:3, :3].T)
        if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
            src.append(a)
            dst.append(b)
    u, _, vt = np.linalg.svd(np.array(dst).T @ np.array(src))
    r_m = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    r_n = fu[0][:3, :3].T @ r_m.T @ fs[0][:3, :3]
    rows, rhs = [], []
    for f_s, f_u in zip(fs, fu):
        rows.append(np.hstack([r_m @ f_u[:3, :3], np.eye(3)]))
        rhs.append(f_s[:3, 3] - r_m @ f_u[:3, 3])
    sol, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r_m, sol[3:]
    n = np.eye(4)
    n[:3, :3], n[:3, 3] = r_n, sol[:3]
    err = max(np.abs(m @ f_u @ n - f_s).max() for f_s, f_u in zip(fs, fu))
    if err > 1e-6:
        raise SystemExit(
            f"solver FK is not a constant base/tool-frame transform of the URDF chain "
            f"(max residual {err:.2e}); the URDF and driver are inconsistent."
        )
    return m, n


def _rot_to_quat(r: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(r, dtype=np.float64).reshape(9))
    return quat


def _fmt(v) -> str:
    return " ".join(f"{x:.9g}" for x in v)


def _triad(length: float = _AXIS_LEN, group: int = 1) -> str:
    r = 0.004
    common = f'size="{r}" contype="0" conaffinity="0" group="{group}"'
    return (
        f'<geom type="capsule" fromto="0 0 0 {length} 0 0" rgba="1 0 0 1" {common}/>'
        f'<geom type="capsule" fromto="0 0 0 0 {length} 0" rgba="0 0.8 0 1" {common}/>'
        f'<geom type="capsule" fromto="0 0 0 0 0 {length}" rgba="0.1 0.1 1 1" {common}/>'
    )


def build_mjcf(arm: Arm, offset: np.ndarray, tool: np.ndarray, use_meshes: bool = True) -> str:
    """Nested-body MJCF for the arm, rooted at the solver's base frame."""
    palette = ["0.75 0.75 0.78 1", "0.35 0.55 0.85 1"]
    bodies_open: List[str] = []
    mesh_assets: List[str] = []
    meshes = arm.meshes if use_meshes else {}

    def mesh_geom(link: str, color: str) -> str:
        stl, xyz, rpy = meshes[link]
        mesh_assets.append(f'<mesh name="m_{link}" file="{stl}"/>')
        return (
            f'<geom type="mesh" mesh="m_{link}" pos="{_fmt(xyz)}" '
            f'quat="{_fmt(_rot_to_quat(_rpy_to_rot(rpy)))}" rgba="{color}" '
            f'contype="0" conaffinity="0"/>'
        )

    qi = 0
    for idx, jt in enumerate(arm.chain):
        pos = _fmt(jt.origin_xyz)
        quat = _fmt(_rot_to_quat(_rpy_to_rot(jt.origin_rpy)))
        nxt = arm.chain[idx + 1].origin_xyz if idx + 1 < len(arm.chain) else (0, 0, 0)
        seg = np.asarray(nxt, dtype=float)
        link_geom = ""
        if jt.child in meshes:
            link_geom = mesh_geom(jt.child, palette[idx % 2])
        elif np.linalg.norm(seg) > 1e-6 and not meshes:
            link_geom = (
                f'<geom type="capsule" fromto="0 0 0 {_fmt(seg)}" size="{_CAPSULE_RADIUS}" '
                f'rgba="{palette[idx % 2]}" contype="0" conaffinity="0"/>'
            )
        joint_xml = ""
        if jt.actuated:
            lo, hi = arm.limits[qi]
            ax = _fmt(gd._normalize_axis(jt.axis, jt.name))
            jtype = "slide" if jt.jtype == "prismatic" else "hinge"
            joint_xml = (
                f'<joint name="q{qi}" type="{jtype}" axis="{ax}" '
                f'range="{lo:.9g} {hi:.9g}" limited="true"/>'
            )
            if not meshes:
                joint_xml += (
                    f'<geom type="sphere" size="{_JOINT_RADIUS}" rgba="0.9 0.5 0.1 1" '
                    f'contype="0" conaffinity="0"/>'
                )
            qi += 1
        bodies_open.append(f'<body name="b{idx}" pos="{pos}" quat="{quat}">{joint_xml}{link_geom}')
    tool_body = (
        f'<body name="tool" pos="{_fmt(tool[:3, 3])}" quat="{_fmt(_rot_to_quat(tool[:3, :3]))}">'
        f'<site name="ee" size="0.01"/>{_triad()}</body>'
    )
    tree = "".join(bodies_open) + tool_body + "</body>" * len(bodies_open)
    base_mesh = mesh_geom(arm.base_link, "0.45 0.45 0.5 1") if arm.base_link in meshes else ""
    base_pad = "" if meshes else (
        '<geom type="cylinder" size="0.07 0.004" pos="0 0 0.002" rgba="0.3 0.3 0.3 1" '
        'contype="0" conaffinity="0"/>'
    )
    asset_xml = "".join(mesh_assets)
    base_pos = _fmt(offset[:3, 3])
    base_quat = _fmt(_rot_to_quat(offset[:3, :3]))
    return f"""
<mujoco model="{arm.name}">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    {asset_xml}
    <texture type="skybox" builtin="gradient" rgb1="0.9 0.93 0.97" rgb2="0.6 0.65 0.72" width="64" height="64"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.84 0.86" rgb2="0.74 0.76 0.8" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="6 6"/>
  </asset>
  <worldbody>
    <light pos="1 -1 3" dir="-0.3 0.3 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="grid" pos="0 0 -0.001" contype="0" conaffinity="0"/>
    {base_pad}
    <body name="base" pos="{base_pos}" quat="{base_quat}">
      {base_mesh}
      {_triad(0.12)}
      {tree}
    </body>
    <body name="target" mocap="true" pos="0 0 0.5">
      {_triad(0.12, group=2)}
      <geom type="sphere" size="0.018" rgba="1 0.2 0.8 0.9" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


# --------------------------------------------------------------------------- #
# IK helpers
# --------------------------------------------------------------------------- #
def sample_q(arm: Arm, rng: np.random.Generator) -> np.ndarray:
    lo = np.array([l for l, _ in arm.limits])
    hi = np.array([h for _, h in arm.limits])
    return rng.uniform(lo, hi)


def _pose_to_tf(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat / np.linalg.norm(quat))
    t[:3, :3] = mat.reshape(3, 3)
    t[:3, 3] = pos
    return t


def _set_target(model, data, tf: np.ndarray) -> None:
    mid = model.body("target").mocapid[0]
    data.mocap_pos[mid] = tf[:3, 3]
    data.mocap_quat[mid] = _rot_to_quat(tf[:3, :3])


def _get_target(model, data) -> np.ndarray:
    mid = model.body("target").mocapid[0]
    return _pose_to_tf(data.mocap_pos[mid].copy(), data.mocap_quat[mid].copy())


def _solve(arm: Arm, tf: np.ndarray) -> List[np.ndarray]:
    return [np.asarray(s) for s in arm.solve(tf.tolist())]


def _ee_tf(model, data) -> np.ndarray:
    sid = model.site("ee").id
    t = np.eye(4)
    t[:3, :3] = data.site_xmat[sid].reshape(3, 3)
    t[:3, 3] = data.site_xpos[sid]
    return t


def _pose_error(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    pos = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    c = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0
    return pos, float(np.arccos(np.clip(c, -1.0, 1.0)))


def apply_q(model, data, q: np.ndarray) -> np.ndarray:
    data.qpos[: len(q)] = q
    mujoco.mj_forward(model, data)
    return _ee_tf(model, data)


def check_fk(arm: Arm, model, data, rng: np.random.Generator, n: int = 20) -> float:
    worst = 0.0
    for _ in range(n):
        q = sample_q(arm, rng)
        pos, ang = _pose_error(apply_q(model, data, q), np.asarray(arm.fk(q.tolist())))
        worst = max(worst, pos, ang)
    return worst


# --------------------------------------------------------------------------- #
# Headless rendering
# --------------------------------------------------------------------------- #
def _write_png(path: Path, img: np.ndarray) -> None:
    h, _w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(tag + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", crc)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", img.shape[1], h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def _camera() -> "mujoco.MjvCamera":
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.3]
    cam.distance = 1.7
    cam.azimuth = 135
    cam.elevation = -22
    return cam


def render_solutions(
    arm: Arm, model, data, target: np.ndarray, sols: List[np.ndarray], path: Path, max_tiles: int = 8
) -> None:
    _set_target(model, data, target)
    renderer = mujoco.Renderer(model, height=360, width=480)
    opt = mujoco.MjvOption()
    tiles = []
    for q in sols[:max_tiles] or [np.zeros(arm.dof)]:
        apply_q(model, data, q)
        renderer.update_scene(data, camera=_camera(), scene_option=opt)
        tiles.append(renderer.render().copy())
    renderer.close()
    cols = min(4, len(tiles))
    rows = -(-len(tiles) // cols)
    tiles += [np.full_like(tiles[0], 255)] * (rows * cols - len(tiles))
    grid = np.vstack([np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)])
    _write_png(path, grid)


# --------------------------------------------------------------------------- #
# Interactive viewer
# --------------------------------------------------------------------------- #
def run_viewer(arm: Arm, model, data, rng: np.random.Generator, target: np.ndarray) -> None:
    import mujoco.viewer as mjv

    state = {"sols": [], "idx": 0, "tf": None, "dirty": True, "random": False}

    def key_cb(keycode: int) -> None:
        ch = chr(keycode).lower() if 0 < keycode < 256 else ""
        if ch == "n" or keycode == 262:  # right arrow
            state["idx"] += 1
            state["dirty"] = True
        elif ch == "p" or keycode == 263:  # left arrow
            state["idx"] -= 1
            state["dirty"] = True
        elif ch == "r":
            state["random"] = True

    with mjv.launch_passive(model, data, key_callback=key_cb) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.3]
        viewer.cam.distance = 1.7
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -22
        _set_target(model, data, target)
        while viewer.is_running():
            if state["random"]:
                state["random"] = False
                _set_target(model, data, np.asarray(arm.fk(sample_q(arm, rng).tolist())))
            tf = _get_target(model, data)
            if state["tf"] is None or np.abs(tf - state["tf"]).max() > 1e-9:
                state["tf"] = tf
                state["sols"] = _solve(arm, tf)
                state["idx"] = 0
                state["dirty"] = True
                if not state["sols"]:
                    print("IK: no solutions (target unreachable or outside limits)")
            if state["dirty"] and state["sols"]:
                state["idx"] %= len(state["sols"])
                q = state["sols"][state["idx"]]
                pos, ang = _pose_error(apply_q(model, data, q), tf)
                print(
                    f"solution {state['idx'] + 1}/{len(state['sols'])}  "
                    f"pos err {pos:.2e} m  rot err {ang:.2e} rad  "
                    f"q = {np.array2string(q, precision=3, suppress_small=True)}"
                )
                state["dirty"] = False
            viewer.sync()
            time.sleep(1 / 60)


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--arm", default="ur5", help="arm name, e.g. ur5 (default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capsules", action="store_true", help="draw capsules even if meshes exist")
    ap.add_argument("--render", type=Path, help="write a PNG of the IK solutions and exit")
    args = ap.parse_args(argv)

    arm = load_arm(args.arm)
    rng = np.random.default_rng(args.seed)
    offset, tool = solver_frame_offsets(arm, rng)
    model = mujoco.MjModel.from_xml_string(build_mjcf(arm, offset, tool, not args.capsules))
    data = mujoco.MjData(model)

    worst = check_fk(arm, model, data, rng)
    status = "OK" if worst < _FK_TOL else "MISMATCH"
    print(f"[{arm.name}] {arm.dof} DOF; MuJoCo vs dexforge FK worst error {worst:.2e} ({status})")
    if worst >= _FK_TOL:
        return 1

    target = np.asarray(arm.fk(sample_q(arm, rng).tolist()))
    sols = _solve(arm, target)
    print(f"[{arm.name}] {len(sols)} IK solution(s) for the random target")
    for i, q in enumerate(sols):
        pos, ang = _pose_error(apply_q(model, data, q), target)
        print(f"  #{i + 1}: pos err {pos:.2e} m, rot err {ang:.2e} rad")

    if args.render:
        render_solutions(arm, model, data, target, sols, args.render)
        print(f"wrote {args.render}")
        return 0
    run_viewer(arm, model, data, rng, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
