#!/usr/bin/env python3
"""Build and inspect ``ssik`` Python IK artifacts (``<arm>_ik.py``).

Deterministic Step 1 of the agent workflow in ``codegen/README.md``:

* ``build``   — verify the installed ``ssik`` matches ``pyproject.toml`` and
  run ``ssik build`` on a URDF, writing ``drivers/<arm>/codegen/<arm>_ik.py``.
* ``inspect`` — load an artifact and report ``DOF``, ``MAX_SOLUTIONS``, the
  ``SOLVER_NAME``, the kinematic solver family and the in-repo reference arm
  whose ``ik_solver.rs`` / ``codegen/`` scripts the agent should mirror.

Usage::

    python codegen/ssik_artifact.py build --urdf my_arm.urdf \\
        --base base_link --ee tool0 \\
        --out repos/dexforge-<oem>/rust/src/drivers/<arm>/codegen/<arm>_ik.py
    python codegen/ssik_artifact.py inspect \\
        repos/dexforge-<oem>/rust/src/drivers/<arm>/codegen/<arm>_ik.py --json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_CODEGEN_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CODEGEN_DIR.parent
if str(_CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_DIR))

from ik_test_data import get_installed_ssik_version  # noqa: E402


# ---------------------------------------------------------------------------
# Kinematic solver families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    """A kinematic solver family and the in-repo driver to use as its template."""

    key: str
    category: int
    title: str
    reference_arm: str
    reference_repo: str
    max_solutions: int
    # The reference bakes sub-chain / polynomial constants via codegen/generate_code.py.
    generate_code: bool
    # Baked files the reference's generate_code.py writes next to ik_solver.rs.
    baked_files: Tuple[str, ...] = ()
    alternates: Tuple[str, ...] = ()
    note: str = ""


FAMILIES = {
    "sp6": Family(
        key="sp6",
        category=1,
        title="Pieper / spherical-wrist 6-DOF (SP6 closed form)",
        reference_arm="ur5",
        reference_repo="repos/dexforge-ur",
        max_solutions=16,
        generate_code=False,
        alternates=("z1",),
    ),
    "general_6r": Family(
        key="general_6r",
        category=2,
        title="General 6R (Raghavan-Roth)",
        reference_arm="piper",
        reference_repo="repos/dexforge-agilex",
        max_solutions=16,
        generate_code=True,
        baked_files=("pq.rs",),
    ),
    "jointlock_sp6": Family(
        key="jointlock_sp6",
        category=3,
        title="Canonical SRS / joint-lock 7-DOF (SP6 inner)",
        reference_arm="franka_panda",
        reference_repo="repos/dexforge-franka",
        max_solutions=128,
        generate_code=True,
        baked_files=("consts.rs",),
        alternates=("xarm7",),
    ),
    "jointlock_general_6r": Family(
        key="jointlock_general_6r",
        category=4,
        title="Non-SRS joint-lock 7-DOF (general-6R / Raghavan-Roth polynomial inner)",
        reference_arm="rizon4",
        reference_repo="repos/dexforge-flexiv",
        max_solutions=256,
        generate_code=True,
        baked_files=("consts.rs", "pq.rs"),
        alternates=("rizon10",),
    ),
    "srs": Family(
        key="srs",
        category=3,
        title="Closed-form SRS 7-DOF (ssik seven_r.srs)",
        reference_arm="franka_panda",
        reference_repo="repos/dexforge-franka",
        max_solutions=128,
        generate_code=True,
        baked_files=("consts.rs",),
        note=(
            "No in-repo driver uses the seven_r.srs solver yet; franka_panda is "
            "the closest 7-DOF reference. Confirm MAX_SOLUTIONS against the "
            "artifact's solver before relying on it."
        ),
    ),
}


def detect_family(solver_name: str, source: str) -> Family:
    """Map an artifact's ``SOLVER_NAME`` (and inner-solver imports) to a :class:`Family`.

    :raises ValueError: For solvers with no matching family.
    """
    if solver_name == "ikgeo.general_6r":
        return FAMILIES["general_6r"]
    if solver_name.startswith("ikgeo."):
        return FAMILIES["sp6"]
    if solver_name.startswith("jointlock."):
        # The locked 6R sub-chain is solved by Raghavan-Roth when it is not SP6.
        if "raghavan_roth" in source:
            return FAMILIES["jointlock_general_6r"]
        return FAMILIES["jointlock_sp6"]
    if solver_name.startswith("seven_r.srs"):
        return FAMILIES["srs"]
    raise ValueError(f"unsupported ssik solver {solver_name!r}: no matching kinematic family")


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def load_artifact(artifact: str | Path) -> types.ModuleType:
    """Import a ssik artifact from a ``*_ik.py`` path or a ``ssik.prebuilt`` name.

    :raises ImportError: If neither resolution strategy succeeds.
    """
    candidate = Path(artifact)
    if candidate.is_file():
        spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load artifact module from {candidate}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        # Keep driver codegen/ directories free of __pycache__.
        prev, sys.dont_write_bytecode = sys.dont_write_bytecode, True
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.dont_write_bytecode = prev
        return mod

    name = str(artifact).removesuffix(".py")
    try:
        return importlib.import_module(f"ssik.prebuilt.{name}")
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"cannot find {str(artifact)!r} as a file or as ssik.prebuilt.{name}"
        ) from exc


def artifact_source(mod: types.ModuleType) -> str:
    path = getattr(mod, "__file__", None)
    return Path(path).read_text(encoding="utf-8") if path else ""


# ---------------------------------------------------------------------------
# Artifact inspection
# ---------------------------------------------------------------------------


@dataclass
class ArtifactInfo:
    """Facts about an artifact that drive scaffolding and agent decisions."""

    arm: str
    artifact: str
    solver_name: str
    dof: int
    max_solutions: int
    base_link: Optional[str]
    ee_link: Optional[str]
    joint_names: List[str]
    # ``None`` per joint when the limit is absent or non-finite (continuous joint).
    joint_limits: List[Optional[Tuple[float, float]]]
    family: str
    category: int
    family_title: str
    reference_arm: str
    reference_repo: str
    reference_driver_dir: str
    generate_code: bool
    baked_files: List[str]
    alternates: List[str]
    note: str


def _joint_limits(mod: types.ModuleType, dof: int) -> List[Optional[Tuple[float, float]]]:
    raw = getattr(mod, "_JOINT_LIMITS", None) or [None] * dof
    out: List[Optional[Tuple[float, float]]] = []
    for lim in raw:
        if lim is None:
            out.append(None)
            continue
        lo, hi = float(lim[0]), float(lim[1])
        out.append((lo, hi) if math.isfinite(lo) and math.isfinite(hi) else None)
    return out


def inspect_artifact(artifact: str | Path, arm: Optional[str] = None) -> ArtifactInfo:
    """Load *artifact* and classify it.

    :param arm: Arm name; defaults to the artifact stem without ``_ik``.
    """
    mod = load_artifact(artifact)
    solver_name = str(getattr(mod, "SOLVER_NAME", ""))
    family = detect_family(solver_name, artifact_source(mod))
    dof = int(getattr(mod, "DOF", len(mod._JOINT_AXES)))
    stem = Path(str(artifact)).stem
    arm = arm or stem.removesuffix("_ik")
    ref_dir = f"{family.reference_repo}/rust/src/drivers/{family.reference_arm}"
    return ArtifactInfo(
        arm=arm,
        artifact=str(artifact),
        solver_name=solver_name,
        dof=dof,
        max_solutions=family.max_solutions,
        base_link=getattr(mod, "BASE_LINK", None),
        ee_link=getattr(mod, "EE_LINK", None),
        joint_names=list(getattr(mod, "_JOINT_NAMES", [])),
        joint_limits=_joint_limits(mod, dof),
        family=family.key,
        category=family.category,
        family_title=family.title,
        reference_arm=family.reference_arm,
        reference_repo=family.reference_repo,
        reference_driver_dir=ref_dir,
        generate_code=family.generate_code,
        baked_files=list(family.baked_files),
        alternates=list(family.alternates),
        note=family.note,
    )


def chain_constants(
    mod: types.ModuleType,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Return ``(t_lefts, axes, t_rights)`` with ``T = Π T_left·Rot(axis, q)·T_right``."""
    t_lefts = [np.asarray(m, dtype=np.float64).reshape(4, 4) for m in mod._JOINT_T_LEFTS]
    axes = [np.asarray(a, dtype=np.float64).reshape(3) for a in mod._JOINT_AXES]
    t_rights = [np.asarray(m, dtype=np.float64).reshape(4, 4) for m in mod._JOINT_T_RIGHTS]
    if not (len(t_lefts) == len(axes) == len(t_rights)):
        raise ValueError("inconsistent _JOINT_T_LEFTS / _JOINT_AXES / _JOINT_T_RIGHTS lengths")
    return t_lefts, axes, t_rights


def nominal_chain(
    mod: types.ModuleType,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Re-express the artifact chain as dexhal ``ChainJoint`` origins and a flange.

    dexhal's ``KinematicChain`` computes ``Π (origin_i · Rot(axis_i, q_i)) · flange``,
    so ``origin_0 = T_left_0``, ``origin_i = T_right_{i-1} · T_left_i`` and
    ``flange = T_right_{n-1}``.
    """
    t_lefts, axes, t_rights = chain_constants(mod)
    origins = [t_lefts[0]]
    for i in range(1, len(t_lefts)):
        origins.append(t_rights[i - 1] @ t_lefts[i])
    return origins, axes, t_rights[-1]


# ---------------------------------------------------------------------------
# Artifact build
# ---------------------------------------------------------------------------


def build_artifact(urdf: Path, base_link: str, ee_link: str, out: Path) -> Path:
    """Run ``ssik build`` on *urdf*, writing the artifact to *out*.

    :raises RuntimeError: On an ssik version mismatch or a failed build.
    """
    get_installed_ssik_version(_REPO_ROOT)
    from ssik.cli import main as ssik_main

    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "ssik", "build", str(urdf),
        "--base", base_link,
        "--ee", ee_link,
        "--out", str(out),
        "--no-validate",
    ]
    saved = sys.argv[:]
    try:
        sys.argv = argv
        ret = ssik_main()
    finally:
        sys.argv = saved
    if ret not in (0, None):
        raise RuntimeError(f"ssik build exited with status {ret}")
    if not out.is_file():
        raise RuntimeError(f"ssik build did not write {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_info(info: ArtifactInfo, as_json: bool) -> None:
    if as_json:
        print(json.dumps(asdict(info), indent=2))
        return
    print(f"arm:            {info.arm}")
    print(f"solver:         {info.solver_name}")
    print(f"DOF:            {info.dof}")
    print(f"MAX_SOLUTIONS:  {info.max_solutions}")
    print(f"base/ee:        {info.base_link} -> {info.ee_link}")
    print(f"family:         category {info.category}: {info.family_title}")
    print(f"reference:      {info.reference_arm} ({info.reference_driver_dir})")
    if info.alternates:
        print(f"alternates:     {', '.join(info.alternates)}")
    gen = ", ".join(info.baked_files) if info.generate_code else "not required"
    print(f"generate_code:  {gen}")
    if info.note:
        print(f"note:           {info.note}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Build <arm>_ik.py from a URDF with ssik.")
    b.add_argument("--urdf", type=Path, required=True)
    b.add_argument("--base", required=True, help="Base link name.")
    b.add_argument("--ee", required=True, help="End-effector (flange) link name.")
    b.add_argument("--out", type=Path, required=True, help="Destination <arm>_ik.py.")
    b.add_argument("--json", action="store_true", help="Print the inspection as JSON.")

    i = sub.add_parser("inspect", help="Classify an existing artifact.")
    i.add_argument("artifact", help="Path to <arm>_ik.py or a ssik.prebuilt name.")
    i.add_argument("--arm", default=None)
    i.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    # ssik's build log prints non-ASCII arrows; don't die on legacy consoles.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    try:
        if args.command == "build":
            out = build_artifact(args.urdf, args.base, args.ee, args.out)
            _print_info(inspect_artifact(out), args.json)
        else:
            _print_info(inspect_artifact(args.artifact, args.arm), args.json)
    except (ImportError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"[ssik_artifact] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
