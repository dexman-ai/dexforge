#!/usr/bin/env python3
"""Scaffold a dexhal vendor crate / arm driver from a URDF and its ssik artifact.

Deterministic Steps 1 and 3 of the agent workflow in ``codegen/README.md``.
Given ``--oem``, ``--arm``, the URDF and its base / flange links this:

* copies the URDF to ``drivers/<arm>/<arm>.urdf`` and builds (or copies) the
  ssik artifact to ``drivers/<arm>/codegen/<arm>_ik.py``;
* creates the vendor crate skeleton when missing (``Cargo.toml``,
  ``README.md``, ``rust/Cargo.toml`` linking ``dexhal``, ``rust/src/lib.rs``,
  ``rust/src/drivers/mod.rs``) and registers the arm in it;
* writes the arm's ``mod.rs``, ``driver.rs`` stub, ``manifest.yaml`` draft and
  ``codegen/generate_test_data.py`` from ``codegen/templates/`` (only when
  missing, or with ``--force``);
* always regenerates ``arm.rs``: the ``dexhal::math::ArmModel`` implementation
  (``NAME``, ``JOINT_LIMITS``, ``HAS_DYNAMICS``, every ``flange_*`` hook) plus
  the nominal ``KinematicChain`` constants.

The coding agent then writes ``ik_solver.rs`` (and ``codegen/generate_code.py``
when the family needs baked constants), implements ``driver.rs``, completes
``manifest.yaml`` and runs the generators listed in the printed next steps.

Usage::

    python codegen/scaffold.py --oem acme --arm acme6 \\
        --urdf path/to/acme6.urdf --base base_link --ee tool0
    # regenerate only arm.rs after the URDF / artifact changed
    python codegen/scaffold.py --oem acme --arm acme6 \\
        --urdf repos/dexforge-acme/rust/src/drivers/acme6/acme6.urdf \\
        --base base_link --ee tool0 --only arm.rs
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_CODEGEN_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CODEGEN_DIR.parent
_TEMPLATES = _CODEGEN_DIR / "templates"
if str(_CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_DIR))

from ssik_artifact import (  # noqa: E402
    ArtifactInfo,
    build_artifact,
    inspect_artifact,
    load_artifact,
    nominal_chain,
)

ARM_OUTPUTS = ("arm.rs", "mod.rs", "driver.rs", "manifest.yaml", "generate_test_data.py", "rust/tests/perf.rs")
VENDOR_OUTPUTS = ("Cargo.toml", "README.md", "rust/Cargo.toml", "rust/src/lib.rs", "rust/src/drivers/mod.rs")

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
_SECTION = re.compile(r"^[ \t]*\{\{#(\w+)\}\}[ \t]*\n(.*?)^[ \t]*\{\{/\1\}\}[ \t]*\n", re.S | re.M)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render(template: str, values: Dict[str, object]) -> str:
    """Substitute ``{{key}}`` placeholders; every placeholder must be provided.

    ``{{#key}}`` ... ``{{/key}}`` sections (each marker on its own line) are kept
    when ``values[key]`` is truthy and dropped otherwise.
    """

    def section(m: re.Match) -> str:
        key = m.group(1)
        if key not in values:
            raise KeyError(f"template section {{{{#{key}}}}} has no value")
        return m.group(2) if values[key] else ""

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in values:
            raise KeyError(f"template placeholder {{{{{key}}}}} has no value")
        return str(values[key])

    return _PLACEHOLDER.sub(sub, _SECTION.sub(section, template))


def render_template(name: str, values: Dict[str, object]) -> str:
    return render((_TEMPLATES / name).read_text(encoding="utf-8"), values)


def struct_name(arm: str) -> str:
    """``franka_panda`` → ``FrankaPanda``, ``xarm7`` → ``Xarm7``."""
    return "".join(part[:1].upper() + part[1:] for part in re.split(r"[_\-]+", arm) if part)


def title(name: str) -> str:
    return " ".join(p[:1].upper() + p[1:] for p in re.split(r"[_\-]+", name) if p)


def rust_f64(x: float) -> str:
    """A Rust ``f64`` literal that round-trips *x* exactly."""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"non-finite constant {x!r}")
    text = repr(x)
    return text if any(c in text for c in ".eE") else text + ".0"


def _limit_f64(x: float) -> str:
    """Joint-limit literal; exact ``±PI``/``±TAU`` use ``core::f64::consts`` (clippy ``approx_constant``)."""
    for name, val in (("PI", math.pi), ("TAU", math.tau)):
        if abs(float(x)) == val:
            return ("-" if x < 0 else "") + f"core::f64::consts::{name}"
    return rust_f64(x)


def _row(values: Sequence[float]) -> str:
    return "[" + ", ".join(rust_f64(v) for v in values) + "]"


def _yaml_list(values: Sequence[Optional[float]]) -> str:
    return ", ".join("TODO" if v is None else repr(float(v)) for v in values)


# ---------------------------------------------------------------------------
# URDF facts
# ---------------------------------------------------------------------------


@dataclass
class UrdfLimits:
    velocity: List[Optional[float]]
    effort: List[Optional[float]]


def urdf_limits(urdf: Path, joint_names: Sequence[str]) -> UrdfLimits:
    """Read ``<limit velocity= effort=>`` for *joint_names* (``None`` when absent / non-positive)."""
    root = ET.parse(urdf).getroot()
    by_name = {j.get("name"): j for j in root.findall("joint")}
    velocity: List[Optional[float]] = []
    effort: List[Optional[float]] = []
    for name in joint_names:
        lim = by_name.get(name)
        lim = lim.find("limit") if lim is not None else None
        for attr, out in (("velocity", velocity), ("effort", effort)):
            raw = lim.get(attr) if lim is not None else None
            value = float(raw) if raw is not None else None
            out.append(value if value is not None and math.isfinite(value) and value > 0 else None)
    return UrdfLimits(velocity=velocity, effort=effort)


def urdf_has_dynamics(urdf: Path, base_link: str, ee_link: str) -> bool:
    """True when every actuated link of the base → flange chain has a positive mass."""
    from generate_dynamics import order_urdf_chain, parse_urdf_joints

    chain = order_urdf_chain(parse_urdf_joints(urdf), base_link, ee_link)
    actuated = [j for j in chain if j.actuated]
    return bool(actuated) and all(
        j.child_inertial is not None and j.child_inertial.mass > 0 for j in actuated
    )


def contract_version(repo_root: Path) -> str:
    """The dexhal ``CONTRACT_VERSION`` as ``"major.minor"``."""
    text = (repo_root / "dexhal" / "src" / "math" / "contract" / "mod.rs").read_text(
        encoding="utf-8"
    )
    m = re.search(
        r"CONTRACT_VERSION:\s*ContractVersion\s*=\s*ContractVersion\s*\{\s*major:\s*(\d+),\s*minor:\s*(\d+)",
        text,
    )
    if not m:
        raise ValueError("cannot find CONTRACT_VERSION in dexhal/src/math/contract/mod.rs")
    return f"{m.group(1)}.{m.group(2)}"


# ---------------------------------------------------------------------------
# File generators
# ---------------------------------------------------------------------------


def generate_arm_rs(info: ArtifactInfo, artifact: Path, has_dynamics: bool) -> str:
    """Render ``arm.rs`` (``ArmModel`` + nominal ``KinematicChain``) for *info*."""
    if info.dof not in (6, 7):
        raise ValueError(f"arm.rs template supports 6- and 7-DOF arms, not {info.dof}-DOF")
    origins, axes, flange = nominal_chain(load_artifact(artifact))
    name = struct_name(info.arm)

    limits = []
    for lim in info.joint_limits:
        limits.append(
            "        None,\n" if lim is None else f"        Some(({_limit_f64(lim[0])}, {_limit_f64(lim[1])})),\n"
        )

    dynamics_impl = ""
    modules = ["generated_kinematics", "ik_solver"]
    if has_dynamics:
        modules.insert(0, "generated_dynamics")
        dynamics_impl = """
    #[inline]
    fn flange_gravity(q: &[f64; DOF], gravity: &[f64; 3]) -> [f64; DOF] {
        generated_dynamics::gravity(q, gravity)
    }

    #[inline]
    fn flange_mass_matrix(q: &[f64; DOF]) -> [[f64; DOF]; DOF] {
        generated_dynamics::mass_matrix(q)
    }

    #[inline]
    fn flange_coriolis(q: &[f64; DOF], qd: &[f64; DOF]) -> [f64; DOF] {
        generated_dynamics::coriolis(q, qd)
    }

    #[inline]
    fn flange_inverse_dynamics(
        q: &[f64; DOF],
        qd: &[f64; DOF],
        qdd: &[f64; DOF],
        gravity: &[f64; 3],
    ) -> [f64; DOF] {
        generated_dynamics::inverse_dynamics(q, qd, qdd, gravity)
    }
"""

    joint_names = info.joint_names or [f"joint_{i + 1}" for i in range(info.dof)]
    return render_template(
        "arm/arm.rs.tmpl",
        {
            "arm": info.arm,
            "arm_title": title(info.arm),
            "struct_name": name,
            "dof": info.dof,
            "max_solutions": info.max_solutions,
            "family_title": info.family_title,
            "base_link": info.base_link or "",
            "ee_link": info.ee_link or "",
            "joint_names": ", ".join(f'"{n}"' for n in joint_names),
            "joint_origins": "".join(f"    {_row(np.asarray(o).reshape(-1))},\n" for o in origins),
            "joint_axes": "".join(f"    {_row(a)},\n" for a in axes),
            "flange": _row(np.asarray(flange).reshape(-1)),
            "joint_limits": "".join(limits),
            "has_dynamics": "true" if has_dynamics else "false",
            "module_uses": "{" + ", ".join(modules) + "}",
            "dynamics_impl": dynamics_impl,
            "six_dof": info.dof == 6,
            "seven_dof": info.dof == 7,
        },
    )


@dataclass
class Layout:
    """Paths of one arm inside its vendor repository."""

    oem: str
    arm: str
    vendor_dir: Path
    crate_dir: Path
    drivers_dir: Path
    driver_dir: Path
    manifest: Path

    @property
    def codegen_dir(self) -> Path:
        return self.driver_dir / "codegen"

    @property
    def artifact(self) -> Path:
        return self.codegen_dir / f"{self.arm}_ik.py"

    @property
    def urdf(self) -> Path:
        return self.driver_dir / f"{self.arm}.urdf"


def _manifest_path(crate_dir: Path, arm: str) -> Path:
    """``rust/manifest.yaml``, or ``rust/manifest_<arm>.yaml`` when another arm owns it."""
    default = crate_dir / "manifest.yaml"
    if not default.is_file():
        return default
    m = re.search(r"^driver:\s*(\S+)", default.read_text(encoding="utf-8"), re.M)
    if m and m.group(1) == arm:
        return default
    return crate_dir / f"manifest_{arm}.yaml"


def make_layout(repos_dir: Path, oem: str, arm: str) -> Layout:
    vendor = repos_dir / f"dexforge-{oem}"
    crate = vendor / "rust"
    drivers = crate / "src" / "drivers"
    return Layout(
        oem=oem,
        arm=arm,
        vendor_dir=vendor,
        crate_dir=crate,
        drivers_dir=drivers,
        driver_dir=drivers / arm,
        manifest=_manifest_path(crate, arm),
    )


def _rel(path: Path, start: Path) -> str:
    """*path* relative to *start* when it lies below it, else absolute (POSIX separators)."""
    path, start = Path(path).resolve(), Path(start).resolve()
    try:
        return path.relative_to(start).as_posix()
    except ValueError:
        return path.as_posix()


def _rel_any(path: Path, start: Path) -> str:
    """*path* relative to *start* (``..`` allowed); absolute across Windows drives."""
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(start).resolve())).as_posix()
    except ValueError:  # different drives on Windows
        return Path(path).resolve().as_posix()


def _is_below(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _write(path: Path, text: str, written: List[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    written.append(path)


def _append_line(path: Path, line: str, written: List[Path]) -> None:
    text = path.read_text(encoding="utf-8")
    if line in text.splitlines():
        return
    if not text.endswith("\n"):
        text += "\n"
    _write(path, text + line + "\n", written)


def _write_perf_test(path: Path, values: Dict[str, object], written: List[Path]) -> None:
    """Create ``rust/tests/perf.rs`` with the shared-harness header, then append one test per arm."""
    block = render_template("arm/rust/tests/perf_arm.rs.tmpl", values)
    if not path.is_file():
        _write(path, render_template("vendor/rust/tests/perf.rs.tmpl", values) + block, written)
        return
    text = path.read_text(encoding="utf-8")
    if f"fn {values['arm']}_arm_timing()" in text:
        return
    _write(path, text.rstrip("\n") + "\n" + block, written)


@dataclass
class ScaffoldResult:
    layout: Layout
    info: ArtifactInfo
    has_dynamics: bool
    written: List[Path]
    skipped: List[Path]


def scaffold(
    oem: str,
    arm: str,
    urdf: Path,
    base_link: str,
    ee_link: str,
    artifact: Optional[Path] = None,
    repos_dir: Optional[Path] = None,
    only: Optional[Sequence[str]] = None,
    force: bool = False,
    n_cases: int = 500,
    repo_root: Path = _REPO_ROOT,
) -> ScaffoldResult:
    """Create / update the vendor crate and arm driver files. See the module docs."""
    repos_dir = repos_dir or repo_root / "repos"
    layout = make_layout(repos_dir, oem, arm)
    wanted = set(only) if only else set(ARM_OUTPUTS) | set(VENDOR_OUTPUTS)
    unknown = wanted - set(ARM_OUTPUTS) - set(VENDOR_OUTPUTS)
    if unknown:
        raise ValueError(f"unknown --only outputs: {sorted(unknown)}")
    written: List[Path] = []
    skipped: List[Path] = []

    # Provenance inputs: the URDF and the ssik artifact live next to the driver.
    layout.codegen_dir.mkdir(parents=True, exist_ok=True)
    urdf = Path(urdf).resolve()
    if urdf != layout.urdf.resolve():
        shutil.copyfile(urdf, layout.urdf)
        written.append(layout.urdf)
    if artifact is not None and Path(artifact).resolve() != layout.artifact.resolve():
        shutil.copyfile(artifact, layout.artifact)
        written.append(layout.artifact)
    elif artifact is None and not layout.artifact.is_file():
        build_artifact(layout.urdf, base_link, ee_link, layout.artifact)
        written.append(layout.artifact)

    info = inspect_artifact(layout.artifact, arm)
    if info.base_link and info.base_link != base_link or info.ee_link and info.ee_link != ee_link:
        raise ValueError(
            f"artifact chain {info.base_link!r} -> {info.ee_link!r} does not match "
            f"--base {base_link!r} --ee {ee_link!r}"
        )
    has_dynamics = urdf_has_dynamics(layout.urdf, base_link, ee_link)
    limits = urdf_limits(layout.urdf, info.joint_names)

    name = struct_name(arm)
    dexhal_dir = repo_root / "dexhal"
    values: Dict[str, object] = {
        "oem": oem,
        "oem_ident": oem.replace("-", "_"),
        "oem_title": title(oem),
        "arm": arm,
        "arm_title": title(arm),
        "struct_name": name,
        "dof": info.dof,
        "max_solutions": info.max_solutions,
        "family_title": info.family_title,
        "reference_driver_dir": info.reference_driver_dir,
        "dexhal_path": _rel_any(dexhal_dir, layout.crate_dir)
        if _is_below(layout.crate_dir, repo_root)
        else dexhal_dir.resolve().as_posix(),
        "contract_version": contract_version(repo_root),
        "has_dynamics": "true" if has_dynamics else "false",
        "manifest_include": _rel_any(layout.manifest, layout.driver_dir),
        "velocity": _yaml_list(limits.velocity),
        "effort": _yaml_list(limits.effort),
        "acceleration": ", ".join(["TODO"] * info.dof),
        "velocity_note": "  (URDF <limit velocity>)" if None not in limits.velocity else "  TODO: missing in URDF",
        "effort_note": "  (URDF <limit effort>)" if None not in limits.effort else "  TODO: missing in URDF",
        "baked_doc": "".join(f", `{f}`" for f in info.baked_files) if info.generate_code else "",
        "baked_mods": "".join(f"pub mod {Path(f).stem};\n" for f in info.baked_files)
        if info.generate_code
        else "",
        "dynamics_parity": "\n#[cfg(test)]\nmod dynamics_parity;\n" if has_dynamics else "",
        "n_cases": n_cases,
        "script_path": _rel(layout.codegen_dir / "generate_test_data.py", repo_root),
        "crate_ident": f"dexforge_{oem.replace('-', '_')}_core",
        "perf_harness": _rel_any(dexhal_dir / "perf" / "arm_timing.rs", layout.crate_dir / "tests"),
    }

    def emit(key: str, path: Path, text_fn, always: bool = False) -> None:
        if key not in wanted:
            return
        if path.is_file() and not (always or force):
            skipped.append(path)
            return
        _write(path, text_fn(), written)

    # Vendor crate skeleton (created once).
    emit("Cargo.toml", layout.vendor_dir / "Cargo.toml", lambda: render_template("vendor/Cargo.toml.tmpl", values))
    emit("README.md", layout.vendor_dir / "README.md", lambda: render_template("vendor/README.md.tmpl", values))
    emit("rust/Cargo.toml", layout.crate_dir / "Cargo.toml", lambda: render_template("vendor/rust/Cargo.toml.tmpl", values))
    emit("rust/src/lib.rs", layout.crate_dir / "src" / "lib.rs", lambda: render_template("vendor/rust/src/lib.rs.tmpl", values))
    emit(
        "rust/src/drivers/mod.rs",
        layout.drivers_dir / "mod.rs",
        lambda: render_template("vendor/rust/src/drivers/mod.rs.tmpl", values),
    )
    if "rust/src/drivers/mod.rs" in wanted and (layout.drivers_dir / "mod.rs").is_file():
        _append_line(layout.drivers_dir / "mod.rs", f"pub mod {arm};", written)
    if "rust/src/lib.rs" in wanted and (layout.crate_dir / "src" / "lib.rs").is_file():
        _append_line(
            layout.crate_dir / "src" / "lib.rs",
            f"pub use drivers::{arm}::{{{name}, {name}Driver}};",
            written,
        )

    # Arm driver files.
    emit("arm.rs", layout.driver_dir / "arm.rs", lambda: generate_arm_rs(info, layout.artifact, has_dynamics), always=True)
    emit("mod.rs", layout.driver_dir / "mod.rs", lambda: render_template("arm/mod.rs.tmpl", values))
    emit("driver.rs", layout.driver_dir / "driver.rs", lambda: render_template("arm/driver.rs.tmpl", values))
    if "rust/tests/perf.rs" in wanted:
        _write_perf_test(layout.crate_dir / "tests" / "perf.rs", values, written)
    emit("manifest.yaml", layout.manifest, lambda: render_template("arm/manifest.yaml.tmpl", values))
    emit(
        "generate_test_data.py",
        layout.codegen_dir / "generate_test_data.py",
        lambda: render_template("arm/codegen/generate_test_data.py.tmpl", values),
    )

    return ScaffoldResult(layout=layout, info=info, has_dynamics=has_dynamics, written=written, skipped=skipped)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _next_steps(res: ScaffoldResult, repo_root: Path) -> str:
    lay, info = res.layout, res.info
    d = _rel(lay.driver_dir, repo_root)
    ref = info.reference_driver_dir
    if info.dof == 6:
        one_api = "`raw_candidates(target, policy) -> ([[f64; 6]; CANDIDATES], usize)` (unfiltered algebraic candidates)"
    else:
        one_api = (
            "`LOCKED_JOINT`, `LOCK_SAMPLES` and "
            f"`lock_sample_solutions(target, lock_value, policy) -> [Option<[f64; 7]>; {info.max_solutions}]`"
        )
    steps = [
        f"Reference arm: {info.reference_arm} ({ref}); family: category {info.category}, {info.family_title}.",
        f"Write {d}/ik_solver.rs from {d}/codegen/{lay.arm}_ik.py following codegen/translate_ssik_artifact.md "
        f"and {ref}/ik_solver.rs; it must export `solve(target, max_solutions, q_seed, respect_limits, "
        f"allow_refinement, policy, refinement_max_iters) -> [Option<[f64; {info.dof}]>; {info.max_solutions}]` "
        f"(flange_all) and {one_api} (flange_one).",
    ]
    if info.generate_code:
        steps.append(
            f"Adapt {ref}/codegen/generate_code.py into {d}/codegen/generate_code.py "
            f"(writes {', '.join(info.baked_files)}) and run it."
        )
    if info.note:
        steps.append(f"Note: {info.note}")
    dyn = (
        f"python codegen/generate_dynamics.py --artifact {d}/codegen/{lay.arm}_ik.py "
        f"--urdf {d}/{lay.arm}.urdf --base {info.base_link} --ee {info.ee_link} "
        f"--arm {lay.arm} --output {d}/generated_dynamics.rs"
    )
    steps.append(dyn)
    steps.append(f"python {d}/codegen/generate_test_data.py")
    if res.has_dynamics:
        steps.append(
            f"python codegen/generate_dynamics_test_data.py --urdf {d}/{lay.arm}.urdf "
            f"--base {info.base_link} --ee {info.ee_link} --output-dir {d} --arm {lay.arm} "
            "--samples 1000 --seed 0"
        )
    steps.append(
        f"Complete the TODOs in {_rel(lay.manifest, repo_root)} and {d}/driver.rs from the OEM documentation."
    )
    steps.append(f"cargo test --manifest-path {_rel(lay.vendor_dir, repo_root)}/Cargo.toml")
    return "\n".join(f"  {i}. {s}" for i, s in enumerate(steps, 1))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oem", required=True, help="Vendor name; the crate is repos/dexforge-<oem>.")
    parser.add_argument("--arm", required=True, help="Arm module name (snake_case), e.g. ur5.")
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--base", required=True, help="URDF base link.")
    parser.add_argument("--ee", required=True, help="URDF flange / end-effector link.")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Existing <arm>_ik.py to copy instead of running `ssik build`.",
    )
    parser.add_argument("--repos-dir", type=Path, default=None, help="Default: <repo>/repos.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=ARM_OUTPUTS + VENDOR_OUTPUTS,
        default=None,
        help="Restrict the outputs written.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing scaffold files.")
    parser.add_argument("--n-cases", type=int, default=500, help="IK reference poses (default 500).")
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.arm):
        parser.error("--arm must be a snake_case Rust module name")
    if not re.fullmatch(r"[a-z][a-z0-9\-]*", args.oem):
        parser.error("--oem must be lowercase (letters, digits, '-')")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")

    try:
        res = scaffold(
            args.oem,
            args.arm,
            args.urdf,
            args.base,
            args.ee,
            artifact=args.artifact,
            repos_dir=args.repos_dir,
            only=args.only,
            force=args.force,
            n_cases=args.n_cases,
        )
    except (ImportError, RuntimeError, ValueError, KeyError, FileNotFoundError) as exc:
        print(f"[scaffold] ERROR: {exc}", file=sys.stderr)
        return 1

    for p in dict.fromkeys(res.written):
        print(f"[scaffold] wrote   {_rel(p, _REPO_ROOT)}")
    for p in res.skipped:
        print(f"[scaffold] kept    {_rel(p, _REPO_ROOT)} (exists; --force to overwrite)")
    print(f"[scaffold] HAS_DYNAMICS = {str(res.has_dynamics).lower()}")
    print("[scaffold] next steps:\n" + _next_steps(res, _REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
