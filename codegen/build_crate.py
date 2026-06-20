"""Build and test module for ik_generator.

This module handles:
- Parsing multi-file LLM output
- Writing generated files to disk
- Running Python helper scripts
- Building and testing the Rust crate
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).parent.parent.resolve()

def drivers_dir(output_root: Path) -> Path:
    """Return the driver directory rooted at *root* for *arm_name*."""
    return output_root / "models"

def driver_dir(output_root: Path, arm_name: str) -> Path:
    """Return the driver directory rooted at *root* for *arm_name*."""
    return drivers_dir(output_root) / arm_name

def _copy_math_kernels(output_root: Path) -> None:
    """Copy the shared math kernels into the generated crate tree."""
    src = _REPO_ROOT / "src" / "math"
    dst = output_root / "math"
    if not src.is_dir():
        raise FileNotFoundError(f"missing math kernels: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for rs in sorted(src.glob("*.rs")):
        shutil.copyfile(rs, dst / rs.name)


def make_crate(
    output_dir: Path,
    oem_name: str,
    arm_name: str,
) -> None:
    """Write the generated crate's mod.rs, Cargo.toml and lib.rs, copy math sources and generate a custom arm.rs impl."""
    _copy_math_kernels(output_dir)
    _cargo_toml(output_dir, oem_name)
    _lib_rs(arm_name)
    _mod_rs(output_dir, oem_name, arm_name)


def _mod_rs(output_root: Path, oem_name:str, arm_name: str) -> None:
    """Create the arm module files in the generated crate if missing."""
    # leaf mod.rs in the arm directory. Include all files in the arm directory as submodules.
    mod_path = driver_dir(output_root, arm_name) / "mod.rs"
    mod_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [f"//! {arm_name} rigid-body dynamics and kinematics model.\n"]
    lines.append("")
    for rs in sorted(mod_path.parent.glob("*.rs")):
        if rs.name != "mod.rs":
            stem = rs.stem
            lines.append(f"pub mod {stem};")
    mod_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Added {mod_path.relative_to(output_root)}")

    # OEM-level mod.rs to re-export the arm module
    oem_mod = driver_dir(output_root, "") / "mod.rs"
    lines: list[str] = [f"//! {oem_name} rigid-body dynamics and kinematics models.\n"]
    for rs in sorted(oem_mod.parent.glob("*")):
        if rs.name != "mod.rs":
            if not rs.is_dir() and not rs.name.endswith(".rs"):
                continue
            stem = rs.stem
            lines.append(f"pub mod {stem};")
    oem_mod.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Added {oem_mod.relative_to(output_root)}")


def _cargo_toml(output_root: Path, oem: str) -> None:
    """Render the stand-alone crate's ``Cargo.toml``."""
    content = f"""\
[package]
name = "dexforge_{oem}"
version = "0.1.0"
edition = "2026"
description = "Analytical IK and rigid-body dynamics model for {oem} arms, compiled by dexforge."
license = "BSD-3-Clause"

# Self-contained: depends only on libm + nalgebra (no_std).
[dependencies]
libm = "0.2"

[dependencies.nalgebra]
version = "0.33"
default-features = false
features = ["libm"]

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
"""
    cargo_toml_path = output_root / "Cargo.toml"
    cargo_toml_path.write_text(content, encoding="utf-8") 
    logger.info(f"Added {cargo_toml_path.relative_to(output_root)}")


def _lib_rs(output_dir: Path, oem: str) -> None:
    """Render the stand-alone crate's ``src/lib.rs``."""
    # capitalize the first letter of the arm name for the Rust module and type names
    librs =  f"""\
//! Stand-alone {oem} rigid-body arm models (kinematics + dynamics), compiled by dexforge.
//!
//! This crate is fully self-contained: it does not depend on the dexforge repository or Python package. 
//! The crate is `#![no_std]`, heap-free, bounded-iteration and deterministic throughout.
#![no_std]
#![forbid(unsafe_code)]

mod models;
mod math;

"""
    # iterate over all arm directories in the models directory and re-export their ArmModel types at the crate root
    for arm_dir in sorted(drivers_dir(output_dir).glob("*")):
        if arm_dir.is_dir():
            arm_name = arm_dir.name
            arm_caps = arm_name[0].upper() + arm_name[1:]
            librs += f"pub use models::{arm_name}::{arm_caps}Model as {arm_caps}Model;\n"

    lib_rs_path = output_dir / "src" / "lib.rs"
    lib_rs_path.parent.mkdir(parents=True, exist_ok=True)
    lib_rs_path.write_text(librs, encoding="utf-8")
    logger.info(f"Added {lib_rs_path.relative_to(output_dir)}")

def _model_rs(output_dir:Path, arm_name: str, dof: int, max_solutions: int) -> None:
    """Render the per-arm ``model.rs`` module containing the ArmModel impl.

    The generated module owns the zero-sized marker type and the concrete
    ``ArmModel`` implementation for the arm's ``fk`` / ``jacobian`` /
    ``solve`` entry points.
    """
    arm_caps = arm_name[0].upper() + arm_name[1:]
    model_impl =  f"""\
use crate::math::arm::{{ArmModel, Pose}};
use crate::math::tolerances::TolerancePolicy;

/// {arm_name} driver marker.
pub struct {arm_caps}Model;

impl ArmModel<{dof}, {max_solutions}> for {arm_caps}Model {{
    #[inline]
    fn flange_fk(q: &[f64; {dof}]) -> Pose {{
        crate::models::{arm_name}::ik_solver::fk(q)
    }}

    #[inline]
    fn flange_jacobian(q: &[f64; {dof}]) -> [[f64; {dof}]; 6] {{
        crate::models::{arm_name}::generated_kinematics::jacobian(q)
    }}

    #[inline]
    fn flange_solve(
        target: &Pose,
        max_solutions: Option<usize>,
        q_seed: Option<&[f64; {dof}]>,
        respect_limits: bool,
        allow_refinement: bool,
        policy: TolerancePolicy,
        refinement_max_iters: usize,
    ) -> [Option<[f64; {dof}]>; {max_solutions}] {{
        crate::models::{arm_name}::ik_solver::solve(
            target,
            max_solutions,
            q_seed,
            respect_limits,
            allow_refinement,
            policy,
            refinement_max_iters,
        )
    #[inline]
    fn flange_gravity(q: &[f64; {dof}]) -> [f64; {dof}] {{
        crate::models::{arm_name}::generated_dynamics::gravity(q)
    }}

    #[inline]
    fn flange_mass_matrix(q: &[f64; {dof}]) -> [[f64; {dof}]; {dof}] {{
        crate::models::{arm_name}::generated_dynamics::mass_matrix(q)
    }}

    #[inline]
    fn flange_coriolis(q: &[f64; {dof}], qd: &[f64; {dof}]) -> [f64; {dof}] {{
        crate::models::{arm_name}::generated_dynamics::coriolis(q, qd)
    }}
}}
"""
    output_file = driver_dir(output_dir, arm_name) / "model.rs"
    output_file.write_text(model_impl, encoding="utf-8")
    logger.info(f"Added {output_file.relative_to(output_dir)}")

def write_generated_files(
    output_dir: Path,
    arm_name: str,
    files: dict[str, str],
) -> dict[str, Path]:
    """Write all generated files to the appropriate locations.

    * ``*.rs`` files → the generated crate's arm directory
    * ``generate_*.py`` files → ``codegen/`` inside the generated arm directory

    :param output_dir: Generated crate root directory.
    :param arm_name: Clean arm name (e.g. ``"my_arm"``).
    :param files: Parsed ``{filename: content}`` from :func:`parse_llm_output`.
    :returns: Mapping of logical name → written :class:`Path`.
    """
    logger.info(f"Saving generated files.")
    driver_dir = driver_dir(output_dir, arm_name)
    driver_dir.mkdir(parents=True, exist_ok=True)
    codegen_dir = driver_dir / "codegen" 
    codegen_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for filename, content in files.items():
        if filename.endswith(".rs"):
            dest = driver_dir / filename
            dest.write_text(content, encoding="utf-8")
            written[filename] = dest
        else:
            # Fallback: write to codegen_path
            dest = codegen_dir / filename
            dest.write_text(content, encoding="utf-8")
            written[filename] = dest

    return written


def remove_generated_files(files: list[Path]) -> None:
    """Remove all files listed in the generated_paths dictionary."""
    logger.info(f"Cleaning up generated files.")
    for path in files:
        if path.exists():
            path.unlink()

def run_py_scripts(output_dir: Path) -> list[tuple[str, str]]:
    """Run all ``generate_*.py`` files that were written to ``codegen/``.

    Each script is executed with the current Python interpreter from the
    generated crate root so that relative paths inside the scripts resolve
    correctly.

    :param output_dir: Generated crate root directory.
    The function will look for ``generate_*.py`` scripts inside the ``codegen/`` directory of each arm.
    :returns:  ``(script_name, error_output)`` tuple when a script fails.
        None means all scripts succeeded.
    """
    generated  = []
    codegen_dir = output_dir / "codegen"
    for path in sorted(codegen_dir.glob("generate_*.py")):
        filename = path.name

        logger.info(f"Running generator script '{filename}'.")

        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            cwd=str(output_dir),
        )
        if result.returncode != 0:
            error = f"Generator script '{filename}' failed with error:\n {(result.stderr + result.stdout).strip()}"
            logger.error(error)
            raise RuntimeError(error)
        generated.append((filename, str(path)))
    return generated


def build_rust_crate(cwd: Path) -> Tuple[bool, str]:
    """Run ``cargo build`` from *cwd*.

    :returns: ``(success, output)`` where *output* is combined stdout+stderr.
    """
    logger.info(f"Running cargo build.")
    result = subprocess.run(
        ["cargo", "build"],
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )
    output = (result.stderr + result.stdout).strip()
    if result.returncode == 0:
        logger.info(f"Cargo build succeeded.")
    else:
        error = f"Cargo build failed:\n{output}"
        logger.error(error)
        raise RuntimeError(error)


def test_rust_crate(cwd: Path) -> Tuple[bool, str]:
    """Run ``cargo test`` from *cwd*.

    :returns: ``(success, output)`` where *output* is combined stdout+stderr.
    """
    logger.info(f"Running cargo test.")
    result = subprocess.run(
        ["cargo", "test"],
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )
    output = (result.stderr + result.stdout).strip()
    if result.returncode == 0:
        logger.info(f"Cargo test succeeded.")
    else:
        error = f"Cargo test failed:\n{output}"
        logger.error(error)
        raise RuntimeError(error)


