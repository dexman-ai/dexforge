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
from typing import Optional, Tuple
import re

# Repository root — all relative paths are anchored here.
_CODEGEN_DIR = Path(__file__).resolve().parent   # codegen/
_REPO_ROOT = _CODEGEN_DIR.parent                 # repository root
if str(_CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_DIR))

from repo_paths import math_src as _math_src  # noqa: E402


def _driver_dir(output_root: Path, oem_name: str, arm_name: str) -> Path:
    """Return the driver directory inside a generated crate for *arm_name*.

    Layout matches ``dexforge build`` / ``plan_export``::

        <crate>/src/drivers/<arm>/

    *oem_name* is retained for provenance naming but no longer nests the path.
    """
    del oem_name  # crate layout is flat under src/drivers/<arm>
    return output_root / "src" / "drivers" / arm_name


def _vendor_math_source(text: str) -> str:
    """Rewrite dexhal-crate paths so vendored math compiles in a stand-alone crate.

    Doctests and comments in ``repos/dexhal`` correctly say ``dexhal::math::…``.
    After copy into a ``dexforge build`` output crate there is no ``dexhal``
    dependency, so those paths become ``crate::math::…``.
    """
    return text.replace("dexhal::math::", "crate::math::")


def _copy_math_kernels(repo_root: Path, output_root: Path) -> None:
    """Copy the shared math kernels into the generated crate tree."""
    src = _math_src(repo_root)
    dst = output_root / "src" / "math"
    dst.mkdir(parents=True, exist_ok=True)
    for rs in sorted(src.glob("*.rs")):
        text = rs.read_text(encoding="utf-8")
        (dst / rs.name).write_text(_vendor_math_source(text), encoding="utf-8")


def _cargo_toml(cargo_suffix: str, arm_name: str) -> str:
    """Render the generated crate's Cargo manifest."""
    return f"""\
[package]
name = "dexforge-{cargo_suffix}"
version = "0.1.0"
edition = "2021"
description = "Rigid Body Kinematics and Dynamics driver for {cargo_suffix} arms, compiled by dexforge."
license = "BSD-3-Clause"

# Keep the generated crate out of any parent Cargo workspace.
[workspace]

[lib]
# Vendored dexhal sources keep ``dexhal::…`` doctests; those do not apply in a
# stand-alone driver crate. Unit tests under ``#[cfg(test)]`` are the gate.
doctest = false

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


def _lib_rs(arm_name: str) -> str:
    """Render the generated crate's src/lib.rs."""
    return f"""\
#![no_std]
#![forbid(unsafe_code)]

pub mod drivers;
pub mod math;

pub use drivers::{arm_name}::ik_solver::{{fk, solve, TolerancePolicy}};
pub use drivers::{arm_name}::generated_kinematics;
pub use drivers::{arm_name}::generated_dynamics;
"""

def _ensure_arm_mod_rs(output_root: Path, oem_name: str, arm_name: str, rust_files: list[str]) -> None:
    """Create the arm module files in the generated crate if missing."""
    drivers_mod = output_root / "src" / "drivers" / "mod.rs"
    drivers_mod.parent.mkdir(parents=True, exist_ok=True)
    drivers_mod.write_text(
        "//! Driver module index (compiled by dexforge build).\n\n"
        f"pub mod {arm_name};\n",
        encoding="utf-8",
    )
    print(f"[ik_generator]   Created {drivers_mod.relative_to(output_root)}")

    mod_path = _driver_dir(output_root, oem_name, arm_name) / "mod.rs"
    mod_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [f"//! {arm_name} IK driver.\n", "", "pub mod ik_solver;"]
    for fname in sorted(rust_files):
        stem = Path(fname).stem
        if stem in ("ik_solver", "mod", "test_data"):
            continue
        # Dynamics / kinematics kernels and the ArmModel marker must be public
        # so crate-root re-exports and flange_* overrides can reach them.
        if stem in ("arm", "generated_dynamics", "generated_kinematics"):
            lines.append(f"pub mod {stem};")
        else:
            lines.append(f"mod {stem};")
    mod_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ik_generator]   Created {mod_path.relative_to(output_root)}")


def copy_driver_provenance(
    dest_dir: Path,
    arm_name: str,
    ssik_artifact: str,
    source_urdf: Path | None = None,
    context: str | None = None,
) -> dict[str, Path]:
    """Copy the Python artifact, optional URDF, and context into a driver directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    py_path = dest_dir / f"{arm_name}_ik.py"
    py_path.write_text(ssik_artifact, encoding="utf-8")
    written["driver_py"] = py_path

    if source_urdf is not None:
        urdf_path = dest_dir / f"{arm_name}.urdf"
        if source_urdf.resolve() != urdf_path.resolve():
            shutil.copyfile(source_urdf, urdf_path)
        written["driver_urdf"] = urdf_path
    if context is not None:
        ctx_path = dest_dir / f"{arm_name}_context.txt"
        ctx_path.write_text(context, encoding="utf-8")
        written["driver_ctx"] = ctx_path

    return written


def write_generated_files(
    output_dir: Path,
    oem_name: str,
    repo_root: Path,
    arm_name: str,
    files: dict[str, str],
    ssik_artifact: str,
    context: str,
    source_urdf: Path | None = None,
) -> dict[str, Path]:
    """Write all LLM-generated files to the appropriate locations.

    * ``*.rs`` files → the generated crate's driver directory
    * The original Python artifact (and URDF, when available) → the driver dir
    * ``generate_*.py`` files → ``codegen/`` inside the generated crate
    * The original Python artifact and context dump → *output_dir*

    :param output_dir: Generated crate root directory.
    :param repo_root: dexforge source tree (provides the shared math kernels).
    :param arm_name: Clean arm name (e.g. ``"my_arm"``).
    :param files: Parsed ``{filename: content}`` from :func:`parse_llm_output`.
    :param ssik_artifact: Source code of the target Python artifact.
    :param context: Assembled prompt context string.
    :returns: Mapping of logical name → written :class:`Path`.
    """
    output_dir = output_dir 
    output_dir.mkdir(parents=True, exist_ok=True)
    driver_dir = _driver_dir(output_dir, oem_name, arm_name)
    driver_dir.mkdir(parents=True, exist_ok=True)
    codegen_path = driver_dir / "codegen" 
    codegen_path.mkdir(parents=True, exist_ok=True)

    _copy_math_kernels(repo_root, output_dir)

    cargo_path = output_dir / "Cargo.toml"
    if not cargo_path.exists():
        cargo_path.write_text(_cargo_toml(oem_name, arm_name), encoding="utf-8")

    lib_path = output_dir / "src" / "lib.rs"
    if not lib_path.exists():
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        lib_path.write_text(_lib_rs(arm_name), encoding="utf-8")

    # Write artifact and context to output_dir
    written = copy_driver_provenance(
        codegen_path, arm_name, ssik_artifact, source_urdf, context
    )

    rust_files_written: list[str] = []
    for filename, content in files.items():
        if filename == "lib.rs":
            dest = output_dir / "lib.rs"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            written[filename] = dest
            print(f"[ik_generator]   Wrote {dest.relative_to(output_dir)}")
        elif filename.endswith(".rs"):
            dest = driver_dir / filename
            dest.write_text(content, encoding="utf-8")
            written[filename] = dest
            rust_files_written.append(filename)
            print(f"[ik_generator]   Wrote {dest.relative_to(output_dir)}")
        elif filename.endswith(".py"):
            # Normalize script names: test-data generators → generate_test_data.py,
            # all other generate_*.py scripts (consts, pq, etc.) → generate_code.py.
            if "test" in filename.lower():
                normalized = "generate_test_data.py"
            else:
                normalized = "generate_code.py"
            dest = codegen_path / normalized
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            written[normalized] = dest
            print(f"[ik_generator]   Wrote {dest.relative_to(output_dir)} (from '{filename}')")
        else:
            # Fallback: write to codegen_path
            dest = codegen_path / filename
            dest.write_text(content, encoding="utf-8")
            written[filename] = dest
            print(f"[ik_generator]   Wrote {dest.relative_to(output_dir)}")

    # Ensure mod.rs exists for the new driver
    _ensure_arm_mod_rs(output_dir, oem_name, arm_name, rust_files_written)

    return written


def _normalize_helper_script_paths(script_path: Path) -> None:
    """Rewrite legacy dest paths so helpers work under ``src/drivers/<arm>/codegen/``.

    Older prompts told the LLM to write::

        join(__file__, "..", "src", "drivers", "<arm>", "test_data.rs")

    which is correct only when the script lives at repo-root ``codegen/``.
    Stand-alone crates place helpers next to the driver, so the dest must be
    ``join(__file__, "..", "test_data.rs")``.
    """
    text = script_path.read_text(encoding="utf-8")
    rewritten = re.sub(
        r'os\.path\.join\(\s*os\.path\.dirname\(__file__\)\s*,\s*"\.\."\s*,\s*"src"\s*,\s*"drivers"\s*,\s*"[^"]+"\s*,\s*("[^"]+\.rs")\s*\)',
        r'os.path.join(os.path.dirname(__file__), "..", \1)',
        text,
    )
    if rewritten != text:
        script_path.write_text(rewritten, encoding="utf-8")
        print(
            f"[ik_generator]   Normalized output paths in {script_path.name} "
            "for stand-alone driver layout."
        )


def run_py_scripts(
    generator_scripts: dict[str, Path], output_dir: Path
) -> Optional[Tuple[str, str]]:
    """Run all ``generate_*.py`` files that were written to ``codegen/``.

    Each script is executed with the current Python interpreter from the
    generated crate root so that relative paths inside the scripts resolve
    correctly.

    :param generator_scripts: The ``{filename: path}`` dict returned by
        :func:`write_generated_files`.
    :returns: ``(script_name, error_output)`` when a script fails, else ``None``.
    """

    for filename, path in sorted(generator_scripts.items()):
        if not filename.startswith("generate_") or not filename.endswith(".py"):
            continue

        _normalize_helper_script_paths(path)
        # Ensure the driver directory exists for scripts that write ``../foo.rs``.
        (path.parent.parent).mkdir(parents=True, exist_ok=True)

        print(f"[ik_generator]   Running {path.relative_to(output_dir)} …")
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            cwd=str(output_dir),
        )
        if result.returncode != 0:
            error = (result.stderr + result.stdout).strip()
            print(f"[ik_generator]   ✗ {filename} failed:\n{error}")
            return (filename, error)
        print(f"[ik_generator]   ✓ {filename} succeeded.")

    return None


def build_rust_crate(cwd: Path = _REPO_ROOT) -> Tuple[bool, str]:
    """Run ``cargo build`` from *cwd*.

    :returns: ``(success, output)`` where *output* is combined stdout+stderr.
    """
    result = subprocess.run(
        ["cargo", "build"],
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )
    output = (result.stderr + result.stdout).strip()
    return result.returncode == 0, output


def test_rust_crate(cwd: Path = _REPO_ROOT) -> Tuple[bool, str]:
    """Run library unit tests from *cwd* (``cargo test --lib``).

    Doctests are disabled on stand-alone crates (see ``doctest = false``); the
    IK/dynamics gate is the ``#[cfg(test)]`` suite.

    :returns: ``(success, output)`` where *output* is combined stdout+stderr.
    """
    result = subprocess.run(
        ["cargo", "test", "--lib"],
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )
    output = (result.stderr + result.stdout).strip()
    return result.returncode == 0, output


def build_and_verify(
    generator_scripts: dict[str, Path],
    output_dir: Path,
) -> Tuple[bool, str]:
    """Run helper scripts, cargo build, and cargo test.

    :returns: ``error_message`` or None.
    """
    # --- Step 7: Run Python helper scripts ---
    print("[ik_generator] Step 7: Running Python helper scripts …")
    result = run_py_scripts(generator_scripts, output_dir)

    if result is not None:
        name, err = result
        error_msg = f"Script '{name}' failed with:\n{err}"
        print(f"[ik_generator]   Generator script {name} failed.")
        return error_msg

    # --- Step 8: Build the Rust crate ---
    print("[ik_generator] Step 8: Building Rust crate …")
    build_ok, build_output = build_rust_crate(output_dir)
    if not build_ok:
        error_msg = (
            f"cargo build failed with the following errors:\n\n{build_output}\n\n"
        )
        print("[ik_generator]   Build failed.")
        return error_msg
    print("[ik_generator]   Build succeeded.")

    # --- Step 9: Run Rust tests ---
    print("[ik_generator] Step 9: Running Rust tests …")
    test_ok, test_output = test_rust_crate(output_dir)
    if not test_ok:
        error_msg = (
            f"cargo test failed with the following output:\n\n{test_output}\n\n"
        )
        print("[ik_generator]   Tests failed.")
        return error_msg
    print("[ik_generator]   All tests passed.")

    return None
