"""Build and test module for generate_ik.

This module handles:
- Parsing multi-file LLM output
- Writing generated files to disk
- Running Python helper scripts
- Building and testing the Rust crate
"""

from __future__ import annotations

import shutil
import re
import subprocess
import sys
from pathlib import Path
from typing import Tuple

# Repository root — all relative paths are anchored here.
_CODEGEN_DIR = Path(__file__).resolve().parent   # codegen/
_REPO_ROOT = _CODEGEN_DIR.parent                 # repository root


def _sanitize_file_content(content: str) -> str:
    """Strip Markdown code fences from a single file's content block.

    :param content: Raw content string that may be wrapped in ``` fences.
    :returns: Clean code with no Markdown artefacts.
    """
    # Extract content from ``` or ```rust / ```python fences if present
    blocks = re.findall(r"```(?:\w+)?[ \t]*\n?(.*?)```", content, flags=re.DOTALL)
    if blocks:
        return "\n".join(block.strip() for block in blocks).strip()
    # No fences: strip any stray fence opener lines
    cleaned = re.sub(r"^```[^\n]*\n?", "", content, flags=re.MULTILINE)
    return cleaned.strip()


def parse_llm_output(llm_response: str) -> dict[str, str]:
    """Parse a multi-file LLM response into a ``{filename: content}`` mapping.

    The LLM is instructed to use ``=== FILE: <name> ===`` separator lines.
    This function splits on those markers and returns all contained files.

    Any Markdown code fences inside individual file blocks are stripped.

    :param llm_response: Raw LLM response text.
    :returns: Ordered mapping of filename → file content strings.
    :raises ValueError: If no ``=== FILE: ... ===`` markers are found.
    """
    pattern = re.compile(r"^=== FILE:\s*(.+?)\s*===$", re.MULTILINE)
    parts = pattern.split(llm_response)
    # parts[0] is any preamble text before the first marker (discard).
    # Remaining parts alternate: [filename, content, filename, content, ...]
    if len(parts) < 3:
        raise ValueError(
            "LLM output does not contain any '=== FILE: <name> ===' markers. "
            "The model may not have followed the output format instructions.\n"
            f"Response preview (first 500 chars):\n{llm_response[:500]}"
        )

    files: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        filename = parts[i].strip()
        raw_content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        files[filename] = _sanitize_file_content(raw_content)

    return files


def _update_drivers_mod(arm_name: str) -> None:
    """Add the new arm module declaration to ``src/drivers/mod.rs`` if absent.

    :param arm_name: Clean arm name (e.g. ``"ur5"``, ``"my_arm"``).
    """
    mod_path = _REPO_ROOT / "src" / "drivers" / "mod.rs"
    if not mod_path.exists():
        return
    content = mod_path.read_text(encoding="utf-8")
    declaration = f"pub mod {arm_name};"
    if declaration not in content:
        content = content.rstrip() + "\n" + declaration + "\n"
        mod_path.write_text(content, encoding="utf-8")
        print(f"[codegen:ik]   Added '{declaration}' to {mod_path.relative_to(_REPO_ROOT)}")


def _ensure_arm_mod_rs(arm_name: str, rust_files: list[str]) -> None:
    """Create ``src/drivers/{arm_name}/mod.rs`` if missing.

    Declares ``ik_solver`` as public and any other ``.rs`` files (except
    ``mod.rs`` and ``test_data.rs``) as private modules.

    :param arm_name: Clean arm name.
    :param rust_files: List of Rust filenames written to the driver directory.
    """
    mod_path = _REPO_ROOT / "src" / "drivers" / arm_name / "mod.rs"
    if mod_path.exists():
        return  # Don't overwrite an existing mod.rs

    lines: list[str] = [f"//! {arm_name} IK driver.\n"]
    lines.append("")
    lines.append("pub mod ik_solver;")
    for fname in sorted(rust_files):
        stem = Path(fname).stem
        if stem in ("ik_solver", "mod", "test_data"):
            continue
        lines.append(f"mod {stem};")
    mod_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[codegen:ik]   Created {mod_path.relative_to(_REPO_ROOT)}"
    )


def copy_driver_provenance(
    dest_dir: Path,
    arm_name: str,
    target_py_code: str,
    source_urdf: Path | None = None,
) -> dict[str, Path]:
    """Copy the Python artifact and optional URDF into a driver directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    py_path = dest_dir / f"{arm_name}_ik.py"
    py_path.write_text(target_py_code, encoding="utf-8")
    written["driver_py"] = py_path

    if source_urdf is not None:
        urdf_path = dest_dir / f"{arm_name}.urdf"
        if source_urdf.resolve() != urdf_path.resolve():
            shutil.copyfile(source_urdf, urdf_path)
        written["driver_urdf"] = urdf_path

    return written


def write_generated_files(
    output_dir: Path,
    arm_name: str,
    files: dict[str, str],
    target_py_code: str,
    context: str,
    source_urdf: Path | None = None,
) -> dict[str, Path]:
    """Write all LLM-generated files to the appropriate locations.

    * ``*.rs`` files → ``src/drivers/{arm_name}/``
    * The original Python artifact (and URDF, when available) → the driver dir
    * ``generate_*.py`` files → ``codegen/``
    * The original Python artifact and context dump → *output_dir*

    :param output_dir: Working output directory for the artifact and context.
    :param arm_name: Clean arm name (e.g. ``"my_arm"``).
    :param files: Parsed ``{filename: content}`` from :func:`parse_llm_output`.
    :param target_py_code: Source of the target Python artifact.
    :param context: Assembled prompt context string.
    :returns: Mapping of logical name → written :class:`Path`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    driver_dir = _REPO_ROOT / "src" / "drivers" / arm_name
    driver_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    # Write artifact and context to output_dir
    py_path = output_dir / f"{arm_name}_ik.py"
    ctx_path = output_dir / f"{arm_name}_context.txt"
    py_path.write_text(target_py_code, encoding="utf-8")
    ctx_path.write_text(context, encoding="utf-8")
    written["py"] = py_path
    written["ctx"] = ctx_path

    provenance_written = copy_driver_provenance(
        driver_dir, arm_name, target_py_code, source_urdf
    )
    written.update(provenance_written)

    rust_files_written: list[str] = []
    for filename, content in files.items():
        if filename.endswith(".rs"):
            dest = driver_dir / filename
            dest.write_text(content, encoding="utf-8")
            written[filename] = dest
            rust_files_written.append(filename)
            print(f"[codegen:ik]   Wrote {dest.relative_to(_REPO_ROOT)}")
        elif filename.endswith(".py"):
            dest = _REPO_ROOT / "codegen" / filename
            dest.write_text(content, encoding="utf-8")
            written[filename] = dest
            print(f"[codegen:ik]   Wrote {dest.relative_to(_REPO_ROOT)}")
        else:
            # Fallback: write to output_dir
            dest = output_dir / filename
            dest.write_text(content, encoding="utf-8")
            written[filename] = dest
            print(f"[codegen:ik]   Wrote {dest}")

    # Ensure mod.rs exists for the new driver
    _ensure_arm_mod_rs(arm_name, rust_files_written)

    # Register the arm module in the top-level drivers mod.rs
    _update_drivers_mod(arm_name)

    return written


def run_py_scripts(written_files: dict[str, Path]) -> list[tuple[str, str]]:
    """Run all ``generate_*.py`` files that were written to ``codegen/``.

    Each script is executed with the current Python interpreter from the
    repository root so that relative paths inside the scripts resolve correctly.

    :param written_files: The ``{filename: path}`` dict returned by
        :func:`write_generated_files`.
    :returns: List of ``(script_name, error_output)`` pairs for failed scripts.
        An empty list means all scripts succeeded.
    """
    failures: list[tuple[str, str]] = []

    for filename, path in sorted(written_files.items()):
        if not filename.startswith("generate_") or not filename.endswith(".py"):
            continue

        print(f"[codegen:ik]   Running {path.relative_to(_REPO_ROOT)} …")
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        if result.returncode != 0:
            error = (result.stderr + result.stdout).strip()
            print(f"[codegen:ik]   ✗ {filename} failed:\n{error}")
            failures.append((filename, error))
        else:
            print(f"[codegen:ik]   ✓ {filename} succeeded.")

    return failures


def build_rust_crate() -> Tuple[bool, str]:
    """Run ``cargo build`` from the repository root.

    :returns: ``(success, output)`` where *output* is combined stdout+stderr.
    """
    result = subprocess.run(
        ["cargo", "build"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    output = (result.stderr + result.stdout).strip()
    return result.returncode == 0, output


def test_rust_crate() -> Tuple[bool, str]:
    """Run ``cargo test`` from the repository root.

    :returns: ``(success, output)`` where *output* is combined stdout+stderr.
    """
    result = subprocess.run(
        ["cargo", "test"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    output = (result.stderr + result.stdout).strip()
    return result.returncode == 0, output
