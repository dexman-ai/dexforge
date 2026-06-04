#!/usr/bin/env python3
"""ik_generator.py — Generate ``#![no_std]`` Rust IK drivers from ssik Python artifacts.

This script generates (or loads) a per-arm ssik Inverse Kinematics Python
artifact, translates it into deterministic, zero-allocation Rust code via an
LLM API, runs the companion Python helper scripts to produce auxiliary Rust
files (``consts.rs``, ``pq.rs``, ``test_data.rs``), then builds and tests the
Rust crate.  Any build or test failure is fed back to the LLM so it can correct
the generated code automatically.

Usage examples::

    # From a ssik prebuilt artifact name (looks up ssik.prebuilt):
    python codegen/ik_generator.py --artifact ur5_ik --output-dir out/

    # From a local artifact .py file:
    python codegen/ik_generator.py --artifact /path/to/my_arm_ik.py --output-dir out/

    # Generate a fresh artifact from a URDF, then translate:
    python codegen/ik_generator.py --urdf robot.urdf --base base_link --ee ee_link \\
        --output-dir out/

    # Dump the full context to disk without invoking an LLM:
    python codegen/ik_generator.py --artifact ur5_ik --dump-context --output-dir out/

Environment variables for offline / air-gapped LLM support::

    DEXFORGE_LLM_KEY       API key (defaults to "nokeyneeded" for local models)
    DEXFORGE_LLM_BASE_URL  OpenAI-compatible base URL (e.g. http://localhost:11434/v1)
    DEXFORGE_LLM_MODEL     Model name (e.g. "qwen2.5-coder:32b", "gpt-4o")
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Repository root — all relative paths are anchored here.
# ---------------------------------------------------------------------------

_CODEGEN_DIR = Path(__file__).resolve().parent   # codegen/
_REPO_ROOT = _CODEGEN_DIR.parent                 # repository root

# Maximum number of LLM retry attempts when scripts, build, or tests fail.
_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Step 0: Obtain source Python artifact
# ---------------------------------------------------------------------------


def load_prebuilt_artifact(artifact_name: str) -> Tuple[str, str]:
    """Load a ssik prebuilt artifact by name or file path.

    :param artifact_name: Either a bare module name (e.g. ``"ur5_ik"``) to
        look up inside ``ssik.prebuilt``, or an absolute / relative path to an
        existing ``*_ik.py`` file.
    :returns: ``(source_code, module_stem)`` where *module_stem* is the bare
        name without the ``_ik`` suffix (e.g. ``"ur5"``).
    :raises FileNotFoundError: If a path argument does not exist on disk.
    :raises ImportError: If a module name is not found in ``ssik.prebuilt``.
    """
    candidate = Path(artifact_name)
    if candidate.exists():
        source_code = candidate.read_text(encoding="utf-8")
        stem = candidate.stem  # e.g. "ur5_ik"
        return source_code, stem

    # Try ssik.prebuilt.<artifact_name>
    module_name = artifact_name if not artifact_name.endswith(".py") else artifact_name[:-3]
    try:
        mod = importlib.import_module(f"ssik.prebuilt.{module_name}")
    except ModuleNotFoundError:
        raise ImportError(
            f"Cannot find '{artifact_name}' as a file or as ssik.prebuilt.{module_name}. "
            "Install ssik (`pip install ssik`) or provide a path to the artifact file."
        )
    import inspect

    source_code = inspect.getsource(mod)
    return source_code, module_name


def generate_artifact_from_urdf(
    urdf_path: Path,
    base_link: str,
    ee_link: str,
    output_dir: Path,
) -> Tuple[str, str]:
    """Generate a ssik artifact from a URDF using ``ssik build``.

    :param urdf_path: Path to the URDF file.
    :param base_link: Name of the base link in the URDF.
    :param ee_link: Name of the end-effector link in the URDF.
    :param output_dir: Directory where the generated ``*_ik.py`` artifact is
        written.
    :returns: ``(source_code, module_stem)``.
    :raises RuntimeError: If ``ssik build`` exits with a non-zero status.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = urdf_path.stem  # e.g. "ur5" from "ur5.urdf"
    out_path = output_dir / f"{stem}_ik.py"

    # Prefer the Python API (avoids shell PATH lookup and subprocess env issues).
    try:
        from ssik.cli import main as _ssik_main
    except ImportError:
        raise ImportError(
            "The 'ssik' package is required to build artifacts from URDF files. "
            "Install it with: pip install ssik urchin"
        )

    saved_argv = sys.argv[:]
    try:
        sys.argv = [
            "ssik",
            "build",
            str(urdf_path),
            "--base",
            base_link,
            "--ee",
            ee_link,
            "--out",
            str(out_path),
            "--no-validate",
        ]
        ret = _ssik_main()
    finally:
        sys.argv = saved_argv

    if ret != 0:
        raise RuntimeError(
            f"ssik build returned exit code {ret}. "
            "Check the output above for details."
        )
    if not out_path.exists():
        raise RuntimeError(
            f"ssik build reported success but the output file '{out_path}' was not created."
        )
    source_code = out_path.read_text(encoding="utf-8")
    return source_code, f"{stem}_ik"


# ---------------------------------------------------------------------------
# Step 1: Dynamic Template Sniffing
# ---------------------------------------------------------------------------


def find_best_example(target_py_code: str) -> Tuple[str, int]:
    """Dynamically determine the kinematic family and maximum IK solution count.

    Inspects the target Python artifact's source code for DOF and solver
    signatures to select the most appropriate existing Rust example and the
    ``MAX_SOLUTIONS`` constant for the ``solve()`` return type.

    :param target_py_code: Full source text of the target ``*_ik.py`` artifact.
    :returns: ``(example_arm_name, max_solutions)`` where *example_arm_name*
        matches a sub-directory of ``src/drivers/`` (e.g. ``"ur5"``,
        ``"piper"``, ``"franka_panda"``).
    :raises ValueError: If the artifact's kinematic topology is not recognised.
    """
    # Use word-boundary patterns so bare numbers (e.g. "128", "16") are not
    # confused with the same digits appearing inside variable names such as
    # "_pq_x128" or floating-point coefficients.
    # 128 = MAX_SOLUTIONS for 7-DOF jointlock arms (16 lock samples × 8 sub-solutions)
    # 16  = MAX_SOLUTIONS for general 6R (Raghavan-Roth polynomial system)
    _has_128 = bool(re.search(r"\b128\b", target_py_code))
    _has_16 = bool(re.search(r"\b16\b", target_py_code))

    # ---- 7-DOF Arms ----
    if "DOF = 7" in target_py_code or _has_128:
        if "husty_pfurner" in target_py_code or "polynomial" in target_py_code:
            # Category 4: Non-SRS 7-DOF (Flexiv, xArm7)
            # Requires the Husty-Pfurner polynomial loop template.
            return "xarm7", 256
        else:
            # Category 3: Canonical SRS / jointlock 7-DOF (Franka, KUKA, xArm7)
            # Requires the standard Joint-Lock + SP6 template.
            return "franka_panda", 128

    # ---- 6-DOF Arms ----
    elif "DOF = 6" in target_py_code or _has_16:
        if "raghavan_roth" in target_py_code or "_ssik_solve_x2_roots_mobius" in target_py_code:
            # Category 2: General 6R (Piper, JACO)
            # Requires the Raghavan-Roth template.
            return "piper", 16
        else:
            # Category 1: Pieper-compliant 6-DOF (UR5, Puma, ABB, Fanuc)
            # Requires the fast SP6 template.
            return "ur5", 8

    else:
        raise ValueError(
            "Unknown kinematic topology generated by SSIK. "
            "Artifact does not match standard 6R or 7R signatures. "
            "Check that 'DOF = 6' or 'DOF = 7' is present in the artifact."
        )


# ---------------------------------------------------------------------------
# Step 2: Dynamic Signature Extraction
# ---------------------------------------------------------------------------


def extract_rust_signatures(rust_file_path: Path) -> str:
    """Extract ``pub fn`` signatures and their ``///`` docstrings from a Rust file.

    Parses the file with regex to pull out only the public API surface:
    docstring comment blocks immediately preceding each ``pub fn`` declaration,
    and the full signature from ``pub fn`` up to the opening ``{`` or ``;``.
    Generics and multi-line signatures are captured as long as the opening
    brace or semicolon is on the same logical block.

    .. note::
        This is a best-effort regex approach.  It may miss signatures with
        deeply nested generics, multi-line ``where`` clauses that themselves
        contain ``{``/``;`` characters, or ``#[...]`` attribute lines that
        span multiple lines.  For the ssik codebase these cases do not arise.

    :param rust_file_path: Absolute path to the Rust source file.
    :returns: A formatted string of ``(docstring, signature)`` pairs, suitable
        for insertion into the LLM prompt context.
    :raises FileNotFoundError: If the Rust file does not exist.
    """
    if not rust_file_path.exists():
        raise FileNotFoundError(f"Rust source file not found: {rust_file_path}")

    source = rust_file_path.read_text(encoding="utf-8")

    # Match a run of /// doc-comment lines optionally followed by #[...] attrs,
    # then the pub fn declaration up to the opening brace or semicolon.
    pattern = re.compile(
        r"((?:[ \t]*///[^\n]*\n)+)"      # group 1: block of /// lines
        r"(?:[ \t]*#\[[^\]]*\]\n)*"       # optional #[...] attribute lines
        r"([ \t]*pub fn [^{;]+(?:\{|;))", # group 2: pub fn ... { or ;
        re.MULTILINE,
    )

    results: list[str] = []
    for match in pattern.finditer(source):
        doc_block = match.group(1).rstrip()
        sig_line = match.group(2).rstrip()
        # Strip trailing opening brace to get a clean signature
        sig_clean = sig_line.rstrip("{").rstrip().rstrip(";").rstrip()
        results.append(f"{doc_block}\n{sig_clean}")

    if not results:
        return f"// No public function signatures found in {rust_file_path.name}\n"

    header = f"// === Signatures from {rust_file_path.name} ===\n\n"
    return header + "\n\n".join(results) + "\n"


# ---------------------------------------------------------------------------
# Step 3: Context and Prompt Assembly
# ---------------------------------------------------------------------------


def _load_example_py_source(example_arm: str) -> str:
    """Load the ssik Python source artifact for *example_arm* from ssik.prebuilt."""
    try:
        mod = importlib.import_module(f"ssik.prebuilt.{example_arm}_ik")
        import inspect

        return inspect.getsource(mod)
    except ModuleNotFoundError:
        return f"# ssik.prebuilt.{example_arm}_ik is not installed — skipped.\n"


def _load_example_rust_source(example_arm: str) -> str:
    """Load the translated Rust IK driver for *example_arm* from src/drivers/."""
    rs_path = _REPO_ROOT / "src" / "drivers" / example_arm / "ik_solver.rs"
    if rs_path.exists():
        return rs_path.read_text(encoding="utf-8")
    return (
        f"// src/drivers/{example_arm}/ik_solver.rs does not exist yet — "
        "no Rust example available for this kinematic family.\n"
    )


def _load_example_scripts(example_arm: str) -> str:
    """Load the Python helper scripts for *example_arm* from scripts/, if present."""
    scripts_dir = _REPO_ROOT / "scripts"
    arm_scripts: list[str] = []
    for pattern in (
        f"generate_{example_arm}_consts.py",
        f"generate_{example_arm}_pq.py",
        f"generate_{example_arm}_test_data.py",
    ):
        path = scripts_dir / pattern
        if path.exists():
            arm_scripts.append(f"--- {pattern} ---\n" + path.read_text(encoding="utf-8"))
    if not arm_scripts:
        return f"# No Python helper scripts found for '{example_arm}' in scripts/ — skipped.\n"
    return "\n\n".join(arm_scripts)


def assemble_context(
    target_py_code: str,
    target_arm_stem: str,
    example_arm: str,
    max_solutions: int,
) -> str:
    """Build the complete context string that will be sent to the LLM.

    Loads all required files at runtime: the prompt template, the target
    Python artifact, the Rust API headers for ``ssik_core`` and ``kinematics``,
    and the one-shot example (Python + Rust pair plus Python helper scripts).

    :param target_py_code: Source text of the target ``*_ik.py`` artifact.
    :param target_arm_stem: Base name of the target arm (e.g. ``"ur5"``).
    :param example_arm: Name of the reference arm selected by
        :func:`find_best_example` (e.g. ``"franka_panda"``).
    :param max_solutions: Maximum number of IK solutions (used in the prompt).
    :returns: The full assembled prompt + context as a single string.
    :raises FileNotFoundError: If the prompt template is missing.
    """
    # --- Prompt template ---
    prompt_path = _CODEGEN_DIR / "translate_ssik_artifact.md"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt template not found at '{prompt_path}'. "
            "Ensure the repository is checked out with the codegen/ directory."
        )
    prompt_template = prompt_path.read_text(encoding="utf-8")

    # --- Rust API headers ---
    ssik_core_sigs = extract_rust_signatures(
        _REPO_ROOT / "src" / "math" / "ssik_core.rs"
    )
    kinematics_sigs = extract_rust_signatures(
        _REPO_ROOT / "src" / "math" / "kinematics.rs"
    )

    # --- One-shot example (Python source + Rust translation + helper scripts) ---
    example_py = _load_example_py_source(example_arm)
    example_rs = _load_example_rust_source(example_arm)
    example_scripts = _load_example_scripts(example_arm)

    # --- Assemble ---
    sections = [
        "=" * 72,
        "DEXFORGE IK GENERATOR — FULL CONTEXT",
        "=" * 72,
        "",
        prompt_template,
        "",
        "=" * 72,
        f"TARGET ARM: {target_arm_stem}   |   MAX_SOLUTIONS = {max_solutions}   |   EXAMPLE ARM: {example_arm}",
        "=" * 72,
        "",
        "--- [SECTION 1] REFERENCE: ssik_core.rs PUBLIC API ---",
        "",
        ssik_core_sigs,
        "",
        "--- [SECTION 2] REFERENCE: kinematics.rs PUBLIC API ---",
        "",
        kinematics_sigs,
        "",
        f"--- [SECTION 3] ONE-SHOT EXAMPLE: Python artifact for '{example_arm}' ---",
        "",
        example_py,
        "",
        f"--- [SECTION 4] ONE-SHOT EXAMPLE: Rust translation for '{example_arm}' ---",
        "",
        example_rs,
        "",
        f"--- [SECTION 5] ONE-SHOT EXAMPLE: Python helper scripts for '{example_arm}' ---",
        "",
        example_scripts,
        "",
        f"--- [SECTION 6] TARGET: Python artifact to translate ('{target_arm_stem}') ---",
        "",
        target_py_code,
        "",
        "=" * 72,
        "END OF CONTEXT — PRODUCE ALL FILES USING THE === FILE: <name> === FORMAT",
        "=" * 72,
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Step 4: LLM Invocation
# ---------------------------------------------------------------------------


class IKGeneratorLLM:
    """Thin wrapper around the OpenAI-compatible chat API for IK generation.

    Supports local, air-gapped models (vLLM, Ollama, LM Studio, …) via
    environment variables::

        DEXFORGE_LLM_KEY       API key (default: "nokeyneeded")
        DEXFORGE_LLM_BASE_URL  Base URL (default: https://api.openai.com/v1)
        DEXFORGE_LLM_MODEL     Model name (default: "gpt-4o")

    :param api_key: Override ``DEXFORGE_LLM_KEY``.
    :param base_url: Override ``DEXFORGE_LLM_BASE_URL``.
    :param model: Override ``DEXFORGE_LLM_MODEL``.
    """

    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    SYSTEM_PROMPT = (
        "You are an expert Rust systems programmer and robotics engineer. "
        "Translate the provided ssik Python IK artifact to #![no_std] Rust "
        "exactly as instructed. "
        "Also generate the required Python helper scripts. "
        "Output ALL files using the exact '=== FILE: <filename> ===' separator "
        "format described in the prompt. Do not include any conversational text."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for LLM generation. "
                "Install it with: pip install openai"
            ) from exc

        self._api_key = api_key or os.environ.get("DEXFORGE_LLM_KEY", "nokeyneeded")
        self._base_url = base_url or os.environ.get(
            "DEXFORGE_LLM_BASE_URL", self.DEFAULT_BASE_URL
        )
        self._model = model or os.environ.get("DEXFORGE_LLM_MODEL", self.DEFAULT_MODEL)

        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def generate(self, messages: list[dict]) -> str:
        """Send a conversation *messages* list to the LLM and return the raw response.

        Temperature is fixed at ``0.0`` (greedy decoding) to ensure
        deterministic output across identical inputs.

        :param messages: OpenAI-style chat messages list (system + user turns).
        :returns: The LLM's response as a raw string.
        :raises RuntimeError: If the API call fails or returns an empty choice.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            messages=messages,
        )
        choices = response.choices
        if not choices:
            raise RuntimeError("LLM returned an empty response (no choices).")
        content = choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned a None message content.")
        return content


# ---------------------------------------------------------------------------
# Step 5: Parse Multi-File LLM Output
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Step 6: Write Output Files
# ---------------------------------------------------------------------------


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
        print(f"[ik_generator]   Added '{declaration}' to {mod_path.relative_to(_REPO_ROOT)}")


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
        f"[ik_generator]   Created {mod_path.relative_to(_REPO_ROOT)}"
    )


def write_generated_files(
    output_dir: Path,
    arm_name: str,
    files: dict[str, str],
    target_py_code: str,
    context: str,
) -> dict[str, Path]:
    """Write all LLM-generated files to the appropriate locations.

    * ``*.rs`` files → ``src/drivers/{arm_name}/``
    * ``generate_*.py`` files → ``scripts/``
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

    rust_files_written: list[str] = []
    for filename, content in files.items():
        if filename.endswith(".rs"):
            dest = driver_dir / filename
            dest.write_text(content, encoding="utf-8")
            written[filename] = dest
            rust_files_written.append(filename)
            print(f"[ik_generator]   Wrote {dest.relative_to(_REPO_ROOT)}")
        elif filename.endswith(".py"):
            dest = _REPO_ROOT / "scripts" / filename
            dest.write_text(content, encoding="utf-8")
            written[filename] = dest
            print(f"[ik_generator]   Wrote {dest.relative_to(_REPO_ROOT)}")
        else:
            # Fallback: write to output_dir
            dest = output_dir / filename
            dest.write_text(content, encoding="utf-8")
            written[filename] = dest
            print(f"[ik_generator]   Wrote {dest}")

    # Ensure mod.rs exists for the new driver
    _ensure_arm_mod_rs(arm_name, rust_files_written)

    # Register the arm module in the top-level drivers mod.rs
    _update_drivers_mod(arm_name)

    return written


# ---------------------------------------------------------------------------
# Step 7: Run Python Helper Scripts
# ---------------------------------------------------------------------------


def run_py_scripts(written_files: dict[str, Path]) -> list[tuple[str, str]]:
    """Run all ``generate_*.py`` files that were written to ``scripts/``.

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

        print(f"[ik_generator]   Running {path.relative_to(_REPO_ROOT)} …")
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        if result.returncode != 0:
            error = (result.stderr + result.stdout).strip()
            print(f"[ik_generator]   ✗ {filename} failed:\n{error}")
            failures.append((filename, error))
        else:
            print(f"[ik_generator]   ✓ {filename} succeeded.")

    return failures


# ---------------------------------------------------------------------------
# Step 8: Build and Test the Rust Crate
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ik_generator",
        description=(
            "Generate a #![no_std] Rust IK driver from a ssik Python artifact using "
            "an LLM.  Provide either --artifact (prebuilt name or file path) or "
            "--urdf (with --base and --ee)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--artifact",
        metavar="NAME_OR_PATH",
        help=(
            "ssik prebuilt module name (e.g. 'ur5_ik') or path to an existing "
            "*_ik.py artifact file."
        ),
    )
    source.add_argument(
        "--urdf",
        metavar="URDF_PATH",
        type=Path,
        help="Path to a URDF file.  Requires --base and --ee.",
    )

    parser.add_argument(
        "--base",
        metavar="LINK",
        help="Base link name (required with --urdf).",
    )
    parser.add_argument(
        "--ee",
        metavar="LINK",
        help="End-effector link name (required with --urdf).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        metavar="DIR",
        type=Path,
        default=Path("out"),
        help="Output directory for the Python artifact and context dump (default: out/).",
    )
    parser.add_argument(
        "--dump-context",
        action="store_true",
        help=(
            "Assemble and write the full prompt + context to disk without invoking "
            "the LLM.  Use this to feed the context to an interactive model manually."
        ),
    )
    parser.add_argument(
        "--max-retries",
        metavar="N",
        type=int,
        default=_MAX_RETRIES,
        help=(
            f"Maximum number of LLM retry attempts when scripts, build, or tests "
            f"fail (default: {_MAX_RETRIES})."
        ),
    )
    parser.add_argument(
        "--llm-key",
        metavar="KEY",
        help="LLM API key (overrides DEXFORGE_LLM_KEY env var).",
    )
    parser.add_argument(
        "--llm-url",
        metavar="URL",
        help="LLM base URL (overrides DEXFORGE_LLM_BASE_URL env var).",
    )
    parser.add_argument(
        "--llm-model",
        metavar="MODEL",
        help="LLM model name (overrides DEXFORGE_LLM_MODEL env var).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the IK generator CLI.

    :param argv: Command-line arguments (defaults to ``sys.argv[1:]``).
    :returns: Exit code (0 = success, non-zero = error).
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # --- Validate --urdf requires --base and --ee ---
    if args.urdf is not None:
        if not args.base or not args.ee:
            parser.error("--urdf requires both --base and --ee to be specified.")
        if not args.urdf.exists():
            parser.error(f"URDF file not found: {args.urdf}")

    # --- Step 0: Obtain source Python artifact ---
    print("[ik_generator] Step 0: Loading Python artifact …")
    try:
        if args.urdf is not None:
            target_py_code, artifact_stem = generate_artifact_from_urdf(
                args.urdf, args.base, args.ee, args.output_dir
            )
            print(f"[ik_generator]   Generated artifact from URDF: {artifact_stem}")
        else:
            target_py_code, artifact_stem = load_prebuilt_artifact(args.artifact)
            print(f"[ik_generator]   Loaded artifact: {artifact_stem}")
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        print(f"[ik_generator] ERROR: {exc}", file=sys.stderr)
        return 1

    arm_name = artifact_stem.removesuffix("_ik")

    # --- Step 1: Detect kinematic family ---
    print("[ik_generator] Step 1: Detecting kinematic family …")
    try:
        example_arm, max_solutions = find_best_example(target_py_code)
    except ValueError as exc:
        print(f"[ik_generator] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"[ik_generator]   → example arm: '{example_arm}', "
        f"MAX_SOLUTIONS = {max_solutions}"
    )

    # Fall back gracefully when the chosen example arm has no Rust driver yet
    rs_example = _REPO_ROOT / "src" / "drivers" / example_arm / "ik_solver.rs"
    if not rs_example.exists():
        fallback = "franka_panda"
        fallback_path = _REPO_ROOT / "src" / "drivers" / fallback / "ik_solver.rs"
        if fallback_path.exists():
            print(
                f"[ik_generator]   WARNING: No Rust driver for '{example_arm}'; "
                f"falling back to '{fallback}' as structural template."
            )
            example_arm = fallback
        else:
            print(
                f"[ik_generator]   WARNING: No Rust driver found for '{example_arm}' "
                "or fallback 'franka_panda'. The one-shot example will be empty.",
                file=sys.stderr,
            )

    # --- Step 2 (internal) + Step 3: Assemble context ---
    print("[ik_generator] Step 3: Assembling context …")
    try:
        context = assemble_context(target_py_code, arm_name, example_arm, max_solutions)
    except FileNotFoundError as exc:
        print(f"[ik_generator] ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[ik_generator]   Context assembled ({len(context):,} chars).")

    # --- Dump context and exit if requested ---
    if args.dump_context:
        print("[ik_generator] --dump-context: skipping LLM invocation.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        ctx_path = args.output_dir / f"{arm_name}_context.txt"
        ctx_path.write_text(context, encoding="utf-8")
        py_path = args.output_dir / f"{arm_name}_ik.py"
        py_path.write_text(target_py_code, encoding="utf-8")
        print(f"[ik_generator] ✓ Context dump : {ctx_path}")
        print(f"[ik_generator] ✓ Python artifact : {py_path}")
        return 0

    # --- Step 4: Invoke LLM with retry loop ---
    print("[ik_generator] Step 4: Invoking LLM …")
    try:
        llm = IKGeneratorLLM(
            api_key=args.llm_key,
            base_url=args.llm_url,
            model=args.llm_model,
        )
    except ImportError as exc:
        print(f"[ik_generator] ERROR: {exc}", file=sys.stderr)
        return 1

    messages: list[dict] = [
        {"role": "system", "content": IKGeneratorLLM.SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]

    max_retries = args.max_retries
    last_raw_response: Optional[str] = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"[ik_generator]   Retry {attempt}/{max_retries} …")

        try:
            raw_response = llm.generate(messages)
        except Exception as exc:  # noqa: BLE001 — surface all API errors
            print(
                f"[ik_generator] LLM error ({type(exc).__name__}): {exc}",
                file=sys.stderr,
            )
            return 1

        last_raw_response = raw_response
        print(f"[ik_generator]   LLM response: {len(raw_response):,} chars.")

        # --- Step 5: Parse multi-file output ---
        try:
            files = parse_llm_output(raw_response)
        except ValueError as exc:
            print(f"[ik_generator]   Parse error: {exc}", file=sys.stderr)
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your output did not follow the required '=== FILE: <name> ===' "
                        "separator format. Please regenerate all files using the exact "
                        "output format specified in the prompt."
                    ),
                })
                continue
            return 1

        print(
            f"[ik_generator]   Parsed {len(files)} file(s): "
            + ", ".join(files.keys())
        )

        # --- Step 6: Write generated files ---
        print("[ik_generator] Step 6: Writing generated files …")
        written = write_generated_files(
            args.output_dir, arm_name, files, target_py_code, context
        )

        # --- Step 7: Run Python helper scripts ---
        print("[ik_generator] Step 7: Running Python helper scripts …")
        script_failures = run_py_scripts(written)

        if script_failures:
            failure_details = "\n\n".join(
                f"Script '{name}' failed with:\n{err}"
                for name, err in script_failures
            )
            error_msg = (
                f"The following Python helper script(s) failed to run:\n\n"
                f"{failure_details}\n\n"
                "Please fix the errors and regenerate all files."
            )
            print(f"[ik_generator]   Script failures detected.")
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": error_msg})
                continue
            print(f"[ik_generator] ERROR: Script failures after all retries.", file=sys.stderr)
            return 1

        # --- Step 8: Build the Rust crate ---
        print("[ik_generator] Step 8: Building Rust crate …")
        build_ok, build_output = build_rust_crate()
        if not build_ok:
            error_msg = (
                f"cargo build failed with the following errors:\n\n{build_output}\n\n"
                "Please fix the Rust code and regenerate all files."
            )
            print(f"[ik_generator]   Build failed.")
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": error_msg})
                continue
            print(f"[ik_generator] ERROR: Build failed after all retries.", file=sys.stderr)
            print(build_output, file=sys.stderr)
            return 1
        print("[ik_generator]   Build succeeded.")

        # --- Step 9: Run Rust tests ---
        print("[ik_generator] Step 9: Running Rust tests …")
        test_ok, test_output = test_rust_crate()
        if not test_ok:
            error_msg = (
                f"cargo test failed with the following output:\n\n{test_output}\n\n"
                "Please fix the Rust code and regenerate all files."
            )
            print(f"[ik_generator]   Tests failed.")
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": error_msg})
                continue
            print(f"[ik_generator] ERROR: Tests failed after all retries.", file=sys.stderr)
            print(test_output, file=sys.stderr)
            return 1
        print("[ik_generator]   All tests passed.")

        # --- All steps succeeded ---
        break

    print(f"\n[ik_generator] ✓ Python artifact : {written.get('py')}")
    print(f"[ik_generator] ✓ Context dump    : {written.get('ctx')}")
    for filename, path in sorted(written.items()):
        if filename not in ("py", "ctx"):
            print(f"[ik_generator] ✓ {filename:<40} → {path.relative_to(_REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
