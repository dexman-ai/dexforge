"""Prompt generation module for ik_generator.

This module handles:
- Loading ssik Python artifacts (from prebuilt or file)
- Generating artifacts from URDF files
- Dynamic example selection based on kinematic family
- Extracting Rust API signatures
- Assembling the complete LLM context/prompt
"""

from __future__ import annotations

import importlib
import inspect
import re
import subprocess
import sys
import shutil
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

# Repository root — all relative paths are anchored here.
_CODEGEN_DIR = Path(__file__).resolve().parent   # codegen/
_REPO_ROOT = _CODEGEN_DIR.parent                 # repository root
if str(_CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_DIR))


class IKGenerationEntrypoint(NamedTuple):
    """Prepared inputs produced by the IK prompt entrypoint."""

    ssik_artifact: str
    arm_name: str
    example_arm: str
    max_solutions: int
    dof: int
    context: str


def _get_locked_ssik_version() -> str:
    """Read the locked ssik version from SSIK_VERSION.txt.

    :returns: Version string (e.g., "1.2.3").
    :raises FileNotFoundError: If SSIK_VERSION.txt is not found.
    """
    version_file = _REPO_ROOT / "SSIK_VERSION.txt"
    if not version_file.exists():
        raise FileNotFoundError(
            f"SSIK_VERSION.txt not found at {version_file}. "
            "Create it with the locked ssik version (e.g., 'ssik==1.2.3')."
        )
    content = version_file.read_text(encoding="utf-8").strip()
    # Parse "ssik==X.Y.Z" or "X.Y.Z" format
    if content.startswith("ssik=="):
        return content[6:]
    elif content.startswith("ssik "):
        return content[5:].split(",")[0].strip()
    return content


def _verify_ssik_version() -> str:
    """Verify that the installed ssik version matches the locked version.

    :returns: The locked ssik version string.
    :raises ImportError: If ssik is not installed or version doesn't match.
    """
    try:
        import ssik
    except ImportError:
        raise ImportError(
            "ssik package is not installed. "
            "Install it with: pip install ssik==<version> (see SSIK_VERSION.txt)"
        )

    locked_version = _get_locked_ssik_version()
    installed_version = getattr(ssik, "__version__", "unknown")

    if installed_version != "unknown" and installed_version != locked_version:
        raise ImportError(
            f"ssik version mismatch: installed={installed_version}, "
            f"locked={locked_version}. "
            f"Update your environment: pip install ssik=={locked_version}"
        )

    return locked_version


def try_load_prebuilt_artifact(artifact_name: str) -> Tuple[str, str]:
    """Load a ssik prebuilt artifact by name or file path.

    :param artifact_name: Either a bare module name (e.g. ``"ur5_ik"``) to
        look up inside ``ssik.prebuilt``, or an absolute / relative path to an
        existing ``*_ik.py`` file.
    :returns: ``(source_code, artifact_stem)`` where *artifact_stem* is:
        - For file paths: the file stem with the ``_ik`` suffix (e.g., ``"ur5_ik"``).
        - For module names: the module name as provided (e.g., ``"ur5_ik"``).
        Callers should use ``.removesuffix("_ik")`` to extract the arm name.
    :raises FileNotFoundError: If a path argument does not exist on disk.
    :raises ImportError: If a module name is not found in ``ssik.prebuilt`` or
        if the installed ssik version does not match the locked version.
    """
    candidate = Path(artifact_name)
    if candidate.exists():
        source_code = candidate.read_text(encoding="utf-8")
        stem = candidate.stem  # e.g. "ur5_ik"
        return source_code, stem

    # Verify ssik version before attempting to import prebuilt artifacts
    _verify_ssik_version()

    # Try ssik.prebuilt.<artifact_name>
    module_name = artifact_name if not artifact_name.endswith(".py") else artifact_name[:-3]
    try:
        mod = importlib.import_module(f"ssik.prebuilt.{module_name}")
    except ModuleNotFoundError:
        return None, None

    source_code = inspect.getsource(mod)
    return source_code, module_name


def generate_artifact_from_urdf(
    urdf_path: Path,
    base_link: str,
    ee_link: str,
    output_dir: Path,
) -> Tuple[str, str]:
    """Generate a ssik artifact from a URDF using ``ssik build``.

    Verifies that the installed ssik version matches the locked version
    before generating the artifact.

    :param urdf_path: Path to the URDF file.
    :param base_link: Name of the base link in the URDF.
    :param ee_link: Name of the end-effector link in the URDF.
    :param output_dir: Directory where the generated ``*_ik.py`` artifact is
        written; the input URDF is copied there as ``<stem>.urdf``.
    :returns: ``(source_code, module_stem)``.
    :raises RuntimeError: If ``ssik build`` exits with a non-zero status.
    :raises ImportError: If the installed ssik version does not match the locked version.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = urdf_path.stem  # e.g. "ur5" from "ur5.urdf"
    out_path = output_dir / f"{stem}_ik.py"

    # Verify ssik version before attempting to use ssik build
    _verify_ssik_version()

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


def find_best_example(ssik_artifact: str) -> Tuple[str, int]:
    """Dynamically determine the kinematic family and maximum IK solution count.

    Inspects the target Python artifact's source code for DOF and solver
    signatures to select the most appropriate existing Rust example and the
    ``MAX_SOLUTIONS`` constant for the ``solve()`` return type.

    :param ssik_artifact: Full source text of the target ``*_ik.py`` artifact.
    :returns: ``(example_arm_name, max_solutions, dof)`` where *example_arm_name*
        matches a known driver family (e.g. ``"ur5"``, ``"piper"``,
        ``"franka_panda"``).
    :raises ValueError: If the artifact's kinematic topology is not recognised.
    """
    # Use word-boundary patterns so bare numbers (e.g. "128", "16") are not
    # confused with the same digits appearing inside variable names such as
    # "_pq_x128" or floating-point coefficients.
    # 128 = MAX_SOLUTIONS for 7-DOF jointlock arms (16 lock samples × 8 sub-solutions)
    # 16  = MAX_SOLUTIONS for general 6R (Raghavan-Roth polynomial system)
    _has_128 = bool(re.search(r"\b128\b", ssik_artifact))
    _has_16 = bool(re.search(r"\b16\b", ssik_artifact))

    # ---- 7-DOF Arms ----
    if "DOF = 7" in ssik_artifact or _has_128:
        if "husty_pfurner" in ssik_artifact or "polynomial" in ssik_artifact:
            # Category 4: Non-SRS 7-DOF (Flexiv, xArm7)
            # Requires the Husty-Pfurner polynomial loop template.
            return "rizon4", 256, 7
        else:
            # Category 3: Canonical SRS / jointlock 7-DOF (Franka, KUKA, xArm7)
            # Requires the standard Joint-Lock + SP6 template.
            return "franka_panda", 128, 7

    # ---- 6-DOF Arms ----
    elif "DOF = 6" in ssik_artifact or _has_16:
        if "raghavan_roth" in ssik_artifact or "_ssik_solve_x2_roots_mobius" in ssik_artifact:
            # Category 2: General 6R (Piper, JACO)
            # Requires the Raghavan-Roth template.
            return "piper", 16, 6
        else:
            # Category 1: Pieper-compliant 6-DOF (UR5, Puma, ABB, Fanuc)
            # Requires the fast SP6 template (16-slot output array).
            return "ur5", 16, 6

    else:
        raise ValueError(
            "Unknown kinematic topology generated by SSIK. "
            "Artifact does not match standard 6R or 7R signatures. "
            "Check that 'DOF = 6' or 'DOF = 7' is present in the artifact."
        )


def _resolve_repo_driver_dir(repo_root: Path, arm: str) -> Optional[Path]:
    """Resolve a driver directory in the repo for an arm name."""
    try:
        from repo_paths import resolve_arm_driver_dir
    except ImportError:  # pragma: no cover - package import path
        from codegen.repo_paths import resolve_arm_driver_dir  # type: ignore

    return resolve_arm_driver_dir(repo_root, str(arm))


def generate_context(
    urdf_path: Path,
    base_link: Optional[str],
    ee_link: Optional[str],
    output_dir: Path,
    repo_root: Path,
    example_arm: Optional[str] = None,
) -> IKGenerationEntrypoint:
    """Main entrypoint for preparing IK generation inputs.

    Runs locked-version verification, Step 0 (artifact generation/loading),
    Step 1 (family detection), and Step 2 (context assembly).
    """
    locked_version = _get_locked_ssik_version()
    print(f"[ik_generator] Locked ssik version: {locked_version}")
    _verify_ssik_version()
    print("[ik_generator] ✓ ssik version verified")

    print("[ik_generator] Step 0: Loading Python artifact …")
    if not base_link or not ee_link:
        raise ValueError("--urdf requires both --base and --ee to be specified.")
    
    # Use a prebuilt artifact only when its BASE_LINK, EE_LINK, and source URDF
    # match the requested inputs; otherwise compile a fresh artifact from the URDF.
    stem_candidate = urdf_path.stem + "_ik"
    ssik_artifact, artifact_stem = try_load_prebuilt_artifact(stem_candidate)
    if ssik_artifact is not None and artifact_stem is not None:
        mb = re.search(r'^BASE_LINK\s*=\s*["\']([^"\']+)["\']', ssik_artifact, re.MULTILINE)
        me = re.search(r'^EE_LINK\s*=\s*["\']([^"\']+)["\']', ssik_artifact, re.MULTILINE)
        art_base = mb.group(1) if mb else None
        art_ee = me.group(1) if me else None
        repo_driver = _resolve_repo_driver_dir(repo_root, urdf_path.stem)
        repo_urdf = (repo_driver / f"{urdf_path.stem}.urdf") if repo_driver is not None else None
        same_urdf = (
            repo_urdf is not None
            and repo_urdf.is_file()
            and Path(urdf_path).read_bytes() == repo_urdf.read_bytes()
        )
        if art_base != base_link or art_ee != ee_link or not same_urdf:
            ssik_artifact, artifact_stem = None, None

    if ssik_artifact is None or artifact_stem is None:
        ssik_artifact, artifact_stem = generate_artifact_from_urdf(
            urdf_path, base_link, ee_link, output_dir
        )
        print(f"[ik_generator]   Generated artifact from URDF: {artifact_stem}")

    arm_name = artifact_stem.removesuffix("_ik")

    print("[ik_generator] Step 1: Detecting kinematic family …")
    best_guess_example_arm, max_solutions, dof = find_best_example(ssik_artifact)
    if example_arm is None:
        example_arm = best_guess_example_arm
    if _resolve_repo_driver_dir(repo_root, example_arm) is None:
        raise FileNotFoundError(
            f"missing example driver directory for arm {example_arm!r}"
        )
    print(
        f"[ik_generator]   → example arm: '{example_arm}', "
        f"MAX_SOLUTIONS = {max_solutions}, DOF = {dof}"
    )

    print("[ik_generator] Step 2: Assembling context …")
    context = assemble_context(
        ssik_artifact,
        arm_name,
        example_arm,
        max_solutions,
        repo_root,
    )
    print(f"[ik_generator]   Context assembled ({len(context):,} chars).")

    return IKGenerationEntrypoint(
        ssik_artifact=ssik_artifact,
        arm_name=arm_name,
        example_arm=example_arm,
        max_solutions=max_solutions,
        dof=dof,
        context=context,
    )


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
    # then the pub fn declaration up to a declaration terminator.
    #
    # Important: Rust type syntax can contain semicolons inside arrays
    # (e.g. &[f64; 3]), so we only treat '{' or ';' as terminators when they
    # appear at the end of a declaration line (before newline/EOF).
    pattern = re.compile(
        r"((?:[ \t]*///[^\n]*\n)+)"      # group 1: block of /// lines
        r"(?:[ \t]*#\[[^\]]*\]\n)*"       # optional #[...] attribute lines
        r"([ \t]*pub fn[\s\S]*?(?:\{|;)[ \t]*(?=\n|$))", # group 2: full pub fn decl
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


def _load_example_py_source(example_arm: str) -> str:
    """Load the ssik Python source artifact for *example_arm* from ssik.prebuilt."""
    try:
        mod = importlib.import_module(f"ssik.prebuilt.{example_arm}_ik")
        return inspect.getsource(mod)
    except ModuleNotFoundError:
        return f"# ssik.prebuilt.{example_arm}_ik is not installed — skipped.\n"


def _load_example_rust_source(example_arm: str, repo_root: Path) -> str:
    """Load the translated Rust IK driver for *example_arm* from the source tree."""
    driver_dir = _resolve_repo_driver_dir(repo_root, example_arm)
    rs_path = (driver_dir / "ik_solver.rs") if driver_dir is not None else None
    if rs_path is not None and rs_path.exists():
        return rs_path.read_text(encoding="utf-8")
    return (
        f"// driver source for {example_arm!r} does not exist yet — "
        "no Rust example available for this kinematic family.\n"
    )


def _load_example_scripts(example_arm: str, repo_root: Path) -> str:
    """Load the Python helper scripts for *example_arm* from the source tree.

    Looks for ``generate_test_data.py`` and ``generate_code.py`` under
    ``src/drivers/<example_arm>/codegen/``.
    """
    driver_dir = _resolve_repo_driver_dir(repo_root, example_arm)
    codegen_dir = (driver_dir / "codegen") if driver_dir is not None else None
    arm_scripts: list[str] = []
    for script_name in ("generate_test_data.py", "generate_code.py"):
        if codegen_dir is None:
            continue
        path = codegen_dir / script_name
        if path.exists():
            arm_scripts.append(f"--- {script_name} ---\n" + path.read_text(encoding="utf-8"))
    if not arm_scripts:
        return f"# No Python helper scripts found for '{example_arm}' in src/drivers/{example_arm}/codegen/ — skipped.\n"
    return "\n\n".join(arm_scripts)


def assemble_context(
    ssik_artifact: str,
    target_arm_stem: str,
    example_arm: str,
    max_solutions: int,
    repo_root: Path,
) -> str:
    """Build the complete context string that will be sent to the LLM.

    Loads all required files at runtime: the prompt template, the target
    Python artifact, the Rust API headers for ``ssik_core`` and ``kinematics``,
    and the one-shot example (Python + Rust pair plus Python helper scripts).

    :param ssik_artifact: Text of the source ``*_ik.py`` artifact being translated.
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
    try:
        from repo_paths import math_src
    except ImportError:  # pragma: no cover
        from codegen.repo_paths import math_src  # type: ignore

    math_dir = math_src(repo_root)
    ssik_core_sigs = extract_rust_signatures(math_dir / "ssik_core.rs")
    kinematics_sigs = extract_rust_signatures(math_dir / "kinematics.rs")

    # --- One-shot example (Python source + Rust translation + helper scripts) ---
    example_py = _load_example_py_source(example_arm)
    example_rs = _load_example_rust_source(example_arm, repo_root)
    example_scripts = _load_example_scripts(example_arm, repo_root)

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
        ssik_artifact,
        "",
        "=" * 72,
        "END OF CONTEXT — PRODUCE ALL FILES USING THE === FILE: <name> === FORMAT",
        "=" * 72,
    ]
    return "\n".join(sections)
