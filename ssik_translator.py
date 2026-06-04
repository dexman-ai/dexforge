#!/usr/bin/env python3
"""ssik_translator.py — Translate ssik Python IK artifacts to ``#![no_std]`` Rust.

This script generates (or loads) a per-arm ssik Inverse Kinematics Python
artifact and translates it into deterministic, zero-allocation Rust code
suitable for a real-time ``#![no_std]`` control loop, using an LLM API.

Usage examples::

    # From a ssik prebuilt artifact name (looks up ssik.prebuilt):
    python ssik_translator.py --artifact ur5_ik --output-dir out/

    # From a local artifact .py file:
    python ssik_translator.py --artifact /path/to/my_arm_ik.py --output-dir out/

    # Generate a fresh artifact from a URDF, then translate:
    python ssik_translator.py --urdf robot.urdf --base base_link --ee ee_link \\
        --output-dir out/

    # Dump the full context to disk without invoking an LLM
    # (use with an interactive model or for debugging):
    python ssik_translator.py --artifact ur5_ik --dump-context --output-dir out/

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
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Repository root — all relative paths are anchored here.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent

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
    Generics, ``where`` clauses, and multi-line signatures are all captured.

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


def assemble_context(
    target_py_code: str,
    target_arm_stem: str,
    example_arm: str,
    max_solutions: int,
) -> str:
    """Build the complete context string that will be sent to the LLM.

    Loads all required files at runtime: the prompt template, the target
    Python artifact, the Rust API headers for ``ssik_core`` and ``kinematics``,
    and the one-shot example (Python + Rust pair).

    :param target_py_code: Source text of the target ``*_ik.py`` artifact.
    :param target_arm_stem: Base name of the target arm (e.g. ``"ur5"``).
    :param example_arm: Name of the reference arm selected by
        :func:`find_best_example` (e.g. ``"franka_panda"``).
    :param max_solutions: Maximum number of IK solutions (used in the prompt).
    :returns: The full assembled prompt + context as a single string.
    :raises FileNotFoundError: If the prompt template is missing.
    """
    # --- Prompt template ---
    prompt_path = _REPO_ROOT / "prompts" / "translate_ssik_artifact.md"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt template not found at '{prompt_path}'. "
            "Ensure the repository is checked out with the prompts/ directory."
        )
    prompt_template = prompt_path.read_text(encoding="utf-8")

    # --- Rust API headers ---
    ssik_core_sigs = extract_rust_signatures(
        _REPO_ROOT / "src" / "math" / "ssik_core.rs"
    )
    kinematics_sigs = extract_rust_signatures(
        _REPO_ROOT / "src" / "math" / "kinematics.rs"
    )

    # --- One-shot example (Python source + Rust translation) ---
    example_py = _load_example_py_source(example_arm)
    example_rs = _load_example_rust_source(example_arm)

    # --- Assemble ---
    sections = [
        "=" * 72,
        "DEXFORGE SSIK TRANSLATOR — FULL CONTEXT",
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
        f"--- [SECTION 5] TARGET: Python artifact to translate ('{target_arm_stem}') ---",
        "",
        target_py_code,
        "",
        "=" * 72,
        "END OF CONTEXT — PRODUCE ONLY RAW RUST CODE BELOW",
        "=" * 72,
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Step 4: LLM Invocation
# ---------------------------------------------------------------------------


class IKTranslatorLLM:
    """Thin wrapper around the OpenAI-compatible chat API for IK translation.

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
                "The 'openai' package is required for LLM translation. "
                "Install it with: pip install openai"
            ) from exc

        self._api_key = api_key or os.environ.get("DEXFORGE_LLM_KEY", "nokeyneeded")
        self._base_url = base_url or os.environ.get(
            "DEXFORGE_LLM_BASE_URL", self.DEFAULT_BASE_URL
        )
        self._model = model or os.environ.get("DEXFORGE_LLM_MODEL", self.DEFAULT_MODEL)

        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def translate(self, context: str) -> str:
        """Send *context* to the LLM and return the raw response text.

        Temperature is fixed at ``0.0`` (greedy decoding) to ensure
        deterministic output across identical inputs.

        :param context: The assembled prompt + context string.
        :returns: The LLM's response as a raw string (may contain markdown
            fences; call :func:`sanitize_rust_output` on the result).
        :raises RuntimeError: If the API call fails or returns an empty choice.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Rust systems programmer and robotics engineer. "
                        "Translate the provided ssik Python IK artifact to #![no_std] Rust "
                        "exactly as instructed. Output only raw, compilable Rust code."
                    ),
                },
                {"role": "user", "content": context},
            ],
        )
        choices = response.choices
        if not choices:
            raise RuntimeError("LLM returned an empty response (no choices).")
        content = choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned a None message content.")
        return content


# ---------------------------------------------------------------------------
# Step 5: Output Sanitization
# ---------------------------------------------------------------------------


def sanitize_rust_output(llm_response: str) -> str:
    """Strip Markdown code fences and surrounding prose, returning only code.

    If the response contains one or more ``rust`` (or generic) code fences,
    extracts and concatenates only their contents.  If no fences are present,
    returns the full response stripped of leading/trailing whitespace.

    :param llm_response: Raw string returned by the LLM.
    :returns: Clean Rust source code with no Markdown artefacts.
    """
    # Collect all ```rust ... ``` or ``` ... ``` code block bodies
    blocks = re.findall(r"```(?:rust)?[ \t]*\n(.*?)```", llm_response, flags=re.DOTALL)
    if blocks:
        return "\n".join(block.strip() for block in blocks).strip()
    # No fences: strip stray backtick-fence lines and return the remainder
    cleaned = re.sub(r"^```[^\n]*\n?", "", llm_response, flags=re.MULTILINE)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_outputs(
    output_dir: Path,
    arm_stem: str,
    target_py_code: str,
    rust_code: Optional[str],
    context: str,
) -> dict[str, Path]:
    """Write the three output artefacts to *output_dir*.

    :param output_dir: Destination directory (created if absent).
    :param arm_stem: Base name of the arm (e.g. ``"ur5"``).
    :param target_py_code: Source Python artifact text.
    :param rust_code: Translated Rust code, or ``None`` when only dumping context.
    :param context: The assembled prompt + context string.
    :returns: Mapping of ``{"py": path, "rs": path, "ctx": path}``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive a clean module name (strip trailing _ik suffix for naming, then re-add)
    name = arm_stem.removesuffix("_ik")

    py_path = output_dir / f"{name}_ik.py"
    rs_path = output_dir / f"{name}_ik_solver.rs"
    ctx_path = output_dir / f"{name}_context.txt"

    py_path.write_text(target_py_code, encoding="utf-8")
    ctx_path.write_text(context, encoding="utf-8")
    if rust_code is not None:
        rs_path.write_text(rust_code, encoding="utf-8")

    return {"py": py_path, "rs": rs_path if rust_code is not None else None, "ctx": ctx_path}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssik_translator",
        description=(
            "Translate a ssik Python IK artifact to #![no_std] Rust using an LLM. "
            "Provide either --artifact (prebuilt name or file path) or --urdf (with "
            "--base and --ee)."
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
        help="Output directory for the three generated files (default: out/).",
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
    """Entry point for the ssik translator CLI.

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
    print("[ssik_translator] Step 0: Loading Python artifact …")
    try:
        if args.urdf is not None:
            target_py_code, artifact_stem = generate_artifact_from_urdf(
                args.urdf, args.base, args.ee, args.output_dir
            )
            print(f"[ssik_translator]   Generated artifact from URDF: {artifact_stem}")
        else:
            target_py_code, artifact_stem = load_prebuilt_artifact(args.artifact)
            print(f"[ssik_translator]   Loaded artifact: {artifact_stem}")
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        print(f"[ssik_translator] ERROR: {exc}", file=sys.stderr)
        return 1

    # --- Step 1: Detect kinematic family ---
    print("[ssik_translator] Step 1: Detecting kinematic family …")
    try:
        example_arm, max_solutions = find_best_example(target_py_code)
    except ValueError as exc:
        print(f"[ssik_translator] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"[ssik_translator]   → example arm: '{example_arm}', "
        f"MAX_SOLUTIONS = {max_solutions}"
    )

    # Fall back gracefully when the chosen example arm has no Rust driver yet
    rs_example = _REPO_ROOT / "src" / "drivers" / example_arm / "ik_solver.rs"
    if not rs_example.exists():
        # Try falling back to franka_panda as a structural template for 7-DOF
        fallback = "franka_panda"
        fallback_path = _REPO_ROOT / "src" / "drivers" / fallback / "ik_solver.rs"
        if fallback_path.exists():
            print(
                f"[ssik_translator]   WARNING: No Rust driver for '{example_arm}'; "
                f"falling back to '{fallback}' as structural template."
            )
            example_arm = fallback
        else:
            print(
                f"[ssik_translator]   WARNING: No Rust driver found for '{example_arm}' "
                "or fallback 'franka_panda'. The one-shot example will be empty.",
                file=sys.stderr,
            )

    # --- Step 3: Assemble context ---
    print("[ssik_translator] Step 3: Assembling context …")
    arm_name = artifact_stem.removesuffix("_ik")
    context = assemble_context(target_py_code, arm_name, example_arm, max_solutions)
    print(f"[ssik_translator]   Context assembled ({len(context):,} chars).")

    # --- Dump context and/or invoke LLM ---
    rust_code: Optional[str] = None

    if args.dump_context:
        print("[ssik_translator] --dump-context: skipping LLM invocation.")
    else:
        print("[ssik_translator] Step 4: Invoking LLM …")
        try:
            llm = IKTranslatorLLM(
                api_key=args.llm_key,
                base_url=args.llm_url,
                model=args.llm_model,
            )
            raw_response = llm.translate(context)
        except ImportError as exc:
            print(f"[ssik_translator] ERROR: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 — surface all API errors
            print(f"[ssik_translator] LLM error: {exc}", file=sys.stderr)
            return 1

        # --- Step 5: Sanitize output ---
        print("[ssik_translator] Step 5: Sanitizing LLM output …")
        rust_code = sanitize_rust_output(raw_response)
        print(f"[ssik_translator]   Rust code: {len(rust_code):,} chars.")

    # --- Write output files ---
    print("[ssik_translator] Writing output files …")
    paths = _write_outputs(
        args.output_dir,
        artifact_stem,
        target_py_code,
        rust_code,
        context,
    )

    print(f"[ssik_translator] ✓ Python artifact : {paths['py']}")
    if paths.get("rs"):
        print(f"[ssik_translator] ✓ Rust artifact   : {paths['rs']}")
    print(f"[ssik_translator] ✓ Context dump    : {paths['ctx']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
