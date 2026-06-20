"""Prompt generation module for ik_generator.

This module handles:
- Loading ssik Python artifacts (from prebuilt or file)
- Generating artifacts from URDF files
- Dynamic example selection based on kinematic family
- Extracting Rust API signatures
- Assembling the complete LLM context/prompt
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional
import logging

logger = logging.getLogger(__name__)

_REPO_CODEGEN_DIR = Path(__file__).resolve().parent   # codegen/
_REPO_ROOT = _REPO_CODEGEN_DIR.parent                 # repository root
_REPO_MATH_DIR = _REPO_ROOT / "src" / "math"
_REPO_DRIVERS_DIR = _REPO_ROOT / "src" / "drivers"


def _resolve_repo_driver_dir(arm: str) -> Optional[Path]:
    """Resolve a driver directory in the repo for an arm name.
    """
    for candidate in sorted(_REPO_DRIVERS_DIR.glob(f"*/{arm}")):
        if candidate.is_dir():
            return candidate
    error = f"No driver directory found in the repository for example arm '{arm}'."
    logger.error(error)
    raise FileNotFoundError(error)


def _extract_rust_signatures(rust_file_path: Path) -> str:
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
    logger.info(f"Extracting Rust signatures from '{rust_file_path}' …")
    if not rust_file_path.exists():
        error = f"Rust source file not found: {rust_file_path}"
        logger.error(error)
        raise FileNotFoundError(error)

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

    header = f"// === Signatures from {rust_file_path.name} ===\n\n"

    if not results:
        warning = f"No public function signatures found in {rust_file_path.name}\n"
        logger.warning(warning)
        return header + f"// No public function signatures.\n"

    return header + "\n\n".join(results) + "\n"


def _load_example_py_source(example_arm: str, example_dir: Path) -> str:
    """Read the ssik Python source artifact for *example_arm* from the example arm's codegen dir."""
    py_path = example_dir / "codegen" / f"{example_arm}_ik.py"
    if py_path.exists():
            return py_path.read_text(encoding="utf-8")
    error = (f"# Python artifact for {example_arm} could not be found in {py_path}.\n")
    logger.error(error)
    raise FileNotFoundError(error)  


def _load_example_rust_source(example_arm: str, example_dir: Path) -> str:
    """Load the translated Rust IK driver for *example_arm* from the source tree."""
    rs_path = example_dir / "ik_solver.rs"
    if rs_path.exists():
        return rs_path.read_text(encoding="utf-8")
    error = (f"Rust driver for {example_arm} could not be found in {rs_path}.\n")
    logger.error(error)
    raise FileNotFoundError(error)  


def _load_example_scripts(example_arm: str, example_dir: Path) -> str:
    """Load the Python helper scripts for *example_arm* from the source tree.

    Looks for ``generate_test_data.py`` and ``generate_code.py`` under
    ``src/drivers/<example_arm>/codegen/``.
    """
    codegen_dir = example_dir / "codegen"
    if not codegen_dir.exists():
        error = f"Codegen directory for example arm '{example_arm}' could not be found at expected location {codegen_dir}.\n"
        logger.error(error)
        raise FileNotFoundError(error)
    arm_scripts: list[str] = []
    for script_name in ("generate_test_data.py", "generate_code.py"):
        path = codegen_dir / script_name
        if path.exists():
            arm_scripts.append(f"--- {script_name} ---\n" + path.read_text(encoding="utf-8"))
    if not arm_scripts:
        error = f"No Python helper scripts found for example arm '{example_arm}' in {codegen_dir}.\n"
        logger.error(error)
        raise FileNotFoundError(error)
    return "\n\n".join(arm_scripts)


def generate_context(
    ssik_artifact: str,
    arm_name: str,
    example_arm: str,
    example_dir: Path,
    max_solutions: int,
) -> str:
    """Build the complete context string that will be sent to the LLM.

    Loads all required files at runtime: the prompt template, the target
    Python artifact, the Rust API headers for ``ssik_core`` and ``kinematics``,
    and the one-shot example (Python + Rust pair plus Python helper scripts).

    :param ssik_artifact: Text of the source ``*_ik.py`` artifact being translated.
    :param arm_name: Name of the target arm (e.g. ``"ur5"``).
    :param example_arm: Name of the reference arm selected by
        :func:`find_best_example` (e.g. ``"franka_panda"``).
    :param max_solutions: Maximum number of IK solutions (used in the prompt).
    :returns: The full assembled prompt + context as a single string.
    :raises FileNotFoundError: If the prompt template is missing.
    """
    logger.info("Assembling LLM context with all required sections …")
    if example_arm is None:
        example_arm = example_dir.name
    elif example_dir is None:
        example_dir = _resolve_repo_driver_dir(example_arm)

    # --- Prompt template ---
    prompt_path = _REPO_CODEGEN_DIR / "translate_ssik_artifact.md"
    if not prompt_path.exists():
        error = f"Prompt template not found at '{prompt_path}'. Ensure the repository is checked out with the codegen/ directory."
        logger.error(error)
        raise FileNotFoundError(error)
    prompt_template = prompt_path.read_text(encoding="utf-8")

    # --- Rust API headers ---
    ssik_core_sigs = _extract_rust_signatures(
        _REPO_MATH_DIR / "ssik_core.rs"
    )
    kinematics_sigs = _extract_rust_signatures(
        _REPO_MATH_DIR / "kinematics.rs"
    )
    tolerances_sigs = _extract_rust_signatures(
        _REPO_MATH_DIR / "tolerances.rs"
    )
    postprocess_sigs = _extract_rust_signatures(
        _REPO_MATH_DIR / "postprocess.rs"
    )

    # --- One-shot example (Python source + Rust translation + helper scripts) ---
    example_py = _load_example_py_source(example_arm, example_dir)
    example_rs = _load_example_rust_source(example_arm, example_dir)
    example_scripts = _load_example_scripts(example_arm, example_dir)

    # --- Assemble ---
    sections = [
        "=" * 72,
        "DEXFORGE IK GENERATOR — FULL CONTEXT",
        "=" * 72,
        "",
        prompt_template,
        "",
        "=" * 72,
        f"TARGET ARM: {arm_name}   |   MAX_SOLUTIONS = {max_solutions}   |   EXAMPLE ARM: {example_arm}",
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
        "--- [SECTION 3] REFERENCE: tolerances.rs PUBLIC API ---",
        "",
        tolerances_sigs,
        "",
        "--- [SECTION 4] REFERENCE: postprocess.rs PUBLIC API ---",
        "",
        postprocess_sigs,
        "",
        f"--- [SECTION 5] ONE-SHOT EXAMPLE: Python artifact for '{example_arm}' ---",
        "",
        example_py,
        "",
        f"--- [SECTION 6] ONE-SHOT EXAMPLE: Rust translation for '{example_arm}' ---",
        "",
        example_rs,
        "",
        f"--- [SECTION 7] ONE-SHOT EXAMPLE: Python helper scripts for '{example_arm}' ---",
        "",
        example_scripts,
        "",
        f"--- [SECTION 8] TARGET: ssik Python artifact to translate ('{arm_name}') ---",
        "",
        ssik_artifact,
        "",
        "=" * 72,
        "END OF CONTEXT — PRODUCE ALL FILES USING THE === FILE: <name> === FORMAT",
        "=" * 72,
    ]
    return "\n".join(sections)


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
        "You are an expert Rust and Python systems programmer and robotics engineer. "
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
        logger.info(f"Initializing LLM client with model='{model}' and base_url='{base_url}'")
        try:
            from openai import OpenAI
        except ImportError as exc:
            error = (
                "The 'openai' package is required for LLM generation. "
                "Install it with: pip install openai"
            )
            logger.error(error)
            raise ImportError(error) from exc

        self._api_key = api_key or os.environ.get("DEXFORGE_LLM_KEY", "nokeyneeded")
        self._base_url = base_url or os.environ.get(
            "DEXFORGE_LLM_BASE_URL", self.DEFAULT_BASE_URL
        )
        self._model = model or os.environ.get("DEXFORGE_LLM_MODEL", self.DEFAULT_MODEL)
        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def send(self, messages: list[dict]) -> str:
        """Send a conversation *messages* list to the LLM and return the raw response.

        Temperature is fixed at ``0.0`` (greedy decoding) to ensure
        deterministic output across identical inputs.

        :param messages: OpenAI-style chat messages list (system + user turns).
        :returns: The LLM's response as a raw string.
        :raises RuntimeError: If the API call fails or returns an empty choice.
        """
        logger.info(f"Invoking '{self._model}' chat LLM.")
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001 — surface all API errors
            error = f"LLM API call failed: {type(exc).__name__}: {exc}"
            logger.error(error)
            raise RuntimeError(error) from exc
        choices = response.choices
        if not choices:
            error = "LLM returned an empty response (no choices)."
            logger.error(error)
            raise RuntimeError(error)
        content = choices[0].message.content
        if content is None:
            error = "LLM returned a None message content."
            logger.error(error)
            raise RuntimeError(error)
        logger.info(f"LLM response received ({len(content):,} characters).")
        return content


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

    
def parse_llm_output(llm_response: str) -> Iterator[tuple[str, str]]:
    """Parse a multi-file LLM response into a ``{filename: content}`` mapping.

    The LLM is instructed to use ``=== FILE: <name> ===`` separator lines.
    This function splits on those markers and returns all contained files.

    Any Markdown code fences inside individual file blocks are stripped.

    :param llm_response: Raw LLM response text.
    :returns: Ordered mapping of filename → file content strings.
    :raises ValueError: If no ``=== FILE: ... ===`` markers are found.
    """
    logger.info("Parsing LLM output for multi-file response.")
    pattern = re.compile(r"^=== FILE:\s*(.+?)\s*===$", re.MULTILINE)
    parts = pattern.split(llm_response)
    # parts[0] is any preamble text before the first marker (discard).
    # Remaining parts alternate: [filename, content, filename, content, ...]
    if len(parts) < 3:
        logger.error(
            "LLM output is improperly formatted: no '=== FILE: <name> ===' markers found. " \
            "The model may not have followed the output format instructions."
        )
        error = (f"Response preview (first 500 chars):\n{llm_response[:500]}")
        logger.debug(error)
        raise ValueError(error)

    for i in range(1, len(parts), 2):
        filename = parts[i].strip()
        raw_content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        logger.info(f"Extracted '{filename}'.")
        yield filename, _sanitize_file_content(raw_content)


