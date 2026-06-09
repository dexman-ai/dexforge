#!/usr/bin/env python3
"""generate_ik.py — Generate ``#![no_std]`` Rust analytical IK from ssik Python artifacts.

This module generates (or loads) a per-arm ssik Inverse Kinematics Python
artifact, translates it into deterministic, zero-allocation Rust code via an
LLM API, runs the companion Python helper scripts to produce auxiliary Rust
files (``consts.rs``, ``pq.rs``, ``test_data.rs``), then builds and tests the
Rust crate.  Any build or test failure is fed back to the LLM so it can correct
the generated code automatically.

**IMPORTANT:** The ssik version is locked in ``SSIK_VERSION.txt`` to ensure
compatibility with the already-translated math kernels (``ssik_core.rs``,
``kinematics.rs``). Install the correct version before running this script:

    pip install ssik==<VERSION_FROM_SSIK_VERSION.txt>

For instructions on upgrading the ssik version, see
``codegen/upgrade_ssik_version.md``.

Usage examples::

    # From a ssik prebuilt artifact name (looks up ssik.prebuilt):
    python codegen/generate_ik.py --artifact ur5_ik --output-dir out/

    # From a local artifact .py file:
    python codegen/generate_ik.py --artifact /path/to/my_arm_ik.py --output-dir out/

    # Use an existing artifact and copy a source URDF alongside it:
    python codegen/generate_ik.py --artifact ur5_ik --urdf robot.urdf --output-dir out/

    # Generate a fresh artifact from a URDF, then translate:
    python codegen/generate_ik.py --urdf robot.urdf --base base_link --ee ee_link \\
        --output-dir out/

    # Dump the full context to disk without invoking an LLM:
    python codegen/generate_ik.py --artifact ur5_ik --dump-context --output-dir out/

Environment variables for offline / air-gapped LLM support::

    DEXFORGE_LLM_KEY       API key (defaults to "nokeyneeded" for local models)
    DEXFORGE_LLM_BASE_URL  OpenAI-compatible base URL (e.g. http://localhost:11434/v1)
    DEXFORGE_LLM_MODEL     Model name (e.g. "qwen2.5-coder:32b", "gpt-4o")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from ik_build import (
    build_rust_crate,
    copy_driver_provenance,
    parse_llm_output,
    run_py_scripts,
    test_rust_crate,
    write_generated_files,
)
from ik_llm import IKGeneratorLLM
from ik_prompt import (
    assemble_context,
    find_best_example,
    generate_artifact_from_urdf,
    load_prebuilt_artifact,
    _get_locked_ssik_version,
    _verify_ssik_version,
)

# ---------------------------------------------------------------------------
# Repository root — all relative paths are anchored here.
# ---------------------------------------------------------------------------

_CODEGEN_DIR = Path(__file__).resolve().parent   # codegen/
_REPO_ROOT = _CODEGEN_DIR.parent                 # repository root

# Default for the --max-retries CLI flag (number of *retry* attempts after the
# initial attempt, so total LLM calls = max_retries + 1).
_MAX_RETRIES = 3
# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_ik",
        description=(
            "Generate a #![no_std] Rust analytical IK module from a ssik Python "
            "artifact using an LLM.  Provide --artifact, --urdf (with --base and "
            "--ee), or both."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--artifact",
        metavar="NAME_OR_PATH",
        help=(
            "ssik prebuilt module name (e.g. 'ur5_ik') or path to an existing "
            "*_ik.py artifact file."
        ),
    )
    parser.add_argument(
        "--urdf",
        metavar="URDF_PATH",
        type=Path,
        help="Path to a URDF file.  Copies the URDF into the target driver dir.",
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

    # --- Check locked ssik version ---
    try:
        locked_version = _get_locked_ssik_version()
        print(f"[codegen:ik] Locked ssik version: {locked_version}")
        _verify_ssik_version()
        print(f"[codegen:ik] ✓ ssik version verified")
    except (ImportError, FileNotFoundError) as exc:
        print(f"[codegen:ik] ERROR: {exc}", file=sys.stderr)
        return 1

    if args.artifact is None and args.urdf is None:
        parser.error("Either --artifact or --urdf must be specified.")
    if args.urdf is not None and not args.urdf.exists():
        parser.error(f"URDF file not found: {args.urdf}")

    # --- Step 0: Obtain source Python artifact ---
    print("[codegen:ik] Step 0: Loading Python artifact …")
    try:
        if args.artifact is None:
            if not args.base or not args.ee:
                parser.error("--urdf requires both --base and --ee to be specified.")
            target_py_code, artifact_stem = generate_artifact_from_urdf(
                args.urdf, args.base, args.ee, args.output_dir
            )
            print(f"[codegen:ik]   Generated artifact from URDF: {artifact_stem}")
        else:
            target_py_code, artifact_stem = load_prebuilt_artifact(args.artifact)
            print(f"[codegen:ik]   Loaded artifact: {artifact_stem}")
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        print(f"[codegen:ik] ERROR: {exc}", file=sys.stderr)
        return 1

    arm_name = artifact_stem.removesuffix("_ik")
    source_urdf = args.urdf

    # --- Step 1: Detect kinematic family ---
    print("[codegen:ik] Step 1: Detecting kinematic family …")
    try:
        example_arm, max_solutions = find_best_example(target_py_code)
    except ValueError as exc:
        print(f"[codegen:ik] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"[codegen:ik]   → example arm: '{example_arm}', "
        f"MAX_SOLUTIONS = {max_solutions}"
    )

    # Fall back gracefully when the chosen example arm has no Rust driver yet
    rs_example = _REPO_ROOT / "src" / "drivers" / example_arm / "ik_solver.rs"
    if not rs_example.exists():
        fallback = "franka_panda"
        fallback_path = _REPO_ROOT / "src" / "drivers" / fallback / "ik_solver.rs"
        if fallback_path.exists():
            print(
                f"[codegen:ik]   WARNING: No Rust driver for '{example_arm}'; "
                f"falling back to '{fallback}' as structural template."
            )
            example_arm = fallback
        else:
            print(
                f"[codegen:ik]   WARNING: No Rust driver found for '{example_arm}' "
                "or fallback 'franka_panda'. The one-shot example will be empty.",
                file=sys.stderr,
            )

    # --- Step 2 (internal) + Step 3: Assemble context ---
    print("[codegen:ik] Step 3: Assembling context …")
    try:
        context = assemble_context(target_py_code, arm_name, example_arm, max_solutions)
    except FileNotFoundError as exc:
        print(f"[codegen:ik] ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[codegen:ik]   Context assembled ({len(context):,} chars).")

    # --- Dump context and exit if requested ---
    if args.dump_context:
        print("[codegen:ik] --dump-context: skipping LLM invocation.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        ctx_path = args.output_dir / f"{arm_name}_context.txt"
        ctx_path.write_text(context, encoding="utf-8")
        py_path = args.output_dir / f"{arm_name}_ik.py"
        py_path.write_text(target_py_code, encoding="utf-8")
        copy_driver_provenance(
            _REPO_ROOT / "src" / "drivers" / arm_name,
            arm_name,
            target_py_code,
            source_urdf,
        )
        print(f"[codegen:ik] ✓ Context dump : {ctx_path}")
        print(f"[codegen:ik] ✓ Python artifact : {py_path}")
        return 0

    # --- Step 4: Invoke LLM with retry loop ---
    print("[codegen:ik] Step 4: Invoking LLM …")
    try:
        llm = IKGeneratorLLM(
            api_key=args.llm_key,
            base_url=args.llm_url,
            model=args.llm_model,
        )
    except ImportError as exc:
        print(f"[codegen:ik] ERROR: {exc}", file=sys.stderr)
        return 1

    messages: list[dict] = [
        {"role": "system", "content": IKGeneratorLLM.SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]

    # max_attempts includes the initial attempt plus all retries:
    #   total calls = args.max_retries (retries) + 1 (initial)
    max_attempts = args.max_retries + 1
    last_raw_response: Optional[str] = None

    for attempt in range(max_attempts):
        if attempt > 0:
            print(f"[codegen:ik]   Retry {attempt}/{args.max_retries} …")

        try:
            raw_response = llm.generate(messages)
        except Exception as exc:  # noqa: BLE001 — surface all API errors
            print(
                f"[codegen:ik] LLM error ({type(exc).__name__}): {exc}",
                file=sys.stderr,
            )
            return 1

        last_raw_response = raw_response
        print(f"[codegen:ik]   LLM response: {len(raw_response):,} chars.")

        # --- Step 5: Parse multi-file output ---
        try:
            files = parse_llm_output(raw_response)
        except ValueError as exc:
            print(f"[codegen:ik]   Parse error: {exc}", file=sys.stderr)
            if attempt < max_attempts - 1:
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
            f"[codegen:ik]   Parsed {len(files)} file(s): "
            + ", ".join(files.keys())
        )

        # --- Step 6: Write generated files ---
        print("[codegen:ik] Step 6: Writing generated files …")
        written = write_generated_files(
            args.output_dir,
            arm_name,
            files,
            target_py_code,
            context,
            source_urdf,
        )

        # --- Step 7: Run Python helper scripts ---
        print("[codegen:ik] Step 7: Running Python helper scripts …")
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
            print(f"[codegen:ik]   Script failures detected.")
            if attempt < max_attempts - 1:
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": error_msg})
                continue
            print(f"[codegen:ik] ERROR: Script failures after all retries.", file=sys.stderr)
            return 1

        # --- Step 8: Build the Rust crate ---
        print("[codegen:ik] Step 8: Building Rust crate …")
        build_ok, build_output = build_rust_crate()
        if not build_ok:
            error_msg = (
                f"cargo build failed with the following errors:\n\n{build_output}\n\n"
                "Please fix the Rust code and regenerate all files."
            )
            print(f"[codegen:ik]   Build failed.")
            if attempt < max_attempts - 1:
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": error_msg})
                continue
            print(f"[codegen:ik] ERROR: Build failed after all retries.", file=sys.stderr)
            print(build_output, file=sys.stderr)
            return 1
        print("[codegen:ik]   Build succeeded.")

        # --- Step 9: Run Rust tests ---
        print("[codegen:ik] Step 9: Running Rust tests …")
        test_ok, test_output = test_rust_crate()
        if not test_ok:
            error_msg = (
                f"cargo test failed with the following output:\n\n{test_output}\n\n"
                "Please fix the Rust code and regenerate all files."
            )
            print(f"[codegen:ik]   Tests failed.")
            if attempt < max_attempts - 1:
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": error_msg})
                continue
            print(f"[codegen:ik] ERROR: Tests failed after all retries.", file=sys.stderr)
            print(test_output, file=sys.stderr)
            return 1
        print("[codegen:ik]   All tests passed.")

        # --- All steps succeeded ---
        break

    print(f"\n[codegen:ik] ✓ Python artifact : {written.get('py')}")
    print(f"[codegen:ik] ✓ Context dump    : {written.get('ctx')}")
    for filename, path in sorted(written.items()):
        if filename not in ("py", "ctx"):
            print(f"[codegen:ik] ✓ {filename:<40} → {path.relative_to(_REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
