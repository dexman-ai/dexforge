"""``dexforge`` command-line interface — a rigid-body *driver compiler*.

``dexforge build`` is an **end-to-end** rigid-body dynamics compiler: point it at
a URDF and it translates the arm's full **kinematics and dynamics** into a
**stand-alone** Rust crate written *outside* the dexforge repository. It calls an
LLM to translate the analytical inverse kinematics and runs the built-in symbolic
dynamics compiler to derive the forward kinematics, geometric Jacobian and — from
the URDF's inertial data — gravity / mass-matrix / Coriolis / inverse-dynamics.The emitted crate is fully self-contained: it copies
the shared math kernels and the per-arm driver sources into the target directory
and generates its own ``Cargo.toml`` + ``lib.rs``, so the result builds and runs
**without** the dexforge repository or the ``dexforge`` Python package.

This lets an OEM use dexforge as a one-shot URDF compiler for their hardware:
hand it a URDF, get back a minimal, zero-allocation ``#![no_std]`` crate exposing
the arm's kinematics (``fk`` / ``solve``) and dynamics that they can vendor
straight into firmware.

Typical usage::

    pip install dexforge

    # End-to-end: compile a kinematics + dynamics driver crate from a URDF.
    dexforge build --urdf my_arm.urdf --base base_link --ee tool0 -o ~/my_arm_ik
    cd ~/my_arm_ik && cargo build --release

    dexforge build --urdf my_arm.urdf --base base_link --ee tool0 \\
        -o ~/my_arm_ik --list   # preview, write nothing

IK generation environment variables::

    DEXFORGE_LLM_KEY       API key (defaults to "nokeyneeded" for local models)
    DEXFORGE_LLM_BASE_URL  OpenAI-compatible base URL (e.g. http://localhost:11434/v1)
    DEXFORGE_LLM_MODEL     Model name (e.g. "qwen2.5-coder:32b", "gpt-4o")
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import logging
import build_crate

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_REPO_ROOT = Path(__file__).parent.parent.resolve()

class ExitCodes(int):
    SUCCESS = 0
    BUILD_ERROR = 1
    ARGS_ERROR = 2

def _codegen_dir(output_root: Path, arm_name: str) -> Path:
    """Return the codegen directory rooted at *root* for *arm_name*."""
    return build_crate.driver_dir(output_root, arm_name) / "codegen"


def verify_crate(out_dir: Path) -> tuple:
    """Build the generated crate to prove it is standalone.

    :returns: ``(success, combined_output)``.
    """
    result = subprocess.run(
        ["cargo", "build", "--release"],
        capture_output=True,
        text=True,
        cwd=str(out_dir),
    )
    return result.returncode == 0, (result.stderr + result.stdout).strip()


def compile(args: argparse.Namespace) -> int:
    """Handle ``dexforge build --urdf``: translate a URDF, then verify the crate."""
    import ssik_build
    import build_crate
    import generate_ik
    import generate_dynamics

    output_dir = Path(args.output).resolve()
    # verify it is an empty directory; if non-existent, create it
    if output_dir.exists():
        if not output_dir.is_dir():
            logger.error(f"ERROR: Output path is not a directory: {output_dir}")
            return ExitCodes.ARGS_ERROR
        if any(output_dir.iterdir()):
            logger.error(f"ERROR: Output directory is not empty: {output_dir}")
            return ExitCodes.ARGS_ERROR
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    urdf = Path(args.urdf)
    if not urdf.is_file():
        logger.error(f"ERROR: URDF file not found: {urdf}")
        return ExitCodes.ARGS_ERROR
    
    arm_name = args.arm_name or urdf.stem
    oem_name = args.oem or output_dir.name 

    if not args.ssik and (not args.base or not args.ee):
        logger.error("ERROR: --base and --ee link names are required when not using a pre-generated ssik artifact.")
        return ExitCodes.ARGS_ERROR

    logger.info(f"========== Setting up code generation environment ==========")
    # set up the minimal crate needed for analytical IK validation: math kernels + mod.rs + Cargo.toml
    build_crate.make_crate(output_dir, oem_name, arm_name)
    urdf_target = build_crate.driver_dir(output_dir, arm_name) / f"{arm_name}.urdf"
    if urdf.resolve() != urdf_target.resolve():
        shutil.copyfile(urdf, urdf_target)

    logger.info(f"========== Extracting ssik artifact ==========")

    artifact_name = args.ssik or arm_name + "_ik"
    ssik_source, ssik_info = ssik_build.get_or_build(
        artifact_name=artifact_name,
        urdf_path=urdf,
        base_link=args.base,
        ee_link=args.ee,
        output_dir=_codegen_dir(output_dir, arm_name),
    )

    if "ik" in args.what:
        logger.info(f"========== Generating Analytical Inverse Kinematics ==========")
        example_arm = args.example_arm or ssik_info.arm_class
        example_dir = _REPO_ROOT / "src" / "drivers" / example_arm
        if not example_dir.is_dir():
            error = f"Example arm directory not found: {example_dir}"
            logger.error(error)
            return ExitCodes.ARGS_ERROR

        arm_model = build_crate.generate_armmodel_impl(arm_name, ssik_info.dof, ssik_info.max_solutions)

        context = generate_ik.generate_context(
            ssik_artifact=ssik_source,
            arm_name=arm_name,
            max_solutions=ssik_info.max_solutions,
            example_dir=example_dir,
            )

        llm = generate_ik.IKGeneratorLLM(
            api_key=args.llm_key,
            base_url=args.llm_url,
            model=args.llm_model,
        )

        messages: list = [
            {"role": "system", "content": generate_ik.IKGeneratorLLM.SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        max_retries = args.max_retries if args.max_retries is not None else _MAX_RETRIES
        max_attempts = max_retries + 1
        for attempt in range(max_attempts):
            if attempt > 0:
                logger.info(f"Retrying, {attempt}/{max_retries}")

            try:
                raw_response = llm.send(messages)
            except Exception as exc:  # noqa: BLE001 — surface all API errors
                if attempt < max_attempts - 1:
                    continue
                raise exc

            try:
                files = generate_ik.parse_llm_output(raw_response)
            except ValueError as exc:
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
                raise exc

            files["model.rs"] = arm_model
            files["context.txt"] = context

            try:
                written = build_crate.write_generated_files(output_dir, arm_name, files,)
                build_crate.build_and_verify(output_dir)
            except Exception as exc:
                error = str(exc)
                if attempt < max_attempts - 1:
                    messages.append({"role": "assistant", "content": raw_response})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The generated code failed to build or test with the following error:\n```\n{error}\n```\n"
                            "Please fix the code to resolve this error and regenerate all files."
                        ),
                    })
                    build_crate.remove_generated_files(list(written.values()))  # clean up generated files before retrying
                    continue
                
                return ExitCodes.BUILD_ERROR

    
    logger.info(f"========== Generating Symbolic Rigid Body Dynamics ==========")
    include_kinematics = "kin" in args.what
    include_dynamics = "dyn" in args.what
    generate_dynamics.generate_rbd(ssik_info, urdf, arm_name, build_crate.driver_dir(output_dir, oem_name, arm_name) / "rbd.rs", no_dynamics=not include_dynamics)

    logger.info(f"========== Verifying Generated Crate ==========")
    build_crate.make_crate(output_dir, oem_name, arm_name, ik=True, kin=include_kinematics, dyn=include_dynamics)
    verify_crate(output_dir)

    return ExitCodes.SUCCESS



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
    """Run ``cargo test`` from *cwd*.

    :returns: ``(success, output)`` where *output* is combined stdout+stderr.
    """
    result = subprocess.run(
        ["cargo", "test"],
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


if __name__ == "__main__":
    args = argparse.Namespace(
        arm_name="ur3e",
        urdf=r"d:\tmp\ur3e.urdf",
        output=r"d:\tmp\ur3e",
        base="base_link",
    )
    compile()