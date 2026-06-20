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
import logging 
import argparse
from typing import Optional, Sequence
from dexforge_build import compile

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('[dexforge] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``dexforge`` argument parser."""
    core_parser = argparse.ArgumentParser(add_help=False)

    core_parser.add_argument(
        "--output",
        "-o",
        required=True,
        metavar="DIR",
        help="Target directory for the generated crate.",
    )
    core_parser.add_argument(
        "--urdf",
        required=True,
        metavar="URDF",
        help=(
            "Compile a driver end-to-end from this URDF file (calls the LLM to translate it)."
        ),
    )
    core_parser.add_argument(
        "--ssik",
        required=False,
        metavar="SSIK",
        help=(
            "Extract analytical IK from this ssik artifact, instead of generating a new one."
        ),
    )
    
    core_parser.add_argument(
        "--oem",
        required=False,
        metavar="OEM",
        help="Name of hardware manufacturer, used to name the generated crate. Inferred from the name of the output directory if absent.",
    )
    
    core_parser.add_argument(
        "--arm",
        required=False,
        metavar="ARM",
        help="Name of the robot arm, if different from the URDF.",
    )

    core_parser.add_argument(
        "--base",
        required=False,
        metavar="LINK",
        help="Base link name in the URDF. ",
    )

    core_parser.add_argument(
        "--ee",
        required=False,
        metavar="LINK",
        help="End-effector link name in the URDF.",
    )

    core_parser.add_argument(
        "--what",
        choices=["ik", "kin","dyn"],
        help=("Specify which components to generate: "
              "'ik' for analytical inverse kinematics, 'kin' for forward kinematics, including the Jacobian and its inverse, 'dyn' for dynamics. "
              "If none specified, all components are generated (default)."),
        default=["ik", "kin","dyn"],
        nargs="+",
    )
    
    llm_parser = argparse.ArgumentParser(add_help=False)
    llm_parser.add_argument(
        "--llm-key",
        metavar="KEY",
        help="LLM API key for --urdf translation (overrides DEXFORGE_LLM_KEY environment variable).",
    )
    llm_parser.add_argument(
        "--llm-url",
        metavar="URL",
        help="LLM base URL for --urdf translation (overrides DEXFORGE_LLM_BASE_URL environment variable).",
    )
    llm_parser.add_argument(
        "--llm-model",
        metavar="MODEL",
        help="LLM model for --urdf translation (overrides DEXFORGE_LLM_MODEL environment variable).",
    )
    llm_parser.add_argument(
        "--max-retries",
        metavar="N",
        type=int,
        help="Max LLM retry attempts for artifact translation.",
    )

    
    parser = argparse.ArgumentParser(
        prog="dexforge",
        description="dexforge — compile stand-alone rigid-body driver crates (kinematics + dynamics).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_p = subparsers.add_parser(
        "build",
        parents=[core_parser, llm_parser],
        help="Compile a stand-alone kinematics + dynamics driver crate.",
        description=(
            "Compile an arm's full rigid-body kinematics and dynamics into a "
            "self-contained Rust crate rooted at the requested output directory. "
            "Point it at a URDF (--urdf, with --base and --ee) to translate new "
            "hardware end-to-end: an LLM translates the analytical IK and the "
            "symbolic compiler derives FK, the Jacobian and (from the URDF's "
            "inertial data) gravity/mass-matrix/Coriolis/inverse-dynamics. The "
            "generated crate does not depend on dexforge."
        ),
    )
    build_p.add_argument(
        "--repo-root",
        metavar="DIR",
        help="Path to the dexforge source tree (auto-detected by default).",
    )
    build_p.set_defaults(func=compile, verify=True)

    prepare_p = subparsers.add_parser(
        "prepare",
        parents=[core_parser],
        help="Validate the source URDF and ssik artifact, generating it if necessary, and construct the LLM prompt used to translate the URDF (for debugging or manual translation).",
        description=(
            "Output the LLM prompt used to translate the URDF, for debugging or manual translation. "
            "This does not call the LLM or generate any files."
        ),
    )
    prepare_p.set_defaults(func=prepare)

    generate_p = subparsers.add_parser(
        "generate",
        parents=[core_parser],
        help="Generate the driver crate from the LLM response.",
        description=(
            "Generate the driver crate from the LLM response. "
            "This skips the validation step and directly emits the crate based on the provided inputs."
        ),
    )
    generate_p.add_argument(
        "response",
        metavar="FILE",
        help="Path to a file containing the LLM response to use for generation (skips the LLM call).",
    )
    
    generate_p.set_defaults(func=compile)

    calibrate_p = subparsers.add_parser(
        "calibrate",
        parents=[core_parser],
        help="Verify that the generated crate builds and passes tests.",
        description=(
            "Verify that the generated crate builds and passes tests. "
            "This is a post-build step that checks the emitted crate for correctness, but does not generate any files."
        ),
    )
    calibrate_p.add_argument(
        "dh_params",
        metavar="FILE",
        help="Path to a file containing the DH parameters to use for calibration.",
    )
    calibrate_p.set_defaults(func=calibrate)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point used by the ``dexforge`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
