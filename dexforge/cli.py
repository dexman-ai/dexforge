"""``dexforge`` command-line interface.

The ``dexforge build`` command runs the full inverse-kinematics **and** dynamics
code-generation pipeline that turns the locked ``ssik`` prebuilt artifacts into
the committed, zero-allocation ``#![no_std]`` Rust drivers under
``src/drivers/<arm>/``.

Typical usage::

    pip install dexforge
    dexforge build                 # regenerate every supported arm
    dexforge build --arm ur5       # regenerate a single arm
    dexforge build --stage kinematics   # only re-run the FK/Jacobian generator
    dexforge build --list          # print the plan without running anything

The individual generators live in ``codegen/`` and require the *locked* ``ssik``
version (see ``SSIK_VERSION.txt``) plus ``sympy``/``numpy`` to be installed::

    pip install $(grep ssik SSIK_VERSION.txt) sympy numpy

``dexforge build`` operates on a checked-out dexforge repository: each generator
writes its output back into ``src/drivers/``. Install the package in editable
mode (``pip install -e .``) from a clone if you intend to regenerate sources.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence

# Arms in canonical (dependency-free) order.
ARMS = ("ur5", "piper", "franka_panda", "xarm7", "rizon4", "rizon10")

# Code-generation stages, in the order they must run. ``consts`` bakes the
# pose-independent chain geometry, ``kinematics`` derives the symbolic FK +
# Jacobian, and the ``*_test_data`` stages capture the golden regression
# datasets the Rust tests assert against.
STAGES = ("consts", "kinematics", "test_data", "dynamics_test_data")


class Step(NamedTuple):
    """A single code-generation command in the build plan."""

    arm: str
    stage: str
    description: str
    script: str  # path relative to the repository root
    args: tuple  # extra command-line arguments for the script

    def command(self, repo_root: Path) -> List[str]:
        """Build the concrete ``subprocess`` argv for this step."""
        return [sys.executable, str(repo_root / self.script), *self.args]


def _kinematics_step(arm: str) -> Step:
    return Step(
        arm=arm,
        stage="kinematics",
        description=f"{arm}: symbolic forward kinematics + geometric Jacobian",
        script="codegen/generate_dynamics.py",
        args=(
            "--artifact",
            f"{arm}_ik",
            "--output",
            f"src/drivers/{arm}/generated_kinematics.rs",
        ),
    )


# Pose-independent baked geometry (consts.rs / pq.rs). UR5 bakes its constants
# directly into ik_solver.rs, so it has no separate consts step.
_CONSTS_STEPS = {
    "franka_panda": [
        Step(
            "franka_panda",
            "consts",
            "franka_panda: baked KinBody sub-chain constants",
            "codegen/generate_franka_consts.py",
            (),
        )
    ],
    "xarm7": [
        Step(
            "xarm7",
            "consts",
            "xarm7: baked KinBody sub-chain constants",
            "codegen/generate_xarm7_consts.py",
            (),
        )
    ],
    "piper": [
        Step(
            "piper",
            "consts",
            "piper: Raghavan-Roth (P, Q) matrix builder",
            "codegen/generate_piper_pq.py",
            (),
        )
    ],
    "rizon4": [
        Step(
            "rizon4",
            "consts",
            "rizon4: baked constants + per-sample (P, Q) builders",
            "codegen/generate_rizon_codegen.py",
            ("rizon4",),
        )
    ],
    "rizon10": [
        Step(
            "rizon10",
            "consts",
            "rizon10: baked constants + per-sample (P, Q) builders",
            "codegen/generate_rizon_codegen.py",
            ("rizon10",),
        )
    ],
}

# IK regression datasets (test_data.rs).
_TEST_DATA_STEPS = {
    "ur5": Step(
        "ur5", "test_data", "ur5: IK regression dataset",
        "codegen/generate_test_data.py", (),
    ),
    "franka_panda": Step(
        "franka_panda", "test_data", "franka_panda: IK regression dataset",
        "codegen/generate_franka_test_data.py", (),
    ),
    "xarm7": Step(
        "xarm7", "test_data", "xarm7: IK regression dataset",
        "codegen/generate_xarm7_test_data.py", (),
    ),
    "piper": Step(
        "piper", "test_data", "piper: IK regression dataset",
        "codegen/generate_piper_test_data.py", (),
    ),
    "rizon4": Step(
        "rizon4", "test_data", "rizon4: IK regression dataset",
        "codegen/generate_rizon_test_data.py", ("rizon4",),
    ),
    "rizon10": Step(
        "rizon10", "test_data", "rizon10: IK regression dataset",
        "codegen/generate_rizon_test_data.py", ("rizon10",),
    ),
}

# Dynamics (inverse-dynamics) golden datasets. Only arms with an arm-only
# inertial URDF have a baked dynamics parity dataset.
_DYNAMICS_TEST_DATA_STEPS = {
    "xarm7": Step(
        "xarm7",
        "dynamics_test_data",
        "xarm7: Pinocchio inverse-dynamics ground truth",
        "codegen/generate_dynamics_test_data.py",
        (
            "--urdf",
            "src/drivers/xarm7/xarm7.urdf",
            "--output",
            "src/drivers/xarm7/dynamics_test_data.rs",
            "--samples",
            "1000",
            "--seed",
            "0",
        ),
    ),
}


def build_plan(
    arms: Sequence[str] = ARMS,
    stages: Sequence[str] = STAGES,
) -> List[Step]:
    """Assemble the ordered list of code-generation steps.

    Steps are grouped by *stage* (so all ``consts`` run before any
    ``kinematics``) and, within a stage, follow the canonical arm order. This
    respects the data dependency that the symbolic kinematics generator reads
    the baked chain geometry produced by the ``consts`` stage.

    :param arms: Arms to include (defaults to every supported arm).
    :param stages: Stages to include (defaults to every stage).
    :returns: The ordered build plan.
    """
    plan: List[Step] = []
    for stage in STAGES:
        if stage not in stages:
            continue
        for arm in arms:
            if stage == "consts":
                plan.extend(_CONSTS_STEPS.get(arm, []))
            elif stage == "kinematics":
                plan.append(_kinematics_step(arm))
            elif stage == "test_data":
                step = _TEST_DATA_STEPS.get(arm)
                if step is not None:
                    plan.append(step)
            elif stage == "dynamics_test_data":
                step = _DYNAMICS_TEST_DATA_STEPS.get(arm)
                if step is not None:
                    plan.append(step)
    return plan


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Locate the dexforge repository root.

    Walks upward from *start* (the directory containing this module by default)
    until it finds a directory that holds both ``Cargo.toml`` and ``codegen``.

    :raises FileNotFoundError: If no repository root is found.
    """
    here = (start or Path(__file__).resolve().parent)
    for candidate in (here, *here.parents):
        if (candidate / "Cargo.toml").is_file() and (candidate / "codegen").is_dir():
            return candidate
    raise FileNotFoundError(
        "could not locate the dexforge repository root (no parent directory "
        "contains both 'Cargo.toml' and 'codegen/'). Run 'dexforge build' from "
        "inside a dexforge checkout, or pass --repo-root."
    )


def _run_step(step: Step, repo_root: Path) -> int:
    """Execute a single step, returning its exit code."""
    script_path = repo_root / step.script
    if not script_path.is_file():
        print(f"[dexforge] ERROR: missing generator script: {step.script}", file=sys.stderr)
        return 1
    command = step.command(repo_root)
    print(f"[dexforge] {step.description}")
    print(f"[dexforge]   $ {' '.join(command[1:])}")
    result = subprocess.run(command, cwd=str(repo_root))
    return result.returncode


def _build(args: argparse.Namespace) -> int:
    try:
        repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    except FileNotFoundError as exc:
        print(f"[dexforge] ERROR: {exc}", file=sys.stderr)
        return 1

    arms = args.arm or list(ARMS)
    for arm in arms:
        if arm not in ARMS:
            print(
                f"[dexforge] ERROR: unknown arm {arm!r}; choose from {', '.join(ARMS)}",
                file=sys.stderr,
            )
            return 2

    stages = args.stage or list(STAGES)
    plan = build_plan(arms, stages)

    if not plan:
        print("[dexforge] nothing to do for the requested arms/stages.")
        return 0

    if args.list:
        print(f"[dexforge] build plan ({len(plan)} step(s)):")
        for step in plan:
            command = step.command(repo_root)
            print(f"  [{step.stage}] {step.description}")
            print(f"      $ {' '.join(command[1:])}")
        return 0

    print(f"[dexforge] repository root: {repo_root}")
    print(f"[dexforge] running {len(plan)} step(s) for arms: {', '.join(arms)}")

    failures: List[Step] = []
    for step in plan:
        code = _run_step(step, repo_root)
        if code != 0:
            print(f"[dexforge]   \u2717 {step.arm}/{step.stage} failed (exit {code}).", file=sys.stderr)
            failures.append(step)
            if not args.keep_going:
                print("[dexforge] aborting (use --keep-going to continue past failures).", file=sys.stderr)
                return 1
        else:
            print(f"[dexforge]   \u2713 {step.arm}/{step.stage} done.")

    if failures:
        print(
            f"[dexforge] {len(failures)} step(s) failed: "
            + ", ".join(f"{s.arm}/{s.stage}" for s in failures),
            file=sys.stderr,
        )
        return 1

    print("[dexforge] all code generation steps completed successfully.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``dexforge`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="dexforge",
        description="dexforge code-generation tooling.",
    )
    subparsers = parser.add_subparsers(dest="command")

    build_p = subparsers.add_parser(
        "build",
        help="Run the IK and dynamics code generators.",
        description=(
            "Regenerate the committed Rust IK/dynamics drivers from the locked "
            "ssik prebuilt artifacts."
        ),
    )
    build_p.add_argument(
        "--arm",
        action="append",
        metavar="ARM",
        help=(
            "Arm to (re)generate; repeatable. Defaults to all arms: "
            + ", ".join(ARMS)
        ),
    )
    build_p.add_argument(
        "--stage",
        action="append",
        choices=STAGES,
        help="Limit generation to a stage; repeatable. Defaults to all stages.",
    )
    build_p.add_argument(
        "--repo-root",
        metavar="DIR",
        help="Path to the dexforge repository root (auto-detected by default).",
    )
    build_p.add_argument(
        "--list",
        "--dry-run",
        dest="list",
        action="store_true",
        help="Print the build plan without running any generator.",
    )
    build_p.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running remaining steps after a failure.",
    )
    build_p.set_defaults(func=_build)
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
