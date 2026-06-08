"""dexforge command line interface."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

_IK_CODEGEN_SCRIPTS: tuple[str, ...] = (
    "generate_test_data.py",
    "generate_franka_test_data.py",
    "generate_xarm7_test_data.py",
    "generate_piper_test_data.py",
    "generate_rizon_test_data.py",
    "generate_franka_consts.py",
    "generate_xarm7_consts.py",
    "generate_piper_pq.py",
    "generate_rizon_codegen.py",
)

_DYNAMICS_ARMS: tuple[str, ...] = (
    "ur5",
    "piper",
    "franka_panda",
    "xarm7",
    "rizon4",
    "rizon10",
)


def _find_repo_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "codegen").is_dir() and (candidate / "src" / "drivers").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find repository root containing 'codegen/' and 'src/drivers/'. "
        "Run this command inside the dexforge repository or pass --repo-root."
    )


def _collect_build_commands(repo_root: Path, python_executable: str) -> list[list[str]]:
    commands: list[list[str]] = []
    codegen_dir = repo_root / "codegen"
    drivers_dir = repo_root / "src" / "drivers"

    for script_name in _IK_CODEGEN_SCRIPTS:
        script = codegen_dir / script_name
        if not script.exists():
            raise FileNotFoundError(f"Missing codegen script: {script}")
        commands.append([python_executable, str(script)])

    dynamics_script = codegen_dir / "generate_dynamics.py"
    if not dynamics_script.exists():
        raise FileNotFoundError(f"Missing dynamics generator: {dynamics_script}")

    for arm in _DYNAMICS_ARMS:
        artifact = drivers_dir / arm / f"{arm}_ik.py"
        output = drivers_dir / arm / "generated_kinematics.rs"
        if not artifact.exists():
            raise FileNotFoundError(f"Missing artifact for arm '{arm}': {artifact}")
        commands.append(
            [
                python_executable,
                str(dynamics_script),
                "--artifact",
                str(artifact),
                "--output",
                str(output),
            ]
        )

    return commands


def _run_command(command: list[str], cwd: Path) -> int:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(cwd), check=False)
    return result.returncode


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="dexforge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="Run the IK and dynamics codegen pipeline for all supported arms.",
    )
    build_parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Path to or inside the dexforge repository (default: current working directory).",
    )
    build_parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run codegen scripts (default: current interpreter).",
    )

    args = parser.parse_args(argv)

    try:
        repo_root = _find_repo_root(args.repo_root)
        commands = _collect_build_commands(repo_root, args.python)
    except FileNotFoundError as exc:
        print(f"dexforge: {exc}", file=sys.stderr)
        return 1

    for command in commands:
        status = _run_command(command, repo_root)
        if status != 0:
            return status

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
