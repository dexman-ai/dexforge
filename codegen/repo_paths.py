"""Monorepo path helpers for dexforge codegen / ``dexforge build``.

Source tree layout::

    <repo>/
      codegen/
      repos/dexhal/src/math/          # shared no_std kernels
      repos/dexforge-<oem>/rust/src/drivers/<arm>/
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

# Shared math kernels copied into every stand-alone driver crate.
MATH_MODULES: Sequence[str] = (
    "arm",
    "kinematics",
    "postprocess",
    "ssik_core",
    "tolerances",
)

# Shared ``crate::drivers::<module>`` helpers some arms pull in.
SHARED_DRIVER_MODULES: Sequence[str] = ("rizon_common",)


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Locate the dexforge repository root.

    :raises FileNotFoundError: If no repository root is found.
    """
    here = start or Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        monorepo_math = candidate / "repos" / "dexhal" / "src" / "math"
        if (candidate / "codegen").is_dir() and monorepo_math.is_dir():
            return candidate
    raise FileNotFoundError(
        "could not locate the dexforge source tree (expected "
        "'repos/dexhal/src/math' + 'codegen/'). Run 'dexforge build' from a "
        "dexforge checkout, or pass --repo-root."
    )


def math_src(repo_root: Path) -> Path:
    """Return the directory holding shared ``math/*.rs`` kernels."""
    mono = repo_root / "repos" / "dexhal" / "src" / "math"
    if mono.is_dir():
        return mono
    raise FileNotFoundError(
        f"missing math kernels under {repo_root / 'repos' / 'dexhal' / 'src' / 'math'}"
    )


def iter_oem_driver_roots(repo_root: Path) -> Iterable[Path]:
    """Yield ``repos/dexforge-*/rust/src/drivers`` directories for every OEM package."""
    repos = repo_root / "repos"
    if repos.is_dir():
        for drivers in sorted(repos.glob("dexforge-*/rust/src/drivers")):
            if drivers.is_dir():
                yield drivers


def resolve_arm_driver_dir(repo_root: Path, arm: str) -> Optional[Path]:
    """Resolve the on-disk driver directory for *arm*, or ``None``."""
    arm = str(arm)
    for drivers_root in iter_oem_driver_roots(repo_root):
        candidate = drivers_root / arm
        if candidate.is_dir():
            return candidate
    return None


def resolve_shared_driver_module(repo_root: Path, module: str) -> Optional[Path]:
    """Locate a shared ``crate::drivers::<module>`` source file."""
    for drivers_root in iter_oem_driver_roots(repo_root):
        candidate = drivers_root / f"{module}.rs"
        if candidate.is_file():
            return candidate
    return None


def list_known_arms(repo_root: Path) -> List[str]:
    """Return arm directory names discovered under OEM driver trees."""
    arms: List[str] = []
    for drivers_root in iter_oem_driver_roots(repo_root):
        for child in sorted(drivers_root.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                arms.append(child.name)
    return sorted(set(arms))
