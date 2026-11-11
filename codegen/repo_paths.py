"""Monorepo path helpers for dexforge codegen / ``dexforge build``.

After the OEM split the source tree looks like::

    <repo>/
      codegen/
      repos/dexhal/src/math/          # shared no_std kernels
      repos/dexforge-<oem>/rust/src/drivers/<arm>/

Older checkouts used a flat ``src/math`` + ``src/drivers/<arm>`` layout; these
helpers accept both.
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

    Prefers the monorepo marker (``codegen/`` + ``repos/dexhal/src/math``), then
    falls back to the legacy ``src/Cargo.toml`` + ``src/math`` layout.

    :raises FileNotFoundError: If no repository root is found.
    """
    here = start or Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        monorepo_math = candidate / "repos" / "dexhal" / "src" / "math"
        if (candidate / "codegen").is_dir() and monorepo_math.is_dir():
            return candidate
        if (candidate / "src" / "Cargo.toml").is_file() and (
            candidate / "src" / "math"
        ).is_dir():
            return candidate
    raise FileNotFoundError(
        "could not locate the dexforge source tree (expected either "
        "'repos/dexhal/src/math' + 'codegen/' or legacy 'src/Cargo.toml' + "
        "'src/math/'). Run 'dexforge build' from a dexforge checkout, or pass "
        "--repo-root."
    )


def math_src(repo_root: Path) -> Path:
    """Return the directory holding shared ``math/*.rs`` kernels."""
    mono = repo_root / "repos" / "dexhal" / "src" / "math"
    if mono.is_dir():
        return mono
    legacy = repo_root / "src" / "math"
    if legacy.is_dir():
        return legacy
    raise FileNotFoundError(
        f"missing math kernels under {repo_root} "
        "(tried repos/dexhal/src/math and src/math)"
    )


def iter_oem_driver_roots(repo_root: Path) -> Iterable[Path]:
    """Yield ``.../rust/src/drivers`` directories for every OEM package."""
    repos = repo_root / "repos"
    if repos.is_dir():
        for drivers in sorted(repos.glob("dexforge-*/rust/src/drivers")):
            if drivers.is_dir():
                yield drivers
    legacy = repo_root / "src" / "drivers"
    if legacy.is_dir():
        yield legacy


def resolve_arm_driver_dir(repo_root: Path, arm: str) -> Optional[Path]:
    """Resolve the on-disk driver directory for *arm*, or ``None``."""
    arm = str(arm)
    # Monorepo: repos/dexforge-*/rust/src/drivers/<arm>
    for drivers_root in iter_oem_driver_roots(repo_root):
        candidate = drivers_root / arm
        if candidate.is_dir():
            return candidate
    # Legacy nested OEM trees: src/drivers/<oem>/drivers/<arm>
    legacy_root = repo_root / "src" / "drivers"
    if legacy_root.is_dir():
        for candidate in sorted(legacy_root.glob(f"*/drivers/{arm}")):
            if candidate.is_dir():
                return candidate
        flat = legacy_root / arm
        if flat.is_dir():
            return flat
    return None


def resolve_shared_driver_module(repo_root: Path, module: str) -> Optional[Path]:
    """Locate a shared ``crate::drivers::<module>`` source file."""
    for drivers_root in iter_oem_driver_roots(repo_root):
        candidate = drivers_root / f"{module}.rs"
        if candidate.is_file():
            return candidate
    legacy = repo_root / "src" / "drivers" / f"{module}.rs"
    if legacy.is_file():
        return legacy
    return None


def list_known_arms(repo_root: Path) -> List[str]:
    """Return arm directory names discovered under OEM driver trees."""
    arms: List[str] = []
    for drivers_root in iter_oem_driver_roots(repo_root):
        for child in sorted(drivers_root.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                arms.append(child.name)
    return sorted(set(arms))
