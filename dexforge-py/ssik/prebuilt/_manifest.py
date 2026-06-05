"""Arm manifest loader for ssik compatibility layer.

This module provides load_manifest() which returns metadata for all available
arms in dexforge_py, matching ssik.prebuilt._manifest API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["Arm", "ArmKnownGap", "load_manifest"]


@dataclass(frozen=True)
class ArmKnownGap:
    """A known coverage gap that's xfailed in the uniform-fuzz test."""

    xfail_reason: str


@dataclass(frozen=True)
class Arm:
    """One prebuilt-arm entry in the manifest.

    Field semantics match ssik.prebuilt._manifest.Arm for test compatibility.
    """

    name: str  # e.g. "ur5_ik"
    display_name: str
    dof: int
    sample_q: tuple[float, ...]
    fk_ceiling_fuzz: float
    known_gaps: ArmKnownGap | None = None
    base_link: str = ""
    ee_link: str = ""
    solver: str = ""
    tier: int = 0


# Hardcoded manifest for dexforge_py arms
# Metadata from SSIK_VERSION 0.8.3 MANIFEST.toml and dexforge memories
_ARMS = {
    "ur5_ik": Arm(
        name="ur5_ik",
        display_name="Universal Robots UR5",
        dof=6,
        sample_q=(0.3, -0.5, 0.7, 0.2, 0.4, -0.1),
        fk_ceiling_fuzz=1e-7,
        known_gaps=None,
        base_link="base_link",
        ee_link="ee_link",
        solver="ikgeo.three_parallel",
        tier=0,
    ),
    "piper_ik": Arm(
        name="piper_ik",
        display_name="AgileX PiPER",
        dof=6,
        sample_q=(0.3, 0.7, -0.5, 0.2, 0.4, -0.1),
        fk_ceiling_fuzz=1e-4,
        known_gaps=ArmKnownGap(
            xfail_reason="RR degenerate-case fallback omitted (ssik#286)"
        ),
        base_link="base_link",
        ee_link="link6",
        solver="ikgeo.general_6r",
        tier=2,
    ),
    "franka_panda_ik": Arm(
        name="franka_panda_ik",
        display_name="Franka Emika Panda",
        dof=7,
        sample_q=(0.2, 0.3, -0.4, -1.2, 0.3, 1.4, 0.5),
        fk_ceiling_fuzz=1e-4,
        known_gaps=None,
        base_link="base_link",
        ee_link="ee_link",
        solver="jointlock.seven_r",
        tier=1,
    ),
}


def load_manifest(path=None) -> dict[str, Arm]:
    """Return manifest of available arms.

    :param path: ignored (for ssik API compatibility)
    :returns: dict mapping arm name to Arm metadata
    """
    return _ARMS
