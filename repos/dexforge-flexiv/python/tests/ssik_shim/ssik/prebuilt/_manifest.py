"""Arm manifest loader for OEM-local ssik compatibility layer."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Arm", "ArmKnownGap", "load_manifest"]


@dataclass(frozen=True)
class ArmKnownGap:
    xfail_reason: str


@dataclass(frozen=True)
class Arm:
    name: str
    display_name: str
    dof: int
    sample_q: tuple[float, ...]
    fk_ceiling_fuzz: float
    known_gaps: ArmKnownGap | None = None
    base_link: str = ""
    ee_link: str = ""
    solver: str = ""
    tier: int = 0


_ARMS = {
    "rizon4_ik": Arm(
        name="rizon4_ik",
        display_name="Flexiv Rizon 4",
        dof=7,
        sample_q=(0.2, 0.3, -0.4, -1.2, 0.3, 1.4, 0.5),
        fk_ceiling_fuzz=1e-4,
        known_gaps=None,
        base_link="base_link",
        ee_link="ee_link",
        solver="jointlock.seven_r",
        tier=1,
    ),
    "rizon10_ik": Arm(
        name="rizon10_ik",
        display_name="Flexiv Rizon 10",
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
    return _ARMS

