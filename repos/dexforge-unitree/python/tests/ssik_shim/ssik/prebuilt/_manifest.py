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
    "z1_ik": Arm(
        name="z1_ik",
        display_name="Unitree Z1",
        dof=6,
        sample_q=(0.3, 0.8, -0.5, 0.2, 0.4, -0.1),
        fk_ceiling_fuzz=1e-7,
        known_gaps=ArmKnownGap(
            xfail_reason="three_parallel SP6 conditioning gap at q[4]=π/2 (Z1-specific; UR5 OK on same fuzz). Filed as #288."
        ),
        base_link="link00",
        ee_link="link06",
        solver="ikgeo.three_parallel",
        tier=0,
    ),
}


def load_manifest(path=None) -> dict[str, Arm]:
    return _ARMS
