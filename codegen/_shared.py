"""Master copy of shared Python binding helpers for OEM packages.

This module is intentionally pure Python and does not import any Rust
extension module. OEM packages can copy this file into their own repos to
reuse conversion and tolerance helper logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = [
    "Pose4x4",
    "as_joint_list",
    "as_pose_4x4",
    "ensure_ssik_tolerances",
    "solution_list_to_numpy",
    "pose_list_to_numpy",
]


Pose4x4 = list[list[float]]


def _as_float_list(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values]


def _tolist_if_available(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


def as_pose_4x4(value: Any, name: str = "T_target") -> Pose4x4:
    """Normalize a pose to a 4x4 nested list of floats.

    Accepts either a NumPy array or a plain Python sequence of sequences.
    Raises ValueError when the shape is not 4x4.
    """

    rows = _tolist_if_available(value)
    if not isinstance(rows, Sequence):
        raise ValueError(f"{name} must be a 4x4 matrix-like object")

    pose: Pose4x4 = []
    for row in rows:
        if not isinstance(row, Sequence):
            raise ValueError(f"{name} must be a 4x4 matrix-like object")
        pose.append(_as_float_list(row))

    if len(pose) != 4 or any(len(row) != 4 for row in pose):
        raise ValueError(f"{name} must be a 4x4 matrix-like object")

    return pose


def as_joint_list(value: Any, dof: int, name: str = "q") -> list[float]:
    """Normalize a joint vector to a Python list of floats."""

    values = _tolist_if_available(value)
    if not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of length {dof}")

    joint_list = _as_float_list(values)
    if len(joint_list) != dof:
        raise ValueError(f"{name} must contain {dof} values")
    return joint_list


@dataclass(frozen=True)
class _Tolerances:
    fk_atol: float
    dedup_atol: float


def ensure_ssik_tolerances(
    policy: Any | None,
    *,
    fk_atol_default: float = 1e-5,
    dedup_atol_default: float = 1e-3,
) -> _Tolerances:
    """Extract solver tolerances from an ssik-compatible policy object."""

    if policy is None:
        return _Tolerances(fk_atol_default, dedup_atol_default)

    fk_atol = getattr(policy, "subproblem_numerical", fk_atol_default)
    dedup_atol = getattr(policy, "subproblem_dedup", dedup_atol_default)
    return _Tolerances(float(fk_atol), float(dedup_atol))


def _numpy_module() -> Any | None:
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return None
    return np


def pose_list_to_numpy(pose: Pose4x4) -> Any:
    """Convert a nested-list pose to a NumPy array when NumPy is available."""

    np = _numpy_module()
    if np is None:
        return pose
    return np.array(pose, dtype=np.float64)


def solution_list_to_numpy(solution: Sequence[float]) -> Any:
    """Convert a joint solution to a NumPy array when NumPy is available."""

    np = _numpy_module()
    if np is None:
        return [float(value) for value in solution]
    return np.array(solution, dtype=np.float64)
