"""Standardized solver return type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = ["RefinementMode", "Solution"]

RefinementMode = Literal["none", "lm"]


@dataclass(frozen=True)
class Solution:
    q: NDArray[np.float64]
    fk_residual: float
    refinement_used: RefinementMode = "none"
