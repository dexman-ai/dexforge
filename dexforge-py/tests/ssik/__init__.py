"""ssik compatibility layer for dexforge_py.

This package provides ssik-compatible APIs wrapping dexforge_py, allowing
ssik tests to run unmodified against the Rust dexforge implementation.
"""

from ssik.core.tolerances import TolerancePolicy, DEFAULT_TOLERANCE_POLICY
from ssik.core.solution import Solution, RefinementMode

__all__ = ["TolerancePolicy", "DEFAULT_TOLERANCE_POLICY", "Solution", "RefinementMode"]
