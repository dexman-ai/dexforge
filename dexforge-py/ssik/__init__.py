"""ssik compatibility layer for dexforge_py.

This package provides ssik-compatible APIs wrapping dexforge_py, allowing
ssik tests to run unmodified against the Rust dexforge implementation.
"""

from ssik.core.tolerances import TolerancePolicy

__all__ = ["TolerancePolicy"]
