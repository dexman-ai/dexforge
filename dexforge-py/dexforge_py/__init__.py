"""dexforge_py Python package structure.

This allows dexforge_py to be imported as a package with submodules.
The actual compiled extension is loaded when this package is imported.
"""

# Import all per-arm functions from the Rust extension
from .dexforge_py import (
    ur5_fk,
    ur5_solve,
    piper_fk,
    piper_solve,
    franka_panda_fk,
    franka_panda_solve,
    xarm7_fk,
    xarm7_solve,
    rizon4_fk,
    rizon4_solve,
    rizon10_fk,
    rizon10_solve,
)

__all__ = [
    "ur5_fk",
    "ur5_solve",
    "piper_fk",
    "piper_solve",
    "franka_panda_fk",
    "franka_panda_solve",
    "xarm7_fk",
    "xarm7_solve",
    "rizon4_fk",
    "rizon4_solve",
    "rizon10_fk",
    "rizon10_solve",
]
