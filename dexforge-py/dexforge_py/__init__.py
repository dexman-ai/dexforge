"""dexforge_py Python package structure.

The compiled extension exposes per-arm submodules, so the public API is
available as ``dexforge_py.ur5.solve``, ``dexforge_py.ur5.fk``, and so on.
"""

from . import dexforge_py as _ext

ur5 = _ext.ur5
piper = _ext.piper
franka_panda = _ext.franka_panda
xarm7 = _ext.xarm7
rizon4 = _ext.rizon4
rizon10 = _ext.rizon10

__all__ = [
    "ur5",
    "piper",
    "franka_panda",
    "xarm7",
    "rizon4",
    "rizon10",
]
