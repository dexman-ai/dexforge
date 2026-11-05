"""Universal Robots Python bindings."""

from . import dexforge_ur as _ext

ur5 = _ext.ur5

__all__ = ["ur5"]