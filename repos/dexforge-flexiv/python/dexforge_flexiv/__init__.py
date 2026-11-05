"""Flexiv Python bindings."""

from . import dexforge_flexiv as _ext

rizon4 = _ext.rizon4
rizon10 = _ext.rizon10

__all__ = ["rizon4", "rizon10"]