"""dexforge — command-line tooling for the dexforge code generators.

The package currently exposes a single entry point, :mod:`dexforge.cli`, whose
``dexforge build`` command runs the inverse-kinematics and dynamics code
generation pipeline that turns the locked ``ssik`` prebuilt artifacts into the
committed ``#![no_std]`` Rust drivers under ``src/drivers/<arm>/``.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
