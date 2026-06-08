"""dexforge — a stand-alone IK *driver compiler*.

The package exposes a single entry point, :mod:`dexforge.cli`, whose
``dexforge build`` command compiles an arm's analytical inverse-kinematics
driver into a self-contained Rust crate written *outside* the dexforge
repository. The emitted crate copies the shared math kernels and per-arm driver
sources and generates its own ``Cargo.toml``/``lib.rs``, so it builds and runs
without the dexforge repository or this package.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
