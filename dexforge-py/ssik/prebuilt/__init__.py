"""ssik.prebuilt compatibility layer for dexforge_py.

This package provides ssik-compatible per-arm IK modules wrapping dexforge_py,
allowing ssik tests to run unmodified against the Rust dexforge implementation.

Available arms:
- ur5_ik
- piper_ik
- franka_panda_ik
- xarm7_ik
- rizon4_ik
- rizon10_ik
"""

__all__ = [
    "ur5_ik",
    "piper_ik",
    "franka_panda_ik",
    "xarm7_ik",
    "rizon4_ik",
    "rizon10_ik",
]
