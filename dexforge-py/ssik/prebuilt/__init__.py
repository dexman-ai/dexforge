"""dexforge_py.prebuilt - Direct access to per-arm IK functions.

This package provides a common interface to dexforge_py's per-arm functions,
making them compatible with the test infrastructure that expects solve() and fk()
methods on arm modules.

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
