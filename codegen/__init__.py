"""dexforge code-generation scripts.

This package collects the offline compiler scripts that turn a URDF into a
stand-alone Rust driver crate: the LLM-assisted IK generator
(:mod:`codegen.ik_generator`), the symbolic kinematics + dynamics compiler
(:mod:`codegen.generate_dynamics`), and the end-to-end driver-build CLI
(:mod:`codegen.dexforge_build`) exposed as the ``dexforge build`` command.
"""
