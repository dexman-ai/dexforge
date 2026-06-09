"""dexforge code-generation scripts.

This package collects the offline compiler scripts that turn a URDF into a
stand-alone Rust crate: the LLM-assisted IK generator
(:mod:`codegen.generate_ik`), the symbolic kinematics + dynamics compiler
(:mod:`codegen.generate_kinematics_dynamics`), and the end-to-end build CLI
(:mod:`codegen.dexforge_build`) exposed as the ``dexforge build`` command.
"""
