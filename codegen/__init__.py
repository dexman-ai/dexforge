"""dexforge code-generation scripts.

This package collects the offline compiler scripts that turn a URDF into a
stand-alone Rust driver crate. The end-to-end driver-build CLI lives in
:mod:`codegen.dexforge_build` (exposed as the ``dexforge build`` command) and
contains both the LLM-assisted IK generator (``_run_ik_generator_core``) and the
symbolic kinematics + dynamics compiler (``_run_generate_dynamics_core``). The IK
generator's support modules (:mod:`codegen.ik_build`, :mod:`codegen.ik_llm`,
:mod:`codegen.ik_prompt`) remain as separate files.
"""
