"""dexforge code-generation tools.

Deterministic scripts and templates used by a coding agent to generate dexhal
OEM driver crates (see ``codegen/README.md`` and ``codegen/new_driver.md``):
:mod:`codegen.ssik_artifact` (build / inspect ssik artifacts),
:mod:`codegen.scaffold` (vendor crate + ``arm.rs`` from templates),
:mod:`codegen.generate_dynamics` (SymPy FK / Jacobian / dynamics),
:mod:`codegen.generate_dynamics_test_data` (Pinocchio ground truth) and
:mod:`codegen.ik_test_data` (ssik IK reference datasets).
"""
