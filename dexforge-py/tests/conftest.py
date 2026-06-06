"""Shared pytest fixtures and CLI options for dexforge-py tests."""

from typing import NamedTuple


class ArmMetadata(NamedTuple):
    """Metadata for a robot arm, mirroring ssik's MANIFEST.toml structure."""
    dof: int
    sample_q: list[float]
    fk_ceiling_fuzz: float


# Arm metadata registry: canonical arm name -> metadata.
#
# Metadata is pulled from ssik's MANIFEST.toml and kept in sync with the
# dexforge_py Python bindings. Each arm that is exposed in dexforge_py should
# have an entry here.
#
# Fields:
#   - dof: Degrees of freedom
#   - sample_q: A known in-limits joint configuration for sanity tests
#   - fk_ceiling_fuzz: Maximum FK residual under 500-pose Hypothesis sweep
#
# FK ceilings come from ssik's per-arm calibrated residual floor and are used
# by test_dexforge_uniform_fuzz.py. These are the worst-case FK residuals
# observed under aggressive non-singular fuzzing, not the average. See
# ssik/docs/arm_coverage.md for per-arm precision floor details.
ARMS_METADATA = {
    "ur5": ArmMetadata(
        dof=6,
        sample_q=[0.3, -0.5, 0.7, 0.2, 0.4, -0.1],
        fk_ceiling_fuzz=1e-7,
    ),
    "piper": ArmMetadata(
        dof=6,
        sample_q=[0.3, 0.7, -0.5, 0.2, 0.4, -0.1],
        fk_ceiling_fuzz=1e-4,
    ),
    "franka_panda": ArmMetadata(
        dof=7,
        sample_q=[0.2, 0.3, -0.4, -1.2, 0.3, 1.4, 0.5],
        fk_ceiling_fuzz=1e-4,
    ),
    "xarm7": ArmMetadata(
        dof=7,
        sample_q=[0.0, -0.6, 0.4, 0.8, -0.5, 0.9, -0.2],
        fk_ceiling_fuzz=1e-4,
    ),
    "rizon4": ArmMetadata(
        dof=7,
        sample_q=[0.0, -0.4, 0.6, -0.9, 0.5, 1.0, -0.3],
        fk_ceiling_fuzz=1e-4,
    ),
    "rizon10": ArmMetadata(
        dof=7,
        sample_q=[0.0, -0.5, 0.5, -0.8, 0.4, 1.1, -0.2],
        fk_ceiling_fuzz=1e-4,
    ),
}


def pytest_addoption(parser):
    parser.addoption(
        "--arm",
        default=None,
        help=(
            "Arm name to test (e.g. ur5, franka_panda). "
            "Default: all known arms."
        ),
    )
    parser.addoption(
        "--n",
        type=int,
        default=50,
        help="Number of random poses to test per arm. Default: 50.",
    )
    parser.addoption(
        "--perf",
        action="store_true",
        default=False,
        help="Print wall-clock timing for Python and Rust solve().",
    )
