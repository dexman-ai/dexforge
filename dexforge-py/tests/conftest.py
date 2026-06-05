"""Shared pytest fixtures and CLI options for dexforge-py tests."""

from typing import NamedTuple


class ArmMetadata(NamedTuple):
    """Metadata for a robot arm supported by dexforge_py."""
    name: str
    dof: int
    sample_q: tuple
    fk_ceiling_fuzz: float


# Arm registry with metadata from SSIK manifest.
# This is the canonical list of arms supported by dexforge_py.
# Metadata (DOF, sample_q, fk_ceiling_fuzz) comes from ssik.prebuilt._manifest,
# which is available at https://github.com/personalrobotics/ssik/blob/main/src/ssik/prebuilt/_manifest.py
# and populated from MANIFEST.toml in that repository.
# These values are baked here to allow dexforge tests to run independently.
ARMS_METADATA = {
    "ur5": ArmMetadata(
        name="ur5",
        dof=6,
        sample_q=(0.3, -0.5, 0.7, 0.2, 0.4, -0.1),
        fk_ceiling_fuzz=1e-7,  # Closed-form 6R: ~1e-12 to 1e-13
    ),
    "piper": ArmMetadata(
        name="piper",
        dof=6,
        sample_q=(0.3, 0.7, -0.5, 0.2, 0.4, -0.1),
        fk_ceiling_fuzz=1e-4,  # RR 6R: ~1e-6
    ),
    "franka_panda": ArmMetadata(
        name="franka_panda",
        dof=7,
        sample_q=(0.2, 0.3, -0.4, -1.2, 0.3, 1.4, 0.5),
        fk_ceiling_fuzz=1e-4,  # 7R via jointlock + LM: ~1e-7 to 1e-9
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
