"""Tests for ``codegen/ssik_artifact.py`` (family detection and chain mapping)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("ssik")

_CODEGEN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODEGEN))
import ssik_artifact as SA  # noqa: E402


@pytest.mark.parametrize(
    "solver, source, family",
    [
        ("ikgeo.three_parallel", "", "sp6"),
        ("ikgeo.spherical_two_parallel", "", "sp6"),
        ("ikgeo.general_6r", "", "general_6r"),
        ("jointlock.seven_r", "from ssik.ikgeo import sp6", "jointlock_sp6"),
        ("jointlock.seven_r", "from ssik.raghavan_roth import solve", "jointlock_general_6r"),
        ("seven_r.srs", "", "srs"),
    ],
)
def test_detect_family(solver, source, family):
    assert SA.detect_family(solver, source).key == family


def test_detect_family_rejects_unknown_solver():
    with pytest.raises(ValueError):
        SA.detect_family("mystery.solver", "")


# Prebuilt artifacts whose families back an in-repo reference driver.
@pytest.mark.parametrize(
    "prebuilt, dof, max_solutions, reference",
    [
        ("ur5_ik", 6, 16, "ur5"),
        ("z1_ik", 6, 16, "ur5"),
        ("piper_ik", 6, 16, "piper"),
        ("franka_panda_ik", 7, 128, "franka_panda"),
        ("xarm7_ik", 7, 128, "franka_panda"),
        ("rizon4_ik", 7, 256, "rizon4"),
        ("rizon10_ik", 7, 256, "rizon4"),
    ],
)
def test_inspect_prebuilt(prebuilt, dof, max_solutions, reference):
    info = SA.inspect_artifact(prebuilt)
    assert info.dof == dof
    assert info.max_solutions == max_solutions
    assert info.reference_arm == reference
    assert info.arm == prebuilt.removesuffix("_ik")
    assert len(info.joint_limits) == dof
    assert len(info.joint_names) == dof


def _rot(axis, q):
    """4x4 homogeneous rotation about a unit *axis* by *q* (Rodrigues)."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    out = np.eye(4)
    out[:3, :3] = np.eye(3) + np.sin(q) * kx + (1.0 - np.cos(q)) * (kx @ kx)
    return out


@pytest.mark.parametrize("prebuilt", ["ur5_ik", "piper_ik", "franka_panda_ik", "rizon4_ik"])
def test_nominal_chain_reproduces_artifact_fk(prebuilt):
    mod = SA.load_artifact(prebuilt)
    origins, axes, flange = SA.nominal_chain(mod)
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, size=len(axes))
        t = np.eye(4)
        for origin, axis, qi in zip(origins, axes, q):
            t = t @ origin @ _rot(axis, qi)
        t = t @ flange
        np.testing.assert_allclose(t, np.asarray(mod.fk(q)).reshape(4, 4), atol=1e-12)
