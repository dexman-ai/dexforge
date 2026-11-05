from __future__ import annotations

import pathlib
import sys

import pytest


THIS_DIR = pathlib.Path(__file__).resolve().parent
SHIM_ROOT = THIS_DIR / "ssik_shim"

p = THIS_DIR
while p != p.parent and not (p / "SSIK_VERSION.txt").exists():
    p = p.parent
WORKSPACE_ROOT = p

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(SHIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SHIM_ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--arm",
        default=None,
        help=(
            "Arm name to test (e.g. ur5, franka_panda, ur5_ik). "
            "Default: all arms in the OEM manifest."
        ),
    )
    parser.addoption(
        "--n",
        action="store",
        type=int,
        default=10,
        help="Number of random pose samples per arm",
    )
    parser.addoption(
        "--perf",
        action="store_true",
        default=False,
        help="Print wall-clock timing for Python and Rust solve().",
    )


@pytest.fixture
def n_samples(request):
    return int(request.config.getoption("--n"))
