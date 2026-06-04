"""Shared pytest fixtures and CLI options for dexforge-py tests."""


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
