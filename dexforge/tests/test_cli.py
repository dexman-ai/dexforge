"""Unit tests for the ``dexforge build`` CLI orchestrator."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexforge import cli  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]


class BuildPlanTests(unittest.TestCase):
    def test_default_plan_covers_every_arm_kinematics(self) -> None:
        plan = cli.build_plan()
        kin = {s.arm for s in plan if s.stage == "kinematics"}
        self.assertEqual(kin, set(cli.ARMS))

    def test_stages_are_ordered_consts_before_kinematics(self) -> None:
        plan = cli.build_plan()
        stages = [s.stage for s in plan]
        last_consts = max(i for i, s in enumerate(stages) if s == "consts")
        first_kin = min(i for i, s in enumerate(stages) if s == "kinematics")
        self.assertLess(last_consts, first_kin)

    def test_ur5_has_no_consts_step(self) -> None:
        plan = cli.build_plan(arms=["ur5"])
        self.assertEqual([s.stage for s in plan if s.stage == "consts"], [])

    def test_kinematics_command_targets_generate_dynamics(self) -> None:
        plan = cli.build_plan(arms=["ur5"], stages=["kinematics"])
        self.assertEqual(len(plan), 1)
        step = plan[0]
        self.assertEqual(step.script, "codegen/generate_dynamics.py")
        self.assertIn("ur5_ik", step.args)
        self.assertIn("src/drivers/ur5/generated_kinematics.rs", step.args)

    def test_dynamics_test_data_only_for_xarm7(self) -> None:
        plan = cli.build_plan(stages=["dynamics_test_data"])
        self.assertEqual([s.arm for s in plan], ["xarm7"])

    def test_stage_filter_limits_plan(self) -> None:
        plan = cli.build_plan(arms=["ur5"], stages=["test_data"])
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].script, "codegen/generate_test_data.py")

    def test_every_referenced_script_exists(self) -> None:
        for step in cli.build_plan():
            self.assertTrue(
                (_REPO_ROOT / step.script).is_file(),
                f"missing generator script for {step.arm}/{step.stage}: {step.script}",
            )


class RepoRootTests(unittest.TestCase):
    def test_find_repo_root_from_package(self) -> None:
        self.assertEqual(cli.find_repo_root(), _REPO_ROOT)

    def test_find_repo_root_raises_when_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            cli.find_repo_root(Path("/"))


class CliMainTests(unittest.TestCase):
    def test_list_does_not_run_subprocess(self) -> None:
        with mock.patch.object(cli.subprocess, "run") as run:
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = cli.main(["build", "--arm", "ur5", "--list"])
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertIn("build plan", buf.getvalue())

    def test_unknown_arm_is_rejected(self) -> None:
        buf = io.StringIO()
        with redirect_stderr(buf):
            code = cli.main(["build", "--arm", "nope"])
        self.assertEqual(code, 2)
        self.assertIn("unknown arm", buf.getvalue())

    def test_build_runs_steps_and_reports_success(self) -> None:
        completed = mock.Mock(returncode=0)
        with mock.patch.object(cli.subprocess, "run", return_value=completed) as run:
            with redirect_stdout(io.StringIO()):
                code = cli.main(["build", "--arm", "ur5", "--stage", "kinematics"])
        self.assertEqual(code, 0)
        run.assert_called_once()
        argv = run.call_args.args[0]
        self.assertIn("generate_dynamics.py", " ".join(argv))

    def test_build_aborts_on_failure(self) -> None:
        completed = mock.Mock(returncode=1)
        with mock.patch.object(cli.subprocess, "run", return_value=completed) as run:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = cli.main(["build", "--stage", "kinematics"])
        self.assertEqual(code, 1)
        # Aborts after the first failing step rather than running them all.
        self.assertEqual(run.call_count, 1)

    def test_no_subcommand_prints_help(self) -> None:
        with redirect_stdout(io.StringIO()):
            code = cli.main([])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
