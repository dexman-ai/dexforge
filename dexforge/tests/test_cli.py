"""Unit tests for the ``dexforge build`` standalone-driver compiler."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexforge import cli  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]


class PlanExportTests(unittest.TestCase):
    def test_plan_copies_math_kernels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export = cli.plan_export("ur5", Path(tmp), _REPO_ROOT)
        copied = {src.name for src, _ in export.copies}
        for name in cli._MATH_MODULES:
            self.assertIn(f"{name}.rs", copied)
        self.assertIn("mod.rs", copied)

    def test_plan_copies_driver_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export = cli.plan_export("ur5", Path(tmp), _REPO_ROOT)
        dsts = {dst.name for _, dst in export.copies}
        self.assertIn("ik_solver.rs", dsts)
        self.assertIn("generated_kinematics.rs", dsts)

    def test_rizon_pulls_in_shared_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export = cli.plan_export("rizon4", Path(tmp), _REPO_ROOT)
        copied = {src.name for src, _ in export.copies}
        self.assertIn("rizon_common.rs", copied)

    def test_non_rizon_has_no_shared_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export = cli.plan_export("ur5", Path(tmp), _REPO_ROOT)
        copied = {src.name for src, _ in export.copies}
        self.assertNotIn("rizon_common.rs", copied)

    def test_generated_files_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export = cli.plan_export("ur5", Path(tmp), _REPO_ROOT)
        gen = {dst.name for dst, _ in export.generated}
        self.assertEqual(gen, {"Cargo.toml", "lib.rs", "mod.rs"})

    def test_lib_rs_reexports_arm_entrypoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export = cli.plan_export("ur5", Path(tmp), _REPO_ROOT)
        lib = next(c for d, c in export.generated if d.name == "lib.rs")
        self.assertIn("pub use drivers::ur5::ik_solver::{fk, solve, TolerancePolicy};", lib)
        self.assertIn("#![no_std]", lib)

    def test_cargo_toml_has_only_minimal_deps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export = cli.plan_export("ur5", Path(tmp), _REPO_ROOT)
        cargo = next(c for d, c in export.generated if d.name == "Cargo.toml")
        self.assertIn("libm", cargo)
        self.assertIn("nalgebra", cargo)
        # No path/git dependency on dexforge itself.
        self.assertNotIn("path =", cargo)

    def test_unknown_arm_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            cli.plan_export("nope", Path("/tmp"), _REPO_ROOT)

    def test_missing_driver_error_points_at_urdf(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            cli.plan_export("nope", Path("/tmp"), _REPO_ROOT)
        self.assertIn("--urdf", str(ctx.exception))

    def test_shared_modules_recorded_for_rizon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export = cli.plan_export("rizon4", Path(tmp), _REPO_ROOT)
        self.assertIn("rizon_common", export.shared_modules)

    def test_drivers_mod_declares_shared_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export = cli.plan_export("rizon4", Path(tmp), _REPO_ROOT)
        mod = next(c for d, c in export.generated if d.name == "mod.rs")
        self.assertIn("pub mod rizon_common;", mod)
        self.assertIn("pub mod rizon4;", mod)


class WriteExportTests(unittest.TestCase):
    def test_write_materialises_self_contained_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ur5_driver"
            export = cli.plan_export("ur5", out, _REPO_ROOT)
            written = cli.write_export(export)
            self.assertEqual(set(written), set(export.files()))
            self.assertTrue((out / "Cargo.toml").is_file())
            self.assertTrue((out / "src" / "lib.rs").is_file())
            self.assertTrue((out / "src" / "math" / "ssik_core.rs").is_file())

    def test_written_tree_resolves_paths_within_crate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ur5_driver"
            cli.write_export(cli.plan_export("ur5", out, _REPO_ROOT))
            rust = "\n".join(p.read_text() for p in out.rglob("*.rs"))
        # Copied sources reference crate::math / crate::drivers, which resolve
        # within the standalone crate itself — never an external crate path back
        # into dexforge (e.g. `dexforge::` or the in-repo crate name `ssik_rust`).
        self.assertNotIn("dexforge::", rust)
        self.assertNotIn("ssik_rust", rust)


class RepoRootTests(unittest.TestCase):
    def test_find_repo_root_from_package(self) -> None:
        self.assertEqual(cli.find_repo_root(), _REPO_ROOT)

    def test_find_repo_root_raises_when_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            cli.find_repo_root(Path("/"))

    def test_is_within_detects_repo_paths(self) -> None:
        self.assertTrue(cli._is_within(_REPO_ROOT / "src" / "x", _REPO_ROOT))
        self.assertFalse(cli._is_within(Path("/tmp/elsewhere"), _REPO_ROOT))


class CliMainTests(unittest.TestCase):
    def _urdf(self, tmp: str) -> Path:
        urdf = Path(tmp) / "ur5.urdf"
        urdf.write_text("<robot name='ur5'></robot>", encoding="utf-8")
        return urdf

    def test_list_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            urdf = self._urdf(tmp)
            with mock.patch.object(
                cli, "generate_driver_from_urdf", return_value=(True, "ur5", "ok")
            ):
                with mock.patch.object(cli, "write_export") as we:
                    with redirect_stdout(io.StringIO()):
                        code = cli.main(
                            ["build", "--urdf", str(urdf), "--base", "b", "--ee", "e",
                             "-o", str(out), "--list"]
                        )
            self.assertEqual(code, 0)
            we.assert_not_called()
            self.assertFalse(out.exists())

    def test_refuses_output_inside_repo(self) -> None:
        inside = _REPO_ROOT / "build_out"
        with tempfile.TemporaryDirectory() as tmp:
            urdf = self._urdf(tmp)
            buf = io.StringIO()
            with redirect_stderr(buf):
                code = cli.main(
                    ["build", "--urdf", str(urdf), "--base", "b", "--ee", "e",
                     "-o", str(inside)]
                )
        self.assertEqual(code, 2)
        self.assertIn("inside the dexforge repository", buf.getvalue())

    def test_build_writes_and_skips_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ur5"
            urdf = self._urdf(tmp)
            with mock.patch.object(
                cli, "generate_driver_from_urdf", return_value=(True, "ur5", "ok")
            ):
                with mock.patch.object(cli, "verify_crate") as vc:
                    with redirect_stdout(io.StringIO()):
                        code = cli.main(
                            ["build", "--urdf", str(urdf), "--base", "b", "--ee", "e",
                             "-o", str(out), "--no-verify"]
                        )
            self.assertEqual(code, 0)
            vc.assert_not_called()
            self.assertTrue((out / "Cargo.toml").is_file())
            self.assertTrue((out / "src" / "lib.rs").is_file())

    def test_verify_failure_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ur5"
            urdf = self._urdf(tmp)
            with mock.patch.object(
                cli, "generate_driver_from_urdf", return_value=(True, "ur5", "ok")
            ):
                with mock.patch.object(cli, "verify_crate", return_value=(False, "boom")):
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        code = cli.main(
                            ["build", "--urdf", str(urdf), "--base", "b", "--ee", "e",
                             "-o", str(out)]
                        )
            self.assertEqual(code, 1)

    def test_no_subcommand_prints_help(self) -> None:
        with redirect_stdout(io.StringIO()):
            code = cli.main([])
        self.assertEqual(code, 1)


class UrdfBuildTests(unittest.TestCase):
    """End-to-end URDF → LLM → standalone driver path."""

    def test_urdf_invokes_generator_then_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            urdf = Path(tmp) / "ur5.urdf"
            urdf.write_text("<robot name='ur5'></robot>", encoding="utf-8")
            out = Path(tmp) / "driver"
            # Pretend the LLM generator produced the 'ur5' driver in-repo.
            with mock.patch.object(
                cli, "generate_driver_from_urdf", return_value=(True, "ur5", "ok")
            ) as gen:
                with redirect_stdout(io.StringIO()):
                    code = cli.main(
                        ["build", "--urdf", str(urdf), "--base", "base_link",
                         "--ee", "tool0", "-o", str(out), "--no-verify"]
                    )
            self.assertEqual(code, 0)
            gen.assert_called_once()
            # Exported as a single self-contained crate at -o (not a subdir).
            self.assertTrue((out / "Cargo.toml").is_file())
            self.assertTrue((out / "src" / "lib.rs").is_file())

    def test_urdf_requires_base_and_ee(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            urdf = Path(tmp) / "ur5.urdf"
            urdf.write_text("<robot/>", encoding="utf-8")
            buf = io.StringIO()
            with redirect_stderr(buf):
                code = cli.main(["build", "--urdf", str(urdf), "-o", str(Path(tmp) / "o")])
            self.assertEqual(code, 2)
            self.assertIn("--base", buf.getvalue())

    def test_urdf_missing_file_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            with redirect_stderr(buf):
                code = cli.main(
                    ["build", "--urdf", str(Path(tmp) / "nope.urdf"),
                     "--base", "b", "--ee", "e", "-o", str(Path(tmp) / "o")]
                )
            self.assertEqual(code, 2)
            self.assertIn("not found", buf.getvalue())

    def test_urdf_generation_failure_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            urdf = Path(tmp) / "ur5.urdf"
            urdf.write_text("<robot/>", encoding="utf-8")
            with mock.patch.object(
                cli, "generate_driver_from_urdf", return_value=(False, "ur5", "boom")
            ):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    code = cli.main(
                        ["build", "--urdf", str(urdf), "--base", "b", "--ee", "e",
                         "-o", str(Path(tmp) / "o")]
                    )
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
