"""Unit tests for the ``dexforge build`` standalone-driver compiler."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dexforge_build as cli  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_minimal_crate(root: Path) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "Cargo.toml").write_text("[package]\nname = \"demo\"\nversion = \"0.1.0\"\n", encoding="utf-8")
    (root / "src" / "lib.rs").write_text("// lib\n", encoding="utf-8")


def _mock_generate_driver(*args, **kwargs):
    output_root = kwargs.get("output_dir")
    if output_root is None and len(args) > 4:
        output_root = args[4]
    if output_root is None:
        raise AssertionError("mock generate_ik expected output_dir")
    _write_minimal_crate(Path(output_root))
    return True, "ur5", "ok"


def _mock_generate_dynamics(*args, **kwargs):
    crate_root = args[2]
    driver = crate_root / "src" / "drivers" / "ur5"
    driver.mkdir(parents=True, exist_ok=True)
    (driver / "generated_kinematics.rs").write_text("// kin\n", encoding="utf-8")
    (driver / "generated_dynamics.rs").write_text("// dyn\n", encoding="utf-8")
    return True, "ok"


def _mock_ensure_parity(*args, **kwargs):
    crate_root = args[2] if len(args) > 2 else kwargs.get("crate_root")
    driver = Path(crate_root) / "src" / "drivers" / "ur5"
    driver.mkdir(parents=True, exist_ok=True)
    (driver / "dynamics_test_data.bin").write_bytes(b"\0" * 8)
    (driver / "dynamics_test_data.rs").write_text("// data\n", encoding="utf-8")
    (driver / "dynamics_parity.rs").write_text("// parity\n", encoding="utf-8")
    return True, "ok"


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
        self.assertIn("arm.rs", dsts)
        self.assertIn("ik_solver.rs", dsts)
        self.assertIn("generated_kinematics.rs", dsts)

    def test_rizon_pulls_in_shared_module(self) -> None:
        if cli._resolve_arm_driver_dir(_REPO_ROOT, "rizon4") is None:
            self.skipTest("rizon4 driver not visible in this checkout/sandbox")
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

    def test_generate_armmodel_impl_renders_arm_module(self) -> None:
        rendered = cli.generate_armmodel_impl("franka_panda", 7, 128)
        self.assertIn("pub struct FrankaPanda;", rendered)
        self.assertIn("impl ArmModel<7, 128> for FrankaPanda", rendered)
        self.assertIn("crate::drivers::franka_panda::ik_solver::fk(q)", rendered)
        self.assertIn(
            "crate::drivers::franka_panda::generated_dynamics::gravity(q, gravity)",
            rendered,
        )
        self.assertIn("fn flange_mass_matrix", rendered)
        self.assertIn("fn flange_coriolis", rendered)

    def test_lib_rs_reexports_arm_entrypoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export = cli.plan_export("ur5", Path(tmp), _REPO_ROOT)
        lib = next(c for d, c in export.generated if d.name == "lib.rs")
        self.assertIn("pub use drivers::ur5::ik_solver::{fk, solve, TolerancePolicy};", lib)
        # Full kinematics + dynamics suite re-exported at the crate root.
        self.assertIn("pub use drivers::ur5::generated_kinematics;", lib)
        self.assertIn("pub use drivers::ur5::generated_dynamics;", lib)
        self.assertIn("#![no_std]", lib)

    def test_lib_rs_reexports_dynamics_only_when_module_exists(self) -> None:
        # An arm without generated_dynamics.rs must still export a crate that compiles.
        self.assertNotIn(
            "generated_dynamics", cli._lib_rs("demo_ik_driver", "demo", with_dynamics=False)
        )
        for arm in ("ur5", "rizon4"):
            has_dynamics = (cli._driver_dir(_REPO_ROOT, arm) / "generated_dynamics.rs").is_file()
            with tempfile.TemporaryDirectory() as tmp:
                export = cli.plan_export(arm, Path(tmp), _REPO_ROOT)
            lib = next(c for d, c in export.generated if d.name == "lib.rs")
            self.assertEqual(
                f"pub use drivers::{arm}::generated_dynamics;" in lib, has_dynamics, arm
            )

    def test_wiring_restores_dynamics_reexport_once(self) -> None:
        # When the dynamics step later writes generated_dynamics.rs, lib.rs re-exports it once.
        with tempfile.TemporaryDirectory() as tmp:
            crate = Path(tmp)
            driver = crate / "src" / "drivers" / "demo"
            driver.mkdir(parents=True)
            (crate / "src" / "lib.rs").write_text(
                cli._lib_rs("demo_ik_driver", "demo", with_dynamics=False), encoding="utf-8"
            )
            (driver / "mod.rs").write_text("pub mod ik_solver;\n", encoding="utf-8")
            (driver / "generated_dynamics.rs").write_text("// dyn\n", encoding="utf-8")
            cli._wire_generated_dynamics_module(driver)
            cli._wire_generated_dynamics_module(driver)
            lib = (crate / "src" / "lib.rs").read_text(encoding="utf-8")
            self.assertEqual(lib.count("pub use drivers::demo::generated_dynamics;"), 1)

    def test_cargo_toml_has_only_minimal_deps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export = cli.plan_export("ur5", Path(tmp), _REPO_ROOT)
        cargo = next(c for d, c in export.generated if d.name == "Cargo.toml")
        self.assertIn("libm", cargo)
        self.assertIn("nalgebra", cargo)
        self.assertIn("doctest = false", cargo)
        # No path/git dependency on dexforge itself.
        self.assertNotIn("path =", cargo)

    def test_unknown_arm_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            cli.plan_export("nope", Path("/tmp"), _REPO_ROOT)

    def test_driver_dir_unknown_arm_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            cli._driver_dir(_REPO_ROOT, "nope")

    def test_missing_driver_error_points_at_urdf(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            cli.plan_export("nope", Path("/tmp"), _REPO_ROOT)
        self.assertIn("--urdf", str(ctx.exception))

    def test_shared_modules_recorded_for_rizon(self) -> None:
        if cli._resolve_arm_driver_dir(_REPO_ROOT, "rizon4") is None:
            self.skipTest("rizon4 driver not visible in this checkout/sandbox")
        with tempfile.TemporaryDirectory() as tmp:
            export = cli.plan_export("rizon4", Path(tmp), _REPO_ROOT)
        self.assertIn("rizon_common", export.shared_modules)

    def test_drivers_mod_declares_shared_modules(self) -> None:
        if cli._resolve_arm_driver_dir(_REPO_ROOT, "rizon4") is None:
            self.skipTest("rizon4 driver not visible in this checkout/sandbox")
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
            rust = "\n".join(p.read_text(encoding="utf-8") for p in out.rglob("*.rs"))
        # Copied sources reference crate::math / crate::drivers, which resolve
        # within the standalone crate itself — never an external crate path back
        # into dexforge (e.g. `dexforge::` or the in-repo crate name `dexforge_examples`).
        self.assertNotIn("dexforge::", rust)
        self.assertNotIn("dexforge_examples", rust)


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
                cli, "generate_ik", return_value=(True, "ur5", "ok")
            ), mock.patch.object(
                cli, "generate_dynamics_for_arm", return_value=(True, "ok")
            ) as dyn:
                with redirect_stdout(io.StringIO()):
                    code = cli.main(
                        ["build", "--urdf", str(urdf), "--base", "b", "--ee", "e",
                         "-o", str(out), "--list"]
                    )
            self.assertEqual(code, 0)
            dyn.assert_not_called()
            self.assertFalse(out.exists())

    def test_allows_output_inside_repo(self) -> None:
        with tempfile.TemporaryDirectory(dir=_REPO_ROOT) as tmp:
            inside = Path(tmp) / "build_out"
            urdf = self._urdf(tmp)
            with mock.patch.object(
                cli, "generate_ik", side_effect=_mock_generate_driver
            ), mock.patch.object(
                cli, "generate_dynamics_for_arm", side_effect=_mock_generate_dynamics
            ), mock.patch.object(
                cli, "ensure_dynamics_parity", side_effect=_mock_ensure_parity
            ), mock.patch.object(cli, "verify_crate", return_value=(True, "ok")):
                with redirect_stdout(io.StringIO()):
                    code = cli.main(
                        ["build", "--urdf", str(urdf), "--base", "b", "--ee", "e",
                         "-o", str(inside), "--no-verify"]
                    )
            self.assertEqual(code, 0)
            self.assertTrue((inside / "Cargo.toml").is_file())
            self.assertTrue((inside / "src" / "drivers" / "ur5" / "generated_kinematics.rs").is_file())

    def test_build_writes_and_skips_verify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ur5"
            urdf = self._urdf(tmp)
            with mock.patch.object(
                cli, "generate_ik", side_effect=_mock_generate_driver
            ), mock.patch.object(
                cli, "generate_dynamics_for_arm", side_effect=_mock_generate_dynamics
            ), mock.patch.object(
                cli, "ensure_dynamics_parity", side_effect=_mock_ensure_parity
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
                cli, "generate_ik", side_effect=_mock_generate_driver
            ), mock.patch.object(
                cli, "generate_dynamics_for_arm", side_effect=_mock_generate_dynamics
            ), mock.patch.object(
                cli, "ensure_dynamics_parity", side_effect=_mock_ensure_parity
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
            with mock.patch.object(
                cli, "generate_ik", side_effect=_mock_generate_driver
            ) as gen, mock.patch.object(
                cli, "generate_dynamics_for_arm", side_effect=_mock_generate_dynamics
            ) as dyn, mock.patch.object(
                cli, "ensure_dynamics_parity", side_effect=_mock_ensure_parity
            ) as parity:
                with redirect_stdout(io.StringIO()):
                    code = cli.main(
                        ["build", "--urdf", str(urdf), "--base", "base_link",
                         "--ee", "tool0", "-o", str(out), "--no-verify"]
                    )
            self.assertEqual(code, 0)
            gen.assert_called_once()
            # dexforge is a kinematics + dynamics compiler: dynamics must run.
            dyn.assert_called_once()
            parity.assert_called_once()
            self.assertTrue((out / "Cargo.toml").is_file())
            self.assertTrue((out / "src" / "lib.rs").is_file())

    def test_dynamics_failure_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            urdf = Path(tmp) / "ur5.urdf"
            urdf.write_text("<robot name='ur5'></robot>", encoding="utf-8")
            out = Path(tmp) / "driver"
            with mock.patch.object(
                cli, "generate_ik", side_effect=_mock_generate_driver
            ), mock.patch.object(
                cli, "generate_dynamics_for_arm", return_value=(False, "kaboom")
            ):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                    code = cli.main(
                        ["build", "--urdf", str(urdf), "--base", "base_link",
                         "--ee", "tool0", "-o", str(out), "--no-verify"]
                    )
            self.assertEqual(code, 1)
            self.assertIn("dynamics", err.getvalue())
            self.assertTrue((out / "Cargo.toml").exists())

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
                cli, "generate_ik", return_value=(False, "ur5", "boom")
            ):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    code = cli.main(
                        ["build", "--urdf", str(urdf), "--base", "b", "--ee", "e",
                         "-o", str(Path(tmp) / "o")]
                    )
            self.assertEqual(code, 1)

    def test_from_existing_skips_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Use the real in-repo ur5.urdf stem so plan_export can resolve the driver.
            urdf = _REPO_ROOT / "repos" / "dexforge-ur" / "rust" / "src" / "drivers" / "ur5" / "ur5.urdf"
            if not urdf.is_file():
                self.skipTest("ur5.urdf not available")
            out = Path(tmp) / "seeded"
            with mock.patch.object(cli, "generate_ik") as gen, mock.patch.object(
                cli, "generate_dynamics_for_arm", return_value=(True, "ok")
            ) as dyn, mock.patch.object(
                cli, "ensure_dynamics_parity", side_effect=_mock_ensure_parity
            ) as parity, mock.patch.object(
                cli, "verify_crate", return_value=(True, "ok")
            ):
                with redirect_stdout(io.StringIO()):
                    code = cli.main(
                        [
                            "build",
                            "--urdf",
                            str(urdf),
                            "--base",
                            "base_link",
                            "--ee",
                            "ee_link",
                            "-o",
                            str(out),
                            "--from-existing",
                            "--no-verify",
                        ]
                    )
            self.assertEqual(code, 0)
            gen.assert_not_called()
            # Dynamics may already be seeded by plan_export; SymPy is skipped then.
            dyn_rs = out / "src" / "drivers" / "ur5" / "generated_dynamics.rs"
            self.assertTrue(
                dyn.called or dyn_rs.is_file(),
                "expected generate_dynamics_for_arm or a seeded generated_dynamics.rs",
            )
            parity.assert_called_once()
            self.assertTrue((out / "Cargo.toml").is_file())
            self.assertTrue((out / "src" / "drivers" / "ur5" / "ik_solver.rs").is_file())
            self.assertIn("[workspace]", (out / "Cargo.toml").read_text())

    def test_from_existing_regenerates_when_urdf_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src_urdf = _REPO_ROOT / "repos" / "dexforge-ur" / "rust" / "src" / "drivers" / "ur5" / "ur5.urdf"
            if not src_urdf.is_file():
                self.skipTest("ur5.urdf not available")
            modified_urdf = Path(tmp) / "ur5.urdf"
            modified_urdf.write_text(
                src_urdf.read_text(encoding="utf-8") + "\n<!-- modified -->\n",
                encoding="utf-8",
            )
            out = Path(tmp) / "seeded_modified"
            with mock.patch.object(cli, "generate_ik") as gen, mock.patch.object(
                cli, "generate_dynamics_for_arm", side_effect=_mock_generate_dynamics
            ) as dyn, mock.patch.object(
                cli, "ensure_dynamics_parity", side_effect=_mock_ensure_parity
            ) as parity, mock.patch.object(
                cli, "verify_crate", return_value=(True, "ok")
            ):
                with redirect_stdout(io.StringIO()):
                    code = cli.main(
                        [
                            "build",
                            "--urdf",
                            str(modified_urdf),
                            "--base",
                            "base_link",
                            "--ee",
                            "ee_link",
                            "-o",
                            str(out),
                            "--from-existing",
                            "--no-verify",
                        ]
                    )
            self.assertEqual(code, 0)
            gen.assert_not_called()
            dyn.assert_called_once()
            parity.assert_called_once()
            self.assertTrue(parity.call_args.kwargs.get("force"))

    def test_from_existing_rejects_frame_or_kinematic_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src_urdf = _REPO_ROOT / "repos" / "dexforge-ur" / "rust" / "src" / "drivers" / "ur5" / "ur5.urdf"
            if not src_urdf.is_file():
                self.skipTest("ur5.urdf not available")
            out = Path(tmp) / "seeded_bad_ee"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                code = cli.main(
                    [
                        "build",
                        "--urdf",
                        str(src_urdf),
                        "--base",
                        "base_link",
                        "--ee",
                        "tool0",
                        "-o",
                        str(out),
                        "--from-existing",
                        "--no-verify",
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("--from-existing", err.getvalue())

            kin_modified = Path(tmp) / "ur5.urdf"
            kin_modified.write_text(
                src_urdf.read_text(encoding="utf-8").replace(
                    'xyz="0.0 0.0 0.089159"', 'xyz="0.0 0.0 0.250000"'
                ),
                encoding="utf-8",
            )
            out2 = Path(tmp) / "seeded_bad_kin"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err2:
                code2 = cli.main(
                    [
                        "build",
                        "--urdf",
                        str(kin_modified),
                        "--base",
                        "base_link",
                        "--ee",
                        "ee_link",
                        "-o",
                        str(out2),
                        "--from-existing",
                        "--no-verify",
                    ]
                )
            self.assertEqual(code2, 1)
            self.assertIn("kinematics", err2.getvalue())


if __name__ == "__main__":
    unittest.main()
