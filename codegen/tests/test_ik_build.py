from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ik_build  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]


class WriteGeneratedFilesTests(unittest.TestCase):
    def test_written_file_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_urdf = tmp_path / "source.urdf"
            source_urdf.write_text("<robot name='demo' />\n", encoding="utf-8")
            oem_name = "OEM"

            written = ik_build.write_generated_files(
                output_dir=tmp_path,
                repo_root=_REPO_ROOT,
                arm_name="demo",
                oem_name=oem_name,
                files={
                    "arm.rs": "// arm\n",
                    "ik_solver.rs": "// solver\n",
                    "generate_code.py": "# codegen\n",
                    "generate_test_data.py": "# test data\n",
                },
                ssik_artifact="# demo artifact\n",
                context="demo context\n",
                source_urdf=source_urdf,
            )

            expected = [
                "Cargo.toml",
                "src/lib.rs",
                "src/math/mod.rs",
                "src/math/arm.rs",
                "src/math/kinematics.rs",
                "src/math/postprocess.rs",
                "src/math/ssik_core.rs",
                "src/math/tolerances.rs",
                "src/drivers/mod.rs",
                "src/drivers/demo/mod.rs",
                "src/drivers/demo/arm.rs",
                "src/drivers/demo/ik_solver.rs",
                "src/drivers/demo/codegen/demo.urdf",
                "src/drivers/demo/codegen/demo_ik.py",
                "src/drivers/demo/codegen/generate_code.py",
                "src/drivers/demo/codegen/generate_test_data.py",
            ]
            for item in expected:
                path = tmp_path / item
                self.assertTrue(path.exists(), f"Expected path missing: {path}")
            self.assertIn("arm.rs", written)
            self.assertIn("ik_solver.rs", written)

    def test_arm_module_is_public_when_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            oem_name = "OEM"

            ik_build.write_generated_files(
                output_dir=tmp_path,
                repo_root=_REPO_ROOT,
                arm_name="demo",
                oem_name=oem_name,
                files={"arm.rs": "// arm\n", "ik_solver.rs": "// solver\n"},
                ssik_artifact="# demo artifact\n",
                context="demo context\n",
                source_urdf=None,
            )

            mod_rs = (tmp_path / "src" / "drivers" / "demo" / "mod.rs").read_text(
                encoding="utf-8"
            )
            self.assertIn("pub mod arm;", mod_rs)
            self.assertIn("pub mod ik_solver;", mod_rs)
            drivers_mod = (tmp_path / "src" / "drivers" / "mod.rs").read_text(
                encoding="utf-8"
            )
            self.assertIn("pub mod demo;", drivers_mod)


    def test_shared_rizon_helper_is_copied_and_declared(self) -> None:
        # A driver modelled on the Rizon example imports crate::drivers::rizon_common.
        for uses_helper in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                solver = (
                    "use crate::drivers::rizon_common::{self, Model};\n"
                    if uses_helper
                    else "// solver\n"
                )
                ik_build.write_generated_files(
                    output_dir=tmp_path,
                    repo_root=_REPO_ROOT,
                    arm_name="demo",
                    oem_name="OEM",
                    files={"ik_solver.rs": solver},
                    ssik_artifact="# demo artifact\n",
                    context="demo context\n",
                    source_urdf=None,
                )
                helper = tmp_path / "src" / "drivers" / "rizon_common.rs"
                drivers_mod = (tmp_path / "src" / "drivers" / "mod.rs").read_text(
                    encoding="utf-8"
                )
                self.assertEqual(helper.is_file(), uses_helper)
                self.assertEqual("pub mod rizon_common;" in drivers_mod, uses_helper)
                self.assertIn("pub mod demo;", drivers_mod)


if __name__ == "__main__":
    unittest.main()
