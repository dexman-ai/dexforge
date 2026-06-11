from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ik_build  # noqa: E402


class WriteGeneratedFilesTests(unittest.TestCase):
    def test_written_file_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_root = Path(__file__).resolve().parents[2]
            source_urdf = tmp_path / "source.urdf"
            source_urdf.write_text("<robot name='demo' />\n", encoding="utf-8")
            oem_name = "OEM"

            written = ik_build.write_generated_files(
                output_dir=tmp_path,
                repo_root=repo_root,
                arm_name="demo",
                oem_name=oem_name,
                files={"arm.rs": "// arm\n", "ik_solver.rs": "// solver\n", "generate_code.py": "# codegen\n", "generate_test_data.py": "# test data\n"},
                ssik_artifact="# demo artifact\n",
                context="demo context\n",
                source_urdf=source_urdf,
            )

            topdir = [
                "drivers", "math", "cargo.toml", #"lib.rs",
                "drivers/OEM", "drivers/OEM/demo", 
                "drivers/OEM/mod.rs",
                "math/arm.rs", "math/kinematics.rs", "math/mod.rs", "math/postprocess.rs", "math/ssik_core.rs", "math/tolerances.rs",
                "drivers/OEM/demo/mod.rs", "drivers/OEM/demo/arm.rs", "drivers/OEM/demo/ik_solver.rs", 
                #"drivers/OEM/demo/consts", "drivers/OEM/demo/test_data.rs",
                "drivers/OEM/demo/codegen", "drivers/OEM/demo/codegen/demo.urdf", "drivers/OEM/demo/codegen/demo_ik.py",
                "drivers/OEM/demo/codegen/generate_code.py", "drivers/OEM/demo/codegen/generate_test_data.py"
            ]

            # check all expected files are written
            for item in topdir:
                path = tmp_path / item
                self.assertTrue(path.exists(), f"Expected file or directory not found: {path}")

    def test_arm_module_is_public_when_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_root = Path(__file__).resolve().parents[2]
            output_dir = tmp_path 
            oem_name = "OEM"
            (repo_root / "src" / "math").mkdir(parents=True, exist_ok=True)
            (repo_root / "src" / "math" / "mod.rs").write_text("// math mod\n", encoding="utf-8")

            ik_build.write_generated_files(
                output_dir=output_dir,
                repo_root=repo_root,
                arm_name="demo",
                oem_name=oem_name,
                files={"arm.rs": "// arm\n", "ik_solver.rs": "// solver\n"},
                ssik_artifact="# demo artifact\n",
                context="demo context\n",
                source_urdf=None,
            )

            mod_rs = (output_dir / "drivers" / oem_name / "demo" / "mod.rs").read_text(encoding="utf-8")
            self.assertIn("pub mod arm;", mod_rs)
            self.assertIn("pub mod ik_solver;", mod_rs)


if __name__ == "__main__":
    unittest.main()
