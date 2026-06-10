from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ik_build  # noqa: E402


class WriteGeneratedFilesTests(unittest.TestCase):
    def test_writes_driver_provenance_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_root = tmp_path / "repo"
            output_dir = tmp_path / "out"
            (repo_root / "src" / "math").mkdir(parents=True, exist_ok=True)
            (repo_root / "src" / "math" / "mod.rs").write_text("// math mod\n", encoding="utf-8")
            source_urdf = tmp_path / "source.urdf"
            source_urdf.write_text("<robot name='demo' />\n", encoding="utf-8")

            written = ik_build.write_generated_files(
                output_dir=output_dir,
                repo_root=repo_root,
                arm_name="demo",
                files={},
                target_py_code="# demo artifact\n",
                context="demo context\n",
                source_urdf=source_urdf,
            )

            self.assertEqual((output_dir / "demo_ik.py").read_text(encoding="utf-8"), "# demo artifact\n")
            self.assertEqual((output_dir / "demo_context.txt").read_text(encoding="utf-8"), "demo context\n")
            self.assertEqual((output_dir / "src" / "math" / "mod.rs").read_text(encoding="utf-8"), "// math mod\n")
            self.assertEqual((output_dir / "src" / "drivers" / "demo" / "demo_ik.py").read_text(encoding="utf-8"), "# demo artifact\n")
            self.assertEqual((output_dir / "src" / "drivers" / "demo" / "demo.urdf").read_text(encoding="utf-8"), "<robot name='demo' />\n")
            self.assertEqual(written["py"], output_dir / "demo_ik.py")
            self.assertEqual(written["ctx"], output_dir / "demo_context.txt")
            self.assertEqual(written["driver_py"], output_dir / "src" / "drivers" / "demo" / "demo_ik.py")
            self.assertEqual(written["driver_urdf"], output_dir / "src" / "drivers" / "demo" / "demo.urdf")

    def test_arm_module_is_public_when_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_root = tmp_path / "repo"
            output_dir = tmp_path / "out"
            (repo_root / "src" / "math").mkdir(parents=True, exist_ok=True)
            (repo_root / "src" / "math" / "mod.rs").write_text("// math mod\n", encoding="utf-8")

            ik_build.write_generated_files(
                output_dir=output_dir,
                repo_root=repo_root,
                arm_name="demo",
                files={"arm.rs": "// arm\n", "ik_solver.rs": "// solver\n"},
                target_py_code="# demo artifact\n",
                context="demo context\n",
                source_urdf=None,
            )

            mod_rs = (output_dir / "src" / "drivers" / "demo" / "mod.rs").read_text(encoding="utf-8")
            self.assertIn("pub mod arm;", mod_rs)
            self.assertIn("pub mod ik_solver;", mod_rs)


if __name__ == "__main__":
    unittest.main()
