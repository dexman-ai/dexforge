from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ik_build  # noqa: E402


class WriteGeneratedFilesTests(unittest.TestCase):
    def test_writes_driver_provenance_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_root = tmp_path / "repo"
            output_dir = tmp_path / "out"
            source_urdf = tmp_path / "source.urdf"
            source_urdf.write_text("<robot name='demo' />\n", encoding="utf-8")

            with patch.object(ik_build, "_REPO_ROOT", repo_root):
                written = ik_build.write_generated_files(
                    output_dir=output_dir,
                    arm_name="demo",
                    files={},
                    target_py_code="# demo artifact\n",
                    context="demo context\n",
                    source_urdf=source_urdf,
                )

            self.assertEqual((output_dir / "demo_ik.py").read_text(encoding="utf-8"), "# demo artifact\n")
            self.assertEqual((output_dir / "demo_context.txt").read_text(encoding="utf-8"), "demo context\n")
            self.assertEqual((repo_root / "src" / "drivers" / "demo" / "demo_ik.py").read_text(encoding="utf-8"), "# demo artifact\n")
            self.assertEqual((repo_root / "src" / "drivers" / "demo" / "demo.urdf").read_text(encoding="utf-8"), "<robot name='demo' />\n")
            self.assertEqual(written["py"], output_dir / "demo_ik.py")
            self.assertEqual(written["ctx"], output_dir / "demo_context.txt")
            self.assertEqual(written["driver_py"], repo_root / "src" / "drivers" / "demo" / "demo_ik.py")
            self.assertEqual(written["driver_urdf"], repo_root / "src" / "drivers" / "demo" / "demo.urdf")


if __name__ == "__main__":
    unittest.main()
