from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ik_generator  # noqa: E402


class BuildArgParserTests(unittest.TestCase):
    def test_allows_artifact_and_urdf_together(self) -> None:
        parser = ik_generator._build_arg_parser()
        with tempfile.TemporaryDirectory() as tmp:
            urdf_path = Path(tmp) / "robot.urdf"
            urdf_path.write_text("<robot name='demo'/>\n", encoding="utf-8")

            args = parser.parse_args(
                [
                    "--artifact",
                    "demo_ik",
                    "--urdf",
                    str(urdf_path),
                ]
            )

        self.assertEqual(args.artifact, "demo_ik")
        self.assertEqual(args.urdf, urdf_path)


if __name__ == "__main__":
    unittest.main()
