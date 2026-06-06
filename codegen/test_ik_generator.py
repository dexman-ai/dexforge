from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ik_generator  # noqa: E402


class ResolveSourceUrdfTests(unittest.TestCase):
    def test_uses_fixture_when_urdf_not_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            codegen_dir = Path(tmp)
            fixture_dir = codegen_dir / "fixtures"
            fixture_dir.mkdir()
            fixture_path = fixture_dir / "demo.urdf"
            fixture_path.write_text("<robot name='demo'/>\n", encoding="utf-8")

            with patch.object(ik_generator, "_CODEGEN_DIR", codegen_dir):
                resolved = ik_generator._resolve_source_urdf("demo", None)

        self.assertEqual(resolved, fixture_path)

    def test_prefers_explicit_urdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            explicit = tmp_path / "explicit.urdf"
            explicit.write_text("<robot name='explicit'/>\n", encoding="utf-8")

            with patch.object(ik_generator, "_CODEGEN_DIR", tmp_path / "codegen"):
                resolved = ik_generator._resolve_source_urdf("demo", explicit)

        self.assertEqual(resolved, explicit)


if __name__ == "__main__":
    unittest.main()
