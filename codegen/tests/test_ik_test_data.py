"""Unit tests for shared ssik IK reference data (``codegen/ik_test_data.py``)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ik_test_data  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]


class IkTestDataTests(unittest.TestCase):
    def test_pack_and_unpack_roundtrip(self) -> None:
        cases = [
            ik_test_data.IKCase(
                q_fk=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                tf=[float(i) for i in range(16)],
                solutions=[
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
                ],
                singular=False,
            ),
            ik_test_data.IKCase(
                q_fk=[0.0] * 6,
                tf=[1.0 if i in (0, 5, 10, 15) else 0.0 for i in range(16)],
                solutions=[[0.0] * 6],
                singular=True,
            ),
        ]
        blob = ik_test_data.pack_ik_test_data(
            ssik_version="1.2.3",
            dof=6,
            cases=cases,
            allow_refinement=True,
            refinement_max_iters=60,
        )
        unpacked = ik_test_data.unpack_ik_test_data(blob)
        self.assertEqual(unpacked.ssik_version, "1.2.3")
        self.assertEqual(unpacked.dof, 6)
        self.assertTrue(unpacked.allow_refinement)
        self.assertEqual(unpacked.refinement_max_iters, 60)
        self.assertEqual(len(unpacked.cases), 2)
        self.assertEqual(unpacked.cases[0], cases[0])
        self.assertEqual(unpacked.cases[1], cases[1])

    def test_all_committed_arms_match_locked_ssik_version(self) -> None:
        locked = ik_test_data.get_locked_ssik_version(_REPO_ROOT)
        arms = ik_test_data.verify_all_arms_ssik_version(_REPO_ROOT)
        self.assertEqual(
            set(arms),
            {"franka_panda", "piper", "rizon10", "rizon4", "ur5", "xarm7", "z1"},
        )
        for arm in arms:
            ds = ik_test_data.load_arm_test_data(arm, _REPO_ROOT)
            self.assertEqual(ds.ssik_version, locked)
            self.assertEqual(len(ds.cases), 500)

    def test_version_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "codegen").mkdir()
            (root / "repos" / "dexhal" / "src" / "math").mkdir(parents=True)
            (root / "SSIK_VERSION.txt").write_text("ssik==9.9.9\n", encoding="utf-8")
            driver_dir = root / "repos" / "dexforge-ur" / "rust" / "src" / "drivers" / "ur5"
            ik_test_data.write_ik_test_data(
                driver_dir=driver_dir,
                arm="ur5",
                dof=6,
                cases=[
                    ik_test_data.IKCase(
                        q_fk=[0.0] * 6,
                        tf=[0.0] * 16,
                        solutions=[[0.0] * 6],
                    )
                ],
                ssik_version="1.2.3",
            )
            with self.assertRaises(RuntimeError) as ctx:
                ik_test_data.verify_all_arms_ssik_version(root)
            self.assertIn("1.2.3", str(ctx.exception))
            self.assertIn("9.9.9", str(ctx.exception))

    def test_unverifiable_ssik_producer_version_raises(self) -> None:
        import types
        from unittest import mock

        fake_ssik = types.ModuleType("ssik")
        with mock.patch.dict(sys.modules, {"ssik": fake_ssik}):
            with self.assertRaises(RuntimeError) as ctx:
                ik_test_data.get_installed_ssik_version(_REPO_ROOT)
            self.assertIn("__version__", str(ctx.exception))

        fake_ssik.__version__ = "unknown"
        with mock.patch.dict(sys.modules, {"ssik": fake_ssik}):
            with self.assertRaises(RuntimeError):
                ik_test_data.get_installed_ssik_version(_REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
