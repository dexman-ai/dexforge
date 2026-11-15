from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ik_prompt  # noqa: E402


class GenerateContextTests(unittest.TestCase):
    def test_generate_context_ur5_has_non_empty_sections(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        urdf_path = (
            repo_root
            / "repos"
            / "dexforge-ur"
            / "rust"
            / "src"
            / "drivers"
            / "ur5"
            / "ur5.urdf"
        )

        self.assertTrue(urdf_path.is_file(), f"Missing URDF fixture: {urdf_path}")

        with tempfile.TemporaryDirectory() as tmp:
            entry = ik_prompt.generate_context(
                urdf_path=urdf_path,
                base_link="base_link",
                ee_link="ee_link",
                output_dir=Path(tmp),
                repo_root=repo_root,
                example_arm="ur5",
            )

        context = entry.context
        marker_pattern = r"^--- \[SECTION (\d+)\]"
        matches = list(re.finditer(marker_pattern, context, flags=re.MULTILINE))

        self.assertGreaterEqual(len(matches), 6, "Expected numbered context sections")

        for i, match in enumerate(matches):
            section_num = int(match.group(1))
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
            section_body = context[start:end].strip()
            self.assertTrue(section_body, f"Section {section_num} is empty")

        # save to file for manual inspection if needed
        # with open("test_context.txt", "w", encoding="utf-8") as f:
        #     f.write(context)


class ExtractRustSignaturesTests(unittest.TestCase):
    def test_does_not_truncate_array_type_signatures(self) -> None:
        rust_source = """
/// 3x3 matrix times 3-vector.
pub fn mat3_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [0.0, 0.0, 0.0]
}

/// Subproblem 1: angle `theta` (least-squares) with `Rot(k, theta) p ~= q`.
pub fn sp1_solve(k: &[f64; 3], p: &[f64; 3], q: &[f64; 3]) -> f64 {
    0.0
}
""".lstrip()

        with tempfile.TemporaryDirectory() as tmp:
            rust_file = Path(tmp) / "sample.rs"
            rust_file.write_text(rust_source, encoding="utf-8")

            extracted = ik_prompt.extract_rust_signatures(rust_file)

        self.assertIn(
            "pub fn mat3_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3]",
            extracted,
        )
        self.assertIn(
            "pub fn sp1_solve(k: &[f64; 3], p: &[f64; 3], q: &[f64; 3]) -> f64",
            extracted,
        )

class FindBestExampleTests(unittest.TestCase):
    """One-shot examples must be self-contained and match their array size."""

    _ARTIFACTS = (
        "DOF = 7\nhusty_pfurner",
        "DOF = 7\nsp6",
        "DOF = 6\nraghavan_roth",
        "DOF = 6\npieper",
    )

    def test_examples_do_not_use_unvendored_helpers(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        for artifact in self._ARTIFACTS:
            arm, max_solutions, _dof = ik_prompt.find_best_example(artifact)
            driver_dir = ik_prompt._resolve_repo_driver_dir(repo_root, arm)
            if driver_dir is None:
                self.skipTest(f"{arm} driver not visible in this checkout")
            rust = (driver_dir / "ik_solver.rs").read_text(encoding="utf-8")
            scripts = ik_prompt._load_example_scripts(arm, repo_root)
            self.assertNotIn("rizon_common", rust, arm)
            self.assertNotIn("_rizon_codegen", scripts, arm)
            declared = re.search(r"MAX_SOLUTIONS:\s*usize\s*=\s*(\d+)", rust)
            if declared is not None:
                self.assertEqual(int(declared.group(1)), max_solutions, arm)


if __name__ == "__main__":
    unittest.main()
