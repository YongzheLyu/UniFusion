from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ReleaseSmokeTests(unittest.TestCase):
    def test_paper_config_has_six_sequences(self) -> None:
        config = yaml.safe_load((ROOT / "configs/paper/exorecon.yaml").read_text())
        self.assertEqual(
            list(config["sequences"]),
            ["bike", "cooking", "cpr", "dance", "piano", "soccer"],
        )

    def test_reproduction_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bike/dataset/frames_output").mkdir(parents=True)
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/reproduce_exorecon.py"), "--data-root", str(root), "--sequences", "bike", "--stages", "preprocess", "--dry-run"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("preprocess_temporal_data.py", result.stdout)

    def test_train_help(self) -> None:
        result = subprocess.run([sys.executable, str(ROOT / "train.py"), "--help"], cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--refinement-only", result.stdout)


if __name__ == "__main__":
    unittest.main()

