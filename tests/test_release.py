from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.reproduce_exorecon import Runner


ROOT = Path(__file__).resolve().parents[1]


class ReleaseSmokeTests(unittest.TestCase):
    def test_runner_does_not_overwrite_completion_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.bin"
            args = argparse.Namespace(resume=False, dry_run=False)
            runner = Runner(args, {})
            runner.run(
                "test",
                "sequence",
                [sys.executable, "-c", f"from pathlib import Path; Path({str(artifact)!r}).write_bytes(b'payload')"],
                artifact,
            )
            self.assertEqual(artifact.read_bytes(), b"payload")

    def test_paper_config_has_six_sequences(self) -> None:
        config = yaml.safe_load((ROOT / "configs/paper/exorecon.yaml").read_text())
        self.assertEqual(
            list(config["sequences"]),
            ["bike", "cooking", "cpr", "dance", "piano", "soccer"],
        )

    def test_model_path_uses_configured_experiment_name(self) -> None:
        config = yaml.safe_load((ROOT / "configs/smoke/bike.yaml").read_text())
        args = argparse.Namespace(
            data_root=Path("/tmp/unifusion-test"),
            results_root=Path("/tmp/unifusion-results"),
        )
        paths = Runner(args, config).sequence_paths("bike")
        self.assertEqual(paths["model"].name, "free_gaussians_smoke_rank4")

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

    def test_prepare_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bike/grouped_by_cams").mkdir(parents=True)
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/reproduce_exorecon.py"), "--data-root", str(root), "--sequences", "bike", "--stages", "prepare", "--dry-run"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("prepare_dataset_pipeline.py", result.stdout)

    def test_train_help(self) -> None:
        result = subprocess.run([sys.executable, str(ROOT / "train.py"), "--help"], cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--refinement-only", result.stdout)

    def test_refinement_receives_depth_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "depth-checkpoints"
            result = subprocess.run(
                [
                    sys.executable, str(ROOT / "train.py"),
                    "--source-path", str(root),
                    "--output-path", str(root / "output"),
                    "--depthanythingv2-checkpoint-dir", str(checkpoint),
                    "--refinement-only", "--dry-run",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(checkpoint), result.stdout)
            self.assertIn("--seed 10086", result.stdout)


if __name__ == "__main__":
    unittest.main()
