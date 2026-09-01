#!/usr/bin/env python3
"""Refine aligned charts into a 2D Gaussian representation."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rich.console import Console


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-s", "--mast3r-scene", "--mast3r_scene", type=Path, required=True)
    parser.add_argument("-o", "--output-path", "--output_path", type=Path)
    parser.add_argument("--white-background", "--white_background", action="store_true")
    parser.add_argument("--dense-data-path", "--dense_data_path", type=Path)
    parser.add_argument("--depthanythingv2-checkpoint-dir", "--depthanythingv2_checkpoint_dir", type=Path, default=ROOT / "Depth-Anything-V2/checkpoints")
    parser.add_argument("--depthanything-encoder", "--depthanything_encoder", default="vitl")
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--dense-regul", "--dense_regul", choices=["default", "strong", "weak", "none"], default="default")
    parser.add_argument("--preprocessed-dir", "--preprocessed_dir", type=Path)
    parser.add_argument("-c", "--config", default="default")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scene = args.mast3r_scene.expanduser().resolve()
    if not scene.is_dir():
        raise SystemExit(f"MASt3R scene does not exist: {scene}")
    output = args.output_path
    if output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ROOT / "outputs" / scene.name / f"refined_free_gaussians_{timestamp}"
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    config_path = ROOT / "configs/free_gaussians_refinement" / f"{args.config}.yaml"
    if not config_path.is_file():
        raise SystemExit(f"refinement config does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    required = ("iterations", "densify_until_iter", "opacity_reset_interval", "depth_ratio", "normal_consistency_from", "distortion_from", "use_mip_filter")
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"missing keys in {config_path}: {', '.join(missing)}")

    command = [
        sys.executable,
        str(ROOT / "2d-gaussian-splatting/train_with_charts_.py"),
        "-s", str(scene),
        "-m", str(output),
        "--iterations", str(config["iterations"]),
        "--densify_until_iter", str(config["densify_until_iter"]),
        "--opacity_reset_interval", str(config["opacity_reset_interval"]),
        "--depth_ratio", str(config["depth_ratio"]),
        "--normal_consistency_from", str(config["normal_consistency_from"]),
        "--distortion_from", str(config["distortion_from"]),
        "--depthanythingv2_checkpoint_dir", str(args.depthanythingv2_checkpoint_dir),
        "--depthanything_encoder", args.depthanything_encoder,
        "--dense_regul", args.dense_regul,
        "--seed", str(args.seed),
    ]
    if config["use_mip_filter"]:
        command.append("--use_mip_filter")
    if args.white_background:
        command.append("--white_background")
    if args.dense_data_path:
        command.extend(["--dense_data_path", str(args.dense_data_path.expanduser().resolve())])
    if args.preprocessed_dir:
        command.extend(["--preprocessed_priors_dir", str(args.preprocessed_dir.expanduser().resolve())])

    console = Console(width=120)
    console.print(f"[INFO] Output: {output}")
    console.print(f"[INFO] Command: {shlex.join(command)}")
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
