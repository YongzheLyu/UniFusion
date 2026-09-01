#!/usr/bin/env python3
"""Run MASt3R-SfM with a checked, reproducible command line."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--source-path", "--source_path", type=Path, required=True)
    parser.add_argument("-o", "--output-path", "--output_path", type=Path)
    parser.add_argument("--n-images", "--n_images", type=int)
    parser.add_argument("--image-idx", "--image_idx", type=int, nargs="*")
    parser.add_argument("--randomize-images", "--randomize_images", action="store_true")
    parser.add_argument("-c", "--config", default="unposed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = args.source_path.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"source path does not exist: {source}")
    if args.n_images is not None and args.image_idx is not None:
        parser.error("--n-images and --image-idx are mutually exclusive")
    use_all = args.n_images is None and args.image_idx is None
    n_images = -1 if use_all else (len(args.image_idx) if args.image_idx is not None else args.n_images)
    output = (args.output_path or ROOT / "outputs" / source.name / "mast3r_sfm").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    config_path = ROOT / "configs/mast3r" / f"{args.config}.yaml"
    if not config_path.is_file():
        raise SystemExit(f"MASt3R config does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    for key in ("weights_path", "retrieval_model"):
        candidate = Path(config[key]).expanduser()
        config[key] = str(candidate if candidate.is_absolute() else (ROOT / candidate).resolve())

    command = [
        sys.executable, str(ROOT / "mast3r/run_mast3r.py"),
        "--scene_path", str(source), "--output_dir", str(output),
        "--weights_path", config["weights_path"], "--retrieval_model", config["retrieval_model"],
        "--min_conf_thr", str(config["min_conf_thr"]), "--matching_conf_thr", str(config["matching_conf_thr"]),
        "--n_coarse_iterations", str(config["n_coarse_iterations"]), "--n_refinement_iterations", str(config["n_refinement_iterations"]),
        "--TSDF_thresh", str(config["TSDF_thresh"]), "--n_images", str(n_images),
        "--image_size", str(config["image_size"]), "--max_window_size", str(config["max_window_size"]),
        "--max_refid", str(config["max_refid"]), "--output_conf_thr", str(config["output_conf_thr"]),
    ]
    for key, flag in (("fix_focal", "--fix_focal"), ("fix_principal_point", "--fix_principal_point"), ("fix_rotation", "--fix_rotation"), ("fix_translation", "--fix_translation"), ("use_calibrated_poses", "--use_calibrated_poses"), ("save_glb", "--save_glb"), ("align_camera_locations", "--align_camera_locations")):
        if config.get(key):
            command.append(flag)
    if use_all:
        command.append("--use_all_images")
    if args.image_idx is not None:
        command.extend(["--image_idx", *(str(index) for index in args.image_idx)])
    if args.randomize_images:
        command.append("--randomize_images")

    print("$", shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

