#!/usr/bin/env python3
"""Extract a multi-resolution TSDF mesh from a trained UniFusion model."""

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
    parser.add_argument("-s", "--mast3r-scene", "--mast3r_scene", type=Path, required=True)
    parser.add_argument("-m", "--model-path", "--model_path", type=Path, required=True)
    parser.add_argument("-o", "--output-path", "--output_path", type=Path)
    parser.add_argument("-c", "--config", default="default")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scene = args.mast3r_scene.expanduser().resolve()
    model = args.model_path.expanduser().resolve()
    output = (args.output_path or model / "tsdf_meshes").expanduser().resolve()
    config_path = ROOT / "configs/multiresolution_tsdf" / f"{args.config}.yaml"
    if not scene.is_dir() or not model.is_dir():
        raise SystemExit(f"scene and model directories must exist: {scene}, {model}")
    if not config_path.is_file():
        raise SystemExit(f"TSDF config does not exist: {config_path}")
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text())
    command = [sys.executable, str(ROOT / "2d-gaussian-splatting/render_multires.py"), "--source_path", str(scene), "--model_path", str(model), "--output_dir", str(output), "--depth_ratio", str(config["depth_ratio"]), "--num_cluster", str(config["num_cluster"]), "--mesh_res", str(config["mesh_res"]), "--multires_factors", *(str(value) for value in config["multires_factors"]), "--skip_train", "--skip_test"]
    print("$", shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

