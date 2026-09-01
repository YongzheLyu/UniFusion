#!/usr/bin/env python3
"""Extract static or temporally deformed tetrahedral meshes from a 2DGS model."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = REPO_ROOT / "2d-gaussian-splatting" / "extract_mesh_adaptive_tsdf.py"


def latest_iteration(model_path: Path) -> int:
    iterations = []
    for path in (model_path / "point_cloud").glob("iteration_*"):
        try:
            iterations.append(int(path.name.rsplit("_", 1)[1]))
        except ValueError:
            pass
    if not iterations:
        raise FileNotFoundError(f"No point_cloud/iteration_* found under {model_path}")
    return max(iterations)


def dataset_timestamps(source_path: Path):
    """Match multipleview_dataset's train split and time normalization."""
    camera_dir = source_path / "cam01"
    if not camera_dir.is_dir():
        raise FileNotFoundError(f"Cannot infer timestamps: missing {camera_dir}")
    frame_count = sum(path.is_file() for path in camera_dir.iterdir())
    if frame_count == 0:
        raise RuntimeError(f"Cannot infer timestamps: {camera_dir} is empty")
    return [frame / frame_count for frame in range(0, frame_count, 3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-o", "--output_path")
    parser.add_argument("-c", "--config", default="default")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--downsample_ratio", type=float, default=0.5)
    parser.add_argument("--truncation_margin", type=float, default=None,
                        help="Override the adaptive TSDF truncation margin from the config.")
    parser.add_argument("--no_filter_mesh", action="store_true",
                        help="Disable Gaussian-scale edge filtering after marching tetrahedra.")
    parser.add_argument("--timestamp", type=float, action="append")
    parser.add_argument("--all_timestamps", action="store_true")
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--max_timestamps", type=int, default=None,
                        help="Limit the number of selected timestamps after applying frame_step.")
    parser.add_argument("--dense_data_path")
    parser.add_argument("--interpolate_views", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip timestamps whose final binary-search mesh already exists.")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    source_path = Path(args.source_path).resolve()
    output_root = Path(args.output_path or model_path / "tetra_meshes_dynamic").resolve()
    config_path = REPO_ROOT / "configs" / "adaptive_tetrahedralization" / f"{args.config}.yaml"
    config = yaml.safe_load(config_path.read_text())
    iteration = latest_iteration(model_path) if args.iteration < 0 else args.iteration
    timestamps = args.timestamp or []
    if args.all_timestamps:
        timestamps = dataset_timestamps(source_path)[::args.frame_step]
    if args.max_timestamps is not None:
        if args.max_timestamps <= 0:
            parser.error("--max_timestamps must be positive")
        timestamps = timestamps[:args.max_timestamps]
    if not timestamps:
        timestamps = [None]

    for timestamp in timestamps:
        frame_dir = output_root if timestamp is None else output_root / f"t_{timestamp:.6f}"
        final_mesh = frame_dir / "tetra_mesh_binary_search_7.ply"
        if args.skip_existing and final_mesh.exists():
            print(f"Skipping existing mesh: {final_mesh}", flush=True)
            continue
        cmd = [
            sys.executable, str(EXTRACTOR), "--source_path", str(source_path),
            "--model_path", str(model_path), "--iteration", str(iteration),
            "--downsample_ratio", str(args.downsample_ratio), "--gaussian_flatness", str(config["gaussian_flatness"]),
            "--depth_ratio", str(config["depth_ratio"]), "--texture_mesh", "--output_dir", str(frame_dir),
            "--interpolation_mode", config["interpolation_mode"], "--truncation_margin",
            str(args.truncation_margin if args.truncation_margin is not None else config["truncation_margin"]),
            "--softmax_temperature", str(config["softmax_temperature"]),
            "--n_neighbors_to_interpolate", str(config["n_neighbors_to_interpolate"]),
            "--n_interpolated_cameras_for_each_neighbor", str(config["n_interpolated_cameras_for_each_neighbor"]),
        ]
        flags = {
            "filter_mesh": "--filter_mesh", "interpolate_depth": "--interpolate_depth",
            "weight_interpolation_by_depth_gradient": "--weight_interpolation_by_depth_gradient",
            "use_dilated_depth": "--use_dilated_depth", "use_sdf_tolerance": "--use_sdf_tolerance",
            "use_unbiased_tsdf": "--use_unbiased_tsdf", "use_binary_opacity": "--use_binary_opacity",
            "filter_with_depth_gradient": "--filter_with_depth_gradient",
            "filter_with_normal_consistency": "--filter_with_normal_consistency",
            "weight_by_softmax": "--weight_by_softmax",
            "weight_by_normal_consistency": "--weight_by_normal_consistency",
        }
        cmd += [
            flag for key, flag in flags.items()
            if config[key] and not (key == "filter_mesh" and args.no_filter_mesh)
        ]
        if timestamp is not None:
            cmd += ["--timestamp", str(timestamp)]
        if args.interpolate_views:
            cmd.append("--interpolate_cameras")
        if args.dense_data_path:
            cmd += ["--dense_data_path", str(Path(args.dense_data_path).resolve())]
        print("Running:", " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
