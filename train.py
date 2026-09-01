#!/usr/bin/env python3
"""Run the UniFusion static/sparse-view reconstruction pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-s", "--source-path", "--source_path", type=Path, required=True)
    parser.add_argument("-o", "--output-path", "--output_path", type=Path)
    parser.add_argument("--n-images", "--n_images", type=int)
    parser.add_argument("--image-idx", "--image_idx", type=int, nargs="*")
    parser.add_argument("--randomize-images", action="store_true")
    parser.add_argument("--dense-supervision", "--dense_supervision", action="store_true")
    parser.add_argument("--dense-regul", "--dense_regul", choices=["default", "strong", "weak", "none"], default="default")
    parser.add_argument("--use-multires-tsdf", "--use_multires_tsdf", action="store_true")
    parser.add_argument("--no-interpolated-views", "--no_interpolated_views", action="store_true")
    parser.add_argument("--sfm-config", "--sfm_config", default="unposed", choices=["unposed", "posed"])
    parser.add_argument("--alignment-config", "--alignment_config", default="default")
    parser.add_argument("--depth-model", "--depth_model", default="depthanythingv2")
    parser.add_argument("--depthanythingv2-checkpoint-dir", "--depthanythingv2_checkpoint_dir", type=Path, default=ROOT / "Depth-Anything-V2" / "checkpoints")
    parser.add_argument("--depthanything-encoder", "--depthanything_encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--free-gaussians-config", "--free_gaussians_config")
    parser.add_argument("--tsdf-config", "--tsdf_config", default="default")
    parser.add_argument("--preprocessed-dir", "--preprocessed_dir", type=Path)
    parser.add_argument("--exp", help="Stable experiment name; defaults to a UTC timestamp")
    parser.add_argument("--tetra-config", "--tetra_config", default="default")
    parser.add_argument("--tetra-downsample-ratio", "--tetra_downsample_ratio", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--sfm-only", "--sfm_only", action="store_true")
    stages.add_argument("--alignment-only", "--alignment_only", action="store_true")
    stages.add_argument("--refinement-only", "--refinement_only", action="store_true")
    stages.add_argument("--mesh-only", "--mesh_only", action="store_true")
    return parser


def append_option(command: list[str], option: str, value: object | None) -> None:
    if value is not None:
        command.extend([option, str(value)])


def run(command: list[str], *, dry_run: bool) -> None:
    print("$", shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = build_parser().parse_args()
    source = args.source_path.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"source path does not exist: {source}")
    output = (args.output_path or ROOT / "outputs" / source.name).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    experiment = args.exp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sfm_scene = output / "mast3r_sfm"
    gaussian_output = output / f"free_gaussians_{experiment}"

    if args.dense_supervision and args.sfm_config != "posed":
        print("[warning] dense supervision requires posed SfM; using --sfm-config posed")
        args.sfm_config = "posed"
    if args.free_gaussians_config is None:
        args.free_gaussians_config = "long" if args.dense_supervision else "default"

    sfm = [sys.executable, str(ROOT / "scripts" / "run_sfm.py"), "--source_path", str(source), "--output_path", str(sfm_scene), "--config", args.sfm_config]
    append_option(sfm, "--n_images", args.n_images)
    if args.image_idx is not None:
        sfm.extend(["--image_idx", *(str(i) for i in args.image_idx)])
    if args.randomize_images:
        sfm.append("--randomize_images")

    alignment = [sys.executable, str(ROOT / "scripts" / "align_charts.py"), "--source_path", str(sfm_scene), "--mast3r_scene", str(sfm_scene), "--output_path", str(sfm_scene), "--config", args.alignment_config, "--depth_model", args.depth_model, "--depthanythingv2_checkpoint_dir", str(args.depthanythingv2_checkpoint_dir), "--depthanything_encoder", args.depthanything_encoder]

    refinement = [sys.executable, str(ROOT / "scripts" / "refine_free_gaussians.py"), "--mast3r_scene", str(sfm_scene), "--output_path", str(gaussian_output), "--config", args.free_gaussians_config, "--dense_regul", args.dense_regul]
    refinement.extend([
        "--depthanythingv2_checkpoint_dir", str(args.depthanythingv2_checkpoint_dir),
        "--depthanything_encoder", args.depthanything_encoder,
        "--seed", str(args.seed),
    ])
    append_option(refinement, "--dense_data_path", source if args.dense_supervision else None)
    append_option(refinement, "--preprocessed_dir", args.preprocessed_dir)

    if args.use_multires_tsdf:
        mesh = [sys.executable, str(ROOT / "scripts" / "extract_tsdf_mesh.py"), "--mast3r_scene", str(sfm_scene), "--model_path", str(gaussian_output), "--output_path", str(output / "tsdf_meshes"), "--config", args.tsdf_config]
    else:
        mesh = [sys.executable, str(ROOT / "scripts" / "extract_tetra_mesh.py"), "--mast3r_scene", str(sfm_scene), "--model_path", str(gaussian_output), "--output_path", str(output / "tetra_meshes"), "--config", args.tetra_config, "--downsample_ratio", str(args.tetra_downsample_ratio)]
        if not args.no_interpolated_views:
            mesh.append("--interpolate_views")
        append_option(mesh, "--dense_data_path", source if args.dense_supervision else None)

    commands = {"sfm": sfm, "alignment": alignment, "refinement": refinement, "mesh": mesh}
    selected = ["sfm", "alignment", "refinement", "mesh"]
    for flag, stage in ((args.sfm_only, "sfm"), (args.alignment_only, "alignment"), (args.refinement_only, "refinement"), (args.mesh_only, "mesh")):
        if flag:
            selected = [stage]

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "source_path": str(source),
        "output_path": str(output),
        "experiment": experiment,
        "stages": selected,
        "commands": {name: shlex.join(commands[name]) for name in selected},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for stage in selected:
        print(f"\n[{stage}]")
        run(commands[stage], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
