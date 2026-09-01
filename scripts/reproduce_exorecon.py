#!/usr/bin/env python3
"""Reproduce UniFusion training and evaluation on ExoReconstruction sequences."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
ALL_STAGES = (
    "prepare", "preprocess", "align", "organize", "finalize",
    "train", "render", "evaluate", "summarize",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    result.add_argument("--config", type=Path, default=ROOT / "configs/paper/exorecon.yaml")
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--semidense-root", type=Path)
    result.add_argument(
        "--evaluate-depth",
        action="store_true",
        help="Also run optional semidense depth evaluation (not part of the main RGB table)",
    )
    result.add_argument("--results-root", type=Path, default=ROOT / "results")
    result.add_argument("--sequences", nargs="+", help="Defaults to every sequence in the config")
    result.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=list(ALL_STAGES))
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--allow-missing", action="store_true", help="Skip missing sequences instead of failing")
    return result


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"configuration does not exist: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("sequences"), dict):
        raise SystemExit(f"invalid experiment configuration: {path}")
    return data


class Runner:
    def __init__(self, args: argparse.Namespace, config: dict[str, Any]) -> None:
        self.args = args
        self.config = config
        self.commands: list[dict[str, str]] = []

    def run(self, stage: str, sequence: str, command: list[str], marker: Path | None = None) -> None:
        if self.args.resume and marker is not None and marker.exists():
            print(f"[{sequence}:{stage}] resume: found {marker}")
            return
        printable = shlex.join(command)
        self.commands.append({"sequence": sequence, "stage": stage, "command": printable})
        print(f"[{sequence}:{stage}] $ {printable}", flush=True)
        if self.args.dry_run:
            return
        subprocess.run(command, cwd=ROOT, check=True)
        if marker is not None:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(datetime.now(timezone.utc).isoformat() + "\n")

    def sequence_paths(self, sequence: str) -> dict[str, Path]:
        rank = int(self.config["temporal_alignment"]["rank"])
        sequence_config = self.config["sequences"][sequence]
        iteration = int(sequence_config.get("iteration", self.config["experiment"]["default_iteration"]))
        dataset = self.args.data_root / sequence / "dataset"
        raw = self.args.data_root / sequence / self.config["preparation"]["raw_subdir"]
        priors = dataset / f"resfield_rank{rank}_priors"
        final = dataset / "final_dataset"
        model = final / f"free_gaussians_resfield_rank{rank}"
        result = model / "test" / f"ours_{iteration}"
        return {
            "dataset": dataset,
            "raw": raw,
            "renamed": dataset / "renamed_images",
            "frames": dataset / "frames_output",
            "preprocessed": dataset / "frames_output" / "preprocessed_temporal_data.pkl",
            "priors": priors,
            "final": final,
            "model": model,
            "result": result,
        }

    def validate_sequence(self, sequence: str, paths: dict[str, Path]) -> bool:
        if "prepare" in self.args.stages and not paths["raw"].is_dir():
            message = f"missing raw multi-camera directory for {sequence}: {paths['raw']}"
            if self.args.allow_missing:
                print(f"[skip] {message}")
                return False
            raise SystemExit(message)
        if "prepare" not in self.args.stages and not paths["dataset"].is_dir():
            message = f"missing dataset directory for {sequence}: {paths['dataset']}"
            if self.args.allow_missing:
                print(f"[skip] {message}")
                return False
            raise SystemExit(message)
        if "preprocess" in self.args.stages and not paths["frames"].is_dir():
            raise SystemExit(f"missing frames directory: {paths['frames']}")
        return True

    def process(self, sequence: str) -> None:
        paths = self.sequence_paths(sequence)
        if not self.validate_sequence(sequence, paths):
            return
        alignment = self.config["temporal_alignment"]
        refinement = self.config["refinement"]
        preparation = self.config["preparation"]
        seq_config = self.config["sequences"][sequence]

        if "prepare" in self.args.stages:
            last_frame = int(seq_config["input_frames"]) - 1
            marker = paths["frames"] / f"frame_{last_frame:05d}" / "mast3r_sfm" / "cameras.json"
            command = [
                sys.executable, str(ROOT / "scripts/prepare_dataset_pipeline.py"),
                str(paths["raw"]), "--output_base", str(paths["dataset"]),
                "--only-step", "1", "2", "--sfm-config", str(preparation["sfm_config"]),
                "--num-workers", str(preparation["sfm_workers"]), "--skip-existing-sfm",
            ]
            if self.args.dry_run:
                command.append("--dry-run")
            self.run("prepare", sequence, command, marker)

        if "preprocess" in self.args.stages:
            command = [sys.executable, str(ROOT / "scripts/preprocess_temporal_data.py"), "--data_dir", str(paths["frames"]), "--output_path", str(paths["preprocessed"]), "--start_frame", str(seq_config["start_frame"]), "--end_frame", str(seq_config["end_frame"]), "--config", alignment["config"]]
            self.run("preprocess", sequence, command, paths["preprocessed"])

        if "align" in self.args.stages:
            if not self.args.dry_run and not paths["preprocessed"].is_file():
                raise SystemExit(f"missing preprocessing artifact: {paths['preprocessed']}")
            command = [sys.executable, str(ROOT / "scripts/align_charts_temporal_from_preprocessed.py"), "--preprocessed_data", str(paths["preprocessed"]), "--output_path", str(paths["priors"]), "--temporal_encoding_type", str(alignment["encoding_type"]), "--temporal_encoding_dim", str(alignment["encoding_dim"]), "--rank", str(alignment["rank"])]
            command.extend(["--start_frame", str(seq_config["start_frame"]), "--end_frame", str(seq_config["end_frame"])])
            if alignment.get("iterations") is not None:
                command.extend(["--alignment_iterations", str(alignment["iterations"])])
            if alignment.get("use_occlusion_loss"):
                command.append("--use_occlusion_loss")
            if alignment.get("occlusion_loss_weight") is not None:
                command.extend(["--occlusion_loss_weight", str(alignment["occlusion_loss_weight"])])
            if alignment.get("depth_order_loss_type"):
                command.extend(["--depth_order_loss_type", str(alignment["depth_order_loss_type"])])
            if alignment.get("use_ssi_loss"):
                command.append("--use_ssi_loss")
            if alignment.get("ssi_loss_weight") is not None:
                command.extend(["--ssi_loss_weight", str(alignment["ssi_loss_weight"])])
            self.run("align", sequence, command, paths["priors"] / ".align-complete")

        if "organize" in self.args.stages:
            marker = paths["priors"] / ".organize-complete"
            if self.args.resume and marker.exists():
                print(f"[{sequence}:organize] resume: found {marker}")
            else:
                command = [sys.executable, str(ROOT / "scripts/organize_priors.py"), str(paths["priors"]), "--priors-folder", str(paths["priors"])]
                if self.args.dry_run:
                    command.append("--dry-run")
                self.run("organize", sequence, command)
                if not self.args.dry_run:
                    if not (paths["priors"] / "charts_data.npz").exists():
                        charts = next(paths["priors"].glob("charts/**/charts_data.npz"), None)
                        if charts is None:
                            raise SystemExit(f"organize stage produced no charts_data.npz in {paths['priors']}")
                        shutil.copy2(charts, paths["priors"] / "charts_data.npz")
                    marker.write_text(datetime.now(timezone.utc).isoformat() + "\n")

        if "finalize" in self.args.stages:
            marker = paths["final"] / "mast3r_sfm" / "preprocessed_priors" / "charts_data.npz"
            command = [
                sys.executable, str(ROOT / "scripts/prepare_dataset_pipeline.py"),
                str(paths["raw"]), "--output_base", str(paths["dataset"]),
                "--only-step", "6", "--renamed-images-dir", str(paths["renamed"]),
                "--priors-dir", str(paths["priors"]),
                "--final-dataset-dir", str(paths["final"]),
            ]
            if self.args.dry_run:
                command.append("--dry-run")
            self.run("finalize", sequence, command, marker)

        if "train" in self.args.stages:
            refinement_config = seq_config.get("refinement_config", refinement["default_config"])
            command = [sys.executable, str(ROOT / "train.py"), "--source-path", str(paths["dataset"]), "--output-path", str(paths["final"]), "--preprocessed-dir", str(paths["priors"]), "--exp", str(refinement["experiment_name"]), "--free-gaussians-config", str(refinement_config), "--refinement-only"]
            if self.args.dry_run:
                command.append("--dry-run")
            self.run("train", sequence, command, paths["model"] / ".train-complete")

        if "render" in self.args.stages:
            command = [sys.executable, str(ROOT / "2d-gaussian-splatting/render.py"), "--model_path", str(paths["model"]), "--iteration", str(seq_config.get("iteration", self.config["experiment"]["default_iteration"])), "--skip_train"]
            self.run("render", sequence, command, paths["result"] / ".render-complete")

        if "evaluate" in self.args.stages:
            rgb = [sys.executable, str(ROOT / "scripts/evaluate_rendering.py"), "--input_dir", str(paths["result"]), "--eval_mode", "full", "--output", str(paths["result"] / "eval_rendering.json")]
            self.run("evaluate-rgb", sequence, rgb, paths["result"] / "eval_rendering.json")
            if self.args.evaluate_depth:
                if self.args.semidense_root is None:
                    raise SystemExit("--semidense-root is required with --evaluate-depth")
                semidense = self.args.semidense_root / seq_config["semidense_id"] / "semidense_points.csv.gz"
                if not self.args.dry_run and not semidense.is_file():
                    raise SystemExit(f"missing sparse-depth ground truth: {semidense}")
                depth = [sys.executable, str(ROOT / "scripts/evaluate_sparse_depth.py"), "--mode", "deformable", "--cameras-json", str(paths["model"] / "cameras.json"), "--semidense-points", str(semidense), "--depth-dir", str(paths["result"] / "depth"), "--output-dir", str(paths["result"] / "sparse_depth_eval"), "--device", str(self.config["evaluation"]["device"]), "--start-frame", str(seq_config["start_frame"]), "--end-frame", str(seq_config["end_frame"]), "--train-stride", str(self.config["evaluation"]["train_stride"]), "--frames-per-camera", str(seq_config["frames_per_camera"])]
                self.run("evaluate-depth", sequence, depth, paths["result"] / "sparse_depth_eval/depth_eval_results.json")

    def summarize(self, sequences: list[str]) -> None:
        if "summarize" not in self.args.stages:
            return
        rows: list[dict[str, Any]] = []
        for sequence in sequences:
            result = self.sequence_paths(sequence)["result"]
            rgb_path = result / "eval_rendering.json"
            depth_path = result / "sparse_depth_eval/depth_eval_results.json"
            if self.args.dry_run:
                print(f"[{sequence}:summarize] read {rgb_path}")
                continue
            if not rgb_path.is_file():
                if self.args.allow_missing:
                    continue
                raise SystemExit(f"missing RGB evaluation output for {sequence}: {rgb_path}")
            rgb = json.loads(rgb_path.read_text())["summary"]
            row = {"sequence": sequence, "psnr": rgb["avg_psnr"], "ssim": rgb["avg_ssim"], "lpips": rgb["avg_lpips"]}
            if self.args.evaluate_depth:
                if not depth_path.is_file():
                    raise SystemExit(f"missing depth evaluation output for {sequence}: {depth_path}")
                depth = json.loads(depth_path.read_text())["per_camera"]["overall"]
                row.update({"depth_abs_rel": depth["abs_rel"], "depth_abs_rel_t95": depth["abs_rel_t95"], "depth_rmse": depth["rmse"], "depth_delta_1": depth["delta_1"], "depth_delta_2": depth["delta_2"], "depth_delta_3": depth["delta_3"], "depth_frames": depth["n_frames"]})
            rows.append(row)
        if not rows:
            return
        numeric = [key for key in rows[0] if key != "sequence"]
        summary = {"rows": rows, "mean": {key: sum(float(row[key]) for row in rows) / len(rows) for key in numeric}}
        destination = self.args.results_root / f"{self.config['experiment']['name']}_summary.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[summary] wrote {destination}")

    def write_manifest(self, sequences: list[str]) -> None:
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": str(self.args.config.resolve()),
            "config_contents": self.config,
            "sequences": sequences,
            "stages": self.args.stages,
            "python": sys.version,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "commands": self.commands,
        }
        destination = self.args.results_root / "manifests" / f"{self.config['experiment']['name']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        if self.args.dry_run:
            print(f"[manifest] would write {destination}")
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    args = parser().parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.semidense_root = args.semidense_root.expanduser().resolve() if args.semidense_root else None
    args.results_root = args.results_root.expanduser().resolve()
    config = load_config(args.config.expanduser().resolve())
    sequences = args.sequences or list(config["sequences"])
    unknown = sorted(set(sequences) - set(config["sequences"]))
    if unknown:
        raise SystemExit(f"unknown sequences: {', '.join(unknown)}")
    runner = Runner(args, config)
    for sequence in sequences:
        runner.process(sequence)
    runner.summarize(sequences)
    runner.write_manifest(sequences)


if __name__ == "__main__":
    main()
