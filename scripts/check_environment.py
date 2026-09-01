#!/usr/bin/env python3
"""Check UniFusion's Python, CUDA, native extensions, and checkpoint layout."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "torch": "2.0.1",
    "torchvision": "0.15.2",
    "numpy": "1.26.4",
    "cv2": "4.11.0",
    "open3d": "0.18.0",
    "pytorch3d": "0.7.4",
    "yaml": None,
    "rich": None,
    "diff_surfel_rasterization": None,
    "simple_knn": None,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-gpu", action="store_true")
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args()

    report: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {},
        "errors": [],
    }
    errors: list[str] = report["errors"]  # type: ignore[assignment]
    packages: dict[str, object] = report["packages"]  # type: ignore[assignment]
    for name, expected in REQUIRED.items():
        try:
            module = importlib.import_module(name)
            actual = getattr(module, "__version__", "installed")
            packages[name] = actual
            if expected and str(actual) != expected:
                errors.append(f"{name}: expected {expected}, found {actual}")
        except Exception as exc:  # import failures may come from native ABI mismatch
            packages[name] = f"ERROR: {type(exc).__name__}: {exc}"
            errors.append(f"cannot import {name}")

    try:
        import torch

        report["torch_cuda"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        report["gpu_count"] = torch.cuda.device_count()
        report["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if args.strict_gpu and not torch.cuda.is_available():
            errors.append("CUDA is not available to PyTorch")
        if torch.version.cuda != "11.8":
            errors.append(f"reference build expects PyTorch CUDA 11.8, found {torch.version.cuda}")
    except Exception:
        pass

    nvcc = shutil.which("nvcc")
    report["nvcc"] = nvcc
    if nvcc:
        result = subprocess.run([nvcc, "--version"], text=True, capture_output=True, check=False)
        report["nvcc_version"] = result.stdout.strip().splitlines()[-1]
    elif args.strict_gpu:
        errors.append("nvcc is not available on PATH")

    checkpoints = {
        "depth_anything_v2": ROOT / "Depth-Anything-V2/checkpoints",
        "mast3r": ROOT / "mast3r/checkpoints",
    }
    report["checkpoints"] = {name: {"path": str(path), "exists": path.is_dir(), "files": len(list(path.glob("*"))) if path.is_dir() else 0} for name, path in checkpoints.items()}

    output = json.dumps(report, indent=2)
    print(output)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(output + "\n")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

