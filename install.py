#!/usr/bin/env python3
"""Build UniFusion's local C++/CUDA dependencies in an existing Conda env."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command: list[str], *, cwd: Path, env_name: str, env: dict[str, str]) -> None:
    printable = " ".join(command)
    print(f"\n[install] ({cwd.relative_to(ROOT) or '.'}) $ {printable}", flush=True)
    subprocess.run(
        ["conda", "run", "--no-capture-output", "-n", env_name, *command],
        cwd=cwd,
        env=env,
        check=True,
    )


def find_cuda_home(explicit: str | None) -> Path:
    candidates = [
        explicit,
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda-11.8",
        "/usr/local/cuda",
    ]
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.insert(0, str(Path(nvcc).resolve().parent.parent))
    for candidate in candidates:
        if candidate and (Path(candidate).expanduser() / "bin" / "nvcc").is_file():
            return Path(candidate).expanduser().resolve()
    raise SystemExit(
        "CUDA toolkit not found. Install CUDA 11.8 including nvcc or pass "
        "--cuda-home /path/to/cuda-11.8."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="unifusion")
    parser.add_argument("--cuda-home", default=None)
    parser.add_argument("--skip-mast3r", action="store_true")
    args = parser.parse_args()

    if shutil.which("conda") is None:
        raise SystemExit("conda is not available on PATH")

    cuda_home = find_cuda_home(args.cuda_home)
    env = os.environ.copy()
    env["CUDA_HOME"] = str(cuda_home)
    env["PATH"] = f"{cuda_home / 'bin'}:{env.get('PATH', '')}"
    env["CPATH"] = f"{cuda_home / 'targets/x86_64-linux/include'}:{env.get('CPATH', '')}"
    env["LD_LIBRARY_PATH"] = (
        f"{cuda_home / 'targets/x86_64-linux/lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    )
    env.setdefault("MAX_JOBS", str(max(1, min(8, os.cpu_count() or 1))))

    submodules = ROOT / "2d-gaussian-splatting" / "submodules"
    run(["python", "-m", "pip", "install", "-e", "."], cwd=submodules / "diff-surfel-rasterization", env_name=args.env_name, env=env)
    run(["python", "-m", "pip", "install", "-e", "."], cwd=submodules / "simple-knn", env_name=args.env_name, env=env)
    run(["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"], cwd=submodules / "tetra-triangulation", env_name=args.env_name, env=env)
    tetra = submodules / "tetra-triangulation"
    run(["cmake", "--build", "build", "--parallel", env["MAX_JOBS"]], cwd=tetra, env_name=args.env_name, env=env)
    artifacts = list((tetra / "build").rglob("tetranerf_cpp_extension*.so"))
    if len(artifacts) != 1:
        raise SystemExit(f"expected one tetra extension after build, found {len(artifacts)}")
    extension_dir = tetra / "tetranerf" / "utils" / "extension"
    extension_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifacts[0], extension_dir / artifacts[0].name)
    run(["python", "-m", "pip", "install", "-e", "."], cwd=tetra, env_name=args.env_name, env=env)

    if not args.skip_mast3r:
        asmk_cython = ROOT / "mast3r" / "asmk" / "cython"
        run(["cythonize", "-i", "*.pyx"], cwd=asmk_cython, env_name=args.env_name, env=env)
        run(["python", "-m", "pip", "install", "."], cwd=ROOT / "mast3r" / "asmk", env_name=args.env_name, env=env)
        run(["python", "setup.py", "build_ext", "--inplace"], cwd=ROOT / "mast3r" / "dust3r" / "croco" / "models" / "curope", env_name=args.env_name, env=env)
        run(["python", "-m", "pip", "install", "-r", "requirements.txt"], cwd=ROOT / "mast3r", env_name=args.env_name, env=env)

    print("\n[install] UniFusion native dependencies installed successfully.")


if __name__ == "__main__":
    main()
