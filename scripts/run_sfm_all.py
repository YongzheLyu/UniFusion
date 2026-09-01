"""
多相机多帧场景，仅运行 MASt3R SfM，生成每帧的 `mast3r_sfm` 输出。

输入假设与 `process_multicams_depth.py` 相同：
dataset_root/
  cam00/*.jpg|png
  cam01/*.jpg|png
  ...

输出目录结构：
output_root/
  frame_00000/
    images/          # 复制/软链后的多相机图像
    mast3r_sfm/      # MASt3R 结果（含 cameras.json 等）
  frame_00001/
    ...
"""

import argparse
import gc
import os
import shutil
import sys
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
MAST3R_DIR = ROOT_DIR / "mast3r"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MAST3R_DIR) not in sys.path:
    sys.path.insert(0, str(MAST3R_DIR))

# 支持的图像后缀
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def natural_key(path: Path) -> List[object]:
    """用于自然排序的 key，保证 cam00, cam01... 按数字顺序排序。"""
    import re

    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.stem)]


def collect_multicam_frames(root: Path) -> Tuple[List[Path], List[List[Path]]]:
    """收集每个相机的帧并确保帧数一致。"""
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    camera_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=natural_key)
    if not camera_dirs:
        raise FileNotFoundError(f"No camera subdirectories found in {root}")

    all_frames: List[List[Path]] = []
    num_frames = None
    for cam_dir in camera_dirs:
        frames = sorted([p for p in cam_dir.iterdir() if p.suffix in VALID_SUFFIXES], key=natural_key)
        if not frames:
            raise FileNotFoundError(f"Camera directory {cam_dir} contains no images")
        if num_frames is None:
            num_frames = len(frames)
        elif len(frames) != num_frames:
            raise ValueError(f"Camera {cam_dir.name} has {len(frames)} frames, expected {num_frames}")
        all_frames.append(frames)

    return camera_dirs, all_frames


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


class ModelCache:
    """缓存 MASt3R 模型，避免重复加载。"""

    def __init__(self, weights_path: str, retrieval_model: str, gpu: int = 0, verbose: bool = True):
        self.weights_path = weights_path
        self.retrieval_model = retrieval_model
        self.gpu = gpu
        self.verbose = verbose
        self.model = None
        self.device = None

    def load_model(self):
        """加载 MASt3R 模型（仅执行一次）。"""
        if self.model is not None:
            return self.model, self.device

        if self.verbose:
            print(f"\n[INFO] Loading MASt3R model from {self.weights_path}...")
            print(f"[INFO] Using GPU {self.gpu}")

        torch.cuda.set_device(self.gpu)
        self.device = torch.device(torch.cuda.current_device())

        # Import here to avoid unused imports
        import argparse
        from mast3r.model import AsymmetricMASt3R

        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([argparse.Namespace])
        self.model = AsymmetricMASt3R.from_pretrained(self.weights_path).to(self.device)

        if self.verbose:
            print("[INFO] MASt3R model loaded and cached.")

        return self.model, self.device

    def cleanup(self):
        """清理模型缓存。"""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            if self.verbose:
                print("[INFO] Model cache cleared.")


def run_mast3r_sfm_cached(
    images_dir: Path,
    output_dir: Path,
    config: dict,
    model_cache: ModelCache,
    not_first_frame: bool = True,
    verbose: bool = True,
):
    """调用 MASt3R SfM，使用缓存的模型。"""
    ensure_dir(output_dir)

    # 从缓存获取模型
    model, device = model_cache.load_model()

    # 导入并调用 run_mast3r_sfm 函数
    from run_mast3r_multiframes import run_mast3r_sfm

    result = run_mast3r_sfm(
        scene_path=str(images_dir),
        output_dir=str(output_dir),
        weights_path=model_cache.weights_path,
        retrieval_model=model_cache.retrieval_model,
        min_conf_thr=config.get('min_conf_thr', 0.),
        matching_conf_thr=config.get('matching_conf_thr', 0.),
        n_coarse_iterations=config.get('n_coarse_iterations', 1000),
        n_refinement_iterations=config.get('n_refinement_iterations', 1000),
        TSDF_thresh=config.get('TSDF_thresh', 0.),
        fix_focal=config.get('fix_focal', False),
        fix_principal_point=config.get('fix_principal_point', False),
        fix_rotation=config.get('fix_rotation', False),
        fix_translation=config.get('fix_translation', False),
        n_images=-1,  # Use all images
        use_all_images=True,
        image_idx=None,
        randomize_images=False,
        image_size=config.get('image_size', 512),
        max_window_size=config.get('max_window_size', 20),
        max_refid=config.get('max_refid', 10),
        use_calibrated_poses=config.get('use_calibrated_poses', False),
        save_glb=config.get('save_glb', False),
        gpu=model_cache.gpu,
        output_conf_thr=config.get('output_conf_thr', 0.1),
        align_camera_locations=config.get('align_camera_locations', False),
        not_first_frame=not_first_frame,
        model=model,  # Pass the cached model
        device=device,  # Pass the cached device
        verbose=verbose,
    )

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run MASt3R SfM on multi-camera multi-frame dataset (no DepthAnything).")
    parser.add_argument("dataset_root", type=Path, help="根目录，包含 camXX 子目录")
    parser.add_argument("output_root", type=Path, help="输出目录，将创建 frame_XXXXX 子目录")
    parser.add_argument("--sfm-config", type=str, default="unposed", help="MASt3R SfM 配置名，传递给 run_sfm_multiframes.py 的 -c")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧索引（含）")
    parser.add_argument("--stop-frame", type=int, default=None, help="结束帧索引（不含），默认处理到最后一帧")
    parser.add_argument("--skip-existing", action="store_true", help="若 mast3r_sfm/cameras.json 已存在则跳过该帧")
    parser.add_argument("--keep-working-images", action="store_true", help="保留 frame_x/images 中的中间图像；默认处理完删除")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device number")
    parser.add_argument("--verbose", "-v", action="store_true", help="增加详细输出")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers (multi-threading)")
    return parser.parse_args()


def process_frame(frame_idx, all_frames, num_cameras, output_root, config, model_cache, skip_existing, keep_working_images, verbose):
    """处理单个帧（用于多线程）。"""
    frame_name = f"frame_{frame_idx:05d}"
    frame_root = output_root / frame_name
    images_dir = frame_root / "images"
    mast3r_output_dir = frame_root / "mast3r_sfm"

    ensure_dir(images_dir)

    # 复制/软链当前帧的多相机图像到 images_dir
    for cam_idx in range(num_cameras):
        src = all_frames[cam_idx][frame_idx]
        dst = images_dir / f"cam{cam_idx:02d}_{src.name}"
        if dst.exists():
            dst.unlink()
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)

    # 是否跳过已有结果
    if skip_existing and (mast3r_output_dir / "cameras.json").exists():
        return frame_idx, "skipped", None

    if verbose:
        print(f"\n[INFO] Processing {frame_name}...")

    not_first_frame = frame_idx != 0
    try:
        result = run_mast3r_sfm_cached(
            images_dir,
            mast3r_output_dir,
            config,
            model_cache,
            not_first_frame=not_first_frame,
            verbose=verbose,
        )
        return frame_idx, "done", result
    except Exception as e:
        return frame_idx, "failed", str(e)
    finally:
        # 清理中间 images
        if not keep_working_images:
            shutil.rmtree(images_dir, ignore_errors=True)


def main():
    args = parse_args()

    camera_dirs, all_frames = collect_multicam_frames(args.dataset_root)
    num_cameras = len(camera_dirs)
    num_frames = len(all_frames[0])
    print(f"[INFO] Cameras: {num_cameras}, Frames: {num_frames}")
    print(f"[INFO] Using {args.num_workers} parallel worker(s)")

    start = max(args.start_frame, 0)
    stop = num_frames if args.stop_frame is None else min(args.stop_frame, num_frames)
    if start >= stop:
        raise ValueError("Invalid frame range")

    ensure_dir(args.output_root)

    # Load config
    import yaml
    config_path = os.path.join('configs/mast3r', args.sfm_config + '.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize model cache (load model once)
    model_cache = ModelCache(
        weights_path=config['weights_path'],
        retrieval_model=config['retrieval_model'],
        gpu=args.gpu,
        verbose=args.verbose,
    )

    try:
        # Pre-load the model
        model_cache.load_model()

        frame_range = list(range(start, stop))

        # 第一帧必须先处理（因为后续帧需要第一帧的 pose）
        first_frame_idx = frame_range[0] if frame_range and frame_range[0] == 0 else None
        remaining_frames = frame_range[1:] if first_frame_idx is not None and len(frame_range) > 1 else frame_range

        # 处理第一帧
        if first_frame_idx is not None:
            if args.verbose:
                print(f"\n[INFO] Processing first frame (frame_{first_frame_idx:05d}) first...")

            first_frame_name = f"frame_{first_frame_idx:05d}"
            frame_root = args.output_root / first_frame_name
            images_dir = frame_root / "images"
            mast3r_output_dir = frame_root / "mast3r_sfm"

            ensure_dir(images_dir)

            # 复制图像
            for cam_idx in range(num_cameras):
                src = all_frames[cam_idx][first_frame_idx]
                dst = images_dir / f"cam{cam_idx:02d}_{src.name}"
                if dst.exists():
                    dst.unlink()
                try:
                    os.symlink(src.resolve(), dst)
                except OSError:
                    shutil.copy2(src, dst)

            # 检查是否跳过
            should_skip = args.skip_existing and (mast3r_output_dir / "cameras.json").exists()

            if should_skip:
                if args.verbose:
                    print(f"[INFO] First frame {first_frame_name} already exists, skipping.")
            else:
                run_mast3r_sfm_cached(
                    images_dir,
                    mast3r_output_dir,
                    config,
                    model_cache,
                    not_first_frame=False,
                    verbose=args.verbose,
                )

            # 清理中间 images
            if not args.keep_working_images:
                shutil.rmtree(images_dir, ignore_errors=True)

        # 多线程处理剩余帧
        if remaining_frames:
            if args.num_workers == 1:
                # 单线程处理
                pbar = tqdm(remaining_frames, desc="Processing frames") if not args.verbose else remaining_frames
                for frame_idx in pbar:
                    status, result = process_frame(
                        frame_idx, all_frames, num_cameras, args.output_root,
                        config, model_cache, args.skip_existing, args.keep_working_images, args.verbose
                    )[1:]
                    if not args.verbose:
                        pbar.set_postfix({"status": status or "done", "frame": f"frame_{frame_idx:05d}"})
                    if status == "failed" and not args.verbose:
                        print(f"[WARNING] Frame {frame_idx} failed: {result}")
            else:
                # 多线程处理
                if args.verbose:
                    print(f"\n[INFO] Processing remaining {len(remaining_frames)} frames with {args.num_workers} workers...")

                with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                    # 提交所有任务
                    future_to_frame = {
                        executor.submit(
                            process_frame,
                            frame_idx, all_frames, num_cameras, args.output_root,
                            config, model_cache, args.skip_existing, args.keep_working_images, args.verbose
                        ): frame_idx for frame_idx in remaining_frames
                    }

                    # 使用进度条跟踪
                    pbar = tqdm(total=len(remaining_frames), desc="Processing frames") if not args.verbose else None

                    completed = 0
                    for future in as_completed(future_to_frame):
                        frame_idx, status, result = future.result()
                        completed += 1

                        if not args.verbose:
                            pbar.update(1)
                            pbar.set_postfix({
                                "status": status or "done",
                                "frame": f"frame_{frame_idx:05d}",
                                "progress": f"{completed}/{len(remaining_frames)}"
                            })

                        if status == "failed" and not args.verbose:
                            print(f"[WARNING] Frame {frame_idx} failed: {result}")
                        elif status == "skipped" and args.verbose:
                            print(f"[INFO] Frame frame_{frame_idx:05d} already exists, skipped.")

                    if not args.verbose:
                        pbar.close()

        if first_frame_idx is not None and remaining_frames:
            total_processed = len(frame_range)
        elif first_frame_idx is not None:
            total_processed = 1
        else:
            total_processed = len(remaining_frames)

        print(f"\n[INFO] Total frames processed: {total_processed}")

    finally:
        # 清理模型缓存
        model_cache.cleanup()

    print(f"[INFO] Output saved to: {args.output_root}")


if __name__ == "__main__":
    main()
