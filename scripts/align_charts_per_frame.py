#!/usr/bin/env python3
"""
逐帧执行 charts alignment

对每个帧独立执行 charts alignment，不依赖其他帧
调用现有的 align_charts.py 脚本处理每个帧
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
import json
import re


def natural_sort_key(filename: str):
    """Return a key list so that Path objects are sorted in natural order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", filename)]


def find_source_images_path(frames_dir):
    """
    查找源图像路径
    尝试从 frames_output 的父父目录查找包含图像的文件夹
    """
    frames_dir = Path(frames_dir)
    possible_paths = [
        frames_dir.parent / "frames",      # dataset/frames
        frames_dir.parent.parent / "frames",  # ../frames
        frames_dir.parent / "renamed_images",  # dataset/renamed_images
        frames_dir / ".." / "frames",
        frames_dir / ".." / "renamed_images",
    ]

    # 还可以尝试查找任意包含图像的目录
    for path in frames_dir.parent.iterdir():
        if path.is_dir() and path.name != "frames_output":
            images_dir = path / "images"
            if images_dir.exists():
                possible_paths.append(path)

    for path in possible_paths:
        path = path.resolve()
        if path.exists():
            # 检查是否包含图像
            has_images = False
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                if list(path.glob(f"*{ext}")) or (path / "images").exists():
                    has_images = True
                    break
            if has_images:
                print(f"[INFO] Found source images at: {path}")
                return path

    print("[WARNING] Could not find source images path automatically")
    return None


def get_num_cameras(mast3r_scene_dir):
    """
    从 mast3r_scene 目录获取相机数量
    读取 cameras.json 或检查 sparse 目录
    """
    cameras_json = Path(mast3r_scene_dir) / "cameras.json"
    if cameras_json.exists():
        try:
            with open(cameras_json, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return len(data.get('cameras', {}))
                elif isinstance(data, list):
                    return len(data)
        except:
            pass

    # 尝试从 COLMAP 数据读取
    colmap_dir = Path(mast3r_scene_dir) / "sparse" / "0"
    if colmap_dir.exists():
        # 可以使用 colmap Python API 或直接猜测
        # 这里简单返回一个合理的默认值
        return 4  # 常见的 GoPro 相机数量

    return 4  # 默认值


def align_single_frame(
    frames_dir,
    frame_dir,
    output_dir,
    source_images_path,
    depthanything_checkpoint_dir,
    depthanything_encoder,
    config,
    verbose=False
):
    """
    对单个帧执行 charts alignment

    Args:
        frames_dir: frames_output 根目录
        frame_dir: 单个帧目录路径 (如 frames_output/frame_00000)
        output_dir: 临时输出目录
        source_images_path: 源图像路径
        depthanything_checkpoint_dir: DepthAnything 模型路径
        depthanything_encoder: DepthAnything 编码器类型
        config: 配置名称
        verbose: 是否显示详细输出

    Returns:
        success: 是否成功
    """
    frame_name = frame_dir.name
    mast3r_sfm_dir = frame_dir / "mast3r_sfm"
    mast3r_scene_dir = mast3r_sfm_dir / "sparse" / "0"

    if not mast3r_scene_dir.exists():
        print(f"[WARNING] mast3r_sfm/sparse/0 directory not found: {mast3r_scene_dir}")
        return False

    # 使用每帧的 mast3r_sfm 目录作为 source_images_path（代码会自动查找 images 子目录）
    frame_source_images_path = mast3r_sfm_dir

    # 创建临时输出目录
    temp_output_dir = Path(output_dir) / frame_name
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    # 调用现有的 align_charts.py 脚本
    script_path = Path(__file__).parent / "align_charts.py"
    if not script_path.exists():
        script_path = Path(__file__).parent.parent / "scripts" / "align_charts.py"

    cmd = [
        sys.executable,
        str(script_path),
        "-s", str(mast3r_sfm_dir),
        "-m", str(mast3r_sfm_dir),
        "-o", str(temp_output_dir),
        "-c", config,
        "--depthanythingv2_checkpoint_dir", depthanything_checkpoint_dir,
        "--depthanything_encoder", depthanything_encoder,
    ]

    if verbose:
        print(f"[INFO] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=False
        )

        if result.returncode != 0:
            print(f"[ERROR] Failed to process {frame_name}")
            if not verbose:
                print(result.stdout)
                print(result.stderr)
            return False

        # 检查输出文件是否存在
        # align_charts.py 会保存 charts_data.npz 和 depth/conf 文件
        if verbose:
            print(f"[INFO] Successfully processed {frame_name}")
        return True

    except Exception as e:
        print(f"[ERROR] Exception processing {frame_name}: {e}")
        return False


def organize_results_to_priors_format(
    temp_output_dir,
    priors_output_dir
):
    """
    将临时输出结果组织到 priors 格式

    临时输出格式:
        temp_output_dir/frame_00000/charts_data.npz
        temp_output_dir/frame_00000/depth/
        temp_output_dir/frame_00000/confidence/

    目标格式:
        priors/charts/frame_00001/charts_data.npz
        priors/depths/cam_0000_0001.npy
        priors/confs/cam_0000_0001.npy
    """
    temp_output_dir = Path(temp_output_dir)
    priors_output_dir = Path(priors_output_dir)

    # 创建目标目录
    depths_dir = priors_output_dir / "depths"
    confs_dir = priors_output_dir / "confs"
    charts_dir = priors_output_dir / "charts"

    depths_dir.mkdir(parents=True, exist_ok=True)
    confs_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有帧目录（自然排序）
    frame_dirs = sorted(
        [d for d in temp_output_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")],
        key=lambda p: natural_sort_key(p.name)
    )

    for idx, frame_dir in enumerate(frame_dirs):
        frame_num = idx + 1  # 输出帧号从 1 开始

        # 处理 charts_data.npz
        charts_data_file = frame_dir / "charts_data.npz"
        if charts_data_file.exists():
            target_charts_dir = charts_dir / f"frame_{frame_num:05d}"
            target_charts_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(charts_data_file), str(target_charts_dir / "charts_data.npz"))

        # 处理 depth 文件
        depth_dir = frame_dir / "depth"
        if depth_dir.exists():
            # 提取相机编号
            depth_files = sorted(depth_dir.glob("*.npy"), key=lambda p: natural_sort_key(p.name))
            for depth_file in depth_files:
                # 从文件名提取相机编号
                # 格式可能是: cam_00_cam_0000_0000_depth.npy 或 cam_0000_0000_depth.npy
                match = re.search(r'cam(\d+)', depth_file.name)
                if match:
                    cam_num = int(match.group(1))
                else:
                    cam_num = 0  # 默认

                target_name = f"cam_{cam_num:04d}_{frame_num:04d}.npy"
                shutil.copy(str(depth_file), str(depths_dir / target_name))

        # 处理 confidence 文件
        conf_dir = frame_dir / "confidence"
        if conf_dir.exists():
            conf_files = sorted(conf_dir.glob("*.npy"), key=lambda p: natural_sort_key(p.name))
            for conf_file in conf_files:
                match = re.search(r'cam(\d+)', conf_file.name)
                if match:
                    cam_num = int(match.group(1))
                else:
                    cam_num = 0

                target_name = f"cam_{cam_num:04d}_{frame_num:04d}.npy"
                shutil.copy(str(conf_file), str(confs_dir / target_name))


def main():
    parser = argparse.ArgumentParser(
        description="逐帧执行 charts alignment"
    )

    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="输入 frames_output 目录路径"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="输出 priors 目录路径"
    )
    parser.add_argument(
        "--source-images", type=str, default=None,
        help="已废弃参数（每帧自动使用自己的 mast3r_sfm 目录）"
    )
    parser.add_argument(
        "--config", "-c", type=str, default="default",
        help="配置文件名 (default: default)"
    )
    parser.add_argument(
        "--depthanything-checkpoint-dir", type=str,
        default="./Depth-Anything-V2/checkpoints/",
        help="DepthAnything 模型检查点目录"
    )
    parser.add_argument(
        "--depthanything-encoder", type=str, default="vitl",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="DepthAnything 编码器类型"
    )
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="起始帧索引 (从 0 开始)"
    )
    parser.add_argument(
        "--end-frame", type=int, default=-1,
        help="结束帧索引 (-1 表示处理所有帧)"
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="保留临时输出目录"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="显示详细输出"
    )

    args = parser.parse_args()

    frames_dir = Path(args.input)
    if not frames_dir.exists():
        print(f"[ERROR] Input directory not found: {frames_dir}")
        return

    # 每帧使用自己的 mast3r_sfm 目录，不需要全局源图像路径
    source_images_path = None

    # 获取帧目录列表
    frame_dirs = sorted(
        [d for d in frames_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")],
        key=lambda p: natural_sort_key(p.name)
    )

    if not frame_dirs:
        print(f"[ERROR] No frame directories found in {frames_dir}")
        return

    # 应用帧范围过滤
    if args.end_frame == -1:
        end_frame = len(frame_dirs) - 1
    else:
        end_frame = min(args.end_frame, len(frame_dirs) - 1)

    frame_dirs = frame_dirs[args.start_frame:end_frame + 1]

    print(f"[INFO] Processing {len(frame_dirs)} frames (from {args.start_frame} to {end_frame})")
    print(f"[INFO] Input: {frames_dir}")
    print(f"[INFO] Output: {args.output}")
    print(f"[INFO] Source images: {source_images_path}")

    # 创建临时输出目录
    temp_output_dir = Path(args.output) / "temp_per_frame_alignment"
    priors_output_dir = Path(args.output)

    success_count = 0
    failed_frames = []

    # 逐帧处理
    for frame_dir in tqdm(frame_dirs, desc="Processing frames"):
        success = align_single_frame(
            frames_dir=frames_dir,
            frame_dir=frame_dir,
            output_dir=temp_output_dir,
            source_images_path=source_images_path,
            depthanything_checkpoint_dir=args.depthanything_checkpoint_dir,
            depthanything_encoder=args.depthanything_encoder,
            config=args.config,
            verbose=args.verbose
        )

        if success:
            success_count += 1
        else:
            failed_frames.append(frame_dir.name)

    print(f"\n[INFO] Processed {success_count}/{len(frame_dirs)} frames successfully")

    if failed_frames:
        print(f"[WARNING] Failed frames: {', '.join(failed_frames)}")

    # 组织结果到 priors 格式
    print(f"[INFO] Organizing results to priors format...")
    organize_results_to_priors_format(temp_output_dir, priors_output_dir)

    # 统计输出文件
    depths_dir = priors_output_dir / "depths"
    confs_dir = priors_output_dir / "confs"
    charts_dir = priors_output_dir / "charts"

    num_depths = len(list(depths_dir.glob("*.npy"))) if depths_dir.exists() else 0
    num_confs = len(list(confs_dir.glob("*.npy"))) if confs_dir.exists() else 0
    num_charts = len(list(charts_dir.glob("*"))) if charts_dir.exists() else 0

    print(f"[INFO] Output files:")
    print(f"  - Depths: {num_depths}")
    print(f"  - Confidences: {num_confs}")
    print(f"  - Charts directories: {num_charts}")

    # 删除临时目录
    if not args.keep_temp:
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)
            print(f"[INFO] Cleaned up temporary directory: {temp_output_dir}")

    print(f"[SUCCESS] Done! Results saved to: {priors_output_dir}")


if __name__ == "__main__":
    main()
